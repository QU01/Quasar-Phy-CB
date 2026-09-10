# M4 — combustor completo + producto: plan de arranque para la próxima sesión

> Escrito el **2026-08-02**, al cierre de la campaña que dejó M1+M2+M3
> HECHOS y verificados. Este documento contiene TODO lo que hace falta
> para arrancar M4 sin releer la historia: qué hay, cómo se usa, qué
> construir, en qué orden, con qué criterios y con qué trampas conocidas.
> El estado MEDIDO vive en `ESTADO.md` (fuente de verdad); el plan
> original de la wave (ya no en el repo; su contenido medido vive en
> `ESTADO.md` §12 y en `docs/Quasar_PhyCB_Science.md` §11.3); las reglas no
> negociables en `docs/IMPLEMENTATION_CONTRACT.md` §10.

---

## 0. Punto de partida (lo que YA hay, verificado)

| pieza | módulo | verificación |
|---|---|---|
| Solver NS + IB ghost-cell sobre SDF, CPU y GPU (CuPy) | `solver.py`, `mesh.py`, `backend.py` | V1–V3 |
| k-ε realizable + wall functions (dos escalas, sobre \|φ\| y contornos) + corrección de curvatura | `turbulence.py` | V4 (−6.6 %), V5 (−12.3 %), canal Dean (−3.7 %) |
| Energía low-Mach + ρ(T) + q_T a la continuidad + escalar pasivo | `energy.py` | M2c (analítico ±1 %), JICF (+19.7 % vs K&G) |
| Especies + min(EDM, Arrhenius) + liberación de calor + **encendido** | `chem.py` | V6 (−2.3 %), V6c (enciende desde frío), V6d CH4 (−8.9 %), V7 (−0.3 %) |
| Mecanismos verificados contra Cantera/Pyrometheus | `chem.MECHANISMS` | propano y metano; jet-A PROVISIONAL (sin termo NASA) |
| Post de V&V (fuerza CV, Strouhal, y⁺) | `post.py` | V3 |
| Post de PRODUCTO (Δp/p, T4/PF, mapa pared, penetraciones, NOx) | `post.py` | **ESQUELETO — esto ES M4** |

`solve()` devuelve `None` a propósito; el sistema degrada etiquetado.
Cambiarlo es la tarea 4.5, la ÚLTIMA (cuando haya producto de verdad).

## 1. Cómo se monta un caso reactivo turbulento (la receta que ya funciona)

Orden de enganche OBLIGADO (lo garantiza `SpeciesTransport.attach`, que
registra la energía él mismo):

```python
from cfd.ibrans import (CartesianMesh, IBRANSSolver, SolverConfig,
                        make_bcs, KEpsilonRealizable, EnergyLowMach,
                        PassiveScalar, chem)
from cfd.ibrans.turbulence import TurbulenceConfig
from cfd.ibrans.energy import EnergyConfig

m = CartesianMesh(lo, hi, dx); m.set_sdf(phi)          # phi del §3
s = IBRANSSolver(m, SolverConfig(...), bcs)
tur = KEpsilonRealizable(m, TurbulenceConfig(...)).attach(s)
sp  = chem.SpeciesTransport(m, chem.ChemConfig(fuel=..., ignition=plan))
en  = EnergyLowMach(m, EnergyConfig(p_thermo_Pa=P3, T_inlet_K=T3,
                                    T_wall_ib_K=None))   # None = adiabática
sp.attach(s, turbulence=tur, energy=en, y_init=..., ignition=True)
info = s.run()          # los modelos avanzan solos (solver.physics)
```

Detalles que cuestan horas si se olvidan (todos MEDIDOS esta campaña):

* **BCs con perfiles por celda**: cualquier componente de `u` y cualquier
  escalar del spec (`T=`, `Z=`, `k=`, `eps=`, `Y_FUEL=`, …) puede ser un
  array del plano de la cara (tamaño `shape_tot` transversal, es decir CON
  fantasmas). Así se hacen ranuras/orificios en una cara. Ver
  `run_v5_jcf.py` como ejemplo completo.
* **Los chorros entran con SU turbulencia**: k/ε de orificio (I ≈ 10 %,
  ℓ = 0.07·d, escalando con u_chorro), no la del plenum. Sin esto el JICF
  sobre-penetraba +26 % (ESTADO §11.4). Aplica a TODAS las filas de
  orificios de M4.
* **Tras sobrescribir campos a mano, llamar `s.apply_bc()`** y no fiarse
  de la primera evaluación de residuales (el driver ya se protege, k>0).
* **Casos estacionarios: `mach_art` 0.2–0.3** (no cambia el estado final,
  acelera ×2–3). No estacionarios: 0.1.
* Los casos turbulentos **estancan el residual en 1e-4–5e-3** (ciclo
  límite del acoplamiento segregado): el criterio de parada útil es
  `max_iter`/`t_end` + medida INTEGRAL promediada (así se midieron V4, V5
  y el JICF). Para la llama, `sp.anchored()` es el criterio del plan.
* GPU: rentable desde ~100k celdas (`backend.GPU_MIN_CELLS`); el caso M4
  (0.5–4 M celdas a dx = 0.5–2 mm) es territorio GPU. FP64 default; para
  M4 (criterio integral, no residual) FP32 es legítimo y ×1.8 —
  cambiarlo exige rehacer la V&V correspondiente (backend.py docstring).
* Coste medido: reactivo laminar ≈ ×4.5 el paso de M1; turbulento frío
  ≈ ×2.2. Presupuesto M4 (500k celdas × ~15–30k pasos, GPU, reactivo):
  extrapolación honesta **30–60 min/caso** — dentro del objetivo <30 min
  sólo si FP32 + malla contenida; medirlo es parte de 4.6.

## 2. Tareas de M4, en orden, con sus criterios

### 4.1 SDF completo del combustor (el bloqueante de todo lo demás)

Hoy `mesh.sdf_from_contract()` da SOLO el anillo del liner
(`liner.profile`) con `meta["missing"] = [swirler, hole_rows, efusion,
domo]`. Hace falta el volumen fluido completo desde el contrato
`phycb-combustor-1` (lo escribe `geometry_generator`; campos en
`docs/IMPLEMENTATION_CONTRACT.md` §4):

* **Orificios de dilución/primarios**: `contract["hole_rows"]` =
  `list[dict(zone, z_mm, r_mm, n, d_mm, Cd, J, Y_mm, Y_over_DL,
  A_row_m2)]`. Cada fila = n cilindros fluidos que atraviesan la pared
  del liner: `sdf_union` del anillo con cilindros radiales en
  `z = z_mm`, ángulos `2πk/n`. ⚠️ Regla de la wave: **resueltos si
  d/dx ≥ 4; si no, fila = fuente de momento/masa calibrada por C_d**
  (comprobar por fila y DECLARAR cuáles van resueltas en el summary).
* **Domo/cúpula**: cierre del perfil en z = 0 con el paso del swirler.
* **Swirler**: opción A (v1, recomendada): NO resolver álabes — anillo de
  entrada en el domo entre `d_hub_mm` y `d_tip_mm` con velocidad
  tangencial impuesta u_θ = u_ax·tan(β) vía perfil por celda en el parche
  (claves REALES del contrato, verificadas: `swirler = {type, n_vanes,
  beta_deg, SN, d_hub_mm, d_tip_mm}`). Comprobar que el número de swirl
  resultante reproduce `swirler.SN` (integral de flujos de momento
  angular/axial en el primer plano — cordura barata). Opción B (upgrade):
  álabes en el SDF (necesita dx ≪ 1 mm local). La A da el swirl correcto,
  que es lo que el PF necesita.
* **Efusión**: NUNCA resuelta (Ø0.3–0.5 mm): pared porosa = banda IB con
  transpiración distribuida. Implementación sugerida: en las celdas de
  banda de la pared exterior del liner, fuente de masa/momento normal
  con el C_d·A del contrato. ⚠️ **T49** (`test_phycb.py`): la
  conductancia C_d·A de la efusión debe ser IDÉNTICA entre
  `physics_core`, `geometry_generator`, `crn_core` y el CFD — hay un
  test de contrato que lo vigila; usar `effusion.wall_area_mm2` y el
  C_d del contrato, no constantes propias.
* El `.vdb` de PicoGK sigue sin binding (T.3): el camino v1 es el SDF
  analítico completo. `sample_vdb()` ya degrada con motivo.

**Criterio de cierre 4.1**: `cold_smoke` sobre el SDF completo con splits
de aire por fila (flujo por cada grupo de orificios / total) coherentes
con los `A_row·Cd` del contrato a ±20 % en frío.

### 4.2 Inyección de combustible

Contrato §9: el gas va a +Z; el inyector está en el domo. v1 = fuente de
masa/especies en la zona del prefilmer/vaporizador (spray NO resuelto,
declarado): región esférica/anular con `Y_FUEL` impuesto o fuente
`source_K_s`-style. Evaporación instantánea declarada, o retardo τ_e de
Rizk & Mongia como retraso de liberación (correlación ya en
`physics_core`). El gancho de fuente volumétrica de energía ya existe
(`EnergyLowMach.source_K_s`); para masa/especies añadir el equivalente en
`SpeciesTransport` (patrón idéntico, ~20 líneas).

### 4.3 Post-proceso de producto (`post.py`, firmas CONGELADAS)

Rellenar los esqueletos (todos reciben `solver` y encuentran los modelos
en `solver.physics`):

* `dp_over_p`: presión total media pesada por caudal entre el parche de
  entrada y el plano 4 (media de p + ½ρu² por plano), /P3.
* `T4_and_pattern_factor`: en el plano 4 (z máx del fluido):
  T4_media (pesada por ρu), T4_máx, PF = (T4max−T4)/(T4−T3),
  RTDF con el perfil radial.
* `adiabatic_wall_map`: T de las celdas de banda IB del liner,
  ETIQUETA fija: ranking, no absoluto (`HONESTY` ya escrito).
* `penetrations`: trayectoria de máximo de exceso de T por fila de
  orificios — la maquinaria es EXACTAMENTE la del JICF
  (`run_v5_jcf.py`), reutilizarla. Y_over_d_measured por fila.
* `nox_post_crn`: clustering k-means en (T, Z) → reactores → `crn_core`
  (Ciencia §9.4). `crn_core` ya existe y funciona (WAVE2).
* `combustor_product`: agrega TODO en las claves del record (contrato
  §4.4): `PF_cfd`, `T4_cfd_K`, `dp_over_p_cfd`, `yplus_hist`,
  `cfd_mesh`, `splits_measured`, `Y_over_d_measured`, `V_zones_measured`.

### 4.4 Encendido en el combustor (el RIESGO #1 de verdad)

El plan (parche + rampa + sub-relajación + `anchored()`) está verificado
en 1D (V6c) pero el combustor real tiene swirl, mezcla imperfecta y
paredes. Parche en la PZ (`IgnitionPlan.patch_center_m` en la zona
primaria del contrato). **La salida de emergencia mixed-is-burned tiene
fecha** (semana 15 del plan de la wave): si el anclaje no es robusto,
degradar a `chem.equilibrium_T_of_Z` sobre el escalar Z (todo escrito,
`PassiveScalar` + esa función) — conserva PF/penetraciones/puntos
calientes ≈ 80 % del valor. NO alargar indefinidamente: decidir y anotar.

### 4.5 `solve()` de verdad + calibración

* Firma que llama `physics_core._try_higher(theta, rec, fid)`:
  `cfd.ibrans.solve(theta, rec) -> dict | None`. El dict se hace
  `rec.update(sol)` y `source="ibrans_L2A"` — devolver SOLO las claves
  del §4.4 + `fidelity_note`/`honesty`. Construir el contrato de
  geometría desde θ con `geometry_generator` (igual que hace
  `cold_smoke`), correr el caso M4, `combustor_product`, y devolver.
  **Nunca lanza**; presupuesto de tiempo con `timeout_s` y degradación
  etiquetada si no converge/enciende (p. ej.
  `ibrans_L2A(no_ignition→crn_L1)` — el sufijo explica el porqué).
* `physics_core.register_hifi_pair(theta, y_hifi)` ya existe (corrección
  afín con datos de alta fidelidad): llamar tras cada `solve()` bueno con
  `y_hifi = {clave: valor}` de las claves compartidas. Ojo: evalúa el
  mismo θ SIN calibrar internamente — no hay que hacer nada especial.
* Actualizar entonces: `MILESTONES["M4"]`, `UNAVAILABLE_REASON` (ya no
  aplica), test M0b (que HOY asegura que degrada — pasará a asegurar que
  produce), `milestone_status()`.

### 4.6 Verificación (los tests que cierran la wave)

* **V9 PRECCINSTA** (quemador swirl DLR, base pública): CON pérdida de
  calor en paredes — `EnergyConfig.T_wall_ib_K` ya existe (isoterma).
  Expectativa DECLARADA por la wave: topología y posición de llama
  CUALITATIVAS (forma M vs V), no perfiles. Si el swirl sale muy mal con
  k-ε+curvatura, el plan B declarado es SST (waves §7) — no inventar
  nada exótico.
* **Caso ancla KJ66** (`pc.THETA_REF`): Δp/p vs red 1D ±20 %; T4 vs
  balance ±5 %; end-to-end desde `--fidelity L2A` en <30 min GPU.
* **Test de RANKING** (el entregable de precisión de L2a): dos variantes
  del ancla (dilución 6 vs 10 orificios) → el PF relativo ordena igual
  que la correlación de L0/L1. Es el criterio de aceptación #3 de la
  wave y lo que justifica la etiqueta de honestidad.
* Sugerencia de IDs (espacio de nombres del PROYECTO, comprobar libre en
  `test_phycb.py` Y en `cfd/tests/`): archivo `test_ibrans_v9.py` con
  M4a (splits fríos vs C_d·A), M4b (encendido del ancla ancla),
  V9a/V9b, M4c (ranking). V8 (Sandia D) puede caer aquí también: sólo
  le falta el piloto anular + caso 3D (mecanismo CH4 YA verificado).

## 3. Trampas conocidas (todas medidas, no teóricas)

1. La malla del combustor a dx = 2 mm NO resuelve orificios de d < 8 mm
   (regla d/dx ≥ 4) → la mayoría de filas van como fuentes C_d en v1.
2. `sdf_annulus_profile` es aproximado en tramos inclinados (O(sinθ),
   sólo importa en la banda IB) y hay que generar el SDF con
   `cap_ends=False` + dominio recortado en Z (si no, el volumen queda
   sellado y el residual da 0 — falso verde ya cazado una vez).
3. Efusión: la pared exterior SOLO (`effusion.walls = ['outer']` en v1).
4. CH4 laminar necesita dx ≤ 100 µm (V6d); en el combustor turbulento
   manda EDM y no aplica, pero cualquier sub-caso laminar sí.
5. El limitador de T de la energía lo fija `SpeciesTransport.attach` a
   [T3−50, 1.05·T_ad]: si el caso usa T3 real del ciclo (600–900 K),
   pasar `EnergyConfig(T_inlet_K=T3)` ANTES de attach.
6. `arrhenius_rate` v1 no soporta exponente de temperatura b≠0 (el paso
   2 del BFER lo lleva): sólo paso 1 efectivo.
7. Los `_S_MASS`/`product_split` difieren <1 % — usar `split["s"]` del
   modelo para consistencia interna (M3d lo vigila).
8. VRAM no es riesgo (guardián `fits_in_vram`; 46 M celdas FP32 caben);
   el riesgo de malla es el swirler → AMR de 2 niveles es upgrade, no v1.

## 4. Comandos de arranque

```bash
cd C:/Users/nickv/projects/Proyectos/Phy-CB
python cfd/tests/test_ibrans_v1_v3.py            # M1 verde (~2 min)
python cfd/tests/test_ibrans_v4_v5.py            # M2 verde (~4 min)
python cfd/tests/test_ibrans_v6_v8.py            # M3 verde (~2 min)
```

Humo de la cadena real θ → contrato → SDF → solver (VERIFICADO
2026-08-02: 65 348 celdas, 4 filas de orificios, swirler en el contrato):

```bash
python -c "import sys; [s.reconfigure(encoding='utf-8') for s in (sys.stdout, sys.stderr)]; import physics_core as pc, geometry_generator as gg, cfd.ibrans as ib; rec = pc.evaluate(pc.THETA_REF); c = gg.build_contract(pc.THETA_REF, rec); r = ib.cold_smoke(c, max_iter=200); print(r['status'], r['cells'], len(c['hole_rows']), 'filas')"
```

El contrato se construye con `geometry_generator.build_contract(theta,
record)` — el `record` sale de `pc.evaluate(theta)` (así lo hará también
`solve()` en 4.5). Ojo con la consola Windows cp1252: reconfigurar
stdout a UTF-8 antes de importar (los tests ya lo hacen solos).
