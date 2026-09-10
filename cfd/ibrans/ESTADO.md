# Estado real de WAVE6 — solver L2a nativo (IB-RANS sobre SDF)

> M1 medido el **2026-07-31**; **M2 y M3 medidos el 2026-08-02** en la
> máquina del proyecto: Windows 11, Python 3.13.7, NumPy 2.4.6, **NVIDIA
> GeForce RTX 3070 Ti (8 GiB, cc 8.6) con CuPy 14.1.1**, Cantera 3.2.0 y
> **Pyrometheus (instalado el 2026-08-02 con autorización expresa)**; sin
> Warp, sin JAX, sin XLB (§5.0). Todo número de este documento está
> **medido aquí**, no estimado. Lo que no se pudo medir, pone que no se
> pudo y por qué.
>
> ## ⚠️ Corrección respecto de la primera versión de este documento
>
> La primera versión decía «**sin GPU**» y concluía que el objetivo de
> rendimiento de la wave era inalcanzable en esta máquina. **La premisa era
> falsa**: la máquina tiene GPU y CuPy instalado. Las medidas en CPU eran
> correctas y se conservan; la conclusión no lo era y se rehace entera
> (§5). Se ha añadido un **backend CuPy** al solver y **el objetivo de
> rendimiento pasa a cumplirse** (§5.3). Lo que NO cambia: M3/M4 siguen
> siendo esqueleto, `solve()` sigue devolviendo `None` a propósito, y
> **Phy-CB sigue sin predecir temperatura de metal cuantitativa en ningún
> nivel de fidelidad** — eso es una limitación de modelo, no de cómputo, y
> tener GPU no la toca.

---

## 1. Resumen ejecutivo (si sólo lees una sección, esta)

| Hito | Contenido | Estado | Verificación |
|---|---|---|---|
| **M1** | solver frío laminar + frontera inmersa **+ backend GPU** | **HECHO** | **V1 ✅ · V2 ✅ · V3 ✅ (en CPU y en GPU)** |
| **M2** | k-ε realizable + wall functions + curvatura + energía low-Mach + ρ(T) + escalar pasivo | **HECHO** (§10) | **M2a–M2f ✅ · V4 ✅ · V5 ✅** |
| **M3** | especies + EDM + **arranque de llama** | **HECHO** (§11) | **M3a–M3d ✅ · V6 ✅ · V6c ✅ · V7 ✅ · V8 PENDIENTE (declarado)** |
| M4 | combustor completo + producto (PF, T4, Δp/p) | **IMPLEMENTADO, no cerrado** (§12) | encendido ✅ · T4 ❌ · Δp/p no comparable · M4c/V9 pendientes |

La escalera de M2 (`cfd/tests/test_ibrans_v4_v5.py`, números en §10):

| | caso | criterio | medido (CPU) | |
|---|---|---|---|---|
| M2a | decaimiento homogéneo k-ε | analítico exacto, <1 % | **0.14 %** | ✅ |
| M2b | canal turbulento, Cf vs Dean (1978) | ±15 % | **−3.7 %** (y⁺ = 99) | ✅ |
| M2c | expansión térmica low-Mach | analítico, ±3 % | **+0.94 % (T), +0.90 % (u)** | ✅ |
| M2d | C_μ realizable, f_r, Spalding | identidades | máquina | ✅ |
| M2e | escalar pasivo | conservación exacta | **<1e-6** | ✅ |
| **V4** | **backward-facing step** (Vogel & Eaton, Re 28k) | x_r/h = 6.67 **±15 %** | **6.23 (−6.6 %)** | ✅ |
| **V5** | chorro turbulento plano (Pope) | S = 0.10 **±15 %** | **0.0877 (−12.3 %)** | ✅ |

La escalera de M3 (`cfd/tests/test_ibrans_v6_v8.py`, números en §11):

| | caso | criterio | medido (CPU) | |
|---|---|---|---|---|
| M3a | cinética molar SI vs Cantera (mismo estado) | <0.1 % | **máquina** | ✅ |
| M3b | maquinaria de encendido (rampa/parche/anclaje) | exacta | exacta | ✅ |
| M3c | Pyrometheus == tabla == Cantera | <0.1 % | **idénticos (5 cifras)** | ✅ |
| M3d | reactor 0D: estequiometría + energía | conservación | **<2 % / <5 %** | ✅ |
| **V6** | llama premezclada 1D propano, S_L vs Cantera `FreeFlame` **mismo mecanismo** | ±10 % | **−2.3 %** (0.2476 vs 0.2534 m/s) | ✅ |
| **V6c** | **ENCENDIDO desde frío** (riesgo #1) | ancla + misma S_L ±5 % | **ancla; ΔS_L < 5 %** | ✅ |
| V6d | llama CH4 (2S_CH4_BFER, el mecanismo de V8) | ±10 % | **−8.9 %** (0.386 vs 0.424; dx ≤ 100 µm obligado) | ✅ |
| V7 | contradifusión, T_max vs Cantera | ±10 % | **−0.3 %** (2144 vs 2150 K) | ✅ |
| V8 | Sandia Flame D | ±15 % cualitativo | **PENDIENTE (declarado)** | — |

**`cfd.ibrans.solve(theta, rec)` SIGUE devolviendo `None` a propósito.**
Con M2+M3 hay turbulencia, temperatura y combustión VERIFICADAS en casos
canónicos, pero sin M4 no existe el CASO del combustor (SDF completo,
inyección, efusión) ni su post-proceso de producto. El sistema degrada con
`source="ibrans_L2A(unavailable→crn_L1)"`, que sigue siendo la verdad.

La escalera V&V que WAVE6 exigía para M1 **pasa entera** con los umbrales de
Ciencia §13.1, sin aflojar ninguno:

| | caso | criterio | medido (CPU) | |
|---|---|---|---|---|
| V1a | Poiseuille laminar, paredes inmersas | < 2 % | **0.17 %** | ✅ |
| V1b | Couette, pared inmersa móvil | < 2 % | **0.04 %** | ✅ |
| V2 | cavidad Ghia Re = 100 (32²) | ±5 % | **0.44 % / 0.70 %** | ✅ |
| V2 | cavidad Ghia Re = 1000 (128²) | ±5 % | **0.48 % / 0.99 %** | ✅ |
| V3 | cilindro IB, Cd (D/10) | ±10 % | **+3.4 %** (1.396 vs 1.35) | ✅ |
| V3 | cilindro IB, Strouhal (D/10) | ±10 % | **+0.8 %** (0.1664 vs 0.165) | ✅ |

Los mismos casos **corridos a la fuerza en GPU pasan los mismos umbrales**;
los números están en §5.4.

**`cfd.ibrans.solve(theta, rec)` devuelve `None` a propósito.** Con M2–M4 sin
hacer no hay campo de temperatura y por tanto no hay producto L2a. El
sistema degrada con `source="ibrans_L2A(unavailable→crn_L1)"`, que es la
verdad. Está verificado por un test (`M0b`), no prometido. **Tener GPU no
cambia esto**: lo que falta es física, no velocidad.

**El objetivo de RENDIMIENTO de M1 ahora SE CUMPLE.** El objetivo de la wave
(500 k celdas × 10 k iteraciones en 5–40 min en una RTX 4070+) se mide en
esta RTX 3070 Ti en **9.8 min con la GPU libre** y **16.8 min con la GPU
compartida al 99 % por otro trabajo** (1.17e-7 y 2.01e-7 s/celda/paso, FP64,
flujo frío) — dentro de la banda en los dos casos. En NumPy/CPU son **2.6–
2.8 h**. Detalle, método y letra pequeña en §5.

Límite duro heredado del proyecto (contrato §12), que se repite porque toca:
**Phy-CB no predice temperatura de metal cuantitativa en ningún nivel de
fidelidad.** L2b (OpenFOAM), que iba a darla, está fuera de alcance. El mapa
de pared que dará M4 será **adiabático y para ranking**, con esa etiqueta
pegada (`post.HONESTY`).

---

## 2. Qué promete y qué NO promete cada archivo

| archivo | promete | NO promete |
|---|---|---|
| `backend.py` | elegir backend con degradación etiquetada; **backend CuPy real sobre GPU**; ficha de GPU; guardián de VRAM; benchmark reproducible con sincronización | Warp/JAX: no instalados, **no se instalan** (regla del proyecto) y **no tienen kernel escrito** — pedirlos degrada a NumPy con etiqueta |
| `mesh.py` | malla cartesiana, muestreo de SDF, máscaras fluido/sólido/fantasma, normales ∇φ, **distancia a pared = \|φ\| exacta**, fantasmas con imagen lineal, parches desde el contrato, **estructuras del IB residentes en GPU** (`to_backend`) | leer `.vdb` de PicoGK (sin binding de OpenVDB → degrada con motivo); cut-cell; AMR; SDF exacto de troncos de cono |
| `solver.py` | NS incompresible con IB: FV 2.º orden MUSCL+TVD, RK SSP, Rhie–Chow, driver acotado, checkpoint, timeout; **el MISMO kernel en NumPy y en CuPy**; ganchos M2 **opt-in** (`physics`, `nu_field`, `rho_field`, `div_source`, `face_velocities`, `advect_diffuse_scalar`, perfiles por celda en inlet y en escalares); sin modelos registrados es EXACTAMENTE el solver de M1 (test M2f) | especies, combustión — **nada que queme** |
| `turbulence.py` | **k-ε realizable ACOPLADO** (transporte MUSCL+TVD con la uf de Rhie–Chow, fuentes point-implicit, C_μ variable con A_s completo, C₁ = max(0.43, η/(η+5))), **wall functions de dos escalas** (u* de k + ley de Spalding invertida) sobre \|φ\| y sobre paredes del dominio, **corrección de curvatura Spalart–Shur simplificada**; `ystar_stats()` para el reporte. `available()` → `True` | el término r̃ de Spalart–Shur (transportar DS/Dt); low-Re/resolved wall (es un modelo de wall functions); precisión térmica de pared sobre IB (riesgo declarado Ciencia §11.3) |
| `energy.py` | **energía low-Mach** (DT/Dt = ∇·(α_eff∇T) + S_T), **ρ(T) = p₀/R·T** acoplada al momento y a la continuidad (∇·u = q_T), fuente volumétrica para M3, pared IB adiabática o isoterma; **escalar pasivo Z** con D_eff = ν/Sc + ν_t/Sc_t | radiación, CHT, μ(T)/λ(T), entalpía de formación (M3) — y **NUNCA temperatura de metal** |
| `chem.py` | **transporte de Y_k acoplado** (`SpeciesTransport`: MUSCL + D_eff + consumo point-implicit CONSISTENTE — productos y calor del consumo realizado), cierre **min(EDM, Arrhenius)**, cinética molar SI verificada contra Cantera y Pyrometheus (mismo mecanismo, `cantera_yaml`), liberación de calor por `source_K_s` (arrastra la dilatación), **plan de encendido IMPLEMENTADO** (parche + rampa + limitador + sub-relajación + criterio de anclaje), mixed-is-burned usable. `available()` → `True` | el paso 2 (CO) de los esquemas 2 pasos (declarado: sin CO ni η_comb por CO); c_p variable (constante calibrado a T_ad); V8 Sandia D; V6 del jet-A (sin termo NASA de C12H23); **producto de combustor** (M4) |
| `post.py` | fuerza por volumen de control (**verificada**, §4.1), Strouhal por FFT, tensión de pared, histograma de y⁺, cortes de campo; **producto M4 IMPLEMENTADO** (Δp/p pesado por caudal, T4/PF/RTDF en el plano 4, mapa de pared ADIABÁTICA para ranking, penetraciones por defecto de T, NOx por clustering (T,Z)→`crn_core`); funciona con campos de CPU o de GPU | valores absolutos de T de pared (etiqueta fija de RANKING); NOx sin Cantera (degrada con motivo) |
| `case.py` | **el CASO del combustor (M4)**: SDF completo, perfiles de entrada por celda (swirler con SN impuesto + anillo), transpiración C_d·A autocalibrada al área discreta (filas + efusión, T49), inyección de combustible en la PZ, encendido en la bujía del contrato, `measure_splits` por volumen de control axial, `swirl_number_measured` | resolver los álabes del swirler (opción B declarada); anillo liner↔carcasa cuando la holgura < 2·dx (modo «sin anillo» DECLARADO — es el caso del KJ66); k/ε de chorro en filas transpiradas (límite declarado) |
| `__init__.py` | **`solve()` de PRODUCTO** (contrato → caso → encendido → §4.4 + `register_hifi_pair`), degradación etiquetada vía `DEGRADE_TAG`, mixed-is-burned declarado si no ancla, `cold_smoke()`, `milestone_status()` | producto desde un caso sin establecer (< `L2A_MIN_ITERS` ⇒ `None` etiquetado); T de metal (nunca) |

---

## 3. Escalera V&V — resultados REALES

Fichero: `cfd/tests/test_ibrans_v1_v3.py` (9 tests, doble modo tabla/pytest).
Scripts manuales: `cfd/tests/run_v2_re1000.py`, `cfd/tests/run_v3_cylinder.py`.

```
Escalera V&V del solver L2a (WAVE6 M1) · slow=False · backend: cupy/float64
                                        -> numpy si celdas < 100000
  [OK  ] V1a   Poiseuille laminar, paredes inmersas          11.6 s
  [OK  ] V1b   Couette, pared inmersa movil                  11.1 s
  [OK  ] V2a   Cavidad Ghia Re=100 (32^2)                    13.0 s
  [SKIP] V2b   Cavidad Ghia Re=1000 (128^2) [lento]           — PHYCB_SLOW=1
  [OK  ] V3a   Cilindro IB malla mini (St)                   11.3 s
  [SKIP] V3b   Cilindro IB D/10 (Cd y St) [lento]             — PHYCB_SLOW=1
  [OK  ] M0a   distancia a pared = |phi| exacta               0.0 s
  [OK  ] M0b   L2A degrada etiquetado (contrato §3)           2.4 s
  [OK  ] M0c   M2/M3 declaran su ausencia                     0.0 s
  [OK  ] M0d   backend etiquetado + paridad CPU/GPU           1.8 s
  TODO EN VERDE   (~51 s)
```

`M0d` es nuevo y es el guardián del port a GPU: comprueba que pedir un
backend ausente degrada con motivo, que `auto` respeta la regla de tamaño y
que **25 pasos del cilindro con IB dan el mismo campo en CPU y en GPU**
(§5.4). Cuesta 2 s, así que vive en la suite rápida.

Los dos `SKIP` **no son checks que pasan trivialmente**: son los casos caros,
corridos y medidos a mano con sus scripts, y sus números están abajo. Es la
regla de la casa aplicada.

Estos casos corren en **CPU aunque haya GPU** porque son pequeños (384 a
38 400 celdas) y por debajo de 100 000 celdas la GPU pierde por coste de
lanzamiento de kernels (§5.2). Forzarla es una variable de entorno, y la
escalera **también pasa así** (§5.4).

### 3.1 V1 — Poiseuille y Couette laminares · **PASA**

Canal de 40 celdas de altura con **las dos paredes inmersas y desplazadas
0.25 dx** respecto de las caras de celda: con las paredes alineadas no se
estaría probando el IB, se estaría probando una BC de caja. Poiseuille por
fuerza volumétrica (presión uniforme ⇒ la solución analítica es exacta);
Couette con la pared superior **inmersa** moviéndose.

| caso | error máx / U_escala | criterio | celdas fantasma huérfanas |
|---|---|---|---|
| V1a Poiseuille | **0.17 %** | < 2 % ✅ | 0 |
| V1b Couette | **0.04 %** | < 2 % ✅ | 0 |

### 3.2 V2 — Cavidad de tapa deslizante (Ghia et al. 1982) · **PASA**

Tablas I y II de *Ghia, U., Ghia, K.N. & Shin, C.T. (1982), J. Comput. Phys.
48(3), 387–411*, metidas como dato en el test con su cita.

El error se mide **en unidades de la velocidad de la tapa** (U = 1), es decir
`max|u_num − u_Ghia|`. Un error relativo punto a punto no significa nada en
un perfil que cruza cero. Se excluyen los dos puntos extremos de las tablas
(y = 0 y y = 1) porque son contorno impuesto, no resultado, y la malla de
centros de celda no llega a ellos. **La interpretación está escrita en el
test antes que los números, no elegida después.**

| malla | Re | max\|Δu\| | max\|Δv\| | criterio | estado | coste (CPU) |
|---|---|---|---|---|---|---|
| 32² | 100 | **0.0044** | **0.0070** | < 0.05 | converge (res_u 8.7e-6) en 13 601 it | 14 s |
| 128² | 1000 | **0.0048** | **0.0099** | < 0.05 | converge (res_u 1.9e-5) en 164 401 it | 856 s |

Re = 1000 **no queda como pendiente**: se corrió entero y pasa. Lo que sí se
declara es que **no cabe en la suite rápida** (14 min), así que vive en
`run_v2_re1000.py` y en `PHYCB_SLOW=1`.

### 3.3 V3 — Cilindro en flujo cruzado con IB · **PASA**

Referencias: Cd(Re = 100) ≈ 1.35 (*Liu, Zheng & Sung 1998, JCP 139, 35–57*;
1.364 en *Braza, Chassaing & Ha Minh 1986, JFM 165, 79–130*); St ≈
0.164–0.166 (*Williamson 1989, JFM 206, 579–627*, experimental).

Dominio 24 D × 16 D (bloqueo 6.2 %), dx = D/10, 38 400 celdas, 40 806
iteraciones, 632 s en CPU.

| magnitud | referencia | medido | error | criterio |
|---|---|---|---|---|
| **Cd** | 1.35 | **1.3958** | **+3.4 %** | ±10 % ✅ |
| **St** (FFT) | 0.165 | **0.16638** | **+0.8 %** | ±10 % ✅ |
| St (cruce por cero, comprobación independiente) | 0.165 | 0.16671 | +1.0 % | — |
| Cl medio (debe ser ≈ 0) | 0 | **+0.0020** | — | cordura ✅ |

Con malla mini (D/5, 3 500 celdas, 15 s) el **Strouhal sigue cumpliendo**
(0.1656, −0.4 %) pero el **Cd no** (1.651, +22 %): es una frecuencia global
frente a una fuerza que depende de la capa límite. Por eso el test de CI a
D/5 comprueba St con ±10 % y el Cd sólo por cordura, y el ±10 % de Cd vive en
el test lento — declarado así en el propio test.

---

## 4. Lo que costó que la V3 pasara (y por qué se cuenta)

La V3 **no pasaba** en la primera versión: Cd = 1.682 a D/10 (+24 %). Se
diagnosticó midiendo, no razonando, y el proceso deja dos resultados útiles.

### 4.1 Primero se verificó la MEDIDA, no el modelo

`post.control_volume_force` cierra el balance de cantidad de movimiento sobre
una caja **interior** al dominio (no sobre el contorno exterior: así la
medida no hereda la calidad de la BC de salida). Si la medida fuese el
problema, el resultado dependería de la caja. Tres cajas muy distintas sobre
el mismo campo:

| caja de control | Cd | Cl |
|---|---|---|
| (−2, 4) × (±2.5) D | 1.6514 | −0.016 |
| (−3, 6) × (±4.0) D | 1.6574 | −0.012 |
| (−1.5, 2.5) × (±1.5) D | 1.6592 | −0.013 |

Dispersión **0.5 %** y Cl ≈ 0. **La medida es correcta**; el sesgo estaba en
el campo, es decir en el IB.

### 4.2 El punto imagen del fantasma era la causa — tunable recalibrado

`mesh.IMAGE_MIN_DX` es la distancia mínima (en dx) a la que se coloca el
punto imagen respecto de la pared. Empujarlo lejos estabiliza el estencil
pero **linealiza el perfil sobre una distancia mayor que la real**, y eso
sesga la tensión de pared. Barrido contra la solución **analítica** de V1 y
contra V3:

| IMAGE_MIN_DX | err V1 Poiseuille | err V1 Couette | Cd V3 (D/5) |
|---|---|---|---|
| 0.05 | 0.76 % | 0.20 % | 1.649 |
| 0.40 | 0.33 % | 0.09 % | 1.649 |
| **0.55** | **0.17 %** | **0.04 %** | 1.651 |
| 0.75 | 0.17 % | 0.04 % | 1.658 |
| 1.00 | 0.21 % | 0.04 % | 1.733 |
| 1.50 (valor inicial) | 0.29 % | — | 1.864 |

Se fijó en **0.55** porque minimiza el error contra la referencia
**analítica** —el único juez exacto disponible— y además cae en la meseta del
Cd. **No se calibró contra el Cd**: eso sería ajustar a la respuesta. Efecto
sobre la V3 completa: Cd 1.682 → **1.396** (dentro del ±10 %).

### 4.3 Otros dos factores medidos, ambos menores

* **Bloqueo lateral**: de 10 % a 4.2 % de bloqueo el Cd sólo baja 3.8 %. **No
  es el factor dominante**; el dominio de ±8 D del test es suficiente.
* **Resolución**: D/5 → D/10 con el tunable ya corregido lleva el Cd de 1.651
  a 1.396. La convergencia es entre 1.er y 2.º orden, lo esperable de un
  ghost-cell IB no conservativo (la frontera está representada por volúmenes
  de celda enteros). El **cut-cell conservativo** —upgrade ya declarado en
  WAVE6 §0— es lo que devolvería el 2.º orden limpio en la frontera.

### 4.4 Fallos reales encontrados y corregidos por el camino

1. **`KeyError: 't_mon'`**: con `monitor_every > 0` y ningún monitor
   registrado, el driver escribía en una clave que sólo creaba `add_monitor`.
   Corregido inicializando el historial completo en el constructor.
2. **Volumen fluido sellado**: el SDF anular del contrato cerraba los
   extremos axiales, así que el fluido quedaba encerrado y el residual salía
   **idénticamente 0** — el solver «convergía» sin mover nada, que es
   justamente el tipo de falso verde que hay que cazar. Se añadió
   `cap_ends=False` y el dominio se recorta en Z al tramo del perfil para que
   los parches de entrada/salida caigan sobre la boca del conducto.
3. **Ley de Spalding con Newton**: el Newton de `u_tau_spalding` no
   convergía desde arranques pobres (con u⁺ ≈ 100 el paso se colapsa y 30
   iteraciones no llegan). Se reescribió con **bisección acotada** sobre una
   función demostradamente monótona: residuo de máquina (1e-13) en las tres
   bandas (log, buffer, subcapa).
4. **Pared inmersa MÓVIL en GPU** (encontrado al portar, 2026-07-31): la
   velocidad de pared del IB se sube al dispositivo una vez, y
   `mesh.apply_dirichlet` la volvía a pasar por `np.asarray` — que sobre un
   array de CuPy **lanza a propósito**. La V1b (Couette) petaba en el
   constructor con `TypeError: Implicit conversion to a NumPy array is not
   allowed`. Corregido usando `xp.asarray`, que sirve para host y para
   dispositivo. Es el fallo típico de un port a GPU y por eso se cuenta.

---

## 5. Presupuesto de rendimiento — RE-MEDIDO con GPU

> Esta sección está **reescrita entera**. La versión anterior partía de
> «esta máquina no tiene GPU», que es falso. Script que produce todo lo de
> abajo: **`python cfd/tests/run_perf_budget.py`**.

### 5.0 Lo que la máquina tiene DE VERDAD

`python cfd/ibrans/backend.py`:

```
Backends disponibles: {'cupy': True, 'warp': False, 'jax': False, 'numpy': True}
  warp: ModuleNotFoundError: No module named 'warp'
  jax:  ModuleNotFoundError: No module named 'jax'
  GPU: NVIDIA GeForce RTX 3070 Ti · cc 8.6 · 6.92/8.00 GiB libres
       · cupy 14.1.1 · CUDA runtime 12090
```

**No se instaló nada** (regla del proyecto): CuPy ya estaba. Warp y JAX
siguen sin estar y **no se instalan**; además ya no hacen falta, porque
CuPy es un reemplazo casi directo de NumPy y el kernel del solver es **el
mismo código** con `xp.` en lugar de `np.` (`solver.py` docstring).

### 5.1 Kernel sintético de advección-difusión (el que la wave manda usar)

Malla 96³ = 884 736 celdas, 20 pasos, con `sync()` antes y después de
cronometrar y 3 pasos de calentamiento en GPU:

| backend | precisión | s/celda/paso | × vs referencia GPU (1e-8) |
|---|---|---|---|
| NumPy/CPU | FP32 | 2.42e-8 | ×2.4 |
| NumPy/CPU | FP64 | 5.07e-8 | ×5.1 |
| **CuPy/GPU** | **FP32** | **1.37e-9** | **×0.14** |
| CuPy/GPU | FP64 | 2.74e-9 | ×0.27 |

Este kernel es **engañosamente bueno** —son 7 operaciones de array por eje y
el solver real hace ~20 veces más trabajo por celda— y se reporta por
transparencia, no como cifra de mérito. Dos avisos de honestidad más:

* en GPU **bate la referencia pública** (×0.14 en FP32). Eso NO significa que
  el solver la bata: ver §5.2.
* el número de CPU FP32 (2.42e-8) es ~2× peor que el 2.13e-8 que anotaba la
  versión anterior de este documento con 64³. Es el mismo código; es un
  microbenchmark ruidoso y sensible al tamaño y al estado térmico de la CPU.
  Se deja la discrepancia a la vista en vez de elegir el número bonito.
* repetido más tarde con la GPU compartida al 99 % por otro trabajo dio
  1.93e-9 (FP32) y 3.78e-9 (FP64): ×1.4 peor, misma conclusión.

### 5.2 El solver real (la cifra que cuenta): CPU vs GPU

Caso idéntico en los dos backends: cubo 3D con **esfera inmersa** (para que
el ghost-cell del IB entre en la cuenta), rk3, FP64, entrada/salida en Z,
12 pasos cronometrados tras 3 de calentamiento.

| celdas | NumPy/CPU s/celda/paso | CuPy/GPU s/celda/paso | quién gana |
|---|---|---|---|
| 8 000 | 7.37e-7 | 7.68e-6 | CPU ×10.4 |
| 32 768 | 5.51e-7 | 1.91e-6 | CPU ×3.5 |
| 64 000 | 5.91e-7 | 9.05e-7 | CPU ×1.5 |
| 140 608 | 2.00e-6 | 4.02e-7 | **GPU ×5.0** |
| 262 144 | 1.96e-6 | 2.04e-7 | **GPU ×9.6** |
| **512 000** | **2.01e-6** | **1.17e-7** | **GPU ×17.1** |
| 1 000 000 | 2.04e-6 | 9.66e-8 | **GPU ×21.1** |

Tres efectos, todos esperables y todos declarados:

1. **Por debajo de ~100 k celdas la GPU PIERDE**, y mucho. Un paso RK3 lanza
   varios cientos de kernels; con mallas pequeñas el paso lo domina el coste
   de lanzarlos, no el de calcular. Por eso `backend="auto"` **se queda en
   CPU** si la malla tiene menos de `backend.GPU_MIN_CELLS = 100 000` celdas
   (el cruce medido cae entre 64 k y 141 k). Es una regla con tabla, no una
   corazonada; forzar GPU sigue siendo posible y la V&V lo hace (§5.4).
2. En CPU, a partir de ~140 k celdas el solver pasa a estar **limitado por
   ancho de banda de memoria** (el conjunto de trabajo excede la caché) y se
   estanca en ~2e-6 s/celda/paso.
3. La GPU sigue mejorando con el tamaño porque los kernels tapan cada vez
   mejor el coste de lanzamiento.

Medidas anteriores en CPU, para comparar sin trampa: la versión anterior
anotaba **2.72e-6** s/celda/paso a 512 k celdas. La nueva mide **2.01e-6**
con el mismo backend. La diferencia **no es la GPU**: al portar se
precomputó la máscara de «sólido profundo» (antes se recalculaba con dos
operaciones booleanas sobre TODO el dominio en cada subpaso RK) y se
sustituyó el indexado booleano por índices planos. Eso acelera **también la
CPU** un ~26 %. Efecto colateral visible: la suite rápida baja de 82 s a
**51 s**, la V2 Re=1000 de 1 473 s a **856 s** y la V3 D/10 de 797 s a
**632 s**, con los MISMOS resultados numéricos (V2: 0.0048/0.0099 y 164 401
it; V3: Cd 1.3958 y St 0.16638 — cifra por cifra los de antes).

**Precisión y VRAM en GPU** (mismo caso, 12 pasos):

| celdas | FP32 s/celda/paso | FP64 s/celda/paso | VRAM FP32 | VRAM FP64 |
|---|---|---|---|---|
| 512 000 | 1.21e-7 | 1.25e-7 | 19.7 MiB | 38.8 MiB |
| 1 000 000 | 5.53e-8 | 9.97e-8 | 37.6 MiB | 74.4 MiB |
| 2 097 152 | 4.38e-8 | 8.92e-8 | 77.5 MiB | 153.5 MiB |

A 512 k celdas FP32 y FP64 empatan (el caso todavía está a medio camino del
régimen limitado por ancho de banda); a partir de 1 M celdas **FP32 es ×1.8
más rápido**, que es justo lo que predice el argumento de Ciencia §11.2. Aun
así el **default es FP64** — el motivo está en §5.4 y es la V&V, no la
comodidad.

**Repetibilidad y contaminación, dicha de frente.** Esta máquina es
compartida: durante parte de la campaña otros trabajos del proyecto tenían
la GPU al **99 % de ocupación** (comprobado con `nvidia-smi`). El caso de
512 k celdas se midió en las dos condiciones:

| condición | CuPy/GPU s/celda/paso | NumPy/CPU s/celda/paso | GPU/CPU |
|---|---|---|---|
| **GPU libre** | **1.17e-7** (y 1.25e-7 en otra tanda) | 2.01e-6 | ×17.1 |
| GPU al 99 % por otro proceso | 2.01e-7 ± 1 % (4 repeticiones) | 1.88e-6 ± 2 % | ×9.4 |

Con la GPU libre el número es el bueno; con la GPU compartida se pierde la
mitad de la ganancia. **Las dos condiciones cumplen el objetivo de la wave**
(9.8 min y 16.8 min frente a la banda 5–40 min), así que la conclusión de
§5.3 no depende de a cuál se mire — pero el número se da con su condición
al lado, no suelto. Dentro de cada condición la medida es **muy estable**
(1–2 % en 4 repeticiones); la dispersión grande es entre condiciones, no
ruido.

**Dónde se va el tiempo en GPU** (caso de 512 k celdas, FP64, medido
cronometrando cada fase por separado con `sync()`):

| fase | ms por llamada | veces por paso rk3 | % del paso |
|---|---|---|---|
| `rhs` (convección MUSCL+TVD, difusión, Rhie–Chow, ∇p) | 17.9 | 3 | **65 %** |
| `apply_bc` completo (contornos del dominio + IB) | 5.1 | 5 | 31 % |
| — de eso, **el ghost-cell del IB** (3 × `apply_dirichlet`) | 1.5 | 5 | 9 % |
| — de eso, `zero_deep_solid` (3 componentes) | 0.22 | 5 | 1.3 % |
| — de eso, `apply_neumann` de la presión | 0.5 | 5 | 3 % |
| `time_step` (reducción de u_max + bajada del escalar) | 0.53 | 1 | 0.6 % |
| **paso completo** | **82.0** | — | 100 % |

Esto responde a la duda que la wave planteaba: **el indexado de las celdas
fantasma y la interpolación al punto imagen SÍ vectorizan en GPU** (son un
*gather* por los 2^ndim nodos del estencil, un producto por pesos y un
*scatter*), y cuestan el **9 %** del paso, no el cuello de botella. **Nada
del bucle se quedó en CPU.** Lo que sí costó llegar ahí:

* el sólido profundo se marcaba con una **máscara booleana**; indexar con
  booleanos obliga a CuPy a hacer un *scan* (cumsum) en cada llamada. Se
  sustituyó por índices planos precomputados. Es también lo que aceleró la
  CPU un 26 %.
* la velocidad de pared del IB se sube **una vez** en el constructor; si se
  subiera en `apply_bc` habría 5 copias host→device por paso.

**Transferencias device↔host en el bucle: 1.33 por paso** (contador
`d2h_per_step` del `info` del solver). Ese 1 es el escalar `u_max` del CFL,
que hay que bajar sí o sí para calcular Δt; el 0.33 son los residuales, que
sólo se evalúan cada `residual_every` pasos. **Ningún campo se mueve dentro
del bucle**: los campos, las máscaras y los estenciles del IB viven en el
dispositivo desde el constructor (`mesh.to_backend`).

### 5.3 El objetivo de la wave, con estos números: **SE CUMPLE**

Objetivo v1 (WAVE6 §5): **500 000 celdas × 10 000 iteraciones**, 5–40 min en
una RTX 4070+. Son 5e9 celda-paso.

| escenario | s/celda/paso | tiempo del objetivo | ¿cumple? |
|---|---|---|---|
| referencia pública GPU FP32 (JAX-Fluids, Ciencia §11.2) | 1e-8 | 50 s | — (vara de medir) |
| **esta máquina, CuPy/GPU, 3D, rk3, FP64, GPU libre** | **1.17e-7** | **9.8 min** | **✅** |
| esta máquina, CuPy/GPU, ídem con la GPU compartida al 99 % | 2.01e-7 | 16.8 min | ✅ |
| esta máquina, CuPy/GPU, 3D, rk3, FP32 | 1.21e-7 | 10.1 min | ✅ |
| esta máquina, CuPy/GPU, malla de 1 M celdas, FP32 | 5.53e-8 | 4.6 min | ✅ |
| esta máquina, NumPy/CPU, 3D, rk3 (medido) | 1.88–2.01e-6 | **2.6–2.8 h** | ❌ |
| *(versión anterior de este documento, CPU)* | *2.72e-6* | *3.8 h* | ❌ |

**Veredicto honesto: el objetivo de rendimiento de M1 se cumple en GPU**, con
×17 sobre la CPU de esta misma máquina (×9.4 si la GPU está compartida) y a
×11.7 de la referencia pública (no ×270 como decía la versión anterior). La distancia que queda con la
referencia es real y tiene explicación medible: el kernel es NumPy
vectorizado sobre CuPy, es decir **una operación de array = un kernel**, sin
fusión; JAX-Fluids fusiona con XLA. Recuperar ese factor exige kernels
fusionados (`cupy.fuse`, `RawKernel` o `ElementwiseKernel`), que es trabajo
opcional de optimización, **no** un bloqueante.

Letra pequeña que NO se oculta:

* Esto es **flujo frío**. Añadir M2 (2 ecuaciones de transporte más) y M3
  (4 especies + fuente EDM) multiplica el coste por ~3–6 ⇒ **30 min – 1 h por
  caso reactivo en GPU**. Sigue dentro de lo utilizable para un lazo de
  optimización nocturno, y **ya no** es 10–25 h como salía en CPU. Es una
  extrapolación, y como tal está etiquetada: el coste real de M2/M3 se
  medirá cuando existan.
* El objetivo de la wave hablaba de una RTX 4070+; esta es una **3070 Ti**,
  y aun así cumple.
* Por debajo de ~100 k celdas el objetivo no aplica y la GPU no sirve: para
  verificación 2D y mallas pequeñas el camino sigue siendo la CPU.

### 5.4 La escalera V&V corrida A LA FUERZA en GPU — mismos umbrales

`PHYCB_L2A_BACKEND=cupy` salta la regla de tamaño y mete en GPU casos donde
la GPU es lenta a propósito: aquí no se mide velocidad, se mide **que el
resultado es el mismo**.

**Resultado: pasan, con los mismos umbrales.** Corrida
`PHYCB_L2A_BACKEND=cupy python cfd/tests/test_ibrans_v1_v3.py`:

```
  [OK  ] V1a   Poiseuille laminar, paredes inmersas         378.8 s   (CPU: 16.0 s)
  [OK  ] V1b   Couette, pared inmersa movil                 411.2 s   (CPU: 14.8 s)
  [OK  ] V2a   Cavidad Ghia Re=100 (32^2)                   316.5 s   (CPU: 17.7 s)
  [OK  ] V3a   Cilindro IB malla mini (St)                  172.8 s   (CPU: 15.1 s)
```

Números medidos en GPU, uno al lado del de CPU (mismo umbral, sin tocar):

| caso | criterio | CPU (FP64) | **GPU (FP64)** | |
|---|---|---|---|---|
| V1a Poiseuille, paredes inmersas | < 2 % | 0.16904 % · 12 301 it | **0.16904 %** · 12 301 it | ✅ |
| V1b Couette, pared inmersa móvil | < 2 % | 0.038530 % · 11 901 it | **0.038530 %** · 11 901 it | ✅ |
| V3a cilindro IB mini, St (FFT) | ±10 % | 0.16564 (−0.4 %) | **0.16564 (−0.4 %)** · 6 848 it | ✅ |
| V3a cilindro IB mini, Cd (cordura 0.8–3.0) | — | 1.6514 | **1.6514** | ✅ |
| V2a cavidad Ghia Re=100 (32²) | ±5 % | 0.0043991 / 0.0069551 · 13 601 it · res_u 8.7408e-6 | **0.0043991 / 0.0069551** · 13 601 it · res_u **8.7408e-6** | ✅ |

Las filas coinciden **hasta la última cifra reportada** y con el MISMO
número de iteraciones hasta convergencia: la diferencia CPU↔GPU en FP64 vive
por debajo de lo que cualquiera de estos criterios puede ver (cuantificado
justo abajo).

Los tiempos son **×11 a ×28 los de CPU, a propósito**: son los casos
pequeños (384–3 500 celdas) donde la GPU pierde (§5.2). Aquí no se mide
velocidad, se mide que el resultado no cambia. **Correr la V&V así no es el
modo normal**: el modo normal es `auto`, que en estos casos elige CPU.

**La primera corrida forzada encontró un fallo real** —la V1b petaba en el
constructor por pasar un array de CuPy a `np.asarray`— y por eso se corre:
un backend nuevo que sólo se prueba en el caso grande esconde exactamente
este tipo de bug. Está contado en §4.4, punto 4.

**Paridad CPU↔GPU cuantificada** (test `M0d`, en la suite rápida): 25 pasos
del cilindro con IB, mismo caso, los dos backends.

| campo | FP64: max\|CPU−GPU\| / escala | FP32: max\|CPU−GPU\| / escala |
|---|---|---|
| u | 4.4e-16 | 3.9e-7 |
| v | 4.8e-16 | 3.7e-7 |
| p | 1.5e-15 | 1.2e-6 |

Es decir: **la diferencia entre backends está en el épsilon de la precisión
usada**, no en el modelo. En FP64 (el default) es ruido de máquina; en FP32
es ruido de FP32. Ninguna de las dos toca el tercer dígito de un Cd o de un
Strouhal.

**Determinismo, comprobado y no supuesto.** Con `SEED = 71` y **el mismo
backend**, dos ejecuciones del mismo caso dan campos **bit-idénticos**
(`np.array_equal` → True, tanto en NumPy como en CuPy). Lo que **NO** es
bit-idéntico es CPU vs GPU: las reducciones (`sum`, `max`) de CuPy acumulan
por bloques en un orden distinto del de NumPy. La magnitud está en la tabla
de arriba, y la advertencia viaja en
`backend.Backend.summary()["determinism"]` para que salga en el record y no
sólo en este documento.

**Por qué el default es FP64 — con el matiz que la medida obligó a poner.**

Al escribir este port supuse que en FP32 (eps ≈ 1.2e-7) el residual
normalizado se estancaría por encima de la tolerancia de parada y que **la
V1 y la V2 dejarían de converger**. Lo medí, y la hipótesis era **demasiado
gruesa**: falsa en los casos fáciles y cierta en el difícil. Mismo caso,
sólo cambiando `PHYCB_L2A_PRECISION`:

| caso | tol de parada | FP64 | FP32 |
|---|---|---|---|
| V1a Poiseuille | 3e-4 | converge, 12 301 it, err **0.16904 %** | converge, 12 301 it, err **0.16904 %** |
| V2a cavidad Re=100 (32²) | 1e-5 | converge, 13 601 it, res_u 8.74e-6, err 0.0044 / 0.0070 | converge, 13 201 it, res_u 9.77e-6, err **0.0044 / 0.0070** |
| **V2b cavidad Re=1000 (128²)** | **2e-5** | **converge**, 164 401 it, res_u 1.94e-5, err 0.0048 / 0.0099 | **NO converge**: `max_iter` a las **600 000 it**, res_u **estancado en 2.75e-5** (×1.4 la tolerancia), err 0.0051 / 0.0101 |
| V3b cilindro D/10 (no estacionario, para por `t_end`) | — | Cd 1.3958 (+3.4 %), St 0.16638 (+0.8 %) | Cd **1.3955** (+3.4 %), St **0.1664** (+0.8 %) |

La última fila es la contraprueba: en un caso **no estacionario**, que para
por tiempo físico y no por residual, FP32 da el mismo resultado hasta la
tercera cifra. El problema de FP32 no es la física, es el **criterio de
parada por residual**.

Es decir, en el caso exigente **el criterio de parada sí se rompe en FP32**:
el residual toca su suelo de redondeo por encima de la tolerancia y el
solver agota el bucle en vez de declarar convergencia. Lo que **no** se
rompe es la precisión: 0.0051 / 0.0101 sigue dentro del ±5 % de Ghia (y a
0.0003 del resultado en FP64). El coste es 2 267 s en vez de 856 s, porque
gasta ×3.6 más iteraciones para no llegar a ninguna parte.

Por eso el default es **FP64**, y ahora el motivo está medido y no supuesto:
**es la única precisión en la que la escalera V&V entera para por
convergencia y no por agotamiento**, y el umbral de parada no se afloja
(regla de la casa). FP32 sigue disponible, es ×1.8 más rápido en GPU y usa
la mitad de VRAM en mallas grandes (§5.2, §5.5), y **la V3 demuestra que en
régimen no estacionario da el mismo resultado**. Por eso es la opción
sensata para los casos que traerán M2/M3 —donde el criterio no es un
residual sino un promedio temporal—; cuando llegue ese momento, cambiar
`backend.GPU_AUTO_PRECISION` es una línea, pero exige rehacer la escalera.

### 5.5 VRAM — medida, no estimada

Ciencia §11.5 estima ~30 MB para 500 k celdas × ~15 campos FP32. Medido con
el pool de CuPy (`used_bytes`, que incluye los temporales del kernel):

| caso | FP32 | FP64 |
|---|---|---|
| 512 000 celdas | **19.7 MiB** | 38.8 MiB |
| 1 000 000 celdas | 37.6 MiB | 74.4 MiB |
| 2 097 152 celdas | 77.5 MiB | 153.5 MiB |

La estimación de Ciencia era **conservadora por ~1.5×**, lo cual es la
dirección correcta para una estimación. Con 6.92 GiB libres sobra sitio: el
caso del objetivo usa el **0.3 %** de la VRAM.

El guardián `backend.fits_in_vram` (margen `VRAM_SAFETY_FRAC = 0.60`) corta
antes de reventar y degrada a CPU **con etiqueta**, nunca con excepción.
Comprobado pidiendo una malla de 97 M celdas:

```
malla de 97 336 000 celdas -> backend numpy
  :: degradado a numpy por VRAM: el caso necesita ~18292 MiB y solo hay
     7087 MiB libres de VRAM (margen 60%). Usa un dx mas grueso.
```

Límite del guardián en esta máquina: **46.4 M celdas en FP32**, 23.2 M en
FP64. Muy por encima de las 200 k–800 k que Ciencia §11.2 pide para una
microturbina.

### 5.6 Lo que la GPU NO arregla

Que conste, porque es donde se pierde la honestidad de un documento así:

* **No hay producto L2a igualmente.** Falta física (M2, M3, M4), no ciclos.
* **No hay temperatura de metal cuantitativa** en ninguna fidelidad
  (contrato §12). Eso no era un problema de coste.
* **No hay Warp ni JAX ni XLB**, y siguen sin instalarse. Lo que hay es
  CuPy.
* **El IB sigue siendo ghost-cell no conservativo**: el orden en la frontera
  (§4.3) no mejora por correr más rápido.
* **La ganancia depende del tamaño.** En 2D pequeño la GPU es hasta ×10 más
  lenta. Quien mire sólo el titular «×17» se equivocará de caso.

---

## 6. La cadena SDF → malla → IB sobre la geometría REAL

`cfd.ibrans.cold_smoke(contract)` sobre el contrato del ancla KJ66
(`THETA_REF`), dx = 3 mm:

```
malla 31 x 31 x 68 = 65 348 celdas
  fluido    17 510
  fantasma  15 936
  huerfanas      0
distancia a pared = |phi| (exacta, gratis)
200 iteraciones en 9.6 s  ->  7.32e-7 s/celda/paso   (backend: numpy;
  auto se queda en CPU porque 65 348 < 100 000 celdas, y lo dice en `note`)
```

**Esto es fontanería, NO producto.** Prueba que la cadena contrato → SDF →
malla cartesiana → máscaras → fantasmas → solver funciona sobre la geometría
de verdad y da su coste real. No da splits, ni penetraciones, ni Δp/p: eso es
M4.

Dos limitaciones declaradas de esta cadena:

1. **No se lee el `.vdb` de PicoGK.** `mesh.sample_vdb()` está escrito, hace
   el import perezoso y degrada con motivo: no hay binding de OpenVDB para
   Windows/CPython 3.13. Lo que se usa es `sdf_from_contract()`, un SDF
   **analítico aproximado** del conducto anular del liner — **sin swirler,
   sin orificios, sin domo**. Para M4 esto no basta y es la tarea 4.1.
2. El SDF anular no es exacto en los tramos inclinados (se toma el mínimo de
   distancias radial y axial). El error es O(sin θ) y sólo importa dentro de
   la banda IB; está anotado en el docstring de `sdf_annulus_profile`.

Nota de tamaño: a dx = 3 mm el caso no llega al régimen donde la GPU paga.
El M4 real irá a dx = 0.5–2 mm (Ciencia §11.2), es decir 0.5–4 M celdas —
justo la banda donde la GPU va ×17 a ×21.

---

## 7. Lo que falta, ordenado, con su riesgo

> **Cambio respecto de la versión anterior**: la tarea «portar los kernels a
> Warp/JAX», que estaba marcada como **alta y bloqueante para todo lo
> demás**, **ya no existe**. Está hecha por otra vía (CuPy) y el objetivo de
> rendimiento se cumple (§5.3). Lo que queda de aquello es una tarea de
> optimización OPCIONAL, sin bloquear nada.

### M2 — turbulencia + ρ variable + energía  → **HECHO (2026-08-02, §10)**

| # | tarea | resultado |
|---|---|---|
| 2.1 | transporte de k y ε (realizable) | ✅ `turbulence.KEpsilonRealizable` — verificado M2a (analítico) y M2b (Dean −3.7 %) |
| 2.2 | primera celda sobre IB | ✅ con **desviación declarada**: dos escalas (u* de k) + Spalding, no Spalding puro — §10.2 |
| 2.3 | corrección de curvatura | ✅ Spalart–Shur **simplificada** (r̃=0, declarado); el juicio real será V9 |
| 2.4 | energía + ρ(T) low-Mach + escalar pasivo | ✅ `energy.py` — verificado M2c (analítico ±1 %) y M2e |
| 2.5 | **V4 backward-facing step** | ✅ **x_r = 6.23h vs 6.67 (−6.6 %, umbral ±15 %)** |
| 2.6 | V5 chorro libre; chorro caliente en cruzado | ✅ chorro plano S = 0.0877 (−12.3 %); JICF: script `run_v5_jcf.py`, resultado en §10.5 |

### M3 — especies + EDM + encendido  → **HECHO (2026-08-02, §11) salvo V8**

| # | tarea | resultado |
|---|---|---|
| 3.1 | transporte de Y_k con D_eff | ✅ `chem.SpeciesTransport` (Le = 1 en laminar vía ν(T) de la energía) |
| 3.2 | fuente EDM acoplada | ✅ min(EDM, Arrhenius); cinética laminar = Arrhenius puro (V6/V7) |
| 3.3 | liberación de calor + consumo point-implicit | ✅ CONSISTENTE (productos y S_T del consumo realizado — M3d cazó la versión inconsistente) |
| 3.4 | **arranque de llama** | ✅ **V6c: un campo frío enciende con el plan de la wave, ancla y da la misma S_L (±5 %)**. La robustez EN EL COMBUSTOR se juzga en M4 — el riesgo #1 sigue vivo hasta V9 |
| 3.5 | tasas con Pyrometheus | ✅ instalado (autorizado 2026-08-02) y verificado idéntico a tabla y Cantera (M3c) |
| 3.6 | V6 / V7 / V8 | V6 ✅ (−2.3 %) · V7 ✅ (T_max −0.3 %) · **V8 PENDIENTE declarado** |

**La salida de emergencia mixed-is-burned sigue escrita y usable** — la
decisión de la semana 15 se tomará mirando V9 (M4), no se pospone.

### M4 — combustor completo + producto  *(el siguiente, sin discusión)*

> **Plan de arranque COMPLETO para la próxima sesión: [`M4_PLAN.md`](M4_PLAN.md)**
> — receta de montaje de un caso reactivo, claves REALES del contrato
> (verificadas contra `geometry_generator.build_contract`), criterios por
> tarea, trampas medidas y comandos de humo. Lo de abajo es el resumen.

| # | tarea | riesgo |
|---|---|---|
| 4.1 | SDF real desde PicoGK (`.vdb`) o generar el SDF completo con swirler, orificios y domo | **alto** — hoy sólo existe el anillo del liner |
| 4.2 | inyección de combustible, orificios por la regla d/dx ≥ 4, efusión como pared porosa | alto |
| 4.3 | `post.dp_over_p`, `T4_and_pattern_factor`, `adiabatic_wall_map`, `penetrations` | medio (firmas ya congeladas) |
| 4.4 | clustering (T, φ) → CRN de post-proceso → NOx (reusa `crn_core`) | medio |
| 4.5 | `solve()` devolviendo producto + `register_hifi_pair` para calibrar L0/L1 | bajo |
| 4.6 | V9 PRECCINSTA con pérdida de calor en pared; ancla KJ66; test de ranking | **alto** — expectativa DECLARADA: cualitativo, no perfiles |

### Transversal

| # | tarea | riesgo |
|---|---|---|
| T.1 | ~~kernels en Warp/JAX sobre GPU~~ → **HECHO con CuPy** (§5). Queda como **opcional**: fusionar kernels (`cupy.fuse`/`RawKernel`) para recortar el ×11.7 que separa de la referencia pública | bajo, **no bloqueante** |
| T.2 | cut-cell conservativo en vez de ghost-cell | medio — recuperaría el 2.º orden en la frontera (§4.3) |
| T.3 | binding de OpenVDB para leer el campo de PicoGK sin aproximar | medio |
| T.4 | reducir el número de kernels por paso (el `apply_bc` lanza ~60) para que la GPU también gane en mallas pequeñas | bajo |

---

## 8. Desviaciones declaradas respecto de lo que pedía la wave

1. **FP64 por defecto en los dos backends** en vez del FP32 de Ciencia
   §11.2. En CPU el FP64 es gratis; en GPU cuesta ×1.8 en mallas grandes.
   Motivo MEDIDO (§5.4): en FP32 la V1 y la V2 a Re = 100 dan errores
   idénticos, pero **la V2 a Re = 1000 no converge** — el residual se
   estanca en 2.75e-5 frente a una tolerancia de 2e-5 y el solver agota las
   600 000 iteraciones. La precisión aguanta (0.0051 vs 0.0048), el
   **criterio de parada no**, y ese umbral no se afloja. FP32 está
   disponible (`precision="fp32"`), medido y reportado.
2. **El backend de producción es CuPy, no Warp.** WAVE6 §0 elegía Warp
   (1.ª opción) o JAX. Ninguno está instalado y la regla del proyecto
   prohíbe instalar; CuPy sí estaba, es un reemplazo casi directo de NumPy
   y permite **un solo kernel** para CPU y GPU. Consecuencia: el benchmark
   «Warp vs JAX de 1 día» de WAVE6 §1 **sigue sin hacerse** porque sigue sin
   haber nada que comparar; `backend.benchmark` está escrito y corre con lo
   que hay (§5.1).
3. **`backend="auto"` NO elige GPU siempre.** Por debajo de
   `GPU_MIN_CELLS = 100 000` celdas se queda en CPU porque la GPU es más
   lenta ahí (tabla medida en §5.2 y en el propio comentario del tunable).
   Se puede forzar con `backend="cupy"` o `PHYCB_L2A_BACKEND=cupy`.
4. **CFL efectivo ≤ 0.6.** Se comprobó que CFL = 0.8 con RK3 desestabiliza el
   acoplamiento presión-velocidad (el término de Rhie–Chow escala con Δt y a
   CFL alto sobre-amortigua hasta oscilar; medido: residuales de 1e2 a 1e4
   sin divergir del todo, que es peor que divergir). Default 0.4–0.5.
5. **`solve()` devuelve `None`** en lugar de un producto parcial. Deliberado:
   devolver un dict etiquetaría el record como `ibrans_L2A` y haría pasar por
   CFD reactivo un solver que sólo resuelve flujo frío. **Esto no cambia por
   tener GPU.**
6. **V2 a Re = 1000 y V3 a D/10 salen de la suite rápida** (14 y 11 min en
   CPU) y viven en scripts manuales + `PHYCB_SLOW=1`. Ambos **se corrieron y
   pasan**; lo que se declara es que no caben en CI, no que no se sepa el
   resultado.
7. **`IMAGE_MIN_DX` y `GPU_MIN_CELLS` son tunables calibrados** (§4.2, §5.2),
   con su barrido y su criterio anotados en el propio código, como manda el
   contrato §10.7.
8. **Los resultados no son bit-idénticos entre CPU y GPU** (§5.4). El
   contrato §10.8 exige determinismo con `SEED = 71`; se cumple **por
   backend**, y la diferencia entre backends se declara en vez de fingir que
   no existe.
9. **(M2) Wall functions de DOS ESCALAS, no Spalding puro.** El plan (tarea
   2.2) preveía `u_tau_spalding` directamente. En separación/readherencia
   u_p → 0 ⇒ u_τ → 0 ⇒ ε_pared → 0 ⇒ ν_t = C_μk²/ε explota — exactamente
   en el punto que V4 mide. Se usa u* = C_μ^¼√k como escala (finita ahí) y
   la MISMA ley de Spalding invertida en el otro sentido (`u_plus_of_ystar`);
   `u_tau_spalding` queda como diagnóstico y juez (test M2d comprueba que
   las dos inversiones describen la misma ley). Detalle en el docstring de
   `turbulence.py`.
10. **(M2) Corrección de curvatura SIMPLIFICADA**: Spalart–Shur con r̃ = 0
    (el término con DS_ij/Dt exigiría transportar el tensor). Cizalla pura
    ⇒ f_r = 1; rotación sólida ⇒ f_r → 0 (lo que preserva la recirculación
    del swirler); recorte [0, 1.25]. Medido en V5: su efecto en el chorro
    plano es <1 % (S 0.0853 sin ella vs 0.0858 con ella a d/3) — el juicio
    real será V9 (swirl fuerte).
11. **(M2) `mach_art` 0.15–0.3 en los casos ESTACIONARIOS** en vez del 0.1
    genérico: la pseudo-compresibilidad no cambia el estado estacionario
    (∂p/∂t = 0 ⇒ ∇·u_f = q_T con el mismo operador espacial), sólo el
    camino, y con c más baja el dt sube. En los no estacionarios (V3)
    sigue 0.1.
12. **(M2) El residual de los casos turbulentos se ESTANCA en 1e-4–5e-3**
    (canal 4.5e-5, V4 6.5e-5, V5 5e-3) en vez de llegar a la tolerancia
    de M1: el acoplamiento segregado momento↔turbulencia deja un ciclo
    límite pequeño (y el chorro tiene además una inestabilidad física de
    baja frecuencia en el borde). Los criterios de M2 son INTEGRALES
    (x_r, S, Cf) medidos sobre el estado cuasi-estacionario, y su
    estabilidad se comprobó alargando la corrida (V4: x_r 6.23 a 120k it
    con res 6.5e-5). Declarado en cada test.
13. **(M2) ρ variable en forma advectiva, no conservativa**: el momento
    resuelve Du/Dt (correcto a ρ variable tras dividir por ρ) añadiendo
    +u·(∇·u_f), y la continuidad entra como ∇·u = q_T en la ecuación de
    presión. μ y λ no dependen de T (declarado en `energy.py`; ν(T)
    potencial disponible SOLO en laminar desde M3, desviación 16). La
    verificación M2c (expansión térmica analítica: T +0.94 %, u +0.90 %,
    ρu conservado al 0.05 %) es la que valida este acoplamiento.
14. **(M3) Pyrometheus SÍ está instalado** (2026-08-02, con autorización
    expresa del usuario — antes la regla del proyecto lo vetaba y la vía
    degradaba). Se usa como VERIFICADOR (M3c: idéntico a la tabla y a
    Cantera a 5 cifras); el acoplamiento usa la tabla porque para 1 paso
    son intercambiables y la tabla no importa nada en el bucle.
15. **(M3) Sólo se acopla el paso 1 efectivo** (fuel → CO₂ + H₂O
    completo): el CO del paso 2 de BFER/WD2 queda para el upgrade
    EDC/multi-paso. Consecuencia declarada: sin CO ni η_comb por CO.
    c_p CONSTANTE por caso, calibrado a la T_ad del mecanismo
    (`calibrate_cp`): la T de llama sale bien en los extremos y el
    interior del frente hereda el sesgo — V6 lo mide y aún así S_L queda
    a −2.3 % de Cantera.
16. **(M3) ν(T) = ν₀·(T/T₀)^n SOLO en régimen laminar** (V6/V7): sin él,
    α en la llama es ~20× menor que el real y S_L saldría varias veces
    baja. Con turbulencia, ν_t domina y el molecular queda constante
    (declarado). El exponente se calibra al α(T) de Cantera en el propio
    test (n = 1.72 para el sistema propano).
17. **(M3) V8 (Sandia D) NO corrió** en la campaña 2026-08-02 y su test
    se salta con motivo (regla de la casa): exige mecanismo de metano en
    la tabla (2S_CH4_BFER, con fuente verificada), entrada anular de
    piloto quemado y un caso 3D de 0.5–1 M celdas en GPU (~1–2 h).
    También queda fuera el V6 del jet-A: no hay termodinámica NASA de
    C12H23 en las bases estándar, así que el combustible VERIFICADO
    contra Cantera es el propano; el jet-A hereda la misma maquinaria con
    sus coeficientes PROVISIONALES (tabla) — declarado.

---

## 9. Cómo reproducir todo esto

```bash
cd C:/Users/nickv/projects/Proyectos/Phy-CB

python cfd/ibrans/backend.py                        # GPU + backends + benchmark
python cfd/tests/run_perf_budget.py                 # presupuesto CPU vs GPU (§5)
python cfd/tests/test_ibrans_v1_v3.py               # escalera M1 rapida (~68 s)
PHYCB_SLOW=1 python cfd/tests/test_ibrans_v1_v3.py  # + los caros (~25 min)
python cfd/tests/run_v2_re1000.py                   # V2 Re=1000 manual
python cfd/tests/run_v3_cylinder.py --dx 0.1        # V3 completa manual
python -m pytest cfd/tests/test_ibrans_v1_v3.py -q  # mismo cuerpo en pytest

# ---- M2 (turbulencia + energia) ----
python cfd/tests/test_ibrans_v4_v5.py               # escalera M2 rapida (~5 min)
PHYCB_SLOW=1 python cfd/tests/test_ibrans_v4_v5.py  # + V4/V5 completos (~50 min)
python cfd/tests/run_v4_bfs.py                      # V4 manual (backward step)
python cfd/tests/run_v5_jet.py                      # V5 manual (chorro plano)
python cfd/tests/run_v5_jcf.py                      # JICF caliente 3D (GPU)
python -m pytest cfd/tests/test_ibrans_v4_v5.py -q  # mismo cuerpo en pytest

# ---- M3 (especies + EDM + encendido) ----
python cfd/tests/test_ibrans_v6_v8.py               # escalera M3 rapida (~2 min)
PHYCB_SLOW=1 python cfd/tests/test_ibrans_v6_v8.py  # + V6b oficial y V7 (~5 min)

# la MISMA escalera forzada en GPU (mismos umbrales, mas lenta a proposito)
PHYCB_L2A_BACKEND=cupy python cfd/tests/test_ibrans_v1_v3.py
PHYCB_L2A_BACKEND=cupy PHYCB_L2A_PRECISION=fp32 python cfd/tests/test_ibrans_v1_v3.py
PHYCB_L2A_BACKEND=cupy python cfd/tests/test_ibrans_v4_v5.py --only=M2a,M2c,M2e
```

Variables de entorno: `PHYCB_L2A_BACKEND` ∈ {auto, numpy, cupy, warp, jax}
y `PHYCB_L2A_PRECISION` ∈ {auto, fp32, fp64}. Pedir un backend que no está
**nunca lanza**: degrada a NumPy y lo dice en `Backend.note`, que viaja en
`record["cfd_mesh"]`/`summary()`.

---

## 10. Hito M2 — lo que se construyó y lo que se midió (2026-08-02)

### 10.1 Arquitectura del acoplamiento

Los modelos de M2 son objetos con `.advance(solver, dt)` registrados en
`solver.physics`; el solver los avanza UNA vez por paso, tras el RK del
momento (**acoplamiento segregado**: ν_t y ρ congelados durante los
subpasos). Todos los ganchos son **opt-in** — sin modelos, el solver es
bit-a-bit el de M1 (la escalera V1–V3 completa se volvió a correr en verde
el 2026-08-02 tras el refactor). Todos los kernels nuevos están escritos
contra `xp` y corren en CPU y en GPU con el mismo código; los escalares
(k, ε, T, Z) usan el MISMO MUSCL+TVD y la MISMA velocidad de cara de
Rhie–Chow que el momento (`solver.advect_diffuse_scalar`), en forma
advectiva (la correcta a ρ variable).

Novedades transversales del solver que M2 obligó a añadir:

* **perfiles por celda en los contornos**: una componente de `u` (o un
  escalar `T=`, `Z=`, `k=`…) puede ser un array sobre el plano de la cara
  (capa límite de entrada del V4, ranura del chorro del V5/JICF). Se suben
  al backend una vez, en `_normalize_bcs`/`scalar_bc_spec`.
* **guardia del falso verde en `run()`**: la primera evaluación de
  residuales ya no puede declarar convergencia — un estado inicial
  uniforme es equilibrio EXACTO del rhs antes de que la física (fuente de
  calor, turbulencia) actúe, y el driver salía en la iteración 1 dando por
  convergido un caso que no había empezado. Pasó DOS veces montando los
  tests de M2; ahora es imposible por construcción (`k > 0`).

### 10.2 k-ε realizable (Shih 1995) — decisiones y desviaciones

* Transporte advectivo explícito + **fuentes point-implicit** (el sumidero
  en el denominador): es lo que aguanta la rigidez de ε con paso explícito.
  El dt acústico de la pseudo-compresibilidad deja el CFL advectivo de los
  escalares en ~0.04 — la estabilidad sobra por ese lado.
* C_μ variable con **A_s completo** (W del tensor, no el valor congelado
  de cizalla del esqueleto de M1). En banda log recupera 0.09 (test M2d).
* C₁ = max(0.43, η/(η+5)); P_k ≤ 10·ε (`PROD_LIMIT_RATIO`, tunable
  documentado).
* **Wall functions de dos escalas** (desviación 9 de §8): ε fija en la
  celda de pared (= u*³/κy), producción de equilibrio, y ν_wall efectiva
  puesta en la celda de pared y su fantasma para que el flujo DISCRETO de
  momento a través de la cara reproduzca τ_w con el perfil lineal del
  ghost-cell. En el IB la distancia es |φ| gratis; en las paredes del
  dominio es dx/2. Los fantasmas de k y ε van por Neumann.
* Corrección de curvatura: desviación 10 de §8.

### 10.3 Energía low-Mach + ρ(T) + escalar pasivo (`energy.py`)

    DT/Dt = ∇·(α_eff ∇T) + S_T      α_eff = ν/Pr + ν_t/Pr_t (Pr_t = 0.85)
    ρ(T)  = p₀/(R_g·T)              (p₀ = presión termodinámica, NO la p del solver)
    ∇·u   = q_T = (∇·(α∇T)+S_T)/T   (exacta para gas ideal a p₀ cte)

q_T entra en la ecuación de presión (∂p/∂t = −ρc²(∇·u_f − q_T)) y ρ en el
1/ρ del gradiente y en la forma advectiva del momento. `S_T` es el gancho
que M3 usará para la liberación de calor EDM. El escalar pasivo Z es el
transporte que M3 convertirá en Y_k. Sin radiación, sin CHT, sin μ(T) —
declarado en el módulo y en §8.13.

### 10.4 Números medidos (CPU, esta máquina, 2026-08-02)

> El tercer caso de la tarea 2.6 (chorro caliente en cruzado, 3D en GPU)
> se midió con la maquinaria de M3 ya disponible y **PASA** (+19.7 % vs
> Kamotani & Greber, umbral ±25 %): detalle y lecciones en §11.4.

| caso | malla | iteraciones | tiempo | resultado | criterio | |
|---|---|---|---|---|---|---|
| M2a decaimiento | 8² | 4 000 pasos fijos | 3 s | err 0.14 % | <1 % analítico | ✅ |
| M2b canal (Re_b 68k, y⁺ 99) | 4×16 | 12 000 | 38 s | Cf −3.7 % vs Dean | ±15 % | ✅ |
| M2c expansión térmica | 48×8 | 14 201 (conv) | 18 s | T +0.94 %, u +0.90 %, ρu 0.05 % | ±3 % | ✅ |
| M2e escalar | 32×6 | 8 000 | 2 s | exacto (<1e-6) | conservación | ✅ |
| **V4 BFS** dx=h/8 | 232×40 | 120 000 | **22 min** | **x_r = 6.23h (−6.6 %)** | 6.67 ±15 % | ✅ |
| V4 mini dx=h/5 (CI) | 105×25 | 15 000 | ~2 min | x_r ≈ 4.9–5.7 (NO convergido, declarado) | cordura 3–9 | ✅ |
| **V5 chorro** dx=d/5 | 200×130 | 60 000 | **15 min** | **S = 0.0877 (−12.3 %)** | 0.10 ±15 % | ✅ |
| V5 mini dx=d/3 (CI) | 75×54 | 20 000 | ~1 min | S ≈ 0.086 | cordura 0.06–0.14 | ✅ |

Notas honestas de los dos grandes:

* **V4**: y* medio 67, máx 100, 90 % en banda [30,150]. A h/5 el caso NO
  converge en 40k it (x_r 4.9→5.7 entre 20k y 40k) — por eso el ±15 % vive
  en el test lento y el mini es sólo humo. Con capa límite de entrada 1/7
  (δ = 1.1h, la condición del experimento); con entrada de tapón x_r cae
  ~0.5h — la sensibilidad a la condición de entrada es real y conocida.
* **V5**: el realizable queda ~12 % corto en el chorro PLANO con esta
  discretización (0.086 a d/3 → 0.088 a d/5, convergiendo hacia ~0.09).
  Pasa el ±15 % pero sin margen de sobra, y se declara en vez de elegir
  una referencia más cómoda. La corrección de curvatura no es la causa
  (medido: <1 % de efecto). El chorro REDONDO (la anomalía clásica de k-ε)
  no se verificó — el plano es el que tiene banda experimental estrecha.
* El coste por paso con turbulencia es ~×2.2 el de M1 (medido 1.2e-6 vs
  5.5e-7 s/celda/paso en CPU a ~9k celdas); con energía + escalar ~×3.

### 10.5 Chorro caliente en cruzado (JICF) — el tercer caso de 2.6

`run_v5_jcf.py`: 3D, chorro a 600 K en cruzado a 300 K, r_eff = 4 con la
densidad del propio campo (ρ_j/ρ_∞ = T_∞/T_j), trayectoria del máximo de
exceso de T vs **Pratte & Baines (1967)** y/rd = 2.05(x/rd)^0.28, criterio
±25 % en x/rd ∈ [1,6] (verifica TENDENCIA con ρ variable + turbulencia +
escalar juntos, no precisión fina — la banda experimental a r = 4 es ancha).
Resultado del run de 184k celdas (backend auto → GPU): ver el JSON/nota al
final de esta sección cuando termine la corrida; el humo de 78k celdas en
CPU corre entero y acotado (T ∈ [300, 598] K).

### 10.6 GPU

La escalera M2 forzada en GPU (`PHYCB_L2A_BACKEND=cupy`, casos pequeños,
lenta a propósito — aquí se mide igualdad, no velocidad): **M2a, M2c y M2e
pasan con los mismos umbrales**. El camino IB+banda de pared en CuPy se
verificó con 800 pasos del V4 mini (campos finitos, y* sano, 2
transferencias device→host por paso: el u_max del CFL y la reducción de
max(ν_eff) del modelo). **No se exige bit-paridad CPU↔GPU en M2**: el dt
depende de max(ν_eff), que es una reducción, y el orden de acumulación
difiere entre backends (misma declaración que §5.4 para los residuales).

---

## 11. Hito M3 — especies + EDM + encendido (2026-08-02)

### 11.1 Dos bugs REALES cazados por la escalera (y por eso se cuenta)

1. **Unidades de Ea (latente desde el esqueleto)**: la tabla `MECHANISMS`
   da Ea en **J/kmol** (1.256e8 = 30 kcal/mol para el propano) y
   `arrhenius_rate` dividía entre R = 8.314 **J/(mol·K)**: exponente
   ~7500 ⇒ tasa CERO absoluto a cualquier temperatura. Nunca se vio
   porque la tasa jamás se había acoplado. La corrección se verificó por
   TRES vías independientes con el mismo estado (M3a/M3c): tabla ==
   Pyrometheus == Cantera a 5 cifras (171.66 kg/m³/s a 1600 K, φ = 1).
2. **Fuente inconsistente (nuevo, cazado por M3d antes de tocar V6)**: el
   sumidero de FUEL era point-implicit pero productos, O₂ y calor usaban
   la ω DEMANDADA: al agotarse el fuel se seguía creando masa (Σ Y llegó
   a 1.47) y calor. Ahora TODO se acopla al consumo REALIZADO del paso
   implícito (`dyf`), acotado además por el O₂ disponible: M3d comprueba
   estequiometría (<2 %), salto de T (<5 %) y Σ Y = 1 (<5e-3).

### 11.2 La auto-validación V6 (la clave de bóveda del hito)

Nuestro solver y Cantera `FreeFlame` reciben **el mismo mecanismo**
(`chem.cantera_yaml` — el YAML declara las unidades de la tabla y Cantera
convierte, así la conversión propia queda vigilada), la misma hipótesis
Le = 1, el mismo T_ad (c_p calibrado al equilibrio HP del sistema de 5
especies) y α(T) potencial ajustado al de Cantera (n = 1.72). Resultado
medido (propano WD2 paso 1, φ = 1, dx = 100 µm):

    S_L = 0.2476 m/s  vs  Cantera 0.2534 m/s  →  −2.3 %   (umbral ±10 %)
    T_max = 2395 K    vs  T_ad 2393 K         →  +0.1 %

**V6c — el riesgo #1 medido**: el MISMO caso arrancado FRÍO con el plan de
encendido de la wave (parche esférico impuesto 800 it + rampa del
pre-exponencial 0.1→1 en 1500 it + sub-relajación 0.3 + limitador de T)
**enciende, ancla** (liberación integral estable ±2 % / 400 it) **y da la
misma S_L que la llama inicializada quemada (±5 %)**. Esto verifica la
MAQUINARIA; la robustez en el combustor real (swirl, mezcla imperfecta,
paredes frías) se juzga en V9/M4 y el riesgo #1 sigue vivo hasta entonces.

**V7 — contradifusión**: chorros opuestos propano(40 %)/aire a baja
tensión; T_max = 2143.7 K vs 2150.1 K del `CounterflowDiffusionFlame` de
Cantera (**−0.3 %**, umbral ±10 %), llama del lado del oxidante en ambos.
Geometrías DISTINTAS (la nuestra plana, la de Cantera axisimétrica):
sólo T_max y el lado de la llama son comparables; los perfiles finos no
se comparan y está declarado en el test. Primera corrida: la llama se
APAGABA — el dominio arrancaba lleno de aire y la franja caliente moría
antes de que llegase el combustible; se arranca con la capa de mezcla ya
formada (anotado en el propio test).

### 11.3 Números y coste (CPU, 2026-08-02)

| caso | malla | iteraciones | tiempo | resultado | |
|---|---|---|---|---|---|
| M3a–M3d | — | — | ~2 s | cinética a nivel máquina; 0D conservativo | ✅ |
| V6a (150 µm) | 200×4 | 10 000 + 200 medida | 30 s | S_L −2.6 % | ✅ |
| V6b (100 µm, oficial) | 300×4 | 20 000 + 200 | 75 s | **S_L −2.3 %** | ✅ |
| V6c encendido | 200×4 | 2×10 000 | 65 s | ancla; ΔS_L < 5 % | ✅ |
| V6d metano (60 µm) | 500×4 | 25 000 + 200 | 126 s | S_L −8.9 % (a 150 µm: −19 %, no resuelta) | ✅ |
| V7 (0.25 mm) | 80×80 | 15 000 | 91 s | **T_max −0.3 %** | ✅ |

El coste reactivo por paso (5 especies + energía + ν(T), laminar) es
~×4.5 el de M1 en la misma malla (3.5e-6 vs 7.4e-7 s/celda/paso en CPU a
~1.2k celdas) — coherente con la extrapolación ×3–6 de §5.3.

### 11.4 Chorro caliente en cruzado (cierre de la tarea 2.6) — **PASA**

Resultado final (`run_v5_jcf.py`, dx = d/4, 184k celdas, GPU, 21 min,
r_eff = 4, T_j/T_∞ = 600/300): **desviación máxima +19.7 % vs la línea
central de temperatura de Kamotani & Greber en x/(r·d) ∈ [1, 4]**
(umbral ±25 %), con el exponente de la trayectoria bien (~0.23 medido vs
0.27). El caso ejercita a la vez k-ε + energía low-Mach + ρ(T) + escalar
+ perfiles de entrada por celda, en GPU.

Tres lecciones MEDIDAS por el camino (por eso se cuentan):

1. **Correlación equivocada**: la primera corrida comparaba contra
   Pratte & Baines (−35 %) — P&B es la línea central visual/de MOMENTO,
   sistemáticamente más alta que la térmica. El comparador correcto para
   una trayectoria medida por el máximo de T es **Kamotani & Greber
   (NASA TN D-6966, 1972, ec. 3)**: Z_c/D =
   0.76·(ρ_j/ρ_∞)^0.15·J^0.52·(x/D)^0.27 — constantes verificadas contra
   el informe original, no de memoria.
2. **No era resolución**: d/4 → d/6 (622k celdas) movió la trayectoria
   <2 % (mesh-converged) y seguía a +26 %.
3. **Era la turbulencia de entrada del chorro**: con la k del cruzado
   (basada en u_∞ = 1) el chorro de u_j = 5.66 entraba casi laminar, se
   mezclaba poco y sobre-penetraba. Con k/ε de salida de tobera (I = 10 %,
   ℓ = 0.07·d, escalando con u_j — perfiles POR CELDA en la ranura, que
   los BCs escalares de M2/M3 ya soportan) la desviación baja de +25.9 %
   a **+19.7 %**. Moraleja para M4: los chorros de dilución deben entrar
   con su turbulencia de orificio, no con la del plenum.

### 11.5 V8 (Sandia Flame D) — PENDIENTE, con el mecanismo YA listo

El **mecanismo de metano ya está en la tabla y verificado**
(2S_CH4_BFER, parámetros confirmados contra la base de CERFACS el
2026-08-02: paso 1 CH4+1.5O2→CO+2H2O, A = 4.9e9 cgs, Ea = 35500 cal/mol,
órdenes 0.5/0.65; el paso 2 lleva b = 0.7 que `arrhenius_rate` v1 no
soporta — se acopla el paso 1 efectivo, como en el resto):

* triple verificación puntual tabla == Pyrometheus == Cantera (idénticos);
* **V6d**: llama CH4 acoplada, S_L = 0.386 vs 0.424 de Cantera
  (**−8.9 %**, umbral ±10 %) a dx = 60 µm. ⚠️ La llama BFER de CH4 es más
  FINA que la WD2 de propano: a 150 µm da −19 % — para cualquier caso con
  CH4 la malla debe resolver el frente (≤100 µm en laminar; en el
  combustor turbulento EDM no aplica esta restricción).

Lo que SIGUE faltando para V8: entrada anular de piloto quemado (T y
composición de equilibrio) y el caso 3D de 0.5–1 M celdas en GPU (~1–2 h)
con EDM, con la expectativa DECLARADA de la wave (±15 % cualitativo en
ejes, documentando que EDM sobre-predice la T de mezcla rápida). El test
existe y SE SALTA con este motivo — no se trivializa.

---

## 12. M4 — combustor completo + producto (medido el 2026-08-03)

**Estado: el caso y el producto están IMPLEMENTADOS y corren de punta a
punta; la verificación cuantitativa del ancla NO está cerrada.** El riesgo
#1 del proyecto (robustez del encendido en el combustor real) queda
RESUELTO. Los números de abajo están medidos en esta máquina (RTX 3070 Ti,
CuPy, FP64) sobre el ancla `pc.THETA_REF` (KJ66) a dx = 2 mm,
206 550 celdas.

### 12.1 Lo que se entrega

| pieza | archivo | qué hace |
|---|---|---|
| SDF completo | `mesh.sdf_combustor_from_contract` | liner ∪ anillo ∪ filas resueltas; regla d/dx ≥ 4; pared inflada a ≥2·dx; modo «sin anillo» declarado |
| Caso del combustor | `case.build_case` | perfiles de entrada por celda (swirler con SN impuesto + anillo), inyección de combustible en la PZ, encendido en la bujía del contrato |
| Transpiración | `case.TranspirationSource` | filas no resueltas + efusión como **fuente volumétrica de masa y momento** calibrada por C_d·A |
| Producto | `post.*` | Δp/p, T4/PF/RTDF, mapa de pared adiabática (ranking), penetraciones, NOx por clustering (T,Z) → `crn_core` |
| Punto de entrada | `ibrans.solve` | contrato → caso → producto §4.4, con `DEGRADE_TAG` y mixed-is-burned declarado |

### 12.2 Números MEDIDOS

**Caso frío** (14 000 iteraciones, 17.8 min):

| medida | valor | criterio |
|---|---|---|
| cierre de masa (plano 4 vs contrato) | **+7.6 %** | converge desde +9.9 % a 6000 it |
| número de swirl | **0.769** vs 0.70 | **+9.9 %**, estable en 3 corridas ✅ |
| caudal inyectado por las fuentes | **0.08801 kg/s** vs 0.08801 | exacto por construcción ✅ |
| Δp/p | 0.0108 | **NO comparable** con la red 1D (§12.4) |
| residual `res_u` | 3.3e-2 | estancado (ciclo límite del segregado) |

**Caso reactivo** (12 501 iteraciones, 21.7 min, GPU):

| medida | valor | criterio de la wave |
|---|---|---|
| **llama anclada** | **SÍ** | `anchored()` ✅ **riesgo #1 resuelto** |
| liberación de calor | 110.5 kW, **oscilación 0.70 %** | ±2 % ✅ |
| presupuesto | **21.7 min** | <30 min GPU ✅ |
| cierre de masa | **−4.2 %** | (era −42 % antes de §12.3) |
| penetraciones Y/d | **0.53–0.72** | (eran 0.003–0.022) |
| **T4** | **1445.8 K** vs 1150 K | **+25.7 %, NO PASA ±5 %** ❌ |
| Δp/p | 0.0189 vs 0.030 | −37 %, **no comparable** (§12.4) |
| PF | 0.162 (L0: 0.350) | sin criterio absoluto; ver §12.5 |
| EI_NOx (CRN de post) | 3.30 | — |

### 12.3 Tres bugs REALES cazados por la escalera de M4

1. **La fila transpirada no inyectaba momento.** El spot de transpiración
   se dimensionaba por comodidad geométrica, no por el área de descarga
   C_d·πd²/4, así que el aire entraba a ~15 m/s en vez de a los ~70 m/s
   del contrato. Consecuencia: no había pérdida de orificio (Δp/p a la
   mitad) ni penetración (Y/d ≈ 0.003). Corregido: `s_r = ½·d·√C_d`.
2. **Una BC de velocidad no entrega el caudal.** Al subir a la U_jet real,
   el Dirichlet de pared entregaba un 17 % menos de masa (el campo decide
   el flujo, no la BC). Sustituido por **fuente volumétrica** (`div_extra`
   y `force_field` del solver, ambos opt-in): el caudal pasa a ser exacto
   por construcción.
3. **La fuente de masa estaba escalada con ρ₃ y no con ρ local.** En la
   continuidad low-Mach el caudal másico es ρ_local·q·V; con q fijo, en la
   llama (ρ ≈ ρ₃/5) entraba cinco veces menos aire de dilución: T4 se iba
   a 1791 K (+56 %) y el cierre de masa a −42 %. **En frío no se veía**
   porque ρ ≡ ρ₃. Corregido reescalando la fuente cada paso.

Además, dos bugs de post-proceso: el mapa de pared leía la T en la celda
FANTASMA (o sea, la propia condición de contorno: daba ~430 K junto a una
llama de 2400 K) y ahora lee el punto imagen; y las penetraciones incluían
esa misma capa, de modo que el máximo déficit de T caía siempre en la
pared y Y/d salía ≈ 0 por construcción.

### 12.4 Límite ESTRUCTURAL del ancla a dx = 2 mm: el modo «sin anillo»

En el KJ66 la holgura liner↔carcasa es ~3 mm y la pared del liner 0.8 mm.
A dx = 2 mm, con la pared inflada a 2 celdas para que sea estanca, **el
anillo no cabe**: el dominio es sólo el interior del liner y todas las
filas entran como fuente. Dos consecuencias que NO son un bug y hay que
decir en voz alta:

* **El reparto de aire es un INPUT, no una predicción.** Sólo sería una
  predicción con el anillo resuelto, donde los caminos compiten por una
  misma Δp (que es justo lo que hace la red 1D de `crn_core`). Por eso el
  test M4a se reformuló: verifica conservación exacta, conductancias del
  contrato (T49), momento de descarga, cierre de masa y swirl — **no**
  recupera el reparto fila a fila, que además el balance de volumen de
  control axial mide mal (±30 %) en cuanto los chorros llevan momento.
* **La pérdida de orificio no ocurre dentro del dominio**, así que Δp/p no
  es comparable con la red 1D. `post.dp_over_p` lo declara con
  `comparable_with_1d=False`. NO se fabrica la parte que falta desde el
  propio C_d: haría el contraste tautológico.

Resolverlo pide **dx ≤ 1 mm** (~1.6 M celdas, 1–2 h/caso): a esa
resolución el anillo cabe (3 celdas) y la fila de dilución ya cumple
d/dx ≥ 4. Es el siguiente paso y está PENDIENTE.

### 12.5 Lo que queda ABIERTO (declarado, no disimulado)

1. **T4 a +25.7 %** del balance (criterio ±5 %). Diagnóstico parcial: con
   el c_p calibrado del modelo (1482 J/kg·K frente a los 1148 de `CP_HOT`
   del ciclo) la T4 de mezcla completa saldría ~1028 K, así que el exceso
   medido no se explica por el c_p sino por **mezcla incompleta**: el 16 %
   de aire de efusión entra distribuido y lento, se queda pegado a la
   pared y pesa poco en una media ponderada por ρu. Falta comprobar el
   balance entálpico en el plano 4 para separar «estratificación real» de
   «error de conservación», y re-medir a dx ≤ 1 mm.
2. **PF no fiable todavía.** La FFT azimutal del plano 4 da 256 K de
   amplitud (bajó desde 442 K) con modo dominante **n = 4** — que es la
   simetría de la MALLA cartesiana, no del hardware (filas de 8, 8 y 10;
   6 inyectores). Mientras el PF esté gobernado por el modo de la malla,
   **el test de ranking M4c mediría ruido**: no se corre.
3. **M4c (ranking)** y **V9 (PRECCINSTA)**: no corridos. Scripts escritos
   (`run_m4_kj66.py --ranking`, `run_v9_preccinsta.py`), tests en SKIP con
   motivo visible.
4. El campo toca el techo del limitador de T (2477 K frente a un techo de
   2478 K): esperable en la zona estequiométrica de una llama de difusión,
   pero conviene vigilar que no esté recortando física.

### 12.6 Veredicto

`MILESTONES["M4"]["done"]` sigue en **False** a propósito. `solve()` SÍ
devuelve producto —y por eso `UNAVAILABLE_REASON` ya no dice «no hay
producto»—, pero el producto lleva `fidelity_note` con el modo de
discretización y la etiqueta de honestidad de siempre, y **los dos
contrastes cuantitativos del ancla (T4 ±5 %, Δp/p ±20 %) no están
superados**. Declararlo cerrado exigiría o aflojar umbrales o callar el
modo «sin anillo»: ninguna de las dos cosas se hace en esta casa.
