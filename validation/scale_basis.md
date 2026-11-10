# Base técnica de la escala 5–25 kg/s de Phy-CB

Fase 17.1 de QSTUDIO, plan 17.1-01. Este informe se versiona en el checkout
original de Phy-CB (D-380) junto con `scale_inlets.json`, `scale_l0.py` y
`test_scale_l0.py`. Distingue tres clases de afirmación:

- **[MEDIDO-SW]** salida de una corrida de software identificada (hash y receta).
- **[CÁLCULO]** deducción algebraica o numérica reproducible desde la fuente
  de Phy-CB en el commit de base, con el script o la prueba que la repite.
- **[CITADO]** dato de una referencia primaria tal como lo transcribió
  `17.1-RESEARCH.md` (QSTUDIO). En esta ejecución no hubo acceso a las fuentes
  primarias para re-verificarlo, así que no se usa como base cuantitativa de
  ninguna constante.

<!-- scale-l0:body -->

<!-- scale-l0:diagnose-8:begin -->
### Traza diagnóstica L0 a 8 kg/s (generada por `scale_l0.py diagnose`)

Fase **diagnóstica**: no cuenta para la matriz de aceptación.

- Entrada: fila `8` de `scale_inlets.json`, contrato Phy-AC `85c016debab3f99c9cdc4ec1896053db63482e0cceecba27a44bb3655eab1745`; P03 = 358961.8805265866 Pa, T03 = 432.4902351398321 K, ṁ = 8.0 kg/s.
- Receta: `python -B -X utf8 phycb_cli.py --optimize --fidelity L0 --mdot 8.0 --p3 358961.8805265866 --t3 432.4902351398321 --t4 1200.0 --arch annular --fuel jet_a --material Inconel-625 --material-casing Inconel-718 --dp-max 0.04 --pf-max 0.4 --seed 71 --n-init 40 --rounds 1 --batch-size 6 --no-figures --no-color --outdir <outdir>` (cwd Phy-CB, tope 60 s).
- Phy-CB `b84d7d7fbf79b45662882abe1f29d859b7363460` (main), clase histórica: fuera de la banda documentada (máx. 1.5 kg/s); límites τ aplicados 4.0–10.0 ms.
- Resultado: factible = **False**; violaciones: `PF` = +0.2031, `tau_max` = +1.0537, `spec_PF_max` = +0.3535, `effusion_band` = +0.1057.
- τ = 20.537 ms, τ_PZ = 16.953 ms, φ_PZ = 0.8673, φ_LBO = 0.6398, margen LBO = 0.3556, D_L = 316.9 mm, L_tot = 720.4 mm, V_c = 0.05681 m³, sizing_driver = aero.
- Contrato Phy-CB `d60e02316bd8c2f7ae86ee67e4e4ad6e3e58d2e1b9ecac34b46d055935383383`; θ `aac547688bd1bd629d12c300a202fe6a9b6d918e02926116aefc512c80049b6a`; DfAM real: dfam_checker.check (real).
- Tiempo: 4.833 s (2026-11-10T21:27:33Z).

```json
{
 "schema": "phycb.scale_diagnose/1",
 "phase": "diagnostic",
 "counts_for_acceptance": false,
 "inlet": {
  "mass_key": "8",
  "mdot_kg_s": 8.0,
  "P03_Pa": 358961.8805265866,
  "T03_K": 432.4902351398321,
  "PR": 3.5426783175582197,
  "contract_sha256": "85c016debab3f99c9cdc4ec1896053db63482e0cceecba27a44bb3655eab1745"
 },
 "recipe": {
  "arch": "annular",
  "fidelity": "L0",
  "t4": 1200.0,
  "fuel": "jet_a",
  "material": "Inconel-625",
  "material_casing": "Inconel-718",
  "dp_max": 0.04,
  "pf_max": 0.4,
  "dfam": true,
  "seed": 71,
  "n_init": 40,
  "rounds": 1,
  "batch_size": 6,
  "timeout_s": 60.0,
  "argv": [
   "python",
   "-B",
   "-X",
   "utf8",
   "phycb_cli.py",
   "--optimize",
   "--fidelity",
   "L0",
   "--mdot",
   "8.0",
   "--p3",
   "358961.8805265866",
   "--t3",
   "432.4902351398321",
   "--t4",
   "1200.0",
   "--arch",
   "annular",
   "--fuel",
   "jet_a",
   "--material",
   "Inconel-625",
   "--material-casing",
   "Inconel-718",
   "--dp-max",
   "0.04",
   "--pf-max",
   "0.4",
   "--seed",
   "71",
   "--n-init",
   "40",
   "--rounds",
   "1",
   "--batch-size",
   "6",
   "--no-figures",
   "--no-color",
   "--outdir",
   "<outdir>"
  ]
 },
 "cb": {
  "head": "b84d7d7fbf79b45662882abe1f29d859b7363460",
  "branch": "main",
  "clean": false
 },
 "run_id": "a5f7cd4025e5",
 "outdir_name": "cb-diag-8-a5f7cd4025e5",
 "returncode": 3,
 "historical_class": {
  "classes_covering": [],
  "documented_max_kgs": 1.5,
  "label": "fuera de la banda documentada (máx. 1.5 kg/s)"
 },
 "tau_limits_ms": {
  "min": 4.0,
  "max": 10.0
 },
 "n_evals": 52,
 "theta": {
  "arch": 1.0,
  "AL_over_Aref": 0.5501342897843737,
  "frac_pz": 0.3843335519426014,
  "frac_sz": 0.003499179852269984,
  "frac_dz": 0.46810573460642746,
  "swirl_SN": 0.9657526794878544,
  "n_holes_pz": 11.083410058816646,
  "n_holes_dz": 8.634162074861559,
  "d_eff_mm": 0.7665076601779888,
  "pitch_over_d_eff": 4.471255236013664,
  "injector": 0.29646626931462605,
  "ALR": 5.740574885123347,
  "L_pz_over_DL": 0.7212632388643625,
  "L_dz_over_DL": 1.0520330359518666,
  "t_liner_mm": 1.2906565372302774,
  "fuel": 0.0,
  "P3_Pa": 358961.8805265866,
  "T3_K": 432.4902351398321,
  "mdot_kgs": 8.0,
  "T4_K": 1200.0
 },
 "theta_sha256": "aac547688bd1bd629d12c300a202fe6a9b6d918e02926116aefc512c80049b6a",
 "feasible": false,
 "g_blocks": {
  "aero": {
   "PF": 0.20312908583975964,
   "T_wall": -0.23236905982518463,
   "LBO": -0.11476442114265828,
   "tau_min": -0.8052262243948685,
   "tau_max": 1.0536645590879106,
   "M_exit": -0.4935215765697226,
   "dp": -0.25,
   "t_liner": -0.6126002653866459,
   "d_eff": -0.34769079817950377,
   "phi_pz_rich": -0.3805277637693444,
   "eta_comb": -0.01783515501032662,
   "mixing": -0.8673778420741771
  },
  "spec": {
   "spec_dp_max": -0.25,
   "spec_PF_max": 0.3535202215697295,
   "spec_envelope_D": -1.0,
   "spec_envelope_L": -1.0
  },
  "struct": {
   "T_wall": -0.23236905982518463,
   "sigma_th": -0.9416797582363637,
   "hoop": -0.9603076479593062,
   "t_wall": -0.6126002653866459,
   "banned": -1.0
  },
  "dfam": {
   "t_min": -0.16666666666666663,
   "d_hole_min": -0.3476910873728649,
   "overhang": -1.0,
   "drainability": -0.2465486848240993,
   "effusion_band": 0.10574899999999995
  }
 },
 "g_dfam_from_contract": [
  -0.16666666666666663,
  -0.3476910873728649,
  -1.0,
  -0.2465486848240993,
  0.10574899999999995
 ],
 "dfam": {
  "checker": "dfam_checker.check (real)",
  "violations": [
   "effusion: P_x/d = 4.471 outside the printable band [5, 8]"
  ]
 },
 "record": {
  "tau_ms": 20.536645590879107,
  "tau_pz_ms": 16.95348421566754,
  "phi_pz": 0.8672611307229178,
  "phi_LBO": 0.6397753408966457,
  "LBO_margin": 0.3555713627653272,
  "LBO_margin_ratio": 1.3555713627653272,
  "V_c_m3": 0.05681059877639984,
  "V_pz_m3": 0.01802466177823757,
  "V_sz_m3": 0.012495203420194834,
  "V_dz_m3": 0.026290733577967434,
  "D_L_mm": 316.87974629918807,
  "D_ref_mm": 427.2284091906743,
  "D_casing_mm": 534.0355114883429,
  "A_ref_m2": 0.14335410361257647,
  "A_ref_aero_m2": 0.14335410361257647,
  "A_ref_chem_m2": 0.040157051639323586,
  "A_L_m2": 0.07886400797858029,
  "A_an_m2": 0.034404984867018346,
  "L_pz_mm": 228.55371214626987,
  "L_sz_mm": 158.43987314959404,
  "L_dz_mm": 333.3679615307921,
  "L_tot_mm": 720.361546826656,
  "U_ref_ms": 19.2969969802401,
  "rho3_kgm3": 2.891945662704109,
  "SMD_um": 18.74203589090473,
  "sizing_driver": "aero",
  "eta_comb": 0.9876142532979774,
  "dp_over_p": 0.03,
  "PF": 0.5414080886278918,
  "T04_K": 1200.0,
  "P04_Pa": 348193.024110789,
  "mdot_air_kgs": 8.0,
  "mdot_fuel_kgs": 0.1831470613667891,
  "mdot4_kgs": 8.183147061366789,
  "I_MW_m3_atm": 39.22077209367196,
  "theta_lefebvre": 210738341.5572924,
  "Omega_load": 3.337950416782396e-09,
  "atomizer_model": "elshanawany_lefebvre",
  "source": "correlations_L0",
  "fidelity": 0,
  "arch": "annular"
 },
 "contract": {
  "sha256": "d60e02316bd8c2f7ae86ee67e4e4ad6e3e58d2e1b9ecac34b46d055935383383",
  "size_bytes": 13258,
  "derived": {
   "T04_K": 1200.0,
   "P04_Pa": 348193.024,
   "mdot_air_kgs": 8.0,
   "mdot_fuel_kgs": 0.183147061,
   "tau_ms": 20.536646,
   "dp_over_p": 0.03,
   "PF": 0.541408089
  }
 },
 "checkpoint_sha256": "e5e5adffc5fcdabe920f110b4f1b35162b3bc14cadfedf5485bd3eade9a7d529",
 "log_sha256": "c2fd8f58609563fa17d2d599b9e14ff3454d455630c4e600fd6a94b5ccbb1672",
 "timing": {
  "started_utc": "2026-11-10T21:27:33Z",
  "finished_utc": "2026-11-10T21:27:39Z",
  "elapsed_s": 4.833
 }
}
```
<!-- scale-l0:diagnose-8:end -->

### Compuerta de la descarga a 5 kg/s (Tarea 2, D-387)

[MEDIDO-SW] Receta común de `scale_l0.py` (objetivo PR 3,2, Phy-AC
`081b6d0`, L0). Criterio: candidato **recomendado** exportado, factible y con
PR observado en [3, 7]; sin elegir puntos del frente ni cambiar receta u
objetivo. Topes por intento según D-387: 150 s el 1.º y 210 s el 2.º.

| Intento | Receta | Tope | Resultado | SHA256 contrato · checkpoint · log |
|---|---|---|---|---|
| 1 (`ac-5-a1-87999e909c96`, 2026-11-10) | semilla 71, 320/5/14 | 150 s | rc 0 en 121,6 s, 390 evaluaciones; recomendado factible, PR 2.700909829562064 (η_pol 0,866) | `de9f4e1b…6f9bfafe` · `ddb7c93e…5eb8f16` · `12814c81…b3913e1` |
| 2 (`ac-5-a2-1751ddb91e55`, tope 150 s anterior a D-387) | semilla 137, 480/7/18 | 150 s | timeout con 588 de 606 evaluaciones; anulado por D-387 | — |
| 2 (`ac-5-a2-3edaddb55c2d`, 2026-11-10) | semilla 137, 480/7/18 | 210 s | rc 0 en 127,6 s, 606 evaluaciones; recomendado factible, PR 2.396008225013416 (η_pol 0,883) | `bcea4a95…b1c8b12f` · `ba8608e9…cb569b95` · `120e2a16…0b3f12ed` |

El intento 2 bajo D-387 reutilizó la evidencia del intento 1 con
`acquire --prior-attempt1`, que comprueba spec, semilla, presupuesto y
fidelidad antes de saltarlo. **Gate agotado.**

[CÁLCULO] Diagnóstico, no evidencia de aceptación: los datasets contienen 177
y 282 diseños factibles, 16 y 24 con PR en [3, 7] (algunos a PR 3,200 con
η 0,83–0,86; máximo factible 4,50 y 4,25). El recomendador de Phy-AC prefiere
η frente a |PR − objetivo| y a 5 kg/s elige PR por debajo del objetivo en las
dos semillas; el mejor verificado por ronda del intento 2 no pasa de PR 2,26.
El fallo no depende del presupuesto (el intento 2 terminó con 82 s de
margen): hace falta cambiar el criterio (punto del frente) o el objetivo o
la receta, decisión que corresponde al fundador.

### Compuerta de la descarga a 5 kg/s con objetivo PR 3,5 (Tarea 2, D-389)

[MEDIDO-SW] D-389 (fundador, 2026-11-10): a 5 kg/s el objetivo de PR pasa de
3,2 a 3,5; receta predefinida, topes D-387 y el mismo criterio (recomendado
factible con PR observado en [3, 7], sin puntos del frente). Los intentos con
objetivo 3,2 quedan como evidencia y no se reutilizan. `scale_l0.py` guarda
`target_PR`, `target_policy: "D-389"` y `superseded_targets: [3.2]` en la
receta y `target_rule` en la procedencia de la fila. Phy-AC `081b6d0`,
limpio antes y después; intérprete de Phy-AC el del venv
(`PHYCB_SCALE_AC_PYTHON`).

| Intento | Receta | Tope | Resultado | SHA256 contrato · checkpoint · log |
|---|---|---|---|---|
| 1 (`ac-5-a1-e6a3b7793283`, 2026-11-10) | semilla 71, 320/5/14 | 150 s | rc 0 en 96,6 s, 390 evaluaciones; recomendado factible, PR 2.700909829562064 (η_pol 0,866) | `e329138e…3fba707876` · `eebc5801…80e39cc296` · `60bc1ea2…be909d927f6e` |
| 2 (`ac-5-a2-a5a72276fa3c`, 2026-11-10) | semilla 137, 480/7/18 | 210 s | rc 0 en 131,6 s, 606 evaluaciones; recomendado factible, PR 2.6072660296604306 (η_pol 0,859) | `d36089cd…ce327157725c` · `23ec6955…e75e2a0da5e` · `648ec55b…504d6b0fe9e8` |

**Gate agotado por tercera vez** (`acquire --mass 5` devuelve 2).

[CÁLCULO] Diagnóstico, no evidencia de aceptación (script
`full_feas5.py`, SHA256 `1796cddd…b6759`, salida `full_feas5.out.json`
`f46f22fd…4fe0c9`, en el directorio de trabajo ignorado por Git):

- Con la semilla 71 el recomendado es **el mismo θ** con objetivo 3,2 y 3,5
  (mismo PR y η; sólo cambia el spec dentro del contrato). Subir el objetivo
  no mueve la recomendación.
- La lectura de la sección anterior era incompleta: el `g` del dataset de
  Phy-AC sólo contiene las 9 restricciones aerodinámicas. Reevaluados con
  `DesignSpec.constraints` completo (topes, bombeo y estructura), **ninguno**
  de los 17 y 34 diseños con PR en [3, 7] es factible, y en todos la
  restricción más violada es la 13, el margen de bombeo en gasto
  (`sm_flow ≥ 0,15`). El mejor margen en banda es 13,2 % a PR 3,03 (g₁₃ =
  +0,117) y 12,3 % a PR 3,68. El recomendador elige bien entre los factibles:
  no hay ninguno en banda.
- Indicio no verificado: en banda la velocidad de punta máxima es ~391 m/s
  (tope 480) y ninguna θ pasa de 24 000 rpm (tope 25 000). A 8 kg/s se llega
  a 447 m/s. A 5 kg/s el tope de rpm podría limitar la carga por etapa y,
  con 8 etapas, el margen de bombeo.
- Opciones que quedan para el fundador (todas cambian la receta común del
  plan): bajar `sm_flow_min` a 5 kg/s, subir `rpm_max` a 5 kg/s o aceptar un
  punto del frente. Cambiar otra vez el objetivo de PR no sirve.

### Descarga a 5 kg/s con `sm_flow_min` 0,12 (Tarea 2, D-390) · compuerta superada

[MEDIDO-SW] D-390 (fundador, 2026-11-10): **sólo a 5 kg/s** `--sm-flow-min`
pasa de 0,15 a 0,12. El resto de la receta D-389 no cambia (objetivo 3,5,
semilla 71, 320/5/14, 150 s; intento 2 sólo si el 1 fallaba). 8 y 25 kg/s
conservan 0,15 (la receta de 8 no se relanzó y la de 25 es la de UAT 14).
Corrida de adquisición nueva por el arnés, no el sondeo:

| Intento | Receta | Tope | Resultado | SHA256 contrato · checkpoint · log |
|---|---|---|---|---|
| 1 (`ac-5-a1-e6686e121c7b`, 2026-11-10T22:35:40Z) | semilla 71, 320/5/14, `--sm-flow-min 0.12` | 150 s | rc 0 en 119,3 s, 390 evaluaciones; recomendado factible con **todas** las g del spec efectivo, PR 3.682040615872552 | `c8799242…99821cde1` · `c3e2d1f0…a7463988be7` · `e0410175…6ce6103cff70` |

Fila congelada: PR 3.682040615872552, P03 = 373082.7654032863 Pa (total,
= 101325·PR), T03 = 443.96438386510135 K, 8 etapas, η_pol 0,8547,
20 025 rpm, U_tip 282,9 m/s, margen de bombeo en gasto `sm_flow` =
0,122569 (reevaluado igual). Phy-AC `081b6d0`, limpio antes y después;
intérprete de Phy-AC el del venv (`PHYCB_SCALE_AC_PYTHON`).

**Regla de factibilidad.** La compuerta reevalúa θ dentro del checkout de
Phy-AC con `DesignSpec(**spec_efectivo).constraints` (19 g: las 9
aerodinámicas que guarda el dataset más topes, bombeo y estructura) y el
`sm_flow_min` del spec efectivo (0,12). No usa la constante fija
`SM_FLOW_MIN` de la CLI de Phy-AC, cuya etiqueta «Surge margin … (min 15%)»
es cosmética y no se corrige en esta fase. `check_row` exige además que el
spec efectivo lleve el umbral de la política del caudal (0,12 a 5, 0,15 a 8
y 25), que la fila, la receta y la procedencia declaren `sm_flow_min: 0.12`
y `sm_flow_policy: "D-390"`, y que el `sm_flow` reevaluado no baje de 0,12.
La excepción no se puede aplicar a otro caudal sin cambiar el arnés.

**Excepción vetable y riesgo (D-390).** El margen de bombeo queda en
12,26 %, 0,26 puntos sobre el umbral 0,12 y por debajo de la práctica del
15–25 % que cita Phy-AC. El diseño usa las 8 etapas del tope y es el único
factible en banda del conjunto evaluado. Se acepta porque Phy-CB necesita
P03/T03 realistas y el compresor no es el objeto de la fase. El informe de
entrega a la fase 18 lo hereda.

[CÁLCULO] Reproducción del sondeo de diagnóstico S2
(`probe5/s2-sm012`, misma receta, script `analyze_probe.py` SHA256
`b55160ee…a7584f6`): θ idéntica y PR, P03, T03, η, rpm y `sm_flow`
idénticos bit a bit. El SHA del contrato difiere (`8e640e97…` en el sondeo)
sólo por los metadatos de la corrida (fecha y ruta). El sondeo no es
evidencia de aceptación.

### Matriz de descargas congelada (Tarea 2)

| Masa | Origen | PR | P03 [Pa] | T03 [K] | sm_flow_min | SHA256 contrato |
|---|---|---|---|---|---|---|
| 5 | adquirida (D-389 + D-390) | 3.682040615872552 | 373082.7654032863 | 443.96438386510135 | 0,12 (excepción vetable) | `c8799242d8df…1c107d99821cde1` |
| 8 | adquirida (plan) | 3.5426783175582197 | 358961.8805265866 | 432.4902351398321 | 0,15 | `85c016debab3…a44bb3655eab1745`[^8] |
| 25 | UAT 14 archivada (`1f9e4fa586c2`) | 3.185506101458321 | 322771.4057302644 | 420.15317853003546 | 0,15 (defecto de Studio) | `cff8c2f09285…a4ec2ab7168526ee` |

[^8]: SHA completo de 8: `85c016debab3f99c9cdc4ec1896053db63482e0cceecba27a44bb3655eab1745`.

La fila de 25 se congeló con `archive-uat14` (sólo lectura) desde
`14-campana-real.json` + `status.json` + el contrato histórico, que
reproduce su SHA publicado. Las tres filas se verifican sin QSTUDIO con
`scale_l0.py check-inlets` y `pytest validation/test_scale_l0.py -k inlet`.
