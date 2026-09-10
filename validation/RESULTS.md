# Resultados de validación — Quasar Phy-CB

Generado por `validation/validate.py` el 2026-09-10 (2026-09-10T23:16:21Z). **Regenerar tras cualquier cambio en `physics_core.py` o en `data/`** — este archivo se versiona como evidencia.

Entorno de esta corrida: Python 3.13.7 · NumPy 2.4.6 · cantera=sí · jax=sí · xlb=NO · warp=NO · matplotlib=sí

Modo: `fidelity=L0`, `use_cache=False`, **`calibrate=False`** (la campaña mide el modelo desnudo; con la calibración afín activa estaríamos validando la calibración, no la física). L1 (CRN Cantera) no incluido en esta corrida (`--l1`).

## 0. Marcador

| PASS | FAIL | de ellos huecos conocidos | NO EVALUABLE | huecos obsoletos |
|---|---|---|---|---|
| 107 | 48 | 48 | 1 | 0 |

*Huecos obsoletos* = checks marcados `known_gap` en `validate.KNOWN_GAPS` que hoy PASAN. Si este número no es 0, el documento miente y hay que quitar la entrada.

Leyenda de `rol`: **derived** = lo calcula el modelo ⇒ puede validar o falsar · **input** = se lo damos nosotros ⇒ solo trazabilidad, jamás evidencia · **data** = se valida un dato del paquete, no el modelo · **declared** = criterio no evaluable.

## 1. Resumen por caso

| Caso | Nivel | Nombre | PASS | FAIL | N/E |
|---|---|---|---|---|---|
| `A1` | A | Clase 0 · microturborreactor 0.1–5 kW | 16 | 6 | 0 |
| `A2` | A | Clase 1 · microturbina 5–50 kW | 13 | 9 | 0 |
| `A3` | A | Clase 2 · microturbina 50–250 kW | 13 | 9 | 0 |
| `A4` | A | Consistencia interna de `class_table.csv` (FAR ↔ T₃,T₄) | 9 | 2 | 0 |
| `B1` | B | KJ66 / clase hobby · anular reverso + vaporizador | 21 | 5 | 0 |
| `B2` | B | KJ66 · sensibilidad al ṁ de comunidad (0.23 kg/s) | 15 | 7 | 0 |
| `C1` | C | Microturborreactor optimizado · reparto PZ 39.35 % → 32 % | 5 | 0 | 0 |
| `D1` | D | Bath/IAAPS/HiETA · IN625, swirl radial, T₃ = 1093 K | 4 | 6 | 0 |
| `D2` | D | Tendencias de emisiones · signo de ∂EI/∂φ_PZ y ∂EI/∂τ | 11 | 4 | 1 |

## 2. Detalle por caso

### `A1` · Clase 0 · microturborreactor 0.1–5 kW

**Fuente.** data/reference/class_table.csv (`@status: verified`), transcrita de docs/Quasar_PhyCB_EstadoDelArte.md §7 «Tabla de valores de referencia — clase microturbina»; las marcas [W] de esa tabla se verificaron en fuente web el 2026-07-29 (URLs en EstadoDelArte §12). Caveat de la propia tabla: las cifras certificadas de Capstone C30/C65 y Turbec T100 NO se verificaron.

**Supuestos declarados.**

* SUPUESTO · punto de operación = punto MEDIO de cada banda de la clase (π, T₃ no recuperada, ṁ, T₄). La tabla da bandas, no un punto: elegir el centro es la opción neutra y es lo que hace falsable el resultado (cualquier otra elección se podría acusar de buscar la banda).
* SUPUESTO · reparto de aire = fila `microturbine` de `data/lefebvre/air_split_by_arch.csv` (frac_pz/sz/dz = 0.30/0.12/0.42, `@status: verified`). NO se ajusta por caso: ajustar frac_pz para centrar φ_PZ sería exactamente la recalibración por-caso que la wave prohíbe (WAVE7 §5).
* SUPUESTO · n_holes_pz = 8. `class_table.csv` no da número de orificios de zona primaria (solo de dilución); 8 es el valor de THETA_REF.
* SUPUESTO · A_L/A_ref, L_PZ/D_L y L_DZ/D_L = defaults de `zone_lengths.csv` (tabla ⚠️ provisional entera, PENDIENTES §2.1 #2).
* SUPUESTO · arquitectura = anular de flujo inverso e inyección por vaporizador: es la práctica dominante de la clase (EstadoDelArte §2.1–2.2, compresor centrífugo + eje corto).

**Notas.** La clase del caso ancla REF_CB10. Es el nivel A «gratis»: no valida contra hardware, valida que el dimensionado L0 cae donde la literatura de la clase dice que cae.

| Check | rol | valor | referencia | veredicto | detalle |
|---|---|---|---|---|---|
| pressure_ratio | input | 2.75 | [2, 3.5] - | PASS | status de la fila: verified |
| T4_K | input | 1175 | [1100, 1250] K | PASS | status de la fila: verified |
| mdot_air_kgs | input | 0.15 | [0.05, 0.25] kg/s | PASS | status de la fila: verified |
| dp_over_p | input | 0.03 | [0.02, 0.04] - | PASS | status de la fila: verified |
| t_liner_mm | input | 0.65 | [0.5, 0.8] mm | PASS | status de la fila: verified |
| d_eff_mm | input | 0.6 | [0.5, 0.7] mm | PASS | status de la fila: verified |
| SN_swirl | input | 0.85 | [0.7, 1] - | PASS | status de la fila: verified |
| FAR_global | derived | 0.02227 | [0.01, 0.018] - | FAIL (hueco conocido) | status de la fila: verified |
| phi_global | derived | 0.3242 | [0.15, 0.27] - | FAIL (hueco conocido) | status de la fila: verified |
| phi_pz | derived | 1.081 | [0.85, 1.15] - | PASS | status de la fila: verified |
| U_ref_ms | derived | 19.35 | [8, 20] m/s | PASS | status de la fila: verified |
| tau_ms | derived | 6.137 | [4, 8] ms | PASS | status de la fila: verified |
| V_c_L | derived | 0.4125 | [0.05, 0.4] L | FAIL (hueco conocido) | status de la fila: verified |
| D_casing_mm | derived | 83.12 | [80, 130] mm | PASS | status de la fila: verified |
| I_MW_m3_atm | derived | 126.9 | [25, 60] MW/m3/atm | FAIL (hueco conocido) | status de la fila: verified |
| eta_comb_design | derived | 0.9812 | [0.97, 0.99] - | PASS | status de la fila: verified |
| PF | derived | 0.35 | [0.35, 0.5] - | PASS | status de la fila: verified |
| frac_cool | derived | 0.16 | [0, 0.15] - | FAIL (hueco conocido) | status de la fila: verified |
| T_wall_liner_K | derived | 976.2 | [900, 1050] K | PASS | status de la fila: verified |
| n_eff_holes | derived | 3145 | [300, 1500] - | FAIL (hueco conocido) | status de la fila: verified |
| SMD_um | derived | 50 | [40, 70] um | PASS | status de la fila: verified |
| cierre de energía T₄→FAR→T₄ | derived | 1175 | 1175 K ±3 % | PASS | Δ = -0.004 % · VERIFICACIÓN, no validación (T₄ es θ[19]) |

**Huecos conocidos de este caso** (explicados en `docs/VALIDATION.md`):

* `FAR_global` — G1 · La tabla de clase se contradice a sí misma: el FAR que impone el balance de entalpía a partir de sus propias bandas de (T₃, T₄) no cabe en su banda de FAR/φ_global (ver A4). φ_global = FAR/FAR_st e I = FAR·LHV·P_atm/(R·T₃·τ) heredan el defecto por identidad. Recalibración: NINGUNA — hay que corregir `class_table.csv`, no el modelo. Tocar una constante para «cuadrar» esto sería calibrar contra un dato inconsistente.
* `phi_global` — G1 · La tabla de clase se contradice a sí misma: el FAR que impone el balance de entalpía a partir de sus propias bandas de (T₃, T₄) no cabe en su banda de FAR/φ_global (ver A4). φ_global = FAR/FAR_st e I = FAR·LHV·P_atm/(R·T₃·τ) heredan el defecto por identidad. Recalibración: NINGUNA — hay que corregir `class_table.csv`, no el modelo. Tocar una constante para «cuadrar» esto sería calibrar contra un dato inconsistente.
* `V_c_L` — G4 · V_c fuera de banda por continuidad: V_c = τ·ṁ/ρ₃. Con τ dentro de banda, V_c queda determinado por ṁ y ρ₃; en el extremo alto de la banda de ṁ de la clase 0 el V_c que exige la continuidad sobrepasa el techo de 0.4 L. Ver la pata 2 de A4.
* `I_MW_m3_atm` — G1 · La tabla de clase se contradice a sí misma: el FAR que impone el balance de entalpía a partir de sus propias bandas de (T₃, T₄) no cabe en su banda de FAR/φ_global (ver A4). φ_global = FAR/FAR_st e I = FAR·LHV·P_atm/(R·T₃·τ) heredan el defecto por identidad. Recalibración: NINGUNA — hay que corregir `class_table.csv`, no el modelo. Tocar una constante para «cuadrar» esto sería calibrar contra un dato inconsistente.
* `frac_cool` — G2 · Dos tablas `verified` del paquete se contradicen en 1 punto: `air_split_by_arch.csv::microturbine.frac_cool_nom` = 0.16 frente a `class_table.csv::clase 0 frac_cool` = 0–0.15. THETA_REF usa 0.16. Es un defecto de DATO, de impacto nulo en la física.
* `n_eff_holes` — G3 · Nº de orificios de efusión por encima de la banda de clase. No es un error de cálculo: n = A_efusión/(π/4·d²) con la porosidad que sale de pitch/d, y `PENDIENTES §4.2` demuestra que hacer FACTIBLE el ancla exige apretar el paso a pitch/d = 4–5, lo que SUBE aún más el número. La banda 300–1500 de la clase 0 y las reglas de efusión densa del propio proyecto son incompatibles. Corregir el DATO. AMPLIADO EL 2026-07-31: la corrección del área mojada de la efusión (physics_core §12.b — π·D_L es el perímetro de un CÍRCULO de área equivalente y no el de la corona real, ×1.375 en el ancla) sube n en las arquitecturas ANULARES en ese mismo factor y mete también a A2 (clase 1, anular inversa: 3696 → 5080 frente a [1000, 4000]) en este hueco. Es la MISMA incompatibilidad de dato, ahora con el área correcta: el número de antes estaba dentro de la banda por un error de geometría, no por acuerdo con la clase. NO se ha movido ningún umbral para taparlo.

### `A2` · Clase 1 · microturbina 5–50 kW

**Fuente.** data/reference/class_table.csv (`@status: verified`), transcrita de docs/Quasar_PhyCB_EstadoDelArte.md §7 «Tabla de valores de referencia — clase microturbina»; las marcas [W] de esa tabla se verificaron en fuente web el 2026-07-29 (URLs en EstadoDelArte §12). Caveat de la propia tabla: las cifras certificadas de Capstone C30/C65 y Turbec T100 NO se verificaron.

**Supuestos declarados.**

* SUPUESTO · punto de operación = punto MEDIO de cada banda de la clase (π, T₃ no recuperada, ṁ, T₄). La tabla da bandas, no un punto: elegir el centro es la opción neutra y es lo que hace falsable el resultado (cualquier otra elección se podría acusar de buscar la banda).
* SUPUESTO · reparto de aire = fila `microturbine` de `data/lefebvre/air_split_by_arch.csv` (frac_pz/sz/dz = 0.30/0.12/0.42, `@status: verified`). NO se ajusta por caso: ajustar frac_pz para centrar φ_PZ sería exactamente la recalibración por-caso que la wave prohíbe (WAVE7 §5).
* SUPUESTO · n_holes_pz = 8. `class_table.csv` no da número de orificios de zona primaria (solo de dilución); 8 es el valor de THETA_REF.
* SUPUESTO · A_L/A_ref, L_PZ/D_L y L_DZ/D_L = defaults de `zone_lengths.csv` (tabla ⚠️ provisional entera, PENDIENTES §2.1 #2).
* SUPUESTO · inyección airblast (default de Phy-CB por encima de la clase hobby, EstadoDelArte §9.2) y T₃ NO recuperada.

**Notas.** La clase de Bath/HiETA (caso D) en su variante NO recuperada. Comparar A2 con D1 aísla el efecto de T₃.

| Check | rol | valor | referencia | veredicto | detalle |
|---|---|---|---|---|---|
| pressure_ratio | input | 3.75 | [3, 4.5] - | PASS | status de la fila: verified |
| T4_K | input | 1225 | [1150, 1300] K | PASS | status de la fila: verified |
| mdot_air_kgs | input | 0.375 | [0.15, 0.6] kg/s | PASS | status de la fila: verified |
| dp_over_p | input | 0.03 | [0.03, 0.05] - | PASS | status de la fila: verified |
| t_liner_mm | input | 0.8 | [0.6, 1] mm | PASS | status de la fila: verified |
| d_eff_mm | input | 0.65 | [0.5, 0.8] mm | PASS | status de la fila: verified |
| SN_swirl | input | 0.75 | [0.6, 0.9] - | PASS | status de la fila: verified |
| FAR_global | derived | 0.0228 | [0.012, 0.02] - | FAIL (hueco conocido) | status de la fila: verified |
| phi_global | derived | 0.3319 | [0.18, 0.3] - | FAIL (hueco conocido) | status de la fila: verified |
| phi_pz | derived | 1.106 | [0.8, 1.1] - | FAIL (hueco conocido) | status de la fila: verified |
| U_ref_ms | derived | 20.01 | [10, 25] m/s | PASS | status de la fila: verified |
| tau_ms | derived | 8.173 | [5, 10] ms | PASS | status de la fila: verified |
| V_c_L | derived | 1.076 | [0.3, 1.5] L | PASS | status de la fila: verified |
| D_casing_mm | derived | 114.4 | [120, 200] mm | FAIL (hueco conocido) | status de la fila: verified |
| I_MW_m3_atm | derived | 91.29 | [20, 50] MW/m3/atm | FAIL (hueco conocido) | status de la fila: verified |
| eta_comb_design | derived | 0.9882 | [0.99, 0.999] - | FAIL (hueco conocido) | status de la fila: verified |
| PF | derived | 0.35 | [0.25, 0.4] - | PASS | status de la fila: verified |
| frac_cool | derived | 0.16 | [0.1, 0.25] - | PASS | status de la fila: verified |
| T_wall_liner_K | derived | 952.2 | [1000, 1150] K | FAIL (hueco conocido) | status de la fila: verified |
| n_eff_holes | derived | 5080 | [1000, 4000] - | FAIL (hueco conocido) | status de la fila: verified |
| SMD_um | derived | 10.69 | [35, 60] um | FAIL (hueco conocido) | status de la fila: verified |
| cierre de energía T₄→FAR→T₄ | derived | 1225 | 1225 K ±3 % | PASS | Δ = -0.000 % · VERIFICACIÓN, no validación (T₄ es θ[19]) |

**Huecos conocidos de este caso** (explicados en `docs/VALIDATION.md`):

* `FAR_global` — G1 · La tabla de clase se contradice a sí misma: el FAR que impone el balance de entalpía a partir de sus propias bandas de (T₃, T₄) no cabe en su banda de FAR/φ_global (ver A4). φ_global = FAR/FAR_st e I = FAR·LHV·P_atm/(R·T₃·τ) heredan el defecto por identidad. Recalibración: NINGUNA — hay que corregir `class_table.csv`, no el modelo. Tocar una constante para «cuadrar» esto sería calibrar contra un dato inconsistente.
* `phi_global` — G1 · La tabla de clase se contradice a sí misma: el FAR que impone el balance de entalpía a partir de sus propias bandas de (T₃, T₄) no cabe en su banda de FAR/φ_global (ver A4). φ_global = FAR/FAR_st e I = FAR·LHV·P_atm/(R·T₃·τ) heredan el defecto por identidad. Recalibración: NINGUNA — hay que corregir `class_table.csv`, no el modelo. Tocar una constante para «cuadrar» esto sería calibrar contra un dato inconsistente.
* `phi_pz` — G9 · φ_PZ fuera de banda. En A2/A3 hereda G1 (el FAR viene inflado). En D1 es el hallazgo de recuperación extrema: con T₃ = 1093 K el φ_global es 0.118 y ni con frac_pz en su MÍNIMO de contrato (0.15) se alcanza φ_PZ ≥ 0.8.
* `D_casing_mm` — G6 · Ø de carcasa por debajo de la banda de clase en las clases 1–2 (y en D1). A_ref sale del criterio aerodinámico con `PLF = 20` (fila `micro_target` de `plf_by_arch.csv`) para TODAS las arquitecturas, mientras la fila de la arquitectura da PLF = 36 (can) o 22.5 (reverse_annular): usar el PLF de la arquitectura subiría A_ref ×√(36/20) = 1.34 en `can`. Es el pendiente de recalibración nº 4.
* `I_MW_m3_atm` — G1 · La tabla de clase se contradice a sí misma: el FAR que impone el balance de entalpía a partir de sus propias bandas de (T₃, T₄) no cabe en su banda de FAR/φ_global (ver A4). φ_global = FAR/FAR_st e I = FAR·LHV·P_atm/(R·T₃·τ) heredan el defecto por identidad. Recalibración: NINGUNA — hay que corregir `class_table.csv`, no el modelo. Tocar una constante para «cuadrar» esto sería calibrar contra un dato inconsistente.
* `eta_comb_design` — G7 · η_comb por debajo de la banda en las clases 1–2. El driver es η_mix = ETA_CEILING·(1−e^{−K_MIX_DP·ΔP_L/q_ref})·penalización de penetración; `K_MIX_DP` es constante de la casa. La banda de clase pide >99 % / >99.5 % y el modelo da 98.8 % / 98.0 %.
* `T_wall_liner_K` — G16 · A2 (clase 1) se sale de la banda de T de pared POR DEBAJO (952 K frente a 1000–1150 K) desde el 2026-08-01, al corregir la corona liner↔carcasa: `A_an` pasó del atajo `A_ref − A_L` a la geométrica π(r_cas²−r_lo²), que es ×0.80 y por tanto refrigera más (más flujo másico por unidad de área en la convección externa C₂, Ciencia §5). La corrección es correcta —es la corona que se fabrica— y en el balance GANA: metió a B1 (994.6 K) y B2 (1012 K) dentro de banda y subió el marcador de 107 a 108 PASS. Pero deja a A2 sub-prediciendo un 4.8 %. Como G11, apunta al mismo sospechoso nº 1 de PENDIENTES §6.1 (`EPS_G_FLOOR_LUMINOUS`), que es quien domina T_w1 y está calibrado a un solo punto: con el término radiante mal anclado, cualquier cambio en el convectivo mueve la pared entera. Phy-CB **no predice T de metal cuantitativa en ningún nivel** (contrato §12): esto es una comprobación de banda.
* `n_eff_holes` — G3 · Nº de orificios de efusión por encima de la banda de clase. No es un error de cálculo: n = A_efusión/(π/4·d²) con la porosidad que sale de pitch/d, y `PENDIENTES §4.2` demuestra que hacer FACTIBLE el ancla exige apretar el paso a pitch/d = 4–5, lo que SUBE aún más el número. La banda 300–1500 de la clase 0 y las reglas de efusión densa del propio proyecto son incompatibles. Corregir el DATO. AMPLIADO EL 2026-07-31: la corrección del área mojada de la efusión (physics_core §12.b — π·D_L es el perímetro de un CÍRCULO de área equivalente y no el de la corona real, ×1.375 en el ancla) sube n en las arquitecturas ANULARES en ese mismo factor y mete también a A2 (clase 1, anular inversa: 3696 → 5080 frente a [1000, 4000]) en este hueco. Es la MISMA incompatibilidad de dato, ahora con el área correcta: el número de antes estaba dentro de la banda por un error de geometría, no por acuerdo con la clase. NO se ha movido ningún umbral para taparlo.
* `SMD_um` — G8 · SMD de airblast ×3–4 por debajo de la banda de clase (10.7 µm frente a 35–60 µm en la clase 1). Las dos correlaciones dimensionales de airblast SÍ están verificadas (arXiv:1906.04042 Tabla 7), así que el conflicto es entre dos fuentes `verified`. CONSECUENCIA GRAVE: φ_LBO ∝ D₃₂², de modo que un SMD 3.5× pequeño hace el margen de LBO ~12× optimista (LBO_margin = 42 en A2, o sea «esta cámara no se apaga nunca», que no es un resultado creíble). Es el pendiente de recalibración nº 2.

### `A3` · Clase 2 · microturbina 50–250 kW

**Fuente.** data/reference/class_table.csv (`@status: verified`), transcrita de docs/Quasar_PhyCB_EstadoDelArte.md §7 «Tabla de valores de referencia — clase microturbina»; las marcas [W] de esa tabla se verificaron en fuente web el 2026-07-29 (URLs en EstadoDelArte §12). Caveat de la propia tabla: las cifras certificadas de Capstone C30/C65 y Turbec T100 NO se verificaron.

**Supuestos declarados.**

* SUPUESTO · punto de operación = punto MEDIO de cada banda de la clase (π, T₃ no recuperada, ṁ, T₄). La tabla da bandas, no un punto: elegir el centro es la opción neutra y es lo que hace falsable el resultado (cualquier otra elección se podría acusar de buscar la banda).
* SUPUESTO · reparto de aire = fila `microturbine` de `data/lefebvre/air_split_by_arch.csv` (frac_pz/sz/dz = 0.30/0.12/0.42, `@status: verified`). NO se ajusta por caso: ajustar frac_pz para centrar φ_PZ sería exactamente la recalibración por-caso que la wave prohíbe (WAVE7 §5).
* SUPUESTO · n_holes_pz = 8. `class_table.csv` no da número de orificios de zona primaria (solo de dilución); 8 es el valor de THETA_REF.
* SUPUESTO · A_L/A_ref, L_PZ/D_L y L_DZ/D_L = defaults de `zone_lengths.csv` (tabla ⚠️ provisional entera, PENDIENTES §2.1 #2).
* SUPUESTO · arquitectura `can`: es la de las microturbinas de generación de esta clase (Capstone, prototipos universitarios — EstadoDelArte §2.2) y la que usa A_L/A_ref tubular 0.66.

**Notas.** Clase de Sierra Turbines (Hastelloy X). El PF medido en combustores pequeños reales, 0.21–0.24 [W], vive en esta banda — es el dato de PF más duro del paquete.

| Check | rol | valor | referencia | veredicto | detalle |
|---|---|---|---|---|---|
| pressure_ratio | input | 4.25 | [3.5, 5] - | PASS | status de la fila: verified |
| T4_K | input | 1275 | [1200, 1350] K | PASS | status de la fila: verified |
| mdot_air_kgs | input | 1 | [0.5, 1.5] kg/s | PASS | status de la fila: verified |
| dp_over_p | input | 0.03 | [0.03, 0.05] - | PASS | status de la fila: verified |
| t_liner_mm | input | 1.15 | [0.8, 1.5] mm | PASS | status de la fila: verified |
| d_eff_mm | input | 0.8 | [0.6, 1] mm | PASS | status de la fila: verified |
| SN_swirl | input | 0.75 | [0.6, 0.9] - | PASS | status de la fila: verified |
| FAR_global | derived | 0.02381 | [0.012, 0.02] - | FAIL (hueco conocido) | status de la fila: verified |
| phi_global | derived | 0.3466 | [0.18, 0.3] - | FAIL (hueco conocido) | status de la fila: verified |
| phi_pz | derived | 1.155 | [0.6, 0.9] - | FAIL (hueco conocido) | status de la fila: verified |
| U_ref_ms | derived | 20.54 | [15, 30] m/s | PASS | status de la fila: verified |
| tau_ms | derived | 11.33 | [5, 10] ms | FAIL (hueco conocido) | status de la fila: verified |
| V_c_L | derived | 3.699 | [1, 4] L | PASS | status de la fila: verified |
| D_casing_mm | derived | 142.3 | [180, 300] mm | FAIL (hueco conocido) | status de la fila: verified |
| I_MW_m3_atm | derived | 65.27 | [20, 45] MW/m3/atm | FAIL (hueco conocido) | status de la fila: verified |
| eta_comb_design | derived | 0.9804 | [0.995, 0.999] - | FAIL (hueco conocido) | status de la fila: verified |
| PF | derived | 0.35 | [0.2, 0.35] - | FAIL (hueco conocido) | status de la fila: verified |
| frac_cool | derived | 0.16 | [0.15, 0.3] - | PASS | status de la fila: verified |
| T_wall_liner_K | derived | 1055 | [1050, 1200] K | PASS | status de la fila: verified |
| n_eff_holes | derived | 5556 | [2000, 8000] - | PASS | status de la fila: verified |
| SMD_um | derived | 11.62 | [30, 50] um | FAIL (hueco conocido) | status de la fila: verified |
| cierre de energía T₄→FAR→T₄ | derived | 1275 | 1275 K ±3 % | PASS | Δ = -0.002 % · VERIFICACIÓN, no validación (T₄ es θ[19]) |

**Huecos conocidos de este caso** (explicados en `docs/VALIDATION.md`):

* `FAR_global` — G1 · La tabla de clase se contradice a sí misma: el FAR que impone el balance de entalpía a partir de sus propias bandas de (T₃, T₄) no cabe en su banda de FAR/φ_global (ver A4). φ_global = FAR/FAR_st e I = FAR·LHV·P_atm/(R·T₃·τ) heredan el defecto por identidad. Recalibración: NINGUNA — hay que corregir `class_table.csv`, no el modelo. Tocar una constante para «cuadrar» esto sería calibrar contra un dato inconsistente.
* `phi_global` — G1 · La tabla de clase se contradice a sí misma: el FAR que impone el balance de entalpía a partir de sus propias bandas de (T₃, T₄) no cabe en su banda de FAR/φ_global (ver A4). φ_global = FAR/FAR_st e I = FAR·LHV·P_atm/(R·T₃·τ) heredan el defecto por identidad. Recalibración: NINGUNA — hay que corregir `class_table.csv`, no el modelo. Tocar una constante para «cuadrar» esto sería calibrar contra un dato inconsistente.
* `phi_pz` — G9 · φ_PZ fuera de banda. En A2/A3 hereda G1 (el FAR viene inflado). En D1 es el hallazgo de recuperación extrema: con T₃ = 1093 K el φ_global es 0.118 y ni con frac_pz en su MÍNIMO de contrato (0.15) se alcanza φ_PZ ≥ 0.8.
* `tau_ms` — G10 · τ fuera de banda. τ = V_c·ρ₃/ṁ es continuidad; hereda el dimensionado (G6) y, en la clase 0 con ṁ alto, el mismo defecto de banda que G4.
* `D_casing_mm` — G6 · Ø de carcasa por debajo de la banda de clase en las clases 1–2 (y en D1). A_ref sale del criterio aerodinámico con `PLF = 20` (fila `micro_target` de `plf_by_arch.csv`) para TODAS las arquitecturas, mientras la fila de la arquitectura da PLF = 36 (can) o 22.5 (reverse_annular): usar el PLF de la arquitectura subiría A_ref ×√(36/20) = 1.34 en `can`. Es el pendiente de recalibración nº 4.
* `I_MW_m3_atm` — G1 · La tabla de clase se contradice a sí misma: el FAR que impone el balance de entalpía a partir de sus propias bandas de (T₃, T₄) no cabe en su banda de FAR/φ_global (ver A4). φ_global = FAR/FAR_st e I = FAR·LHV·P_atm/(R·T₃·τ) heredan el defecto por identidad. Recalibración: NINGUNA — hay que corregir `class_table.csv`, no el modelo. Tocar una constante para «cuadrar» esto sería calibrar contra un dato inconsistente.
* `eta_comb_design` — G7 · η_comb por debajo de la banda en las clases 1–2. El driver es η_mix = ETA_CEILING·(1−e^{−K_MIX_DP·ΔP_L/q_ref})·penalización de penetración; `K_MIX_DP` es constante de la casa. La banda de clase pide >99 % / >99.5 % y el modelo da 98.8 % / 98.0 %.
* `PF` — G5 · PF = 0.35000 en TODOS los casos, con cualquier clase y cualquier punto de operación. No es una predicción: es la calibración devuelta. PF = exp[−k·(L_DZ/D_L)·(ΔP_L/q_ref)] con k calibrada a UN punto, y con el criterio aerodinámico dimensionando A_ref el cociente ΔP_L/q_ref = 15 es una identidad algebraica (PENDIENTES §5). Es el pendiente de recalibración nº 1 de esta campaña.
* `SMD_um` — G8 · SMD de airblast ×3–4 por debajo de la banda de clase (10.7 µm frente a 35–60 µm en la clase 1). Las dos correlaciones dimensionales de airblast SÍ están verificadas (arXiv:1906.04042 Tabla 7), así que el conflicto es entre dos fuentes `verified`. CONSECUENCIA GRAVE: φ_LBO ∝ D₃₂², de modo que un SMD 3.5× pequeño hace el margen de LBO ~12× optimista (LBO_margin = 42 en A2, o sea «esta cámara no se apaga nunca», que no es un resultado creíble). Es el pendiente de recalibración nº 2.

### `A4` · Consistencia interna de `class_table.csv` (FAR ↔ T₃,T₄)

**Fuente.** data/reference/class_table.csv (`@status: verified`), transcrita de docs/Quasar_PhyCB_EstadoDelArte.md §7 «Tabla de valores de referencia — clase microturbina»; las marcas [W] de esa tabla se verificaron en fuente web el 2026-07-29 (URLs en EstadoDelArte §12). Caveat de la propia tabla: las cifras certificadas de Capstone C30/C65 y Turbec T100 NO se verificaron.

**Supuestos declarados.**

* El balance usado es el MISMO del ciclo (CONTEXTO §4 y `physics_core._far_from_T4`): f = (cp_h·T₄ − cp_c·T₃) / (η_b·LHV − cp_h·T₄) con cp_h = 1148, cp_c = 1005, η_b = 0.995, LHV = 43.1 MJ/kg. Es álgebra, no una correlación: no hay ningún coeficiente provisional dentro.
* SUPUESTO · combustible = Jet-A. Con propano o H₂ el LHV cambia y con él la banda de FAR; la tabla de clase no declara combustible, así que se toma el default del proyecto.

**Notas.** Este caso NO valida el modelo: valida el DATO. Si la banda de FAR/φ_global de la tabla y su banda de (T₃, T₄) no pueden ser ciertas a la vez, ninguna comparación contra esas bandas significa nada. Se comprueba en las dos ramas de T₃ (no recuperada y recuperada).

| Check | rol | valor | referencia | veredicto | detalle |
|---|---|---|---|---|---|
| clase 0 · T₃ no recuperada | data | [0.01899, 0.02492] | solapa con [0.01, 0.018] | FAIL (hueco conocido) | T₃ ∈ [400, 470] K · T₄ ∈ [1100, 1250] K |
| clase 0 · V_c ↔ (τ, ṁ, ρ₃) | data | [0.06474, 1.331] L | solapa con [0.05, 0.4] L | PASS | continuidad V_c = τ·ṁ/ρ₃ (álgebra) · en el CENTRO de las bandas V_c = 0.4032 L |
| clase 0 · I ↔ (FAR, T₃, τ) | data | [40.47, 171.2] MW/m³·atm | solapa con [25, 60] | PASS | identidad I = FAR·LHV·P_atm/(R·T₃·τ) (physics_core §7, auditoría 2026-07-30) · en el CENTRO de las bandas I = 81.62 |
| clase 1 · T₃ no recuperada | data | [0.01967, 0.02561] | solapa con [0.012, 0.02] | PASS | T₃ ∈ [430, 500] K · T₄ ∈ [1150, 1300] K |
| clase 1 · T₃ recuperada | data | [0.008792, 0.01663] | solapa con [0.012, 0.02] | PASS | T₃ ∈ [800, 950] K · T₄ ∈ [1150, 1300] K |
| clase 1 · V_c ↔ (τ, ṁ, ρ₃) | data | [0.203, 2.832] L | solapa con [0.3, 1.5] L | PASS | continuidad V_c = τ·ṁ/ρ₃ (álgebra) · en el CENTRO de las bandas V_c = 0.9878 L |
| clase 1 · I ↔ (FAR, T₃, τ) | data | [36.52, 141.5] MW/m³·atm | solapa con [20, 50] | PASS | identidad I = FAR·LHV·P_atm/(R·T₃·τ) (physics_core §7, auditoría 2026-07-30) · en el CENTRO de las bandas I = 69.81 |
| clase 2 · T₃ no recuperada | data | [0.02036, 0.02655] | solapa con [0.012, 0.02] | FAIL (hueco conocido) | T₃ ∈ [450, 530] K · T₄ ∈ [1200, 1350] K |
| clase 2 · T₃ recuperada | data | [0.006556, 0.01561] | solapa con [0.012, 0.02] | PASS | T₃ ∈ [900, 1100] K · T₄ ∈ [1200, 1350] K |
| clase 2 · V_c ↔ (τ, ṁ, ρ₃) | data | [0.6373, 6.434] L | solapa con [1, 4] L | PASS | continuidad V_c = τ·ṁ/ρ₃ (álgebra) · en el CENTRO de las bandas V_c = 2.449 L |
| clase 2 · I ↔ (FAR, T₃, τ) | data | [34.45, 135.3] MW/m³·atm | solapa con [20, 45] | PASS | identidad I = FAR·LHV·P_atm/(R·T₃·τ) (physics_core §7, auditoría 2026-07-30) · en el CENTRO de las bandas I = 66.25 |

**Huecos conocidos de este caso** (explicados en `docs/VALIDATION.md`):

* `clase 0 · T₃ no recuperada` — G1 · La tabla de clase se contradice a sí misma: el FAR que impone el balance de entalpía a partir de sus propias bandas de (T₃, T₄) no cabe en su banda de FAR/φ_global (ver A4). φ_global = FAR/FAR_st e I = FAR·LHV·P_atm/(R·T₃·τ) heredan el defecto por identidad. Recalibración: NINGUNA — hay que corregir `class_table.csv`, no el modelo. Tocar una constante para «cuadrar» esto sería calibrar contra un dato inconsistente.
* `clase 2 · T₃ no recuperada` — G1 · La tabla de clase se contradice a sí misma: el FAR que impone el balance de entalpía a partir de sus propias bandas de (T₃, T₄) no cabe en su banda de FAR/φ_global (ver A4). φ_global = FAR/FAR_st e I = FAR·LHV·P_atm/(R·T₃·τ) heredan el defecto por identidad. Recalibración: NINGUNA — hay que corregir `class_table.csv`, no el modelo. Tocar una constante para «cuadrar» esto sería calibrar contra un dato inconsistente.

### `B1` · KJ66 / clase hobby · anular reverso + vaporizador

**Fuente.** Arquitectura y geometría del vaporizador: docs/Quasar_PhyCB_EstadoDelArte.md §2.2, marca [W] VERIFICADA («6–12 tubos *walking stick* de ~64 mm con codo ~30°»), planos públicos de la comunidad (gtba.co.uk; plano Englar). Punto de operación: `THETA_REF` del contrato §1 (caso ancla REF_CB10).

**Supuestos declarados.**

* SUPUESTO · punto de operación (P₃ = 250 kPa, T₃ = 430 K, ṁ = 0.13 kg/s, T₄ = 1150 K) = `THETA_REF` del contrato §1. NO es una medición del KJ66: es el punto que el proyecto declara como ancla de clase hobby.
* DISCREPANCIA DECLARADA · las cifras de comunidad del KJ66 que cita `Phy-CC/docs/VALIDACION_ES.md` §2b son π ≈ 2.2, ṁ ≈ 0.23 kg/s, ~117 krpm — ṁ casi el DOBLE del punto ancla. El caso se corre en el punto ancla y la variante 0.23 kg/s se reporta como sensibilidad (caso B2), sin decidir cuál es «el» KJ66: ninguna de las dos está verificada en esta sesión.
* NO HAY DATO · el paquete no contiene ninguna medición de prestaciones del KJ66 (empuje, EGT, consumo, Δp frío, PF). Por tanto B **no puede falsar prestaciones**; solo arquitectura y bandas de clase. Decirlo es el entregable.

**Notas.** Comprueba además que el modelo devuelve la geometría de vaporizador verificada (nº de tubos, longitud, codo). Eso es CONSISTENCIA, no predicción: el dato entra por `physics_core` y sale por el record — vale para detectar que alguien lo rompa, no para validar física.

| Check | rol | valor | referencia | veredicto | detalle |
|---|---|---|---|---|---|
| pressure_ratio | input | 2.467 | [2, 3.5] - | PASS | status de la fila: verified |
| T4_K | input | 1150 | [1100, 1250] K | PASS | status de la fila: verified |
| mdot_air_kgs | input | 0.13 | [0.05, 0.25] kg/s | PASS | status de la fila: verified |
| dp_over_p | input | 0.03 | [0.02, 0.04] - | PASS | status de la fila: verified |
| t_liner_mm | input | 0.8 | [0.5, 0.8] mm | PASS | status de la fila: verified |
| d_eff_mm | input | 0.6 | [0.5, 0.7] mm | PASS | status de la fila: verified |
| SN_swirl | input | 0.7 | [0.7, 1] - | PASS | status de la fila: verified |
| FAR_global | derived | 0.0216 | [0.01, 0.018] - | FAIL (hueco conocido) | status de la fila: verified |
| phi_global | derived | 0.3145 | [0.15, 0.27] - | FAIL (hueco conocido) | status de la fila: verified |
| phi_pz | derived | 1.048 | [0.85, 1.15] - | PASS | status de la fila: verified |
| U_ref_ms | derived | 19.24 | [8, 20] m/s | PASS | status de la fila: verified |
| tau_ms | derived | 6.05 | [4, 8] ms | PASS | status de la fila: verified |
| V_c_L | derived | 0.3882 | [0.05, 0.4] L | PASS | status de la fila: verified |
| D_casing_mm | derived | 81.46 | [80, 130] mm | PASS | status de la fila: verified |
| I_MW_m3_atm | derived | 126.3 | [25, 60] MW/m3/atm | FAIL (hueco conocido) | status de la fila: verified |
| eta_comb_design | derived | 0.9846 | [0.97, 0.99] - | PASS | status de la fila: verified |
| PF | derived | 0.35 | [0.35, 0.5] - | PASS | status de la fila: verified |
| frac_cool | derived | 0.16 | [0, 0.15] - | FAIL (hueco conocido) | status de la fila: verified |
| T_wall_liner_K | derived | 994.6 | [900, 1050] K | PASS | status de la fila: verified |
| n_eff_holes | derived | 3020 | [300, 1500] - | FAIL (hueco conocido) | status de la fila: verified |
| SMD_um | derived | 50 | [40, 70] um | PASS | status de la fila: verified |
| vaporizador KJ66 (nº tubos) | data | 6 | [6, 12] tubos ✅ [W] | PASS | CONSISTENCIA con el dato, no predicción |
| vaporizador KJ66 (L tubo) | data | 64 | 64 ± 5 mm ✅ [W] | PASS | ídem |
| vaporizador KJ66 (codo) | data | 30 | 30 ± 5 deg ✅ [W] | PASS | ídem |
| arquitectura | input | reverse_annular | reverse_annular | PASS | — |
| cierre de energía T₄→FAR→T₄ | derived | 1150 | 1150 K ±3 % | PASS | Δ = +0.000 % · VERIFICACIÓN, no validación (T₄ es θ[19]) |

**Huecos conocidos de este caso** (explicados en `docs/VALIDATION.md`):

* `FAR_global` — G1 · La tabla de clase se contradice a sí misma: el FAR que impone el balance de entalpía a partir de sus propias bandas de (T₃, T₄) no cabe en su banda de FAR/φ_global (ver A4). φ_global = FAR/FAR_st e I = FAR·LHV·P_atm/(R·T₃·τ) heredan el defecto por identidad. Recalibración: NINGUNA — hay que corregir `class_table.csv`, no el modelo. Tocar una constante para «cuadrar» esto sería calibrar contra un dato inconsistente.
* `phi_global` — G1 · La tabla de clase se contradice a sí misma: el FAR que impone el balance de entalpía a partir de sus propias bandas de (T₃, T₄) no cabe en su banda de FAR/φ_global (ver A4). φ_global = FAR/FAR_st e I = FAR·LHV·P_atm/(R·T₃·τ) heredan el defecto por identidad. Recalibración: NINGUNA — hay que corregir `class_table.csv`, no el modelo. Tocar una constante para «cuadrar» esto sería calibrar contra un dato inconsistente.
* `I_MW_m3_atm` — G1 · La tabla de clase se contradice a sí misma: el FAR que impone el balance de entalpía a partir de sus propias bandas de (T₃, T₄) no cabe en su banda de FAR/φ_global (ver A4). φ_global = FAR/FAR_st e I = FAR·LHV·P_atm/(R·T₃·τ) heredan el defecto por identidad. Recalibración: NINGUNA — hay que corregir `class_table.csv`, no el modelo. Tocar una constante para «cuadrar» esto sería calibrar contra un dato inconsistente.
* `frac_cool` — G2 · Dos tablas `verified` del paquete se contradicen en 1 punto: `air_split_by_arch.csv::microturbine.frac_cool_nom` = 0.16 frente a `class_table.csv::clase 0 frac_cool` = 0–0.15. THETA_REF usa 0.16. Es un defecto de DATO, de impacto nulo en la física.
* `n_eff_holes` — G3 · Nº de orificios de efusión por encima de la banda de clase. No es un error de cálculo: n = A_efusión/(π/4·d²) con la porosidad que sale de pitch/d, y `PENDIENTES §4.2` demuestra que hacer FACTIBLE el ancla exige apretar el paso a pitch/d = 4–5, lo que SUBE aún más el número. La banda 300–1500 de la clase 0 y las reglas de efusión densa del propio proyecto son incompatibles. Corregir el DATO. AMPLIADO EL 2026-07-31: la corrección del área mojada de la efusión (physics_core §12.b — π·D_L es el perímetro de un CÍRCULO de área equivalente y no el de la corona real, ×1.375 en el ancla) sube n en las arquitecturas ANULARES en ese mismo factor y mete también a A2 (clase 1, anular inversa: 3696 → 5080 frente a [1000, 4000]) en este hueco. Es la MISMA incompatibilidad de dato, ahora con el área correcta: el número de antes estaba dentro de la banda por un error de geometría, no por acuerdo con la clase. NO se ha movido ningún umbral para taparlo.

### `B2` · KJ66 · sensibilidad al ṁ de comunidad (0.23 kg/s)

**Fuente.** Cifras de comunidad citadas en `Phy-CC/docs/VALIDACION_ES.md` §2b (KJ66: PR ≈ 2.2, 0.23 kg/s, ~117 krpm). ⚠️ NO verificadas en esta sesión.

**Supuestos declarados.**

* SUPUESTO · T₃ = 430 K y T₄ = 1150 K se mantienen del punto ancla: la fuente de comunidad no da ninguna de las dos.
* Es una SENSIBILIDAD, no un caso independiente: mide cuánto se mueve el dimensionado al duplicar el gasto dentro de la misma clase. Su verdict no debe leerse como validación del KJ66.

**Notas.** ṁ = 0.23 kg/s sigue dentro de la banda de clase 0 (0.05–0.25 kg/s), así que el caso es interno a la clase.

| Check | rol | valor | referencia | veredicto | detalle |
|---|---|---|---|---|---|
| pressure_ratio | input | 2.2 | [2, 3.5] - | PASS | status de la fila: verified |
| T4_K | input | 1150 | [1100, 1250] K | PASS | status de la fila: verified |
| mdot_air_kgs | input | 0.23 | [0.05, 0.25] kg/s | PASS | status de la fila: verified |
| dp_over_p | input | 0.03 | [0.02, 0.04] - | PASS | status de la fila: verified |
| t_liner_mm | input | 0.8 | [0.5, 0.8] mm | PASS | status de la fila: verified |
| d_eff_mm | input | 0.6 | [0.5, 0.7] mm | PASS | status de la fila: verified |
| SN_swirl | input | 0.7 | [0.7, 1] - | PASS | status de la fila: verified |
| FAR_global | derived | 0.02151 | [0.01, 0.018] - | FAIL (hueco conocido) | status de la fila: verified |
| phi_global | derived | 0.3132 | [0.15, 0.27] - | FAIL (hueco conocido) | status de la fila: verified |
| phi_pz | derived | 1.044 | [0.85, 1.15] - | PASS | status de la fila: verified |
| U_ref_ms | derived | 19.24 | [8, 20] m/s | PASS | status de la fila: verified |
| tau_ms | derived | 8.522 | [4, 8] ms | FAIL (hueco conocido) | status de la fila: verified |
| V_c_L | derived | 1.085 | [0.05, 0.4] L | FAIL (hueco conocido) | status de la fila: verified |
| D_casing_mm | derived | 114.7 | [80, 130] mm | PASS | status de la fila: verified |
| I_MW_m3_atm | derived | 89.34 | [25, 60] MW/m3/atm | FAIL (hueco conocido) | status de la fila: verified |
| eta_comb_design | derived | 0.9883 | [0.97, 0.99] - | PASS | status de la fila: verified |
| PF | derived | 0.35 | [0.35, 0.5] - | PASS | status de la fila: verified |
| frac_cool | derived | 0.16 | [0, 0.15] - | FAIL (hueco conocido) | status de la fila: verified |
| T_wall_liner_K | derived | 1012 | [900, 1050] K | PASS | status de la fila: verified |
| n_eff_holes | derived | 5993 | [300, 1500] - | FAIL (hueco conocido) | status de la fila: verified |
| SMD_um | derived | 50 | [40, 70] um | PASS | status de la fila: verified |
| cierre de energía T₄→FAR→T₄ | derived | 1150 | 1150 K ±3 % | PASS | Δ = -0.000 % · VERIFICACIÓN, no validación (T₄ es θ[19]) |

**Huecos conocidos de este caso** (explicados en `docs/VALIDATION.md`):

* `FAR_global` — G1 · La tabla de clase se contradice a sí misma: el FAR que impone el balance de entalpía a partir de sus propias bandas de (T₃, T₄) no cabe en su banda de FAR/φ_global (ver A4). φ_global = FAR/FAR_st e I = FAR·LHV·P_atm/(R·T₃·τ) heredan el defecto por identidad. Recalibración: NINGUNA — hay que corregir `class_table.csv`, no el modelo. Tocar una constante para «cuadrar» esto sería calibrar contra un dato inconsistente.
* `phi_global` — G1 · La tabla de clase se contradice a sí misma: el FAR que impone el balance de entalpía a partir de sus propias bandas de (T₃, T₄) no cabe en su banda de FAR/φ_global (ver A4). φ_global = FAR/FAR_st e I = FAR·LHV·P_atm/(R·T₃·τ) heredan el defecto por identidad. Recalibración: NINGUNA — hay que corregir `class_table.csv`, no el modelo. Tocar una constante para «cuadrar» esto sería calibrar contra un dato inconsistente.
* `tau_ms` — G10 · τ fuera de banda. τ = V_c·ρ₃/ṁ es continuidad; hereda el dimensionado (G6) y, en la clase 0 con ṁ alto, el mismo defecto de banda que G4.
* `V_c_L` — G4 · V_c fuera de banda por continuidad: V_c = τ·ṁ/ρ₃. Con τ dentro de banda, V_c queda determinado por ṁ y ρ₃; en el extremo alto de la banda de ṁ de la clase 0 el V_c que exige la continuidad sobrepasa el techo de 0.4 L. Ver la pata 2 de A4.
* `I_MW_m3_atm` — G1 · La tabla de clase se contradice a sí misma: el FAR que impone el balance de entalpía a partir de sus propias bandas de (T₃, T₄) no cabe en su banda de FAR/φ_global (ver A4). φ_global = FAR/FAR_st e I = FAR·LHV·P_atm/(R·T₃·τ) heredan el defecto por identidad. Recalibración: NINGUNA — hay que corregir `class_table.csv`, no el modelo. Tocar una constante para «cuadrar» esto sería calibrar contra un dato inconsistente.
* `frac_cool` — G2 · Dos tablas `verified` del paquete se contradicen en 1 punto: `air_split_by_arch.csv::microturbine.frac_cool_nom` = 0.16 frente a `class_table.csv::clase 0 frac_cool` = 0–0.15. THETA_REF usa 0.16. Es un defecto de DATO, de impacto nulo en la física.
* `n_eff_holes` — G3 · Nº de orificios de efusión por encima de la banda de clase. No es un error de cálculo: n = A_efusión/(π/4·d²) con la porosidad que sale de pitch/d, y `PENDIENTES §4.2` demuestra que hacer FACTIBLE el ancla exige apretar el paso a pitch/d = 4–5, lo que SUBE aún más el número. La banda 300–1500 de la clase 0 y las reglas de efusión densa del propio proyecto son incompatibles. Corregir el DATO. AMPLIADO EL 2026-07-31: la corrección del área mojada de la efusión (physics_core §12.b — π·D_L es el perímetro de un CÍRCULO de área equivalente y no el de la corona real, ×1.375 en el ancla) sube n en las arquitecturas ANULARES en ese mismo factor y mete también a A2 (clase 1, anular inversa: 3696 → 5080 frente a [1000, 4000]) en este hueco. Es la MISMA incompatibilidad de dato, ahora con el área correcta: el número de antes estaba dentro de la banda por un error de geometría, no por acuerdo con la clase. NO se ha movido ningún umbral para taparlo.

### `C1` · Microturborreactor optimizado · reparto PZ 39.35 % → 32 %

**Fuente.** Wiley, International Journal of Aerospace Engineering 2021:6616537 (optimización de un microturborreactor real). Dato citado y marcado [W] verificado en docs/Quasar_PhyCB_EstadoDelArte.md §1 y §12, y transcrito a la cabecera de `data/lefebvre/air_split_by_arch.csv`: la optimización bajó el aire de zona primaria de **39.35 % a 32 %** (φ_PZ **0.89 → 1.09**) para estabilizar la llama durante la aceleración.

**Supuestos declarados.**

* DERIVADO de la fuente (no supuesto) · φ_global = φ_PZ·frac_pz = 0.89·0.3935 = 0.35022 y 1.09·0.32 = 0.34880; media 0.34951, dispersión ±0.20 %. Los dos puntos publicados son mutuamente consistentes con la definición de Ciencia §3.
* SUPUESTO · P₃ = 250 kPa, T₃ = 430 K, ṁ = 0.13 kg/s: el paper no da el punto de operación en la cita disponible. Se toman los del ancla (clase 0, dentro de banda). **No afectan al check**: φ_PZ = FAR/(frac_pz·FAR_st) y el runner ajusta T₄ para clavar la φ_global derivada, así que el caso mide el reparto de aire, no el punto.
* SUPUESTO · T₄ se resuelve por bisección acotada (60 pasos, 800–1800 K) para que `record['phi_global']` = 0.34951. Es la misma técnica que `Phy-AC/validation/validate.build_theta` (bisección de φ1 para clavar el annulus publicado): se le da al modelo el dato medido y se califica lo que él deduce.
* SUPUESTO · frac_sz y frac_dz se reescalan proporcionalmente para dejar frac_cool = 0.16 (nominal `microturbine`) en los dos puntos: la fuente solo publica frac_pz.

**Notas.** Es el caso MÁS falsable del paquete: dos puntos publicados de la misma máquina con el mismo φ_global. C1a comprueba el INVARIANTE (φ_PZ,base/φ_PZ,opt = frac_pz,opt/frac_pz,base) que no depende del punto de operación; C1b comprueba el valor ABSOLUTO de φ_PZ en cada punto.

| Check | rol | valor | referencia | veredicto | detalle |
|---|---|---|---|---|---|
| invariante φ_PZ,opt/φ_PZ,base | derived | 1.208 | 1.225 ±2 % | PASS | Δ = -1.32 % · independiente del punto de operación: mide SOLO la cadena reparto→φ_PZ |
| φ_PZ · base (frac_pz = 0.3935) | derived | 0.9038 | 0.89 ±5 % | PASS | Δ = +1.55 % |
| φ_PZ · optimizado (frac_pz = 0.3200) | derived | 1.092 | 1.09 ±5 % | PASS | Δ = +0.20 % |
| signo · menos aire de PZ ⇒ φ_PZ mayor | derived | 0.9038 → 1.092 | creciente | PASS | tendencia publicada: 0.89 → 1.09 |
| T₄ resuelto por bisección | input | 1234 | — | PASS | φ_global objetivo = 0.3495 (derivada de la fuente) |

| punto | frac_pz | φ_PZ publicada | φ_PZ modelo | Δ |
|---|---|---|---|---|
| base | 0.3935 | 0.890 | 0.9038 | +1.55 % |
| optimizado | 0.3200 | 1.090 | 1.0922 | +0.20 % |

### `D1` · Bath/IAAPS/HiETA · IN625, swirl radial, T₃ = 1093 K

**Fuente.** Energy Conversion and Management 248:114805 (2021) e International Journal of Engine Research (2023), vía docs/Quasar_PhyCB_EstadoDelArte.md §3.4 (marca [W] verificada): combustor tubular cónico con swirl radial e inyección de combustible por los álabes del swirler («solo fabricable por AM»), **Inconel 625**, CFD flamelet validado, −75 % NOx / −40 % CO / +20 % calidad de mezcla con upstream fuelling + lattice de tres filas, integrado con recuperador AM de HiETA que eleva **T₃ a 820 °C = 1093 K** — clase Bladon 12 kW. El T₃ recuperado está además en la cabecera de `data/reference/class_table.csv`.

**Supuestos declarados.**

* VERIFICADO · T₃ = 1093 K (820 °C) y material IN625 son dato publicado [W]. Todo lo demás de este caso es supuesto.
* SUPUESTO · ṁ = 0.15 kg/s (borde bajo de la banda de clase 1). Con T₃ = 1093 K y T₄ = 1250 K el balance da FAR ≈ 0.0081 ⇒ ~52 kW térmicos ⇒ ~23 % de rendimiento eléctrico para los 12 kW de la clase Bladon. Es plausible pero NO está medido.
* SUPUESTO · T₄ = 1250 K (banda de clase 1: 1150–1300 K). El paper no da TIT en la cita disponible.
* SUPUESTO · π = 3.75 (medio de la banda de clase 1).
* SUPUESTO · frac_pz = 0.15, que es el **mínimo del contrato §1**. No es una elección: es el borde. Con φ_global = 0.118 (la que impone el balance a T₃ = 1093 K) ni siquiera el borde alcanza la banda de φ_PZ de la clase — ver el hallazgo de docs/VALIDATION.md sobre φ_PZ en recuperación extrema.
* CIERRE DE DISEÑO DECLARADO · frac_sz = frac_dz = 0.10 ⇒ frac_cool = 0.65. θ[2..4] son VARIABLES DE DISEÑO, no constantes del modelo: fijarlas es diseñar, no calibrar. El reparto «ingenuo» (mandar el aire sobrante a dilución: frac_dz = 0.57) se probó primero y **colapsa**: Y/D_L = 1.45 frente al objetivo 0.5 (sobre-penetración masiva), η_mix cae a su suelo 0.049 y η_comb a 0.05, y ni llevando n_holes_dz al tope del contrato (18) baja Y/D_L de 1.01. Es un hallazgo del caso, no un ajuste: en recuperación extrema el aire sobrante NO puede ir por los chorros de dilución. Se reporta en docs/VALIDATION.md.
* SUPUESTO · n_holes_dz = 16 (banda de clase 1: 8–12 por fila; aquí hace falta más para repartir el chorro). Se declara que queda fuera de la banda de la tabla.
* SUPUESTO · arquitectura `can` como aproximación del tubular cónico del paper. Phy-CB no tiene gate para «tubular cónico con swirl radial»; `can` es la más cercana (un solo tubo, un inyector). El PLF y U_ref que hereda son los de `can`.
* NO MODELABLE · el −75 % NOx / −40 % CO del lattice de tres filas es un efecto de HETEROGENEIDAD de mezcla. El NOx de L0 es ZONAL (una sola φ_PZ, Ciencia §8) y no tiene submodelo de pockets ricos: **no puede reproducir ese número y no se intenta**. Lo que sí se califica es el SIGNO de ∂EI/∂φ_PZ y ∂EI/∂τ (caso D2), que es el criterio de la wave.

**Notas.** El caso de estrés térmico del paquete: con el aire de refrigeración a 1093 K el liner arranca ya caliente. Bath lo FABRICÓ y lo hizo funcionar en IN625; si el térmico 1D de Phy-CB declara infactible ese material, el modelo está conservador y el primer sospechoso es el suelo de ε_g (PENDIENTES §6.1), no el material.

| Check | rol | valor | referencia | veredicto | detalle |
|---|---|---|---|---|---|
| T_wall_liner_K | derived | 1466 | [1000, 1150] K | FAIL (hueco conocido) | BANDA, no valor: Phy-CB no predice T de metal cuantitativa en ningún nivel (contrato §12) |
| Inconel-625 soporta la pared predicha | derived | 1466 | ≤ 1203 K (Inconel-625 − margen 50 K) | FAIL (hueco conocido) | Bath FABRICÓ y operó este combustor en IN625: si aquí sale infactible, el conservadurismo es del modelo |
| T_w1 > T₃ (el refrigerante ya llega a 1093 K) | derived | 1466 | > 1093 K | PASS | test de cordura del térmico 1D en recuperación extrema |
| phi_pz | derived | 0.7937 | [0.8, 1.1] | FAIL (hueco conocido) | frac_pz = 0.150 = MÍNIMO del contrato §1 (bound inferior 0.15) |
| tau_ms | derived | 3.822 | [5, 10] ms | FAIL (hueco conocido) | — |
| V_c_L | derived | 0.4733 | [0.3, 1.5] L | PASS | — |
| D_casing_mm | derived | 71.69 | [120, 200] mm | FAIL (hueco conocido) | — |
| eta_comb_design | derived | 0.9882 | [0.99, 0.999] - | FAIL (hueco conocido) | — |
| PF | derived | 0.35 | [0.25, 0.4] - | PASS | — |
| cierre de energía T₄→FAR→T₄ | derived | 1250 | 1250 K ±3 % | PASS | Δ = -0.000 % · VERIFICACIÓN, no validación (T₄ es θ[19]) |

**Huecos conocidos de este caso** (explicados en `docs/VALIDATION.md`):

* `T_wall_liner_K` — G11 · T de pared fuera de la banda de clase. **Solo queda D1**, y CAMBIÓ DE SIGNO el 2026-08-01 al corregir la corona geométrica: era 1466 K (+27 % sobre 1000–1150 K) y ahora es 952 K (−4.8 % por DEBAJO). Está mucho más cerca, pero el modelo pasó de sobre- a sub-predecir. B1 y B2 salieron del hueco (994.6 K y 1012 K, dentro de banda). Sospechoso nº 1 de PENDIENTES §6.1: el suelo `EPS_G_FLOOR_LUMINOUS = 0.15`, calibración de la casa que hoy SUSTITUYE a la física declarada (ε_g calculado = 0.0065) y que al barrer su banda de literatura mueve T_w1 entre 808 K y 1573 K. Phy-CB **no predice T de metal cuantitativa en ningún nivel** (contrato §12): esto es una comprobación de banda.
* `Inconel-625 soporta la pared predicha` — G12 · El modelo declara IN625 insuficiente (T_w1 = 1466 K frente al límite 1253 K − 50 K de margen) en el caso Bath/HiETA, que está FABRICADO y OPERADO en IN625. Es la falsación más limpia de la campaña y apunta al mismo sitio que G11.
* `phi_pz` — G9 · φ_PZ fuera de banda. En A2/A3 hereda G1 (el FAR viene inflado). En D1 es el hallazgo de recuperación extrema: con T₃ = 1093 K el φ_global es 0.118 y ni con frac_pz en su MÍNIMO de contrato (0.15) se alcanza φ_PZ ≥ 0.8.
* `tau_ms` — G10 · τ fuera de banda. τ = V_c·ρ₃/ṁ es continuidad; hereda el dimensionado (G6) y, en la clase 0 con ṁ alto, el mismo defecto de banda que G4.
* `D_casing_mm` — G6 · Ø de carcasa por debajo de la banda de clase en las clases 1–2 (y en D1). A_ref sale del criterio aerodinámico con `PLF = 20` (fila `micro_target` de `plf_by_arch.csv`) para TODAS las arquitecturas, mientras la fila de la arquitectura da PLF = 36 (can) o 22.5 (reverse_annular): usar el PLF de la arquitectura subiría A_ref ×√(36/20) = 1.34 en `can`. Es el pendiente de recalibración nº 4.
* `eta_comb_design` — G7 · η_comb por debajo de la banda en las clases 1–2. El driver es η_mix = ETA_CEILING·(1−e^{−K_MIX_DP·ΔP_L/q_ref})·penalización de penetración; `K_MIX_DP` es constante de la casa. La banda de clase pide >99 % / >99.5 % y el modelo da 98.8 % / 98.0 %.

### `D2` · Tendencias de emisiones · signo de ∂EI/∂φ_PZ y ∂EI/∂τ

**Fuente.** Criterio de WAVE7 §3.2 y Ciencia §13.2: de las emisiones **nunca** se promete valor absoluto, solo tendencia y orden de magnitud (×2). Signos esperados por física de Zeldovich y de oxidación de CO (Ciencia §8): ∂EI_NOx/∂φ_PZ > 0 en el lado POBRE (T_ad sube con φ hasta ≈1.05), ∂EI_NOx/∂τ > 0, ∂EI_CO/∂τ < 0 y ∂EI_CO/∂φ_PZ < 0 en el lado pobre. Banda de orden de magnitud para queroseno de clase micro: 0.1–5 g/kg, derivada por la casa en `physics_core.py` §3 a partir de `emissions_coeffs.csv::reference_micro` (NOx 35–80 ppm@15 %O₂ ≈ 5 g/kg en el techo).

**Supuestos declarados.**

* SUPUESTO · el barrido se hace alrededor de THETA_REF. Las tendencias son locales: no se afirma nada fuera de ese entorno.
* DECLARADO · el signo se evalúa en el lado POBRE (φ_PZ < 1). En el entorno de φ_PZ ≈ 1.05 (donde vive el ancla) la derivada del NOx frente a φ es ≈ 0 por construcción física (pico de T_ad): pedir un signo ahí sería pedir ruido.
* La banda de orden de magnitud 0.1–5 g/kg es una derivación de la casa (ppm@15 %O₂ → g/kg), NO una medición directa. Se usa solo para el criterio ×2, que es lo que la wave permite.

**Notas.** Cubre además el abanico completo de EI_NOx (5 correlaciones) y, si hay Cantera, el EI_NOx del CRN L1 — que es el número menos empírico del paquete.

| Check | rol | valor | referencia | veredicto | detalle |
|---|---|---|---|---|---|
| signo ∂/∂φ_PZ · EI_NOx (cabecera = lefebvre) | derived | -2.068 | > 0 (Zeldovich: T_ad ↑ con φ en el lado pobre) | FAIL (hueco conocido) | variación relativa 16.33 %  · φ_PZ ∈ [0.6044, 0.7966], 5 puntos pobres |
| signo ∂/∂φ_PZ · EI_NOx_lefebvre | derived | -2.068 | > 0 (Zeldovich: T_ad ↑ con φ en el lado pobre) | FAIL (hueco conocido) | variación relativa 16.33 %  · φ_PZ ∈ [0.6044, 0.7966], 5 puntos pobres |
| signo ∂/∂φ_PZ · EI_NOx_rizk_mongia | derived | 1.451e-15 | > 0 (Zeldovich: T_ad ↑ con φ en el lado pobre) | FAIL (hueco conocido) | variación relativa 0.00 % ⇒ INSENSIBLE (< 2 %): el signo no significa nada · φ_PZ ∈ [0.6044, 0.7966], 5 puntos pobres |
| signo ∂/∂φ_PZ · EI_NOx_zonal | derived | 5.242 | > 0 (Zeldovich: T_ad ↑ con φ en el lado pobre) | PASS | variación relativa 358.05 %  · φ_PZ ∈ [0.6044, 0.7966], 5 puntos pobres |
| signo ∂/∂φ_PZ · EI_NOx_odgers | derived | 14.17 | > 0 (Zeldovich: T_ad ↑ con φ en el lado pobre) | PASS | variación relativa 184.42 %  · φ_PZ ∈ [0.6044, 0.7966], 5 puntos pobres |
| signo ∂/∂φ_PZ · EI_NOx_laviolette | derived | 2.671 | > 0 (Zeldovich: T_ad ↑ con φ en el lado pobre) | PASS | variación relativa 296.50 %  · φ_PZ ∈ [0.6044, 0.7966], 5 puntos pobres |
| signo ∂/∂φ_PZ · EI_CO (lefebvre) | derived | -965.9 | < 0 (más T ⇒ más oxidación de CO) | PASS | variación relativa 87.71 %  · φ_PZ ∈ [0.6044, 0.7966], 5 puntos pobres |
| signo ∂/∂τ · EI_NOx (cabecera = lefebvre) | derived | 0.3306 | > 0 (más tiempo ⇒ más NOx térmico) | PASS | variación relativa 49.23 %  · τ ∈ [4.859, 8.033] ms |
| signo ∂/∂τ · EI_NOx_lefebvre | derived | 0.3306 | > 0 (más tiempo ⇒ más NOx térmico) | PASS | variación relativa 49.23 %  · τ ∈ [4.859, 8.033] ms |
| signo ∂/∂τ · EI_NOx_rizk_mongia | derived | 0.392 | > 0 (más tiempo ⇒ más NOx térmico) | PASS | variación relativa 33.17 %  · τ ∈ [4.859, 8.033] ms |
| signo ∂/∂τ · EI_NOx_zonal | derived | 0.001687 | > 0 (más tiempo ⇒ más NOx térmico) | FAIL (hueco conocido) | variación relativa 0.14 % ⇒ INSENSIBLE (< 2 %) · τ ∈ [4.859, 8.033] ms |
| signo ∂/∂τ · EI_NOx_odgers | derived | 0.5831 | > 0 (más tiempo ⇒ más NOx térmico) | PASS | variación relativa 20.51 %  · τ ∈ [4.859, 8.033] ms |
| signo ∂/∂τ · EI_NOx_laviolette | derived | 0.07916 | > 0 (más tiempo ⇒ más NOx térmico) | PASS | variación relativa 8.01 %  · τ ∈ [4.859, 8.033] ms |
| signo ∂/∂τ · EI_CO (lefebvre) | derived | -19.06 | < 0 (más tiempo ⇒ más CO quemado) | PASS | variación relativa 108.37 %  · τ ∈ [4.859, 8.033] ms |
| orden de magnitud EI_NOx (abanico L0) | derived | [2, 8.911] g/kg | solapa [0.05, 10] g/kg (×2) | PASS | abanico de 5 correlaciones, dispersión ×4.456 |
| orden de magnitud EI_NOx (CRN L1) | declared | — | — | N/E | NO EVALUADO: L1 cuesta ~2.2 s/punto — usa `--l1` |

**Huecos conocidos de este caso** (explicados en `docs/VALIDATION.md`):

* `signo ∂/∂φ_PZ · EI_NOx (cabecera = lefebvre)` — G13 · La cabecera `record['EI_NOx']` es EI_NOx_lefebvre (physics_core línea 3042) y tiene el signo EQUIVOCADO frente a φ_PZ: la forma de Lefebvre no depende de φ_PZ (va con T_st, P₃, V_c y ṁ) y el −25 % que se ve al barrer φ_PZ es puro artefacto del denominador (g NOx por kg de combustible, y el combustible cambia con η_comb). Las tres formas ZONALES —zonal, Odgers y Laviolette— SÍ dan el signo correcto y el pico en φ_PZ ≈ 1.05. ACCIÓN: la cabecera de NOx debe ser la zonal, no la de Lefebvre. Es el pendiente de recalibración nº 3 y **no requiere ningún dato nuevo** — es una decisión de modelo.
* `signo ∂/∂φ_PZ · EI_NOx_lefebvre` — G13 · La cabecera `record['EI_NOx']` es EI_NOx_lefebvre (physics_core línea 3042) y tiene el signo EQUIVOCADO frente a φ_PZ: la forma de Lefebvre no depende de φ_PZ (va con T_st, P₃, V_c y ṁ) y el −25 % que se ve al barrer φ_PZ es puro artefacto del denominador (g NOx por kg de combustible, y el combustible cambia con η_comb). Las tres formas ZONALES —zonal, Odgers y Laviolette— SÍ dan el signo correcto y el pico en φ_PZ ≈ 1.05. ACCIÓN: la cabecera de NOx debe ser la zonal, no la de Lefebvre. Es el pendiente de recalibración nº 3 y **no requiere ningún dato nuevo** — es una decisión de modelo.
* `signo ∂/∂φ_PZ · EI_NOx_rizk_mongia` — G14 · Rizk & Mongia global es INSENSIBLE a φ_PZ (variación relativa 0.00 % en todo el barrido). No es un fallo de implementación: la forma global no lleva φ. Se reporta para que nadie la use como indicador de tendencia.
* `signo ∂/∂τ · EI_NOx_zonal` — G15 · La forma zonal es INSENSIBLE a τ (variación < 2 %): es función de φ y T de zona, no del tiempo de residencia. Complementaria de G13: para tendencia en φ, la zonal; para tendencia en τ, Odgers o Lefebvre.

**Barrido de φ_PZ** (por `frac_pz`, lado pobre) — todo en g/kg:

| φ_PZ | EI_NOx | EI_NOx_lefebvre | EI_NOx_rizk_mongia | EI_NOx_zonal | EI_NOx_odgers | EI_NOx_laviolette | EI_CO |
|---|---|---|---|---|---|---|---|
| 0.6044 | 2.619 | 2.619 | 3.627 | 0.01317 | 0.5267 | 0.02254 | 304.4 |
| 0.6315 | 2.551 | 2.551 | 3.627 | 0.02898 | 0.7179 | 0.03874 | 264.7 |
| 0.6702 | 2.46 | 2.46 | 3.627 | 0.08149 | 1.088 | 0.0801 | 216.5 |
| 0.7236 | 2.35 | 2.35 | 3.627 | 0.2734 | 1.802 | 0.1937 | 165.8 |
| 0.7966 | 2.221 | 2.221 | 3.627 | 1.048 | 3.251 | 0.5436 | 117 |

**Barrido de τ [ms]** (por `L_dz/D_L`) — todo en g/kg:

| τ [ms] | EI_NOx | EI_NOx_lefebvre | EI_NOx_rizk_mongia | EI_NOx_zonal | EI_NOx_odgers | EI_NOx_laviolette | EI_CO |
|---|---|---|---|---|---|---|---|
| 4.8595 | 1.606 | 1.606 | 3.109 | 3.754 | 8.038 | 3.03 | 97.07 |
| 5.6529 | 1.869 | 1.869 | 3.463 | 3.755 | 8.648 | 3.133 | 66.6 |
| 6.4463 | 2.131 | 2.131 | 3.784 | 3.757 | 9.149 | 3.204 | 50.69 |
| 7.2397 | 2.393 | 2.393 | 4.08 | 3.758 | 9.559 | 3.252 | 40.92 |
| 8.0330 | 2.655 | 2.655 | 4.356 | 3.759 | 9.896 | 3.285 | 34.3 |

## 3. Anclas de regresión (NO son mediciones)

Congelan la salida ACTUAL del modelo para detectar deriva silenciosa de la física. Se congelan con `python test_phycb.py --freeze-anchors` y moverlas exige justificación escrita (contrato §10.9).

| Ancla | Estado | Detalle |
|---|---|---|
| REF_CB10 | PASS | D_ref_mm = 65.1649 (esp. 65.1649); EI_CO = 57.5679 (esp. 57.5679); EI_NOx = 1.99973 (esp. 1.99973); L_tot_mm = 166.289 (esp. 166.289); PF = 0.35 (esp. 0.35); SMD_um = 50 (esp. 50); T04_K = 1150 (esp. 1150); T_wall_max_K = 994.566 (esp. 994.566); dp_over_p = 0.03 (esp. 0.03); eta_comb = 0.984568 (esp. 0.984568); phi_pz = 1.04826 (esp. 1.04826); tau_ms = 6.04958 (esp. 6.04958) |
| REF_CB10_L1 | AUSENTE | Ancla L1 (CRN Cantera). No está congelada: Cantera es OPCIONAL y una CI sin Cantera no podría reproducirla. Se comprueba en su lugar la reproducibilidad del CRN en esta máquina y se reporta el número, sin congelarlo. |
| REF_CB10_L2A | AUSENTE | Ancla L2a (IB-RANS nativo). NO PUEDE EXISTIR: `cfd/ibrans/` está vacío en esta máquina y no hay Warp/JAX instalados, así que L2a degrada con etiqueta `ibrans_L2A(unavailable→…)`. WAVE6 la creará. |

## 4. UQ mínima — bandas sobre el caso ancla REF_CB10

σ_total = √(σ_modelo² + σ_ensemble²). **APROXIMACIÓN DECLARADA**: supone errores relativos, gaussianos, independientes y modelo lineal en el entorno del punto; ninguna de las cuatro está demostrada. σ_ensemble = 0 en la campaña (no hay surrogate entrenado) ⇒ la banda es **solo** incertidumbre de modelo físico.

| Magnitud | valor | tipo | banda | origen |
|---|---|---|---|---|
| phi_LBO | 0.5245 | sigma | [0.2885, 0.7606] | σ_modelo = ±45 %, σ_ens = ±0 % |
| LBO_margin | 0.9985 | sigma | [0.5492, 1.448] | σ_modelo = ±45 %, σ_ens = ±0 % |
| EI_NOx | 2 | sigma | [0.9999, 3] | σ_modelo = ±50 %, σ_ens = ±0 % |
| EI_CO | 57.57 | sigma | [28.78, 86.35] | σ_modelo = ±50 %, σ_ens = ±0 % |
| dp_over_p | 0.03 | sigma | [0.024, 0.036] | σ_modelo = ±20 %, σ_ens = ±0 % |
| T_wall_max_K | 994.6 | barrido medido | [808, 1573] | suelo de ε_g (EPS_G_FLOOR_LUMINOUS) — barrido del suelo por la banda de literatura 0.05–0.60; `g[1]` cambia de signo (−0.33 ↔ +0.31). PENDIENTES §5 |
| PF | 0.35 | barrido medido | [0.1225, 0.5915] | k_dilution (pf_constant.csv) — k×2 → PF −65 %; k×0.5 → PF +69 %. k está calibrada a UN punto y ese punto ES el ancla. PENDIENTES §5 y §6.2 |

## 5. Criterios v1 que NO son evaluables (declarados)

Regla de la casa: un criterio que no se puede medir se DECLARA; nunca se convierte en un check que aprueba trivialmente.

| Criterio | Por qué no |
|---|---|
| `dp_over_p` | Δp/p ±20 % NO es evaluable en L0. En Phy-CB el Δp/p es **entrada** del dimensionado (`plf_by_arch.csv::micro_target.dp_over_p_nom` = 3 %, o `evaluate(dp_over_p_target=…)`): `_holes()` reparte el área de los orificios PARA cumplirlo, de modo que `record['dp_over_p']` reproduce el objetivo por construcción (identidad, no predicción). Validarlo exige `dp_over_p_cfd` de L1.5/L2a — y en esta máquina no hay JAX/XLB/Warp ni `cfd/ibrans/`, así que ambos degradan. Lo único que se comprueba aquí es que el objetivo cae en la banda de clase (2–4 %), que es trazabilidad, no validación. |
| `T4` | T4 ±3 % NO es evaluable como validación: T4 es θ[19] (punto de operación pineado por `DesignSpec.fix_operating_point`) y el FAR sale de él por balance de entalpía. Sería validable con un caso que publicase (T3, ṁ_aire, ṁ_combustible, T4) a la vez, y **no hay ninguno en el paquete**. Se evalúa en su lugar el CIERRE del balance (round-trip T4 → FAR → T4), que es VERIFICACIÓN. |
| `LBO` | φ_LBO ×1.5 NO es evaluable: no hay ningún dato de extinción pobre medido en el paquete ni en las fuentes de EstadoDelArte §12. La constante `C_LBO_SI` está calibrada a UN punto ancla con ±45 % declarado (PENDIENTES §2.1 #14): en el ancla no predice, devuelve su calibración. Cierra con banco propio (nivel F) — el ensayo de extinción pobre es barato y muy discriminante. |
| `T_metal` | Phy-CB **no predice temperatura de metal cuantitativa en ningún nivel de fidelidad** (contrato §12): L2b, que iba a darla con CHT+radiación, quedó fuera de alcance. El térmico 1D de L0 da tendencia y margen frente al límite del material; L2a daría mapas de pared adiabática para *ranking*. Toda comparación de T de pared de esta campaña es una comprobación de BANDA, no de valor. |

## 6. Niveles de la escalera declarados y no corridos

| Nivel | Caso | Motivo |
|---|---|---|
| E | Sandia D / PRECCINSTA (bases TNF/DLR) | Valida el CFD **L2a**, no el diseño preliminar: pertenece a la escalera V1–V9 de WAVE6 (`cfd/tests/`). En esta máquina `cfd/ibrans/` y `cfd/tests/` están VACÍOS y no hay JAX/XLB/Warp instalados, así que L1.5 y L2a degradan con etiqueta. No se simula nada y no se reporta ningún número. |
| F | Banco propio (cámara ancla impresa, IN625 ~10 kW) | NO EXISTE. Sin impresora LPBF ni banco no hay dato. Es el único nivel que cerraría los tres pendientes críticos (suelo de ε_g, k del PF, C_LBO) — ver §«especificación de impresión» de docs/VALIDATION.md, lista para cuando haya proveedor. |
| L2b | Backend OpenFOAM (WAVE5-B) | FUERA DE ALCANCE por decisión de proyecto (contrato §12): máquina Windows, exigiría WSL2/Docker. `Fidelity.L2B` se conserva en el enum (es ABI) y degrada con `source='openfoam_L2B(out_of_scope→…)'`. Consecuencia que se repite en todas partes: **Phy-CB no predice temperatura de metal cuantitativa en ningún nivel de fidelidad**. |

## 7. Cómo añadir un caso

1. Reunir el dato publicado **con su URL o cita**; lo que la fuente no diga va en `assumptions` con la palabra SUPUESTO.
2. Añadir la entrada a `validation/cases.py` marcando cada parámetro comparado como `derived` o `input`.
3. `python validation/validate.py` y versionar este archivo.
4. Si algo falla: **no se recalibra por caso**. Se abre entrada en `docs/VALIDATION.md` §prioridad de recalibración, se toca **una** constante global y se re-corren TODOS los casos.
