# Phy-CB a mayor escala · resultados L0 de la matriz 5–25 kg/s

Informe generado por `validation/scale_l0.py report` desde `validation/scale_acceptance.json` (no se edita a mano). Fase 17.1 de QSTUDIO, planes 17.1-02 y 17.1-03. **Todas las cifras son salidas de software L0 (correlaciones), no ensayos experimentales.**

## Identidad

- Fuente del modelo (física Python que ejecutó la matriz): `c7a9bda6ad369e0e853b09a71bb141ce76736ce4`
- Arnés de las corridas: `71fc741ced5ddcaff46a0697e63a27ba07e91762`, `c7a9bda6ad369e0e853b09a71bb141ce76736ce4` (sólo rutas de soporte respecto a la fuente)
- Generador C#/PicoGK de los STL: `e4eb7dce7c15973fedf8c32ec539c8d60204c1e4` · launcher `bcaeb77bfc06d9cb5adb83f725a279f002cf7b2db4c3abb5c3c002fe26a0af6c` · biblioteca `36bb646e6dbc70c2026465d423541f05aed5e0621936930b958591578ba513b6`
- `cad_evidence_sha256`: `c484f837a8580a15f8a2a4d36245cf9961bd1d1f6a562296efb329cec5ba53e5`
- Descargas Phy-AC congeladas: `validation/scale_inlets.json` (`inlets_sha256` 75cc65ee46115a6be99c60813847590582d7127a409a3c65dcf08f6eac0d3d09)
- Revisión de política: `phycb-scale/2 (17.1-02, D-373..D-376; física D-388)`
- Receta: L0, T04 objetivo 1200 K, tope 1250 K, Jet-A, Inconel-625/718, Δp/p máx 0.04, PF máx 0.4, DfAM activo, workers 1. Etapa 1 semilla 71 · 320/5/14; escalada única semilla 137 · 480/7/18.
- Tope por celda: D-391: 240 s por celda e intento en 5 y 25 kg/s; 8 kg/s conserva 120 s (17.1-02)

## Etiquetas de las salidas

- **T04**: impuesto por la receta (1200 K) y verificado ≤ 1250 K.
- **P04**: calculado, P04 = P03 · (1 − Δp/p), con cierre comprobado.
- **Caudal de salida (A7)**: calculado, aire + combustible.
- **Δp/p**: impuesto por el cierre de diseño L0 (`DP_OVER_P_DESIGN` = 0,030); no es una pérdida calculada de forma independiente y no se usa para ordenar arquitecturas.
- **τ, η, PF, LBO, T pared**: calculados por correlaciones L0 con la base D-388 (banda de τ 5–10 ms y LBO intensiva por transferencia, vetable).

## Matriz de aceptación (6/6)

| Celda | Etapa | Tiempo (s) | Evals | P03 (Pa) | T03 (K) | PR AC | T04 (K) | P04 (Pa) | ṁ salida (kg/s) | Δp/p | η | PF | τ (ms) [cota] | LBO margen | g máx (nombre) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5/annular | 1 | 76.646 | 460 | 373 082.77 | 443.96 | 3.682 | 1200.0 | 361 890.282 | 5.112975977 | 0.0300 | 0.98839 | 0.36720 | 8.656 [5.0–10.0] | 7.142 | -0.0186 (eta_comb) |
| 5/reverse_annular | 1 | 78.948 | 460 | 373 082.77 | 443.96 | 3.682 | 1200.0 | 361 890.282 | 5.112975977 | 0.0300 | 0.98839 | 0.36720 | 8.656 [5.0–10.0] | 7.142 | -0.0186 (eta_comb) |
| 8/annular | 1 | 82.818 | 460 | 358 961.88 | 432.49 | 3.543 | 1200.0 | 348 193.024 | 8.184921118 | 0.0300 | 0.97842 | 0.35029 | 8.492 [5.0–10.0] | 0.320 | -0.0086 (eta_comb) |
| 8/reverse_annular | 1 | 79.239 | 460 | 358 961.88 | 432.49 | 3.543 | 1200.0 | 348 193.024 | 8.184921118 | 0.0300 | 0.97842 | 0.35029 | 8.492 [5.0–10.0] | 0.320 | -0.0086 (eta_comb) |
| 25/annular | 1 | 77.17 | 460 | 322 771.41 | 420.15 | 3.186 | 1200.0 | 313 088.264 | 25.579412638 | 0.0300 | 0.98835 | 0.38286 | 8.392 [5.0–10.0] | 0.852 | -0.0186 (eta_comb) |
| 25/reverse_annular | 1 | 77.335 | 460 | 322 771.41 | 420.15 | 3.186 | 1200.0 | 313 088.264 | 25.579412638 | 0.0300 | 0.98835 | 0.38286 | 8.392 [5.0–10.0] | 0.852 | -0.0186 (eta_comb) |

Detalle por bloque (máximo de cada bloque de g, todos ≤ 0) y dimensiones:

| Celda | aero | spec | struct | dfam | D_L (mm) | D_casing (mm) | L_tot (mm) | h_L (mm) | T pared máx (K) | SMD (µm) / atomizador | Corona: copas × d_tip/d_hub (mm), A por copa (mm²) | Envolvente D×L (mm), V (m³) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5/annular | -0.0186 (eta_comb) | -0.0820 (spec_PF_max) | -0.1872 (T_wall) | -0.1667 (t_min) | 293.1 | 416.8 | 219.1 | 68.1 | 977.8 | None / micromix | 15 × 38.12/20.37, 815.2 (anillo r 157.7) | 422.1 × 371.6, 0.0520 |
| 5/reverse_annular | -0.0186 (eta_comb) | -0.0820 (spec_PF_max) | -0.1872 (T_wall) | -0.1667 (t_min) | 293.1 | 416.8 | 219.1 | 68.1 | 977.8 | None / micromix | 15 × 38.12/20.37, 815.2 (anillo r 157.7) | 422.1 × 559.7, 0.0783 |
| 8/annular | -0.0086 (eta_comb) | -0.1243 (spec_PF_max) | -0.2407 (T_wall) | -0.0135 (d_hole_min) | 361.8 | 534.0 | 228.5 | 79.4 | 913.5 | 50.0 / vaporizer | 6 × 69.67/24.38, 3344.9 (anillo r 205.9) | 538.8 × 415.4, 0.0947 |
| 8/reverse_annular | -0.0086 (eta_comb) | -0.1243 (spec_PF_max) | -0.2407 (T_wall) | -0.0135 (d_hole_min) | 361.8 | 534.0 | 228.5 | 79.4 | 913.5 | 50.0 / vaporizer | 6 × 69.67/24.38, 3344.9 (anillo r 205.9) | 538.8 × 660.4, 0.1506 |
| 25/annular | -0.0186 (eta_comb) | -0.0428 (spec_PF_max) | -0.1939 (T_wall) | -0.1329 (effusion_band) | 588.2 | 988.4 | 288.4 | 107.9 | 969.8 | None / micromix | 23 × 87.49/62.04, 2989.0 (anillo r 400.7) | 992.4 × 588.5, 0.4553 |
| 25/reverse_annular | -0.0186 (eta_comb) | -0.0428 (spec_PF_max) | -0.1939 (T_wall) | -0.1329 (effusion_band) | 588.2 | 988.4 | 288.4 | 107.9 | 969.8 | None / micromix | 23 × 87.49/62.04, 2989.0 (anillo r 400.7) | 992.4 × 1055.9, 0.8168 |

DfAM analítico del contrato exportado (checker real, `g_dfam` = t_min, d_hole_min, overhang, drainability, effusion_band):

- 5/annular: imprimible = True · g_dfam [-0.1667, -0.4400, -1.0000, -0.2388, -0.1667]
- 5/reverse_annular: imprimible = True · g_dfam [-0.1667, -0.4400, -0.2222, -0.2439, -0.1667]
- 8/annular: imprimible = True · g_dfam [-0.1667, -0.0135, -1.0000, -0.2392, -0.1667]
- 8/reverse_annular: imprimible = True · g_dfam [-0.1667, -0.0135, -0.2222, -0.2448, -0.1667]
- 25/annular: imprimible = True · g_dfam [-0.1667, -0.1617, -1.0000, -0.2415, -0.1329]
- 25/reverse_annular: imprimible = True · g_dfam [-0.1667, -0.1617, -0.2222, -0.2467, -0.1329]

Contratos (SHA256 de los bytes archivados) y semillas:

- 5/annular: `007ac3e116dc0acfb2bd517c7c525cfd5827ea5441b7e717ca462dc3cf525e4c` · semilla 71 · descarga AC `c8799242d8df7d65…` · fila del checkpoint 56
- 5/reverse_annular: `fc21ca7ada45740d72fc61408baa4fefc7ac8d5b424afb52a7289a74140b80ae` · semilla 71 · descarga AC `c8799242d8df7d65…` · fila del checkpoint 56
- 8/annular: `ef2f9032b3bb1f61bab07955f72762994cd8903f1afe99d19e0d3078ad97d8bd` · semilla 71 · descarga AC `85c016debab3f99c…` · fila del checkpoint 70
- 8/reverse_annular: `f299fb76229470d9d49b7df11989dfdc67a796083d67c2c9310e12c6ac5e1205` · semilla 71 · descarga AC `85c016debab3f99c…` · fila del checkpoint 70
- 25/annular: `f2e3e985d74a164a5788fc4e5131c7cf4ac88a5612df5808b729d69504072a7b` · semilla 71 · descarga AC `cff8c2f092857d68…` · fila del checkpoint 176
- 25/reverse_annular: `9c6421d14c0a2b2079ab23f390b6cddf10a8079da5eab1bebd58ca82e84c4748` · semilla 71 · descarga AC `cff8c2f092857d68…` · fila del checkpoint 176

## Comparación D-377 a 8 kg/s y recomendación

| Magnitud | annular | reverse_annular |
|---|---|---|
| Δp/p (impuesta) | 0.0300 | 0.0300 |
| T04 (K) | 1200.0 | 1200.0 |
| P04 (Pa) | 348193.024 | 348193.024 |
| ṁ salida (kg/s) | 8.184921118 | 8.184921118 |
| g máx | -0.008601 | -0.008601 |
| D envolvente (mm) | 538.8 | 538.8 |
| L envolvente (mm) | 415.4 | 660.4 |
| V envolvente (m³) | 0.0947 | 0.1506 |
| g_dfam | [-0.167, -0.014, -1.000, -0.239, -0.167] | [-0.167, -0.014, -0.222, -0.245, -0.167] |
| DfAM vóxel g (overhang, polvo, drenaje) | [+4.343, -1.000, -3.500] | [+5.627, -1.000, -2.500] |

Regla aplicada (en orden):

- R1 pérdida de presión: empate por construcción: Δp/p = 0,030 impuesto en L0 en ambas; no se usa para ordenar
- R2 peor g global: empate: g_max = -0.008601 (eta_comb) en ambas
- R3 márgenes ordenados: posición 9 del vector ordenado: annular -0.239249 frente a reverse_annular -0.222222; gana annular
- R4 embalaje: V_env annular 0.0947 m³ frente a reverse_annular 0.1506 m³; menor: annular

**Recomendación confirmada por el fundador (D-393, 2026-11-10): `8/annular`** (la regla decide en R3; etapa 1; contrato `ef2f9032b3bb1f61bab07955f72762994cd8903f1afe99d19e0d3078ad97d8bd`; fila 70). sin compromiso: el embalaje confirma la regla. En L0 las dos anulares comparten la física: la decisión sale de los márgenes DfAM del contrato y del embalaje, no de una diferencia aerotérmica.

D-393: Confirmado por el fundador: la referencia del piloto a 8 kg/s es 8/annular; Studio la prueba en 17.1-04 y se entrega a la fase 18. Las dos anulares siguen en la matriz y en la compuerta PicoGK.

El DfAM sobre vóxeles as-built se registra como evidencia separada del gate analítico: annular imprimible = False, reverse_annular imprimible = False. No entra en la regla porque mide la geometría generada a vóxel 0,8 mm (paredes subidas a 2 vóxeles), no el diseño verificado.

## Geometría PicoGK (D-386)

Compilación única `dotnet build CombustorDesigner.sln -t:Rebuild -c Debug --no-restore -nodeReuse:false -p:UseSharedCompilation=false` en 3.877 s; vóxel 0.8 mm para los pilotos y 1,2 mm para el smoke KJ66 (T29). Ejecución en serie con tope de 180 s y 18 GiB por geometría.

| Caso | Tiempo (s) | Pico RAM (GiB) | Piezas | DfAM vóxel g | Imprimible | Topología |
|---|---|---|---|---|---|---|
| kj66 | 5.847 | 0.225 | 9 | [+2.921, -1.000, -2.000] | False | — |
| annular | 112.984 | 1.76 | 9 | [+4.343, -1.000, -3.500] | False | 6 copas OK, paso central abierto |
| reverse_annular | 129.804 | 2.576 | 9 | [+5.627, -1.000, -2.500] | False | 6 copas OK, paso central abierto |

**kj66** · contrato `d44718a47dba178b323710df4cafe45450075600f3970d21c8767f701b1c603f` · reporte as-built `8c184706bb7e0336438bf3e8db0781df7141f979c4d1cd2acc95138c794ced08`

| Pieza | Volumen vóxel (mm³) | Caja (mm) | Triángulos | STL SHA256 |
|---|---|---|---|---|
| Liner | 201 622 | [-39.3, -37.5, 0.0, 39.3, 37.5, 230.8] | 339724 | `e635e156573970bc…` |
| Casing | 167 097 | [-55.1, -55.1, 0.0, 55.1, 55.1, 233.7] | 239544 | `a7d17cd4cca4055a…` |
| Vaporizer1 | 2 684 | [27.5, -3.0, 133.1, 34.5, 3.0, 216.1] | 3864 | `3df109603efd78d1…` |
| Vaporizer2 | 2 722 | [12.3, 23.4, 133.1, 18.7, 30.3, 216.1] | 3868 | `8070cf8da7bf4aae…` |
| Vaporizer3 | 2 722 | [-18.7, 23.4, 133.1, -12.3, 30.3, 216.1] | 3868 | `f1c58962b9a7a7d6…` |
| Vaporizer4 | 2 686 | [-34.5, -3.0, 133.1, -27.5, 3.0, 216.1] | 3864 | `c795b7e25357c243…` |
| Vaporizer5 | 2 721 | [-18.7, -30.3, 133.1, -12.3, -23.4, 216.1] | 3868 | `a24dde26b2578158…` |
| Vaporizer6 | 2 722 | [12.3, -30.3, 133.1, 18.7, -23.4, 216.1] | 3868 | `28a7fcd5f25c94d6…` |
| Assembly | 390 387 | [-55.1, -55.1, 0.0, 55.1, 55.1, 233.7] | 596284 | `2d14c4b248861738…` |

Incumplimientos voxelizados registrados: overhang: part 'Casing' has 1.961 % of its solid cells with no support within 45 deg beneath them in build 'z+' (tolerance 0.5 %). Either re-orient the build, chamfer the down-skins, or raise dfam.overhang_frac_max if supports are acceptable on that surface.

**annular** · contrato `ef2f9032b3bb1f61bab07955f72762994cd8903f1afe99d19e0d3078ad97d8bd` · reporte as-built `7d3a93d3b7fcea70cbe77d50501d6b38740692d540c63b21e20555419f9dbe0c`

| Pieza | Volumen vóxel (mm³) | Caja (mm) | Triángulos | STL SHA256 |
|---|---|---|---|---|
| Liner | 1 626 919 | [-245.7, -245.7, -180.8, 245.7, 245.7, 232.5] | 9116916 | `454c2096ba5f9086…` |
| Casing | 2 145 860 | [-281.4, -281.4, -184.9, 281.4, 281.4, 235.5] | 7276296 | `94e1b03dd32cc00a…` |
| Vaporizer1 | 2 501 | [201.8, -3.0, -11.3, 208.9, 3.0, 71.4] | 9904 | `6653af21a5ccfef3…` |
| Vaporizer2 | 2 276 | [99.4, 174.4, -11.3, 105.9, 181.3, 71.4] | 10628 | `31f1be662bc441d7…` |
| Vaporizer3 | 2 280 | [-105.9, 174.4, -11.3, -99.4, 181.3, 71.4] | 10628 | `8b3692d10a2d7450…` |
| Vaporizer4 | 2 500 | [-208.9, -3.0, -11.3, -201.8, 3.0, 71.4] | 9904 | `d907c43604ee88f8…` |
| Vaporizer5 | 2 278 | [-105.9, -181.3, -11.3, -99.4, -174.4, 71.4] | 10628 | `1d791619ff46c687…` |
| Vaporizer6 | 2 280 | [99.4, -181.3, -11.3, 105.9, -174.4, 71.4] | 10628 | `ad359ef35b830647…` |
| Assembly | 3 784 504 | [-281.4, -281.4, -184.9, 281.4, 281.4, 235.5] | 16455532 | `a4a79b8f272cabf2…` |

Incumplimientos voxelizados registrados: overhang: part 'Casing' has 2.672 % of its solid cells with no support within 45 deg beneath them in build 'z+' (tolerance 0.5 %). Either re-orient the build, chamfer the down-skins, or raise dfam.overhang_frac_max if supports are acceptable on that surface.

**reverse_annular** · contrato `f299fb76229470d9d49b7df11989dfdc67a796083d67c2c9310e12c6ac5e1205` · reporte as-built `ff222598b5a35d468de6637e7799d64ca03a5b2e8e688e02e2db045d1685e647`

| Pieza | Volumen vóxel (mm³) | Caja (mm) | Triángulos | STL SHA256 |
|---|---|---|---|---|
| Liner | 2 363 113 | [-245.7, -245.7, -0.0, 245.7, 245.7, 657.8] | 13682540 | `b4d8e582035aa12b…` |
| Casing | 2 666 133 | [-281.4, -281.4, -3.0, 281.4, 281.4, 660.4] | 9153196 | `552bb0db1ae056cb…` |
| Vaporizer1 | 2 516 | [201.8, -3.0, 406.1, 208.9, 3.0, 488.9] | 9912 | `9dfe2152eeddef3c…` |
| Vaporizer2 | 2 293 | [99.4, 174.4, 406.1, 105.9, 181.3, 488.9] | 10624 | `22b0abaa2d2566ec…` |
| Vaporizer3 | 2 295 | [-105.9, 174.4, 406.1, -99.4, 181.3, 488.9] | 10624 | `f611e788c25dae49…` |
| Vaporizer4 | 2 515 | [-208.9, -3.0, 406.1, -201.8, 3.0, 488.9] | 9912 | `a28aaad6b95d1190…` |
| Vaporizer5 | 2 294 | [-105.9, -181.3, 406.1, -99.4, -174.4, 488.9] | 10624 | `cd6ce9df31d1cad3…` |
| Vaporizer6 | 2 296 | [99.4, -181.3, 406.1, 105.9, -174.4, 488.9] | 10624 | `868c5e1ceac2a456…` |
| Assembly | 5 043 456 | [-281.4, -281.4, -3.0, 281.4, 281.4, 660.4] | 22898056 | `351e00023462155c…` |

Incumplimientos voxelizados registrados: overhang: part 'Casing' has 3.314 % of its solid cells with no support within 45 deg beneath them in build 'z+' (tolerance 0.5 %). Either re-orient the build, chamfer the down-skins, or raise dfam.overhang_frac_max if supports are acceptable on that surface.

Ensayo negativo: contrato de escala con `swirler.count` ≠ `injectors.count` → rc 3 antes de voxelizar (Invalid contract: scale cups (operating_class annular_5_25): swirler.count = 5 cups but injectors.count = 6: one cup per injector station is required).

Los STL quedan en `C:\Users\nickv\projects\Proyectos\Software\QSTUDIO\backend\app\workspace\_tmp\phase-17.1\cad-790da485` (directorio de ejecución, fuera de Git), ligados por SHA256. Los tres STL de campaña G4 corresponden a las corridas finales de la fase 18.

## Limitaciones de la evidencia L0

- Correlaciones L0 sin CFD ni ensayo; la base de residencia y LBO es una transferencia (D-388) vetable.
- En L0 annular y reverse_annular comparten la física: misma θ salvo la gate y mismos g aerotérmicos; sólo el contrato y el generador las distinguen.
- A 8 kg/s el vaporizador conserva `n_tubes` = 6 copas; 5 y 25 kg/s usan micromix con el conteo por sector.
- El generador C# coloca cada copa con eje paralelo a Z sobre la estación del domo hemisférico; la camisa atraviesa toda la pared oblicua del domo. Los tubos del vaporizador siguen las cotas `tube_z0/z1` del contrato (plano del domo), no la estación de la copa: es la convención actual del contrato.
- El DfAM as-built usa paredes de 2 vóxeles (1,6 mm a 0,8 mm), más gruesas que el diseño verificado.

## Reproducción

```
python -B -X utf8 validation/scale_l0.py verify --masses 5,8,25
python -B -X utf8 validation/scale_l0.py accept --mass 5,25 --out-root DIR   # tope D-391
python -B -X utf8 validation/scale_l0.py cad --out-root DIR --evidence DIR/cad_evidence.json
python -B -X utf8 validation/scale_l0.py archive-cad --evidence DIR/cad_evidence.json
python -B -X utf8 validation/scale_l0.py report
python -B -X utf8 validation/scale_l0.py verify-handoff --wave 3 --summary 17.1-03-SUMMARY.md
```
