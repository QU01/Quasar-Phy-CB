# Phy-CB — Campaña de validación: plan, resultados y veredicto

> **Entregable de WAVE7.** Este documento dice qué se ha comparado contra
> datos publicados, con qué supuestos, qué cuadra, qué NO cuadra y en qué
> orden hay que tocar las constantes si algo falla. Los fallos se
> documentan, no se esconden: un caso que no cuadra y se explica vale más
> que un caso maquillado.
>
> **Fecha de corte:** 2026-07-31 · **Máquina:** Windows 11, Python 3.13.7,
> NumPy 2.4.6, Cantera 3.2.0, matplotlib 3.11, dotnet 10.0.100.
> **Sin** JAX / XLB / Warp ⇒ L1.5 y L2a degradan con etiqueta.
>
> **Reproducir** (la trampa de entorno del contrato §0.b obliga a lanzarlo
> desde la raíz de Phy-CB; desde otro `cwd` un `import physics_core`
> resuelve al de Phy-AC):
>
> ```
> cd "C:/Users/nickv/projects/Proyectos/Phy-CB"
> python validation/validate.py --l1          # tabla + validation/RESULTS.md
> python validation/validate.py --strict      # el que corre la CI
> ```
>
> Los números de este documento están en `validation/RESULTS.md`, que se
> regenera con el script y se versiona como evidencia.

---

## 1. Qué es esto y qué no

La familia Phy-\* mantiene una distinción que este documento **no puede
relajar**:

| | Pregunta | Dónde vive | Estado |
|---|---|---|---|
| **Verificación** | ¿resolvemos BIEN las ecuaciones? | `test_phycb.py` (T1–T40 y T49) y la escalera V1–V9 del CFD | **355/355 checks OK · 1 SKIP declarado** (T29, caro; con `--slow` y `dotnet` presente sube a 359) |
| **Validación** | ¿son las ecuaciones CORRECTAS? | `validation/` — **este documento** | **107 PASS · 49 FAIL** (49 de 49 explicados aquí) · 0 huecos obsoletos. Era 108/48 hasta el 2026-07-31: A2 salió de la banda de `n_eff_holes` al corregirse el área mojada de la efusión (hueco G3) |

Un test verde en `test_phycb.py` no dice nada sobre si el modelo acierta.
Esta campaña es la que lo dice, y su respuesta corta es:

> **Phy-CB acierta el reparto de aire (el único caso con dos puntos
> publicados de la misma máquina) con un error del 0.2–1.6 %. Todo lo
> demás está sin anclar: cinco constantes gobiernan los números de portada
> y tres de ellas están calibradas a UN solo punto. Además, la campaña ha
> falsado un dato del propio paquete — la tabla de referencia por clase se
> contradice a sí misma — y una elección de modelo — la correlación de NOx
> de cabecera tiene el signo equivocado frente a φ_PZ.**

Regla que se hereda de Phy-CC y Phy-AC y que aquí se aplica sin
excepciones: **nunca se recalibra por caso.** Si hicieran falta constantes
distintas por máquina, el modelo estaría mal, no la constante.

---

## 2. La escalera de casos (WAVE7 §2)

| Nivel | Caso | Estado | Qué valida |
|---|---|---|---|
| **A** | Tabla de referencia por clase (`data/reference/class_table.csv`, EstadoDelArte §7) | **CORRIDO** — `A1`, `A2`, `A3` (una por clase) + `A4` (consistencia interna del dato) | Dimensionado L0: V_c, τ, Ø, U_ref, PF, η, T de pared, emisiones en banda de clase |
| **B** | KJ66 / clase hobby (vaporizador + anular reverso) | **CORRIDO** — `B1` (punto ancla REF_CB10) + `B2` (sensibilidad al ṁ de comunidad) | Arquitectura y geometría del vaporizador; bandas de clase 0 |
| **C** | Microturborreactor optimizado (Wiley IJAE 2021:6616537) | **CORRIDO** — `C1`, dos puntos publicados | Reparto de aire → φ_PZ. **El único caso con dato duro de una máquina real** |
| **D** | Bath / IAAPS / HiETA (ECM 248:114805 + IJER 2023) | **CORRIDO** — `D1` (térmico a T₃ = 1093 K) + `D2` (tendencias de emisiones) | Térmico con T₃ extrema; signo de ∂EI/∂φ_PZ y ∂EI/∂τ; orden de magnitud |
| **E** | Sandia D / PRECCINSTA (bases TNF/DLR) | **DECLARADO, NO CORRIDO** | Valida el CFD **L2a**, no el diseño preliminar: pertenece a V1–V9 de WAVE6. `cfd/ibrans/` y `cfd/tests/` están **vacíos** en esta revisión y no hay JAX/XLB/Warp ⇒ L1.5 y L2a degradan con etiqueta. No se simula nada y no se reporta ningún número |
| **F** | Banco propio (cámara ancla impresa, IN625) | **NO EXISTE** | Es el único nivel que cierra los tres pendientes críticos. Especificación de impresión y de ensayo lista en §8 |
| — | **L2b (OpenFOAM)** | **FUERA DE ALCANCE** (contrato §12) | Máquina Windows: exigiría WSL2/Docker. `Fidelity.L2B` se conserva en el enum (es ABI) y degrada con `source="openfoam_L2B(out_of_scope→…)"`. T46–T48 van como SKIP declarado |

### 2.1 Procedencia de los datos y supuestos

Cada caso lleva en `validation/cases.py` su fuente citada y **todo lo que
la fuente no dice va marcado con la palabra SUPUESTO**. Resumen:

* **A1–A4** — `class_table.csv` (`@status: verified`, transcrita de
  EstadoDelArte §7 con marcas [W] verificadas en web el 2026-07-29). El
  punto de operación es el **centro de cada banda**: es la elección neutra
  y es la que hace el resultado falsable. El reparto de aire es la fila
  `microturbine` de `air_split_by_arch.csv`, sin ajustar por caso.
* **B1/B2** — geometría del vaporizador ✅ [W] verificada (6–12 tubos
  «walking stick» de ~64 mm con codo ~30°, planos públicos gtba.co.uk).
  El punto de operación es `THETA_REF` del contrato §1, **que no es una
  medición del KJ66**. Discrepancia declarada: las cifras de comunidad que
  cita `Phy-CC/docs/VALIDACION_ES.md` §2b dan ṁ ≈ 0.23 kg/s, casi el doble
  — de ahí `B2`. **En el paquete no hay ninguna medición de prestaciones
  del KJ66** (empuje, EGT, consumo, Δp frío, PF): B no puede falsar
  prestaciones y no pretende hacerlo.
* **C1** — Wiley IJAE 2021:6616537. Los dos puntos publicados
  (frac_pz 39.35 % → 32 %, φ_PZ 0.89 → 1.09) implican la MISMA φ_global
  (0.35022 y 0.34880, dispersión ±0.20 %): esa auto-consistencia es la
  prueba de que la fuente usa la definición de Ciencia §3 y de que el caso
  es comparable. T₄ se resuelve por bisección acotada para clavar
  φ_global = 0.34951 — misma técnica que la bisección de φ₁ de
  `Phy-AC/validation/validate.py`.
* **D1/D2** — ECM 248:114805 e IJER 2023 vía EstadoDelArte §3.4. Verificado
  [W]: **T₃ = 1093 K (820 °C)** e **Inconel 625**. Todo lo demás (ṁ, T₄,
  π, reparto) es supuesto declarado.

---

## 3. Criterios cuantitativos v1 y su estado real

WAVE7 §3.2 fija cinco criterios. **Dos de ellos NO son evaluables en esta
máquina, y decirlo es parte del entregable.**

| Criterio v1 | Estado | Resultado |
|---|---|---|
| **Δp/p ±20 %** | **NO EVALUABLE** | En L0 el Δp/p es **entrada**, no salida: `plf_by_arch.csv::micro_target.dp_over_p_nom` = 3 % (o `evaluate(dp_over_p_target=…)`), y `_holes()` reparte el área de los orificios PARA cumplirlo. `record["dp_over_p"]` reproduce el objetivo **por construcción**. Validarlo exige `dp_over_p_cfd` de L1.5/L2a, que aquí degradan. Lo único comprobado es que el objetivo cae en la banda de clase (2–4 %) — trazabilidad, no validación |
| **T4 ±3 %** | **NO EVALUABLE como validación** | T₄ es θ[19], punto de operación pineado; el FAR sale de él por balance de entalpía. No hay en el paquete **ningún** caso que publique (T₃, ṁ_aire, ṁ_combustible, T₄) a la vez. Se evalúa en su lugar el **cierre** del balance (round-trip T₄→FAR→T₄), que es VERIFICACIÓN: **cierra a < 4e-3 % en los 6 casos** (máximo −3.6e-3 % en A1) |
| **PF en la banda de su clase** | **EVALUADO — y el resultado es que PF no predice** | PF = 0.35000 en TODOS los casos, de las tres clases y de los dos puntos de KJ66. No es una coincidencia: es la calibración devuelta (§4, hueco G5) |
| **τ, V_c, Ø en banda de clase** | **EVALUADO — parcial** | τ ✓ en A1, A2, B1; ✗ en A3 (11.3 ms), B2 (8.5 ms) y D1 (3.8 ms). V_c ✓ en A2, A3, B1, D1; ✗ en A1 (0.41 L, +3 %) y B2 (1.09 L). Ø de carcasa ✓ en clase 0; **✗ sistemáticamente bajo en las clases 1–2 y en D1** (hueco G6) |
| **Emisiones: tendencia + orden de magnitud ×2** | **EVALUADO — y falsa la elección de cabecera** | Orden de magnitud ✓ (abanico L0 en 2.0–8.9 g/kg y CRN L1 en 3.07 g/kg, frente a la banda de referencia 0.1–5 g/kg con el factor ×2). Tendencias: ∂EI/∂τ ✓ en 4 de 5 correlaciones y en el CO; **∂EI_NOx/∂φ_PZ ✗ en la correlación de CABECERA** (hueco G13) |
| **LBO: φ_LBO dentro de ×1.5 del dato** | **NO EVALUABLE** | No hay ningún dato de extinción pobre medido en el paquete ni en las fuentes de EstadoDelArte §12. `C_LBO_SI` está calibrada a UN punto ancla con ±45 % declarado: en el ancla no predice, devuelve su calibración. Y la campaña ha destapado algo peor: en los casos con airblast el margen de LBO sale **×42 (A2) y ×476 (D1)** — números que no son creíbles y que apuntan al SMD (hueco G8) |

---

## 4. Los huecos, uno a uno

Cada hueco tiene su identificador `G#` en `validation/validate.py::KNOWN_GAPS`.
`validate.py --strict` **no falla por ellos** (para que la CI detecte
regresiones, no el estado conocido) **pero sí falla si uno empieza a
pasar**: un hueco cerrado hay que quitarlo de aquí, no dejarlo mintiendo.

### 4.1 Huecos de DATO (el modelo no tiene la culpa)

**G1 · `class_table.csv` se contradice a sí misma.** Es el hallazgo
principal del nivel A y explica 17 de los 49 FAIL.

El balance de entalpía del ciclo (`physics_core._far_from_T4`, idéntico al
de CONTEXTO §4) es álgebra, no una correlación: no lleva dentro ningún
coeficiente provisional. Aplicado a las propias bandas de la tabla:

| Clase | T₃ [K] | T₄ [K] | FAR que IMPONE el balance | FAR que DECLARA la tabla | ¿solapan? |
|---|---|---|---|---|---|
| 0 · no recuperada | 400–470 | 1100–1250 | **0.01899–0.02492** | 0.010–0.018 | **NO** |
| 1 · no recuperada | 430–500 | 1150–1300 | 0.01967–0.02561 | 0.012–0.020 | sí, por un pelo |
| 1 · recuperada | 800–950 | 1150–1300 | 0.00879–0.01663 | 0.012–0.020 | sí |
| 2 · no recuperada | 450–530 | 1200–1350 | **0.02036–0.02655** | 0.012–0.020 | **NO** |
| 2 · recuperada | 900–1100 | 1200–1350 | 0.00656–0.01561 | 0.012–0.020 | sí |

Lectura: la banda de FAR/φ_global de la tabla **solo es compatible con
operación RECUPERADA**. En la clase 0, que por definición no tiene
recuperador, es directamente imposible. De ahí salen en cadena, por pura
identidad, los FAIL de `FAR_global`, `phi_global` e `I_MW_m3_atm` en A1,
A2, A3, B1 y B2 — y de ahí salen también los φ_PZ altos de A2 y A3.

Esto **confirma y extiende** la auditoría de intensidad de combustión que
ya está anotada en `physics_core._sizing` (2026-07-30): allí se demostró que
`I = FAR·LHV·P_atm/(R·T₃·τ)` y que las bandas de I (20–60) y de τ (5–10 ms)
solo son compatibles para FAR ≲ 0.014. La campaña añade la otra mitad: ese
FAR ≲ 0.014 es incompatible con la banda de (T₃, T₄) de la propia tabla.

> **Acción: corregir el DATO, no el modelo.** Tocar una constante para
> «cuadrar» esto sería calibrar contra un dato inconsistente — el peor
> error posible en una campaña de validación. Concretamente: revisar en la
> fuente de EstadoDelArte §7 si la banda de FAR es de operación de crucero
> o de ralentí, y si el T₄ es TIT o EGT.

**G2 · Dos tablas `verified` se contradicen en 1 punto.**
`air_split_by_arch.csv::microturbine.frac_cool_nom` = 0.16 frente a
`class_table.csv::clase 0 frac_cool` = 0–0.15. `THETA_REF` usa 0.16.
Impacto físico nulo; hay que decidir cuál manda y alinear la otra.

**G3 · Nº de orificios de efusión fuera de banda** (3145 en A1, 5080 en A2,
5993 en B2 y 3020 en B1, frente a 300–1500 de la clase 0 y 1000–4000 de la
clase 1). No es un error de cálculo:
`n = A_efusión/(π/4·d²)` con la porosidad que da `pitch/d`, y
`PENDIENTES §4.2` demuestra que hacer FACTIBLE el ancla exige apretar el
paso a `pitch/d` = 4–5, lo que **sube aún más** el número. La banda de la
tabla y las reglas de efusión densa del propio proyecto son incompatibles.

> **AMPLIADO EL 2026-07-31 — y hay que decirlo en voz alta: este hueco
> creció con la corrección del área mojada de la efusión** (`physics_core`
> §12.b). `π·D_L` es el perímetro de un CÍRCULO de área equivalente, no el
> de la corona real, así que en las arquitecturas ANULARES el área mojada
> —y con ella `n`— estaba **×1.375 corta**. Al corregirla, **A2 (clase 1,
> anular inversa) pasa de 3696 (dentro de [1000, 4000]) a 5080 (fuera)**.
> El marcador del nivel A baja de PASS = 108 a PASS = 107: es una
> REGRESIÓN del marcador y NO se ha tapado moviendo ningún umbral. Lo
> honesto es lo contrario de lo que parece: A2 aprobaba por un error de
> geometría, no por acuerdo con la clase. La conclusión de G3 no cambia —
> hay que corregir el DATO de `class_table.csv`— pero ahora afecta a
> cuatro casos en vez de a tres.

**G4 · V_c fuera de banda por continuidad.** `V_c = τ·ṁ/ρ₃`: con τ dentro
de banda, V_c queda determinado. En el extremo alto de la banda de ṁ de la
clase 0 la continuidad exige hasta 1.33 L frente a un techo de 0.4 L.

### 4.2 Huecos de MODELO (aquí sí hay que tocar algo)

**G5 · El PF no predice: devuelve su calibración.** PF = 0.35000 en los
seis casos con clase, sea cual sea la clase, el gasto o la presión. La
razón está medida en `PENDIENTES §5`: `PF = exp[−k·(L_DZ/D_L)·(ΔP_L/q_ref)]`
con `k = 0.0388823` **calibrada a un único punto**, y con el criterio
aerodinámico dimensionando A_ref el cociente `ΔP_L/q_ref = 15` es una
**identidad algebraica**. Con L_DZ/D_L fijo en 1.8, el PF no tiene ningún
grado de libertad que dependa del caso. La banda de PF de la clase 2
(0.20–0.35) lo deja fuera por 8.57e-9 en valor absoluto — marginal en número, total
en significado.

**G8 · El SMD de airblast es ×3–4 pequeño, y eso rompe el LBO.** En A2 el
modelo da SMD = 10.7 µm frente a la banda de clase 35–60 µm; en A3, 11.6 µm
frente a 30–50 µm. Las dos correlaciones dimensionales de airblast
(El-Shanawany & Lefebvre, Rizkalla & Lefebvre) **sí están verificadas**
contra arXiv:1906.04042 Tabla 7, así que el conflicto es entre dos fuentes
`verified`, no entre una verificada y una provisional.

La consecuencia es grave porque `φ_LBO ∝ D₃₂²`:

| Caso | SMD [µm] | φ_LBO | LBO_margin | ¿creíble? |
|---|---|---|---|---|
| B1 (vaporizador, d₅₀ = 50 µm verificado) | 50.0 | 0.525 | **1.00** | sí |
| A1 (vaporizador) | 50.0 | 0.507 | 1.13 | sí |
| A2 (airblast) | 10.7 | 0.025 | **42.5** | **no** |
| A3 (airblast) | 11.6 | 0.044 | 25.3 | **no** |
| D1 (airblast, T₃ = 1093 K) | 10.4 | 0.0017 | **475.8** | **no** |

Un margen de LBO de ×476 dice «esta cámara no se apaga jamás». Eso no es
un resultado, es un síntoma. Los casos con vaporizador se salvan porque el
SMD no sale de una correlación sino del d₅₀ = 50 µm verificado del KJ66.

**G6 · Ø de carcasa sistemáticamente bajo en las clases 1–2.** A2:
114.4 mm vs 120–200. A3: 142.3 mm vs 180–300. D1: 71.7 mm vs 120–200.
Sospechoso concreto y comprobado: `coeffs()["PLF"] = 20.0`, que es la fila
`micro_target` de `plf_by_arch.csv`, se usa para **todas** las
arquitecturas, mientras la fila de la arquitectura da PLF = 36 (`can`) o
22.5 (`reverse_annular`). Como `A_ref ∝ √(PLF/(Δp/p))`, usar el PLF de la
arquitectura subiría A_ref ×√(36/20) = 1.34 en `can` — que es justo el
orden del déficit.

**G7 · η_comb por debajo de banda en las clases 1–2** (0.9882 vs ≥0.99 y
0.9804 vs ≥0.995). El driver es `η_mix`, con `K_MIX_DP` constante de la
casa y una penalización de penetración cuadrática.

**G9 · φ_PZ fuera de banda.** En A2/A3 hereda G1. En D1 es un hallazgo
propio: con T₃ = 1093 K el φ_global que impone el balance es 0.118, y **ni
con `frac_pz` en su mínimo de contrato (0.15)** se alcanza φ_PZ ≥ 0.8
(sale 0.794). Traducción de diseño: en recuperación extrema un combustor
real no puede repartir el aire con las reglas de esta tabla — necesita
piloto/escalonamiento o una PZ efectiva mucho menor que el mínimo del θ.

**G10 · τ fuera de banda** en A3 (11.3 ms), B2 (8.5 ms) y D1 (3.8 ms).
Hereda el dimensionado (G6) y, en la clase 0, el mismo defecto de banda
que G4.

**G11 · T de pared por encima de banda.** Marginal en B1 (1059 K vs ≤1050,
+0.9 %) y B2 (1073 K, +2.2 %); **catastrófico en D1: 1466 K frente a
1000–1150 K**. Sospechoso nº 1: el suelo `EPS_G_FLOOR_LUMINOUS = 0.15`,
calibración de la casa que hoy **sustituye** a la física declarada (el ε_g
calculado vale 0.0065, dos órdenes por debajo de la banda 0.2–0.6 de
literatura) y que, barrido por esa banda, mueve T_w1 entre **808 K y
1573 K** con `g[1]` cambiando de signo (`PENDIENTES §5`).

**G12 · La falsación más limpia de la campaña.** En D1 el modelo declara
que **Inconel 625 no aguanta** (T_w1 = 1466 K frente al límite 1253 K menos
50 K de margen). Bath/HiETA **fabricó y operó ese combustor en Inconel
625**. O el modelo es conservador, o el caso está mal reconstruido — y como
lo único verificado del caso son precisamente T₃ = 1093 K y el material,
el peso de la prueba cae del lado del modelo. Nótese que T_w1 se mueve
apenas 1466–1540 K en un barrido de 18 combinaciones de reparto de aire y
número de orificios: **el resultado es robusto frente a los supuestos del
caso**, que es lo que lo convierte en evidencia.

**G13 · La correlación de NOx de cabecera tiene el signo equivocado.**
`record["EI_NOx"]` es `EI_NOx_lefebvre` (`physics_core` línea 3042), y en un
barrido por el lado pobre da pendiente **negativa** frente a φ_PZ:

| φ_PZ | lefebvre (cabecera) | rizk_mongia | zonal | odgers | laviolette | EI_CO |
|---|---|---|---|---|---|---|
| 0.604 | 2.619 | 3.627 | 0.013 | 0.527 | 0.023 | 304.4 |
| 0.670 | 2.460 | 3.627 | 0.081 | 1.088 | 0.080 | 216.5 |
| 0.797 | 2.221 | 3.627 | 1.048 | 3.251 | 0.544 | 117.0 |
| 0.899 | 2.087 | 3.627 | 3.498 | 5.991 | 1.584 | 77.6 |
| **1.048** | **2.000** | 3.627 | **3.756** | **8.911** | **3.172** | 57.6 |
| 1.260 | 2.102 | 3.627 | 0.246 | 5.594 | 1.405 | 81.5 |
| 1.599 | 2.376 | 3.627 | 0.000 | 1.597 | 0.157 | 177.1 |

La forma de Lefebvre no depende de φ_PZ (va con T_st, P₃, V_c y ṁ): el
−25 % que se ve es puro artefacto del denominador (g de NOx por kg de
combustible, y el combustible cambia con η_comb). Las tres formas
**zonales** —zonal, Odgers y Laviolette— sí dan el signo correcto **y el
pico en φ_PZ ≈ 1.05**, justo donde está el máximo de T_ad. El CO también
va bien (304 → 58 g/kg al enriquecer, y vuelve a subir al pasarse).

> **Acción, y no cuesta ningún dato nuevo: la cabecera de NOx debe ser la
> zonal, no la de Lefebvre.** Es una decisión de modelo, no una
> calibración. La de Lefebvre sigue reportándose en el abanico (donde es
> útil: da la tendencia con τ y con P₃).

**G14 · Rizk & Mongia global es insensible a φ_PZ** (variación relativa
0.00 % en todo el barrido). No es un fallo de implementación: la forma
global no lleva φ. Se marca para que nadie la use como indicador de
tendencia.

**G15 · La forma zonal es insensible a τ** (variación < 2 %): es función de
φ y de T de zona, no del tiempo de residencia. Complementaria de G13: para
tendencia en φ, la zonal; para tendencia en τ, Odgers o Lefebvre.

### 4.3 Lo que SÍ cuadra

**C1 es el mejor resultado del paquete y el único con dato duro de una
máquina real.** Dos puntos publicados del mismo microturborreactor, un solo
modelo, sin recalibrar nada:

| punto | frac_pz | φ_PZ publicada | φ_PZ Phy-CB | Δ |
|---|---|---|---|---|
| base | 0.3935 | 0.890 | 0.9038 | **+1.55 %** |
| optimizado | 0.3200 | 1.090 | 1.0922 | **+0.20 %** |
| invariante φ_PZ,opt/φ_PZ,base | — | 1.225 | 1.208 | **−1.32 %** |

El invariante es el check fuerte: **no depende del punto de operación**, de
modo que mide exclusivamente la cadena reparto de aire → φ_PZ de Ciencia §3.
Además el modelo reproduce la razón física por la que el paper hizo el
cambio (menos aire de PZ ⇒ φ_PZ mayor ⇒ llama estable en aceleración).

Otros PASS con contenido:

* **Cierre del balance de energía**: < 4e-3 % en los seis casos
  (máximo −3.6e-3 % en A1; el residuo es el punto fijo de η_comb, que
  entra en el FAR).
  Es verificación, pero es la verificación que sostiene todo el
  dimensionado.
* **Clase 0 completa** (A1, B1): π, T₄, ṁ, Δp/p, φ_PZ, U_ref, τ, Ø de
  carcasa, η_comb, PF y SMD **todos en banda**. La clase para la que el
  proyecto está pensado es la que mejor se comporta.
* **Geometría del vaporizador** (B1): 6 tubos de 64 mm con codo de 30°,
  dentro del dato ✅ [W]. Es CONSISTENCIA, no predicción — el dato entra por
  `physics_core` y sale por el record; vale para detectar que alguien lo
  rompa.
* **Orden de magnitud de NOx**: abanico L0 en 2.0–8.9 g/kg y **CRN L1
  (Cantera, el número menos empírico del paquete) en 3.07 g/kg**, con
  EI_CO_crn = 19.6 g/kg y η_comb_crn = 0.9954. Todo dentro de la banda de
  referencia de queroseno micro (0.1–5 g/kg) con el factor ×2 que permite
  la wave.
* **Ancla de regresión REF_CB10**: reproducida bit a bit (rtol < 1e-6).

---

## 5. Prioridad de recalibración

Si algo falla **no se toca todo a la vez**. Se toca en este orden, se
registra el movimiento con su justificación, y **se re-corren TODOS los
casos** después de cada ajuste (regla de la familia). Esta lista
re-prioriza la de `PENDIENTES §6` con lo que la campaña ha medido.

| # | Qué tocar | Por qué es el primer sospechoso | Con qué se cierra | Huecos que cierra |
|---|---|---|---|---|
| **0** | **`class_table.csv`** (no es una constante: es el dato) | Antes de calibrar nada contra la tabla hay que arreglar la tabla: sus bandas de FAR y de (T₃, T₄) no pueden ser ciertas a la vez (§4.1). Calibrar contra un dato inconsistente es peor que no calibrar | Volver a la fuente de EstadoDelArte §7: ¿el FAR es de crucero o de ralentí? ¿el T₄ es TIT o EGT? | G1, G2, G3, G4 |
| **1** | **Cabecera de EI_NOx: `lefebvre` → `zonal`** | Signo equivocado frente a φ_PZ, demostrado con barrido (§4.2 G13). **No requiere ningún dato nuevo**: es una decisión de modelo, gratis y de impacto inmediato en la utilidad del optimizador | Ninguna medición. Cambiar la línea 3042 de `physics_core.py` y re-correr | G13 |
| **2** | **Suelo de ε_g** (`EPS_G_FLOOR_LUMINOUS = 0.15`) y detrás el coeficiente 0.29 | Calibración de la casa que hoy SUSTITUYE a la física declarada, y lo que más mueve T de pared: ±765 K al barrer la banda de literatura, con `g[1]` cambiando de signo. Es lo que hace que el modelo declare inviable un combustor que existe (G12) | Lefebvre & Ballal **cap. 8** (ε_g de llama luminosa) o termopares en el liner del banco propio (nivel F) | G11, G12 |
| **3** | **Cadena SMD de airblast → φ_LBO** | El SMD sale ×3–4 pequeño frente a la banda de clase y, como φ_LBO ∝ D₃₂², produce márgenes de LBO de ×42 a ×476 (§4.2 G8). Ojo: las correlaciones de SMD están VERIFICADAS, así que el sospechoso puede ser la banda de la tabla, el ALR efectivo o el propio C_LBO | Comparar la banda 30–60 µm de EstadoDelArte §7 con su fuente; ensayo de extinción pobre en banco (barato y muy discriminante) | G8 |
| **4** | **k del PF** (`pf_constant.csv::k_dilution`) | Calibrada a UN punto, y ese punto ES el ancla. PF = 0.35000 en los seis casos: no predice (§4.2 G5) | Lefebvre & Ballal **cap. 5**; y con ≥2 cámaras medidas (PF medido en pequeñas reales: 0.21–0.24) reajustar k con L_DZ/D_L variable | G5 |
| **5** | **PLF por arquitectura** (`plf_by_arch.csv`) | Hoy se usa `micro_target` (PLF = 20) para todas las arquitecturas; la fila de la arquitectura da 36 (`can`) y 22.5 (`reverse_annular`). Explica el déficit de Ø de carcasa de las clases 1–2 (§4.2 G6) | Cap. 4; decisión de modelo + comprobar contra las tres clases a la vez | G6, G10 |
| **6** | **`K_MIX_DP`** y la penalización de penetración de `η_mix` | η_comb queda por debajo de banda en las clases 1–2 y es el driver en 4 de 6 casos. Además es lo que colapsa D1 con el reparto ingenuo | Cap. 5 + la red 1D de WAVE2, que resuelve el reparto en vez de imponerlo | G7 |
| **7** | **`C_LBO_SI`** | Calibrada a UN punto con ±45 % declarado. Queda detrás del SMD: arreglar el SMD primero cambia el diagnóstico | Cap. 5, o ensayo de extinción pobre. Upgrade previsto: modelo Flame Volume (±15 %) como correlación enchufable | — |
| **8** | **Carta η_c(θ)** (`eta_vs_theta.csv`) | Un solo punto de anclaje. Hoy no muerde (el criterio aerodinámico gana en los seis casos, `sizing_driver = "aero"`), pero si la carta real es más exigente cambia TODA la geometría (A_ref +59 %, V_c y τ ×2) | Digitalizar la carta completa a `data/` (capa 0), no un ajuste | — (riesgo latente) |

**Lo que NO se va a hacer:** ajustar una constante por caso hasta que
cuadre. Está prohibido por diseño (WAVE7 §5) y es lo que distingue esta
campaña de un ejercicio de curve-fitting.

---

## 6. UQ mínima (WAVE7 §3.5)

Las bandas se calculan con `σ_total = √(σ_modelo² + σ_ensemble²)` y son una
**APROXIMACIÓN DECLARADA**: suponen errores relativos, gaussianos,
independientes entre sí y del ensemble, y modelo lineal en el entorno del
punto. **Ninguna de las cuatro está demostrada.** En la campaña
σ_ensemble = 0 porque no hay surrogate entrenado ⇒ la banda es *solo*
incertidumbre de modelo físico. Sobre el caso ancla `REF_CB10`:

| Magnitud | valor | banda | origen de la σ |
|---|---|---|---|
| `phi_LBO` | 0.5245 | [0.289, 0.761] | ±45 % ✅ precisión declarada del modelo de LBO de Lefebvre (EstadoDelArte §1, [W]) |
| `LBO_margin` | 0.9985 | [0.549, 1.448] | hereda la de φ_LBO (propagación de 1.er orden) |
| `EI_NOx` | 2.000 g/kg | [1.000, 3.000] | ±50 % (banda ±30–50 % de correlaciones/CRN, extremo conservador) |
| `EI_CO` | 57.57 g/kg | [28.8, 86.4] | ±50 % |
| `dp_over_p` | 0.030 | [0.024, 0.036] | ±20 % — **pero en L0 es entrada**, ver §3 |

Dos magnitudes NO llevan σ gaussiana porque su incertidumbre no es la de
una correlación sino el **barrido medido** de un coeficiente calibrado a un
solo punto. Fingir una σ ahí sería inventar:

| Magnitud | valor | intervalo medido | quién manda |
|---|---|---|---|
| `T_wall_max_K` | 1059 K | **[808, 1573] K** | suelo de ε_g barrido por la banda de literatura 0.05–0.60 (`PENDIENTES §5`); `g[1]` cambia de signo |
| `PF` | 0.3500 | **[0.1225, 0.5915]** | `k_dilution` ×2 y ×0.5 (`PENDIENTES §5`) |

**Calibración afín (WAVE7 §3.6):** sigue en la **identidad**.
`HiFiCalibration` necesita ≥2 pares hi-fi y hoy hay **cero**: no hay L2a
(`cfd/ibrans/` vacío, sin Warp/JAX) ni L2b (fuera de alcance) ni banco. La
campaña corre siempre con `calibrate=False` a propósito — con la afín
activa estaríamos validando la calibración, no la física.

---

## 7. Límites declarados

Se repiten aquí y en el bloque «Límites declarados» de todo `report.html`:

1. **Phy-CB no predice temperatura de metal cuantitativa en ningún nivel
   de fidelidad** (contrato §12). L2b, que iba a darla con CHT + radiación
   (±10 % documentado), quedó fuera de alcance al ser la máquina objetivo
   Windows. El térmico 1D de L0 da tendencia y margen frente al límite del
   material; L2a daría mapas de pared **adiabática** para *ranking* entre
   diseños. Toda comparación de T de pared de esta campaña es una
   comprobación de BANDA, no de valor — y el caso D1 muestra hasta qué
   punto eso importa (G11/G12). Cerrar el hueco exige banco (nivel F) o
   reactivar L2b en una máquina con WSL2/Docker; SU2 v8 (LGPL) sigue
   documentado en Ciencia §11.4 como la alternativa empaquetable.
2. **Δp/p no se predice, se impone** (§3). En microturbina es restricción
   dura del ciclo.
3. **De las emisiones nunca se promete valor absoluto**: solo tendencia y
   orden de magnitud ×2. Y hoy ni siquiera la tendencia frente a φ_PZ es
   correcta en la cabecera (G13).
4. **El LBO no está validado** y hoy da márgenes no creíbles con airblast
   (G8).
5. **El PF no está validado**: devuelve su calibración (G5).
6. **Termoacústica fuera del alcance de L0–L2.** Mitigación pasiva
   (resonadores de Helmholtz integrables por AM) + ensayo. Declarado.
7. **42 avisos de coeficiente provisional** llegan a los números del caso
   ancla (`record["limitations"]`); el inventario completo con su impacto
   MEDIDO está en `docs/PENDIENTES_Lverificar.md`.

---

## 8. Nivel F — especificación de ensayo, lista para cuando haya proveedor

WAVE7 §5 lo pide explícitamente: sin impresora LPBF el paquete es completo,
pero la especificación tiene que estar escrita. Es lo único que cierra los
tres pendientes críticos (suelo de ε_g, k del PF, C_LBO) y G8/G11/G12.

**Pieza:** cámara ancla `REF_CB10` (anular de flujo inverso, vaporizador,
Ø carcasa 81.5 mm, L 166.3 mm), tal como la emite
`geometry_generator.generate()` (`combustor.json` + STL + `*_field.vdb`).

**Fabricación**
* Material: **Inconel 625** (`structures_core.MATERIALS`, T_max pared
  1253 K; «polvo preferido para piezas de combustor», AMS7000, el de
  Bath/HiETA).
* Proceso LPBF con control según **SAE AMS7003**; aleación según
  **AMS7000**.
* Espesor de liner 0.8 mm (θ[14]); Ø de efusión 0.6 mm (θ[8]).
  **Corregir `pitch/d` de 6.0 a 4–5 antes de imprimir**: es lo que devuelve
  el blowing ratio a la banda del mapa y hace factible el ancla
  (`PENDIENTES §4.2`), a cambio de subir el nº de orificios.
* Orientación de build: la que emita `dfam_checker.check(...)["orientation"]`.
  Los orificios con eje ⟂ a la dirección de construcción salen en **gota**,
  no circulares (patente de efusión autosoportante).
* Postproceso: alivio de tensiones **antes** de retirar de placa; HIP
  1160–1200 °C / 100–150 MPa / 3–4 h; solución + envejecido según aleación.
* Inspección: **CT** + **verificación de permeabilidad de los ~2200
  orificios de efusión** (con ese número, «orificios abiertos» es requisito
  de calidad, no un detalle). Comprobar los ≥2 puertos de drenaje por
  cavidad que coloca el checker: polvo atrapado sinteriza en el HIP y la
  obstrucción es permanente.

**Instrumentación mínima (por orden de valor para esta campaña)**
1. **Termopares de pared del liner** (≥5 axiales en la línea caliente,
   tipo K o N con unión soldada por puntos). Cierran G11/G12 y el suelo de
   ε_g — el pendiente nº 2 de §5. *Es el dato más valioso del banco.*
2. **Ensayo de extinción pobre**: rampa de caudal de combustible a P₃ y T₃
   fijos hasta apagado, ≥3 repeticiones por punto. Cierra C_LBO y arbitra
   G8. Barato y muy discriminante.
3. **Δp frío** (sin combustión, barrido de ṁ): valida el criterio
   aerodinámico y el reparto por C_d·A_h — y es el único camino a validar
   Δp/p sin CFD.
4. **Rejilla de termopares en el plano de salida** (≥12 puntos, 2 radios):
   PF y RTDF medidos. Cierra el pendiente nº 4 (k del PF).
5. **Analizador de gases** (NOx/CO/O₂ secos, corregidos a 15 % O₂): pasa
   las emisiones de «tendencia» a «orden de magnitud verificado».
6. **Medición de SMD** del inyector en banco frío (difracción láser) si se
   ensaya la variante airblast: arbitraría G8 directamente.

Con solo (1) y (2) —un arranque y una extinción— esta campaña pasaría de
«una sola ancla real» a tener anclado el térmico y la estabilidad, que son
los dos huecos que hoy más peso tienen en el veredicto.

---

## 9. CI y cómo se mantiene esto vivo

`.github/workflows/ci.yml`:

| Job | Qué corre |
|---|---|
| `python-nocantera` (Linux + Windows) | manifiesto de la capa 0 → `test_phycb.py` → `validate.py --strict` → `python -m build`. **Solo NumPy**: comprueba que toda la degradación etiquetada del contrato §3 funciona |
| `python-cantera` (Linux + Windows) | lo mismo con Cantera de **conda-forge** (nunca pip en Windows) y `validate.py --strict --l1` (incluye el CRN) |
| `cfd` | escalera V&V con **mallas mini** (`PHYCB_CFD_MESH=mini`). Si `cfd/tests/` está vacío —como hoy— lo dice y sigue. **Los V&V completos NO corren en CI**: son script manual |
| `dotnet` | `dotnet build CombustorDesigner.sln` + smoke del contrato (T29). Si PicoGK no arranca en el runner, T29 **se salta con motivo**, nunca aprueba en falso |

No hay job de OpenFOAM: L2b está fuera de alcance (contrato §12).

**Contrato de mantenimiento del documento.** `validate.py --strict` falla
si:
* un check `derived` o `data` falla **sin** estar en `KNOWN_GAPS` (⇒ hay
  una regresión o un hallazgo nuevo sin documentar), **o**
* un `known_gap` **empieza a pasar** (⇒ el hueco se cerró y este documento
  ha quedado desactualizado: hay que borrar la entrada), **o**
* el ancla `REF_CB10` se mueve o desaparece.

Es decir: **no se puede arreglar el código sin actualizar este documento, ni
al revés.** Es deliberado.

---

## 10. Veredicto

**La campaña está corrida, es reproducible y ha encontrado cosas.** Ese es
el criterio de aceptación de WAVE7 §4.1, y se cumple: ≥3 casos de niveles
A–D evaluados (son 9), veredictos honestos, y los fallos documentados con
plan de recalibración priorizado.

Lo que se puede afirmar hoy con evidencia:

* La cadena **reparto de aire → φ_PZ** está validada contra una máquina
  real con error del 0.2–1.6 % y sin recalibrar nada (C1).
* El **balance de energía** cierra exactamente y el **dimensionado de la
  clase 0** cae dentro de todas las bandas de la literatura de su clase.
* El **orden de magnitud del NOx** es correcto, incluido el CRN L1 de
  Cantera (3.07 g/kg).

Lo que NO se puede afirmar:

* **Nada sobre la temperatura de pared**, en ningún nivel de fidelidad.
* **Nada sobre el PF**: devuelve su calibración.
* **Nada sobre el LBO**, y con airblast los números que salen no son
  creíbles.
* **Nada sobre Δp/p** hasta que exista L1.5 o L2a.
* Nada sobre las clases 1 y 2 más allá de que el dimensionado sale
  sistemáticamente pequeño.

Y una conclusión que la campaña no esperaba encontrar: **la tabla de
referencia por clase, que es el ancla nº 1 del plan de validación, está
internamente inconsistente.** Arreglarla es el paso 0 de cualquier
recalibración, porque hoy una parte de los «fallos» del modelo son en
realidad fallos del patrón contra el que se le mide.

> **Estado de WAVE7: campaña COMPLETA, sistema NO VALIDADO.** Igual que
> WAVE1 quedó «completa pero no cerrada», aquí la infraestructura de
> validación está entera y corriendo en CI, pero el sistema tiene **una
> sola ancla real** (el reparto de aire de C1) y cinco constantes de
> portada sin anclar. Decirlo es el entregable.
