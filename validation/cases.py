"""
QUASAR Phy-CB · validation/cases.py
===================================
Base de datos de la campaña de VALIDACIÓN física (`docs/VALIDATION.md`):
casos con datos de entrada, **procedencia** y **supuestos declarados**.

Distinción de la familia (WAVE7 §1), que este módulo NO puede relajar:

* **Verificación** — «¿resolvemos bien las ecuaciones?» → `test_phycb.py`
  (T1–T40, 350/350) y la escalera V1–V9 del CFD.
* **Validación** — «¿son las ecuaciones CORRECTAS?» → esto: comparación
  contra datos publicados **sin recalibración por-caso**.

Patrón: `Phy-AC/validation/machines.py`. Cada entrada declara la ESPEC
publicada; `validate.py` construye el θ 20-D y corre `evaluate`.

Convenciones de este archivo
----------------------------
* **Sin imports.** Es dato puro e inspeccionable; los gates van como
  literales con su cita del contrato §1 (`arch` 0=can · 1=annular ·
  2=reverse_annular; `injector` 0=airblast · 1=simplex · 2=vaporizer ·
  3=micromix; `fuel` 0=jet_a · 1=propane · 2=h2).
* Todo número que **no** venga de una fuente aparece en `assumptions`
  con la palabra SUPUESTO y su justificación. Nunca se rellena de memoria.
* Cada parámetro comparado se etiqueta `derived` o `input`:
  - `derived` — lo **calcula** el modelo ⇒ puede validar o falsar.
  - `input`   — se lo damos nosotros ⇒ es trazabilidad, **no** evidencia.
  Un caso cuyo veredicto se apoye en un `input` sería un caso maquillado.
* `known_gap=True` marca un check que HOY falla y está explicado en
  `docs/VALIDATION.md`. `validate.py --strict` no falla por él (para que
  la CI detecte REGRESIONES, no el estado conocido) **pero sí falla si un
  `known_gap` empieza a pasar**: un hueco que se cierra hay que quitarlo
  del documento, no dejarlo mintiendo.

Añadir un caso = añadir una entrada aquí y re-correr
`python validation/validate.py`.
"""

# ---------------------------------------------------------------------------
# 1. Criterios cuantitativos v1 — WAVE7 §3.2, SIN recalibración por-caso
# ---------------------------------------------------------------------------
#: Δp/p: ±20 %. ⚠️ Ver `NON_EVALUABLE["dp_over_p"]`: en L0 Δp/p es ENTRADA
#: (`plf_by_arch.csv::micro_target.dp_over_p_nom`, restricción dura del
#: ciclo), no salida — el criterio no es evaluable sin L1.5/L2a.
DP_TOL_REL = 0.20
#: T4: ±3 % (balance de energía: «debe clavar»). ⚠️ En Phy-CB T4 es θ[19];
#: lo que se puede medir es el CIERRE del balance (round-trip FAR↔T4).
T4_TOL_REL = 0.03
#: φ_PZ frente al dato de reparto de aire publicado (nivel C).
PHI_PZ_TOL_REL = 0.05
#: Invariante de reparto φ_PZ·frac_pz = φ_global (nivel C, ratio).
PHI_RATIO_TOL_REL = 0.02
#: LBO: φ_LBO dentro de ×1.5 del dato «cuando exista» — hoy NO existe.
LBO_TOL_FACTOR = 1.5
#: Emisiones: nunca valor absoluto. Solo signo de la tendencia y orden de
#: magnitud ×2 frente a la banda de referencia.
EI_ORDER_FACTOR = 2.0

# ---------------------------------------------------------------------------
# 2. UQ mínima — WAVE7 §3.5
# ---------------------------------------------------------------------------
#: σ relativa de MODELO declarada por cada correlación. Se suma en
#: cuadratura con la σ epistémica del ensemble del surrogate:
#:     σ_tot = √(σ_modelo² + σ_ensemble²)
#: **APROXIMACIÓN DECLARADA**: supone (a) errores relativos, (b)
#: gaussianos, (c) independientes entre sí y del ensemble, (d) modelo
#: lineal en el entorno del punto. Ninguna de las cuatro está demostrada;
#: la banda es un orden de magnitud de la incertidumbre, no un intervalo
#: de confianza. En la campaña σ_ensemble = 0 (no hay surrogate entrenado).
UQ_SIGMA_REL = {
    "phi_LBO": 0.45,      # ✅ precisión declarada del modelo de LBO de
    #                       Lefebvre (EstadoDelArte §1, [W] verificado).
    "LBO_margin": 0.45,   # hereda la de φ_LBO (propagación de 1er orden).
    "EI_NOx": 0.50,       # banda ±30–50 % de las correlaciones/CRN
    "EI_CO": 0.50,        # (WAVE7 §3.5) — se toma el extremo conservador.
    "dp_over_p": 0.20,    # ±20 % declarado; en L0 es entrada (ver arriba).
}

#: Números cuya incertidumbre NO es una σ de correlación sino el barrido
#: MEDIDO de un coeficiente provisional (docs/PENDIENTES_Lverificar.md §5).
#: Se reportan como intervalo medido, no como banda gaussiana: fingir una
#: σ sobre una constante calibrada a un solo punto sería inventar.
UQ_MEASURED_SWEEP = {
    "T_wall_max_K": dict(
        lo_K=808.0, hi_K=1573.0, driver="suelo de ε_g (EPS_G_FLOOR_LUMINOUS)",
        note=("barrido del suelo por la banda de literatura 0.05–0.60; "
              "`g[1]` cambia de signo (−0.33 ↔ +0.31). PENDIENTES §5")),
    "PF": dict(
        lo_rel=-0.65, hi_rel=+0.69, driver="k_dilution (pf_constant.csv)",
        note=("k×2 → PF −65 %; k×0.5 → PF +69 %. k está calibrada a UN "
              "punto y ese punto ES el ancla. PENDIENTES §5 y §6.2")),
}

# ---------------------------------------------------------------------------
# 3. Criterios que NO se pueden evaluar en esta máquina — declarados
# ---------------------------------------------------------------------------
#: Regla de la casa: un criterio que no se puede medir se DECLARA, nunca se
#: convierte en un check que aprueba trivialmente.
NON_EVALUABLE = {
    "dp_over_p": (
        "Δp/p ±20 % NO es evaluable en L0. En Phy-CB el Δp/p es **entrada** "
        "del dimensionado (`plf_by_arch.csv::micro_target.dp_over_p_nom` = "
        "3 %, o `evaluate(dp_over_p_target=…)`): `_holes()` reparte el área "
        "de los orificios PARA cumplirlo, de modo que `record['dp_over_p']` "
        "reproduce el objetivo por construcción (identidad, no predicción). "
        "Validarlo exige `dp_over_p_cfd` de L1.5/L2a — y en esta máquina no "
        "hay JAX/XLB/Warp ni `cfd/ibrans/`, así que ambos degradan. "
        "Lo único que se comprueba aquí es que el objetivo cae en la banda "
        "de clase (2–4 %), que es trazabilidad, no validación."),
    "T4": (
        "T4 ±3 % NO es evaluable como validación: T4 es θ[19] (punto de "
        "operación pineado por `DesignSpec.fix_operating_point`) y el FAR "
        "sale de él por balance de entalpía. Sería validable con un caso "
        "que publicase (T3, ṁ_aire, ṁ_combustible, T4) a la vez, y **no hay "
        "ninguno en el paquete**. Se evalúa en su lugar el CIERRE del "
        "balance (round-trip T4 → FAR → T4), que es VERIFICACIÓN."),
    "LBO": (
        "φ_LBO ×1.5 NO es evaluable: no hay ningún dato de extinción pobre "
        "medido en el paquete ni en las fuentes de EstadoDelArte §12. La "
        "constante `C_LBO_SI` está calibrada a UN punto ancla con ±45 % "
        "declarado (PENDIENTES §2.1 #14): en el ancla no predice, devuelve "
        "su calibración. Cierra con banco propio (nivel F) — el ensayo de "
        "extinción pobre es barato y muy discriminante."),
    "T_metal": (
        "Phy-CB **no predice temperatura de metal cuantitativa en ningún "
        "nivel de fidelidad** (contrato §12): L2b, que iba a darla con "
        "CHT+radiación, quedó fuera de alcance. El térmico 1D de L0 da "
        "tendencia y margen frente al límite del material; L2a daría mapas "
        "de pared adiabática para *ranking*. Toda comparación de T de pared "
        "de esta campaña es una comprobación de BANDA, no de valor."),
}

# ---------------------------------------------------------------------------
# 4. Niveles de la escalera declarados y NO implementados (WAVE7 §2)
# ---------------------------------------------------------------------------
DECLARED_NOT_RUN = [
    dict(level="E", name="Sandia D / PRECCINSTA (bases TNF/DLR)",
         why=("Valida el CFD **L2a**, no el diseño preliminar: pertenece a "
              "la escalera V1–V9 de WAVE6 (`cfd/tests/`). En esta máquina "
              "`cfd/ibrans/` y `cfd/tests/` están VACÍOS y no hay "
              "JAX/XLB/Warp instalados, así que L1.5 y L2a degradan con "
              "etiqueta. No se simula nada y no se reporta ningún número.")),
    dict(level="F", name="Banco propio (cámara ancla impresa, IN625 ~10 kW)",
         why=("NO EXISTE. Sin impresora LPBF ni banco no hay dato. Es el "
              "único nivel que cerraría los tres pendientes críticos "
              "(suelo de ε_g, k del PF, C_LBO) — ver §«especificación de "
              "impresión» de docs/VALIDATION.md, lista para cuando haya "
              "proveedor.")),
    dict(level="L2b", name="Backend OpenFOAM (WAVE5-B)",
         why=("FUERA DE ALCANCE por decisión de proyecto (contrato §12): "
              "máquina Windows, exigiría WSL2/Docker. `Fidelity.L2B` se "
              "conserva en el enum (es ABI) y degrada con "
              "`source='openfoam_L2B(out_of_scope→…)'`. Consecuencia que "
              "se repite en todas partes: **Phy-CB no predice temperatura "
              "de metal cuantitativa en ningún nivel de fidelidad**.")),
]

# ---------------------------------------------------------------------------
# 5. Plantilla de θ (contrato §1, NDIM = 20)
# ---------------------------------------------------------------------------
#: Índices del vector de diseño. Se replican aquí (en vez de importarlos)
#: para que este archivo siga siendo dato puro; `validate.py` comprueba
#: contra `physics_core.VAR_NAMES` que el orden no se ha movido.
THETA_INDEX = {
    "arch": 0, "AL_over_Aref": 1, "frac_pz": 2, "frac_sz": 3, "frac_dz": 4,
    "swirl_SN": 5, "n_holes_pz": 6, "n_holes_dz": 7, "d_eff_mm": 8,
    "pitch_over_d_eff": 9, "injector": 10, "ALR": 11, "L_pz_over_DL": 12,
    "L_dz_over_DL": 13, "t_liner_mm": 14, "fuel": 15, "P3_Pa": 16,
    "T3_K": 17, "mdot_kgs": 18, "T4_K": 19,
}

#: Neutro de partida: es `THETA_REF` (= caso ancla REF_CB10, contrato §1).
#: Cada caso sobreescribe SOLO lo que su fuente o su supuesto declara.
THETA_BASE = [
    2.0, 0.70, 0.30, 0.12, 0.42, 0.70, 8.0, 10.0, 0.60, 6.0,
    2.0, 4.0, 0.75, 1.80, 0.80, 0.0, 250_000.0, 430.0, 0.13, 1150.0,
]


def theta_from_spec(spec: dict) -> list:
    """θ 20-D desde el `spec` de un caso: base + sobreescrituras nombradas.

    Dato puro: no importa `physics_core` ni NumPy. `validate.py` valida el
    orden contra `physics_core.VAR_NAMES` antes de usarlo (si el contrato
    §1 se moviera, el runner para en seco en vez de evaluar basura).
    """
    theta = list(THETA_BASE)
    for name, value in spec.items():
        if name not in THETA_INDEX:
            raise KeyError(f"'{name}' no es una variable de θ (contrato §1)")
        theta[THETA_INDEX[name]] = float(value)
    return theta


# ---------------------------------------------------------------------------
# 6. NIVEL A — tabla de referencia por clase (`data/reference/class_table.csv`)
# ---------------------------------------------------------------------------
#: Parámetros de `class_table.csv` que se comparan, y CÓMO se obtienen.
#:   key    → fila `param` de la tabla
#:   rec    → clave del record de Phy-CB (o expresión derivada, ver
#:            `validate._class_value`)
#:   role   → "derived" (el modelo lo calcula ⇒ valida) ·
#:            "input" (se lo damos ⇒ solo trazabilidad)
CLASS_PARAMS = [
    dict(key="pressure_ratio",      rec="_pi",            role="input"),
    dict(key="T4_K",                rec="T04_K",          role="input"),
    dict(key="mdot_air_kgs",        rec="mdot_air_kgs",   role="input"),
    dict(key="dp_over_p",           rec="dp_over_p",      role="input"),
    dict(key="t_liner_mm",          rec="t_liner_mm",     role="input"),
    dict(key="d_eff_mm",            rec="d_eff_mm",       role="input"),
    dict(key="SN_swirl",            rec="swirl_SN",       role="input"),
    dict(key="FAR_global",          rec="f_fuel_air",     role="derived"),
    dict(key="phi_global",          rec="phi_global",     role="derived"),
    dict(key="phi_pz",              rec="phi_pz",         role="derived"),
    dict(key="U_ref_ms",            rec="U_ref_ms",       role="derived"),
    dict(key="tau_ms",              rec="tau_ms",         role="derived"),
    dict(key="V_c_L",               rec="_V_c_L",         role="derived"),
    dict(key="D_casing_mm",         rec="D_casing_mm",    role="derived"),
    dict(key="I_MW_m3_atm",         rec="I_MW_m3_atm",    role="derived"),
    dict(key="eta_comb_design",     rec="eta_comb",       role="derived"),
    dict(key="PF",                  rec="PF",             role="derived"),
    dict(key="frac_cool",           rec="frac_cool",      role="derived"),
    dict(key="T_wall_liner_K",      rec="T_wall_max_K",   role="derived"),
    dict(key="n_eff_holes",         rec="n_eff_holes",    role="derived"),
    dict(key="SMD_um",              rec="SMD_um",         role="derived"),
]

_CLASS_SOURCE = (
    "data/reference/class_table.csv (`@status: verified`), transcrita de "
    "docs/Quasar_PhyCB_EstadoDelArte.md §7 «Tabla de valores de referencia "
    "— clase microturbina»; las marcas [W] de esa tabla se verificaron en "
    "fuente web el 2026-07-29 (URLs en EstadoDelArte §12). Caveat de la "
    "propia tabla: las cifras certificadas de Capstone C30/C65 y Turbec "
    "T100 NO se verificaron.")

_CLASS_ASSUMPTIONS_COMMON = [
    "SUPUESTO · punto de operación = punto MEDIO de cada banda de la clase "
    "(π, T₃ no recuperada, ṁ, T₄). La tabla da bandas, no un punto: elegir "
    "el centro es la opción neutra y es lo que hace falsable el resultado "
    "(cualquier otra elección se podría acusar de buscar la banda).",
    "SUPUESTO · reparto de aire = fila `microturbine` de "
    "`data/lefebvre/air_split_by_arch.csv` (frac_pz/sz/dz = 0.30/0.12/0.42, "
    "`@status: verified`). NO se ajusta por caso: ajustar frac_pz para "
    "centrar φ_PZ sería exactamente la recalibración por-caso que la wave "
    "prohíbe (WAVE7 §5).",
    "SUPUESTO · n_holes_pz = 8. `class_table.csv` no da número de orificios "
    "de zona primaria (solo de dilución); 8 es el valor de THETA_REF.",
    "SUPUESTO · A_L/A_ref, L_PZ/D_L y L_DZ/D_L = defaults de "
    "`zone_lengths.csv` (tabla ⚠️ provisional entera, PENDIENTES §2.1 #2).",
]

CASES = [
    # -- A · tabla de referencia por clase ---------------------------------
    dict(
        cid="A1", level="A", kind="class_bands", class_id=0,
        name="Clase 0 · microturborreactor 0.1–5 kW",
        spec=dict(arch=2.0, injector=2.0, fuel=0.0,
                  P3_Pa=278_643.75,   # π = 2.75 (medio de 2–3.5) × P_atm
                  T3_K=435.0,         # medio de 400–470 K (no recuperada)
                  mdot_kgs=0.15,      # medio de 0.05–0.25 kg/s
                  T4_K=1175.0,        # medio de 1100–1250 K
                  t_liner_mm=0.65, d_eff_mm=0.60, swirl_SN=0.85,
                  n_holes_dz=8.0, ALR=3.5),
        source=_CLASS_SOURCE,
        assumptions=_CLASS_ASSUMPTIONS_COMMON + [
            "SUPUESTO · arquitectura = anular de flujo inverso e inyección "
            "por vaporizador: es la práctica dominante de la clase "
            "(EstadoDelArte §2.1–2.2, compresor centrífugo + eje corto).",
        ],
        notes=("La clase del caso ancla REF_CB10. Es el nivel A «gratis»: "
               "no valida contra hardware, valida que el dimensionado L0 "
               "cae donde la literatura de la clase dice que cae."),
    ),
    dict(
        cid="A2", level="A", kind="class_bands", class_id=1,
        name="Clase 1 · microturbina 5–50 kW",
        spec=dict(arch=2.0, injector=0.0, fuel=0.0,
                  P3_Pa=379_968.75,   # π = 3.75 (medio de 3–4.5)
                  T3_K=465.0,         # medio de 430–500 K (no recuperada)
                  mdot_kgs=0.375,     # medio de 0.15–0.6 kg/s
                  T4_K=1225.0,        # medio de 1150–1300 K
                  t_liner_mm=0.80, d_eff_mm=0.65, swirl_SN=0.75,
                  n_holes_dz=10.0, ALR=4.5),
        source=_CLASS_SOURCE,
        assumptions=_CLASS_ASSUMPTIONS_COMMON + [
            "SUPUESTO · inyección airblast (default de Phy-CB por encima de "
            "la clase hobby, EstadoDelArte §9.2) y T₃ NO recuperada.",
        ],
        notes=("La clase de Bath/HiETA (caso D) en su variante NO "
               "recuperada. Comparar A2 con D1 aísla el efecto de T₃."),
    ),
    dict(
        cid="A3", level="A", kind="class_bands", class_id=2,
        name="Clase 2 · microturbina 50–250 kW",
        spec=dict(arch=0.0, injector=0.0, fuel=0.0,
                  AL_over_Aref=0.66,  # tubular (zone_lengths.csv) ⚠️
                  P3_Pa=430_631.25,   # π = 4.25 (medio de 3.5–5)
                  T3_K=490.0,         # medio de 450–530 K (no recuperada)
                  mdot_kgs=1.0,       # medio de 0.5–1.5 kg/s
                  T4_K=1275.0,        # medio de 1200–1350 K
                  t_liner_mm=1.15, d_eff_mm=0.80, swirl_SN=0.75,
                  n_holes_dz=13.0, ALR=4.5),
        source=_CLASS_SOURCE,
        assumptions=_CLASS_ASSUMPTIONS_COMMON + [
            "SUPUESTO · arquitectura `can`: es la de las microturbinas de "
            "generación de esta clase (Capstone, prototipos universitarios "
            "— EstadoDelArte §2.2) y la que usa A_L/A_ref tubular 0.66.",
        ],
        notes=("Clase de Sierra Turbines (Hastelloy X). El PF medido en "
               "combustores pequeños reales, 0.21–0.24 [W], vive en esta "
               "banda — es el dato de PF más duro del paquete."),
    ),
    dict(
        cid="A4", level="A", kind="table_consistency",
        name="Consistencia interna de `class_table.csv` (FAR ↔ T₃,T₄)",
        source=_CLASS_SOURCE,
        assumptions=[
            "El balance usado es el MISMO del ciclo (CONTEXTO §4 y "
            "`physics_core._far_from_T4`): f = (cp_h·T₄ − cp_c·T₃) / "
            "(η_b·LHV − cp_h·T₄) con cp_h = 1148, cp_c = 1005, η_b = 0.995, "
            "LHV = 43.1 MJ/kg. Es álgebra, no una correlación: no hay "
            "ningún coeficiente provisional dentro.",
            "SUPUESTO · combustible = Jet-A. Con propano o H₂ el LHV cambia "
            "y con él la banda de FAR; la tabla de clase no declara "
            "combustible, así que se toma el default del proyecto.",
        ],
        notes=("Este caso NO valida el modelo: valida el DATO. Si la banda "
               "de FAR/φ_global de la tabla y su banda de (T₃, T₄) no "
               "pueden ser ciertas a la vez, ninguna comparación contra "
               "esas bandas significa nada. Se comprueba en las dos ramas "
               "de T₃ (no recuperada y recuperada)."),
    ),

    # -- B · KJ66 y clase hobby --------------------------------------------
    dict(
        cid="B1", level="B", kind="class_bands", class_id=0,
        name="KJ66 / clase hobby · anular reverso + vaporizador",
        spec=dict(),   # = THETA_BASE = THETA_REF = ancla REF_CB10
        source=(
            "Arquitectura y geometría del vaporizador: "
            "docs/Quasar_PhyCB_EstadoDelArte.md §2.2, marca [W] VERIFICADA "
            "(«6–12 tubos *walking stick* de ~64 mm con codo ~30°»), planos "
            "públicos de la comunidad (gtba.co.uk; plano Englar). Punto de "
            "operación: `THETA_REF` del contrato §1 (caso ancla REF_CB10)."),
        assumptions=[
            "SUPUESTO · punto de operación (P₃ = 250 kPa, T₃ = 430 K, "
            "ṁ = 0.13 kg/s, T₄ = 1150 K) = `THETA_REF` del contrato §1. NO "
            "es una medición del KJ66: es el punto que el proyecto declara "
            "como ancla de clase hobby.",
            "DISCREPANCIA DECLARADA · las cifras de comunidad del KJ66 que "
            "cita `Phy-CC/docs/VALIDACION_ES.md` §2b son π ≈ 2.2, "
            "ṁ ≈ 0.23 kg/s, ~117 krpm — ṁ casi el DOBLE del punto ancla. "
            "El caso se corre en el punto ancla y la variante 0.23 kg/s "
            "se reporta como sensibilidad (caso B2), sin decidir cuál es "
            "«el» KJ66: ninguna de las dos está verificada en esta sesión.",
            "NO HAY DATO · el paquete no contiene ninguna medición de "
            "prestaciones del KJ66 (empuje, EGT, consumo, Δp frío, PF). "
            "Por tanto B **no puede falsar prestaciones**; solo arquitectura "
            "y bandas de clase. Decirlo es el entregable.",
        ],
        notes=("Comprueba además que el modelo devuelve la geometría de "
               "vaporizador verificada (nº de tubos, longitud, codo). Eso "
               "es CONSISTENCIA, no predicción: el dato entra por "
               "`physics_core` y sale por el record — vale para detectar "
               "que alguien lo rompa, no para validar física."),
    ),
    dict(
        cid="B2", level="B", kind="class_bands", class_id=0,
        name="KJ66 · sensibilidad al ṁ de comunidad (0.23 kg/s)",
        spec=dict(mdot_kgs=0.23, P3_Pa=222_915.0),   # π = 2.2 × P_atm
        source=("Cifras de comunidad citadas en "
                "`Phy-CC/docs/VALIDACION_ES.md` §2b (KJ66: PR ≈ 2.2, "
                "0.23 kg/s, ~117 krpm). ⚠️ NO verificadas en esta sesión."),
        assumptions=[
            "SUPUESTO · T₃ = 430 K y T₄ = 1150 K se mantienen del punto "
            "ancla: la fuente de comunidad no da ninguna de las dos.",
            "Es una SENSIBILIDAD, no un caso independiente: mide cuánto se "
            "mueve el dimensionado al duplicar el gasto dentro de la misma "
            "clase. Su verdict no debe leerse como validación del KJ66.",
        ],
        notes=("ṁ = 0.23 kg/s sigue dentro de la banda de clase 0 "
               "(0.05–0.25 kg/s), así que el caso es interno a la clase."),
    ),

    # -- C · microturborreactor optimizado (dato de reparto de aire) -------
    dict(
        cid="C1", level="C", kind="split_anchor",
        name="Microturborreactor optimizado · reparto PZ 39.35 % → 32 %",
        source=(
            "Wiley, International Journal of Aerospace Engineering "
            "2021:6616537 (optimización de un microturborreactor real). "
            "Dato citado y marcado [W] verificado en "
            "docs/Quasar_PhyCB_EstadoDelArte.md §1 y §12, y transcrito a la "
            "cabecera de `data/lefebvre/air_split_by_arch.csv`: la "
            "optimización bajó el aire de zona primaria de **39.35 % a "
            "32 %** (φ_PZ **0.89 → 1.09**) para estabilizar la llama "
            "durante la aceleración."),
        # Los dos puntos publicados: (frac_pz, φ_PZ).
        points=[dict(label="base",      frac_pz=0.3935, phi_pz_pub=0.89),
                dict(label="optimizado", frac_pz=0.3200, phi_pz_pub=1.09)],
        #: φ_global implícita en cada punto: φ_PZ·frac_pz. Que los dos
        #: puntos den la MISMA φ_global (0.3502 vs 0.3488, +0.4 %) es la
        #: prueba de que la fuente y Phy-CB usan la misma definición
        #: (Ciencia §3: φ_PZ = FAR_global/(frac_pz·FAR_st)) — sin eso el
        #: caso no sería comparable y habría que descartarlo.
        phi_global_implied=0.34951,
        spec=dict(arch=2.0, injector=0.0, fuel=0.0,
                  P3_Pa=250_000.0, T3_K=430.0, mdot_kgs=0.13),
        assumptions=[
            "DERIVADO de la fuente (no supuesto) · φ_global = φ_PZ·frac_pz "
            "= 0.89·0.3935 = 0.35022 y 1.09·0.32 = 0.34880; media 0.34951, "
            "dispersión ±0.20 %. Los dos puntos publicados son mutuamente "
            "consistentes con la definición de Ciencia §3.",
            "SUPUESTO · P₃ = 250 kPa, T₃ = 430 K, ṁ = 0.13 kg/s: el paper "
            "no da el punto de operación en la cita disponible. Se toman "
            "los del ancla (clase 0, dentro de banda). **No afectan al "
            "check**: φ_PZ = FAR/(frac_pz·FAR_st) y el runner ajusta T₄ "
            "para clavar la φ_global derivada, así que el caso mide el "
            "reparto de aire, no el punto.",
            "SUPUESTO · T₄ se resuelve por bisección acotada (60 pasos, "
            "800–1800 K) para que `record['phi_global']` = 0.34951. Es la "
            "misma técnica que `Phy-AC/validation/validate.build_theta` "
            "(bisección de φ1 para clavar el annulus publicado): se le da "
            "al modelo el dato medido y se califica lo que él deduce.",
            "SUPUESTO · frac_sz y frac_dz se reescalan proporcionalmente "
            "para dejar frac_cool = 0.16 (nominal `microturbine`) en los "
            "dos puntos: la fuente solo publica frac_pz.",
        ],
        notes=("Es el caso MÁS falsable del paquete: dos puntos publicados "
               "de la misma máquina con el mismo φ_global. C1a comprueba el "
               "INVARIANTE (φ_PZ,base/φ_PZ,opt = frac_pz,opt/frac_pz,base) "
               "que no depende del punto de operación; C1b comprueba el "
               "valor ABSOLUTO de φ_PZ en cada punto."),
    ),

    # -- D · Bath / HiETA ---------------------------------------------------
    dict(
        cid="D1", level="D", kind="thermal_extreme", class_id=1,
        name="Bath/IAAPS/HiETA · IN625, swirl radial, T₃ = 1093 K",
        source=(
            "Energy Conversion and Management 248:114805 (2021) e "
            "International Journal of Engine Research (2023), vía "
            "docs/Quasar_PhyCB_EstadoDelArte.md §3.4 (marca [W] "
            "verificada): combustor tubular cónico con swirl radial e "
            "inyección de combustible por los álabes del swirler («solo "
            "fabricable por AM»), **Inconel 625**, CFD flamelet validado, "
            "−75 % NOx / −40 % CO / +20 % calidad de mezcla con upstream "
            "fuelling + lattice de tres filas, integrado con recuperador AM "
            "de HiETA que eleva **T₃ a 820 °C = 1093 K** — clase Bladon "
            "12 kW. El T₃ recuperado está además en la cabecera de "
            "`data/reference/class_table.csv`."),
        spec=dict(arch=0.0, injector=0.0, fuel=0.0,
                  AL_over_Aref=0.66,
                  frac_pz=0.15,        # mínimo del contrato §1 — ver notas
                  frac_sz=0.10, frac_dz=0.10,   # ⇒ frac_cool = 0.65
                  P3_Pa=379_968.75,    # π = 3.75 (medio clase 1)
                  T3_K=1093.0,         # ✅ VERIFICADO (820 °C medidos)
                  mdot_kgs=0.15,       # borde bajo de la banda de clase 1
                  T4_K=1250.0,
                  t_liner_mm=0.80, d_eff_mm=0.65, swirl_SN=0.75,
                  n_holes_dz=16.0, ALR=4.5),
        material="Inconel-625",        # ✅ VERIFICADO (el material de Bath)
        assumptions=[
            "VERIFICADO · T₃ = 1093 K (820 °C) y material IN625 son dato "
            "publicado [W]. Todo lo demás de este caso es supuesto.",
            "SUPUESTO · ṁ = 0.15 kg/s (borde bajo de la banda de clase 1). "
            "Con T₃ = 1093 K y T₄ = 1250 K el balance da FAR ≈ 0.0081 ⇒ "
            "~52 kW térmicos ⇒ ~23 % de rendimiento eléctrico para los "
            "12 kW de la clase Bladon. Es plausible pero NO está medido.",
            "SUPUESTO · T₄ = 1250 K (banda de clase 1: 1150–1300 K). El "
            "paper no da TIT en la cita disponible.",
            "SUPUESTO · π = 3.75 (medio de la banda de clase 1).",
            "SUPUESTO · frac_pz = 0.15, que es el **mínimo del contrato §1**. "
            "No es una elección: es el borde. Con φ_global = 0.118 (la que "
            "impone el balance a T₃ = 1093 K) ni siquiera el borde alcanza "
            "la banda de φ_PZ de la clase — ver el hallazgo de "
            "docs/VALIDATION.md sobre φ_PZ en recuperación extrema.",
            "CIERRE DE DISEÑO DECLARADO · frac_sz = frac_dz = 0.10 ⇒ "
            "frac_cool = 0.65. θ[2..4] son VARIABLES DE DISEÑO, no "
            "constantes del modelo: fijarlas es diseñar, no calibrar. El "
            "reparto «ingenuo» (mandar el aire sobrante a dilución: "
            "frac_dz = 0.57) se probó primero y **colapsa**: Y/D_L = 1.45 "
            "frente al objetivo 0.5 (sobre-penetración masiva), η_mix cae a "
            "su suelo 0.049 y η_comb a 0.05, y ni llevando n_holes_dz al "
            "tope del contrato (18) baja Y/D_L de 1.01. Es un hallazgo del "
            "caso, no un ajuste: en recuperación extrema el aire sobrante "
            "NO puede ir por los chorros de dilución. Se reporta en "
            "docs/VALIDATION.md.",
            "SUPUESTO · n_holes_dz = 16 (banda de clase 1: 8–12 por fila; "
            "aquí hace falta más para repartir el chorro). Se declara que "
            "queda fuera de la banda de la tabla.",
            "SUPUESTO · arquitectura `can` como aproximación del tubular "
            "cónico del paper. Phy-CB no tiene gate para «tubular cónico "
            "con swirl radial»; `can` es la más cercana (un solo tubo, un "
            "inyector). El PLF y U_ref que hereda son los de `can`.",
            "NO MODELABLE · el −75 % NOx / −40 % CO del lattice de tres "
            "filas es un efecto de HETEROGENEIDAD de mezcla. El NOx de L0 "
            "es ZONAL (una sola φ_PZ, Ciencia §8) y no tiene submodelo de "
            "pockets ricos: **no puede reproducir ese número y no se "
            "intenta**. Lo que sí se califica es el SIGNO de ∂EI/∂φ_PZ y "
            "∂EI/∂τ (caso D2), que es el criterio de la wave.",
        ],
        notes=("El caso de estrés térmico del paquete: con el aire de "
               "refrigeración a 1093 K el liner arranca ya caliente. Bath "
               "lo FABRICÓ y lo hizo funcionar en IN625; si el térmico 1D "
               "de Phy-CB declara infactible ese material, el modelo está "
               "conservador y el primer sospechoso es el suelo de ε_g "
               "(PENDIENTES §6.1), no el material."),
    ),
    dict(
        cid="D2", level="D", kind="emission_trends",
        name="Tendencias de emisiones · signo de ∂EI/∂φ_PZ y ∂EI/∂τ",
        source=(
            "Criterio de WAVE7 §3.2 y Ciencia §13.2: de las emisiones "
            "**nunca** se promete valor absoluto, solo tendencia y orden de "
            "magnitud (×2). Signos esperados por física de Zeldovich y de "
            "oxidación de CO (Ciencia §8): ∂EI_NOx/∂φ_PZ > 0 en el lado "
            "POBRE (T_ad sube con φ hasta ≈1.05), ∂EI_NOx/∂τ > 0, "
            "∂EI_CO/∂τ < 0 y ∂EI_CO/∂φ_PZ < 0 en el lado pobre. "
            "Banda de orden de magnitud para queroseno de clase micro: "
            "0.1–5 g/kg, derivada por la casa en `physics_core.py` §3 a "
            "partir de `emissions_coeffs.csv::reference_micro` "
            "(NOx 35–80 ppm@15 %O₂ ≈ 5 g/kg en el techo)."),
        spec=dict(),                    # barrido alrededor del ancla
        sweep=dict(
            # φ_PZ se mueve por frac_pz (φ_PZ = FAR/(frac_pz·FAR_st)), de
            # modo que el barrido recorre el lado POBRE: frac_pz ALTO ⇒
            # φ_PZ BAJO. Se evita el entorno de φ ≈ 1.05 (pico de T_ad)
            # donde la derivada cambia de signo y el criterio del signo no
            # significa nada.
            frac_pz=[0.60, 0.55, 0.50, 0.45, 0.40],
            # τ se mueve por L_dz_over_DL (alarga la zona de dilución ⇒
            # más volumen ⇒ más τ) sin tocar el punto de operación.
            L_dz_over_DL=[1.20, 1.60, 2.00, 2.40, 2.80],
        ),
        #: El signo se evalúa **por correlación**, no solo sobre la
        #: cabecera: `record["EI_NOx"]` es EI_NOx_lefebvre (physics_core
        #: línea 3042) y el abanico tiene cinco miembros con
        #: sensibilidades distintas. Agregar antes de mirar escondería
        #: exactamente lo que la wave pide medir.
        #: clave del record → signo esperado de ∂EI/∂x
        trend_keys=[
            dict(key="EI_NOx", label="EI_NOx (cabecera = lefebvre)",
                 phi_sign=+1, tau_sign=+1),
            dict(key="EI_NOx_lefebvre", label="EI_NOx_lefebvre",
                 phi_sign=+1, tau_sign=+1),
            dict(key="EI_NOx_rizk_mongia", label="EI_NOx_rizk_mongia",
                 phi_sign=+1, tau_sign=+1),
            dict(key="EI_NOx_zonal", label="EI_NOx_zonal",
                 phi_sign=+1, tau_sign=+1),
            dict(key="EI_NOx_odgers", label="EI_NOx_odgers",
                 phi_sign=+1, tau_sign=+1),
            dict(key="EI_NOx_laviolette", label="EI_NOx_laviolette",
                 phi_sign=+1, tau_sign=+1),
            dict(key="EI_CO", label="EI_CO (lefebvre)",
                 phi_sign=-1, tau_sign=-1),
        ],
        assumptions=[
            "SUPUESTO · el barrido se hace alrededor de THETA_REF. Las "
            "tendencias son locales: no se afirma nada fuera de ese "
            "entorno.",
            "DECLARADO · el signo se evalúa en el lado POBRE (φ_PZ < 1). En "
            "el entorno de φ_PZ ≈ 1.05 (donde vive el ancla) la derivada "
            "del NOx frente a φ es ≈ 0 por construcción física (pico de "
            "T_ad): pedir un signo ahí sería pedir ruido.",
            "La banda de orden de magnitud 0.1–5 g/kg es una derivación de "
            "la casa (ppm@15 %O₂ → g/kg), NO una medición directa. Se usa "
            "solo para el criterio ×2, que es lo que la wave permite.",
        ],
        notes=("Cubre además el abanico completo de EI_NOx (5 "
               "correlaciones) y, si hay Cantera, el EI_NOx del CRN L1 — "
               "que es el número menos empírico del paquete."),
    ),
]

# ---------------------------------------------------------------------------
# 7. Anclas de regresión (NO son mediciones)
# ---------------------------------------------------------------------------
#: Congelan la salida actual del modelo para detectar deriva silenciosa.
#: Viven en `validation/anchors.json` (las escribe/congela
#: `test_phycb.py --freeze-anchors`, ancla REF_CB10 del contrato §11 T16).
#: Aquí solo se declara qué anclas DEBERÍAN existir y cuáles faltan y por
#: qué — para que la ausencia sea visible en el reporte, no silenciosa.
EXPECTED_ANCHORS = [
    dict(name="REF_CB10", fidelity="L0", required=True,
         why="Ancla L0 del contrato §11 (T16). Debe existir siempre."),
    dict(name="REF_CB10_L1", fidelity="L1", required=False,
         why=("Ancla L1 (CRN Cantera). No está congelada: Cantera es "
              "OPCIONAL y una CI sin Cantera no podría reproducirla. "
              "Se comprueba en su lugar la reproducibilidad del CRN en "
              "esta máquina y se reporta el número, sin congelarlo.")),
    dict(name="REF_CB10_L2A", fidelity="L2A", required=False,
         why=("Ancla L2a (IB-RANS nativo). NO PUEDE EXISTIR: `cfd/ibrans/` "
              "está vacío en esta máquina y no hay Warp/JAX instalados, "
              "así que L2a degrada con etiqueta `ibrans_L2A(unavailable→…)`. "
              "WAVE6 la creará.")),
]
