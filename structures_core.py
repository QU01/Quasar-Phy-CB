"""
QUASAR Phy-CB · structures_core.py
==================================
Capa 1s del CEM: NÚCLEO ESTRUCTURAL-TÉRMICO del liner y la carcasa (nivel L0s
— analítico, NumPy puro, <1 ms por evaluación), espejo de la escalera de
fidelidad aerodinámica (L0s analítico → L1s FEA axisimétrico → L2s FEA 3D).

Contrato (IMPLEMENTATION_CONTRACT §5.2 y §8; WAVE2 §3):

    evaluate_structural(theta, record, material="Inconel-625",
                        material_casing=None) -> dict

    theta            vector de diseño 20-D del contrato §1.
    record           record de `physics_core.evaluate` del MISMO theta. Se lee
                     SIEMPRE de forma DEFENSIVA (`record.get(clave, default)`):
                     este módulo se escribió antes de que existiese el record
                     real, y los artefactos JSON envejecen respecto al código
                     (trampa CONTEXTO §10.3).
    material         pared CALIENTE (liner). Default IN625 — «polvo preferido
                     para piezas de combustor» (EstadoDelArte §4.1).
    material_casing  pared FRÍA (carcasa a presión). Default IN718: la carcasa
                     SÍ puede ser IN718 porque vive a T3, no a T de llama
                     (WAVE2 §3.4).

Contenido (L0s), todo en Ciencia §12:

  * Biblioteca de MATERIALES con derating térmico k(T) piecewise-lineal
    (clamp en los extremos, sin extrapolación) + límites de servicio, de pared
    de llama, espesor mínimo LPBF, emisividad y conductividad k(T).
  * Térmico: margen de T de pared contra `T_max_flame_wall_K − T_WALL_MARGIN_K`.
  * Tensión térmica de gradiente σ_th = E·α·(T_w1−T_w2)/(2(1−ν)) contra la
    fluencia derateada k(T)·σ_y — consume el campo `alpha` (CTE) que Phy-AC
    dejó sin usar.
  * Proxy de LCF arranque-parada (Coffin-Manson simplificado): **MÉTRICA
    REPORTADA, NO restricción en v1** — mismo criterio que el cribado de
    flutter de Phy-AC («demasiado grueso para gatear»). Va en la clave
    `N_lcf_cycles_metric` y en el bloque `lcf` con `is_constraint=False`.
  * Presión: hoop del liner (ΔP_liner) y de la carcasa (P3 − P_atm) contra el
    derateado de CADA pared a SU temperatura.
  * Regla dura de material prohibido en pared caliente (`HOT_WALL_BANNED`).

El vector `g_struct <= 0` (longitud fija `N_STRUCT_CONSTRAINTS = 5`, orden
CONGELADO por el contrato §5.2) es RESTRICCIÓN DURA del optimizador:
`DesignSpec` (neural_optimizer) lo concatena al g aero/espec/DfAM. Es CONTINUO
y **nunca lanza**: entrada degenerada (NaN/inf/ceros/negativos, record vacío,
material desconocido) devuelve violación grande y continua `G_DEGENERATE`.

Descargo de responsabilidad (el mismo que Phy-AC): los valores de material son
tipo handbook (MMPDS/ASM/fichas de fabricante) para un proxy de diseño
preliminar. *Substitute certified specs for real design.*
"""

from __future__ import annotations

import math

import numpy as np

# ==========================================================================
# 0. Constantes de la casa
# ==========================================================================
SEED = 71                  # semilla de la familia (contrato §10.8). Este
#                            módulo es determinista y NO usa aleatoriedad;
#                            se declara para que el import no la reintroduzca.
IN2_TO_M2 = 6.4516e-4      # in² → m² (heredado de Phy-AC: unidad de an2_max)
P_ATM_PA_FALLBACK = 101_325.0   # = physics_core.P_ATM (contrato §6). Se usa
#                            solo si `physics_core` aún no existe/importa —
#                            la capa 1s no debe acoplarse a la capa 1.

_P_ATM_CACHE: float | None = None


def _import_sibling(module_name: str):
    """Importa un módulo de Phy-CB verificando que sea REALMENTE de Phy-CB.

    ⚠️ TRAMPA REAL de esta máquina (2026-07-30): Phy-AC está instalado como
    paquete editable (`__editable__.phyac-0.1.0.pth` + `_EditableFinder` en
    `sys.meta_path`), de modo que un `import physics_core` desde un directorio
    que aún no tiene el suyo resuelve **al de Phy-AC** en silencio — con un θ
    de 15 posiciones y otra física. Este helper compara el directorio del
    módulo importado con el de ESTE archivo y devuelve `None` si no coinciden.

    Devuelve el módulo o `None` (nunca lanza): patrón de import perezoso con
    degradación elegante (invariante CONTEXTO §2.3).
    """
    import importlib
    import os
    try:
        mod = importlib.import_module(module_name)
        here = os.path.dirname(os.path.abspath(__file__))
        there = os.path.dirname(os.path.abspath(getattr(mod, "__file__", "")))
        return mod if os.path.normcase(here) == os.path.normcase(there) \
            else None
    except Exception:
        return None


def _p_atm() -> float:
    """Presión ambiente [Pa]: la de `physics_core` si está disponible.

    Import perezoso y verificado (ver `_import_sibling`): la capa 1s tiene que
    poder evaluarse aunque `physics_core` todavía no exista (es el caso
    mientras WAVE1 no ha aterrizado) y NUNCA debe consumir el de Phy-AC.
    """
    global _P_ATM_CACHE
    if _P_ATM_CACHE is None:
        mod = _import_sibling("physics_core")
        value = getattr(mod, "P_ATM", None) if mod is not None else None
        try:
            _P_ATM_CACHE = float(value)
        except (TypeError, ValueError):
            _P_ATM_CACHE = P_ATM_PA_FALLBACK
        if not math.isfinite(_P_ATM_CACHE) or _P_ATM_CACHE <= 0.0:
            _P_ATM_CACHE = P_ATM_PA_FALLBACK
    return _P_ATM_CACHE


# ==========================================================================
# 1. Biblioteca de materiales
# ==========================================================================
# Propiedades a temperatura ambiente + curva de derating k(T) aplicada a
# sigma_y y sigma_uts. `an2_max_in2rpm2` en in²·rpm² (unidad de la práctica
# industrial) se conserva por compatibilidad con Phy-AC aunque un combustor no
# gira: mantiene el dict intercambiable entre hermanos.
#
# Los CINCO primeros materiales son los de `Phy-AC/structures_core.py` con sus
# valores EXACTOS (rho, E, nu, alpha, sigma_y, sigma_uts, sigma_e,
# an2_max_in2rpm2, derate, note). Phy-CB solo AÑADE campos nuevos a todos:
#
#   T_max_service_K     T de servicio continuo de la ESTRUCTURA cargada
#                       (fluencia/sobre-envejecimiento). Gobierna el decaimiento
#                       extra de k(T) más allá de la tabla (ver material_at_T).
#   T_max_flame_wall_K  T de metal admisible de una pared de llama LIGERAMENTE
#                       CARGADA (el liner no es un elemento portante). Por eso
#                       en las superaleaciones es ≥ T_max_service_K; en las
#                       aleaciones que no son de pared caliente coincide.
#   lpbf_t_min_mm       espesor mínimo LPBF con calidad aceptable [mm]
#                       (EstadoDelArte §4.2-4.3).
#   emissivity_wall     emisividad de la pared oxidada — la consume el térmico
#                       1D de WAVE1 (Ciencia §5): fuente única de verdad aquí.
#   k_thermal_W_mK      conductividad térmica: lista [(T_K, k_W_mK)]
#                       interpolable (misma convención que `derate`).
#
# ⚠️ TODOS los números de abajo son de handbook y NO están verificados contra
# certificados de colada ni contra probetas LPBF propias. Recalibrar en WAVE7
# (campaña de validación) antes de usarlos para dimensionar hardware.
MATERIALS = {
    "Al-2618-T61": dict(
        rho=2760.0, E=74.5e9, nu=0.33, alpha=22.0e-6,
        sigma_y=372e6, sigma_uts=441e6, sigma_e=150e6, an2_max_in2rpm2=2.0e10,
        derate=[(293, 1.00), (423, 0.90), (473, 0.65), (523, 0.40),
                (573, 0.25)],
        T_max_service_K=473.0, T_max_flame_wall_K=473.0,
        lpbf_t_min_mm=1.00, emissivity_wall=0.20,
        k_thermal_W_mK=[(293, 146.0), (423, 155.0), (573, 165.0)],
        note="aleación ligera; NO es material de pared caliente"),
    "Al-7075-T6": dict(
        rho=2810.0, E=71.7e9, nu=0.33, alpha=23.4e-6,
        sigma_y=503e6, sigma_uts=572e6, sigma_e=160e6, an2_max_in2rpm2=2.5e10,
        derate=[(293, 1.00), (373, 0.95), (423, 0.75), (473, 0.45),
                (523, 0.25)],
        T_max_service_K=423.0, T_max_flame_wall_K=423.0,
        lpbf_t_min_mm=1.00, emissivity_wall=0.20,
        k_thermal_W_mK=[(293, 130.0), (473, 150.0)],
        note="alta resistencia en frío; poco soldable/imprimible (fisura)"),
    "Ti-6Al-4V": dict(
        rho=4430.0, E=113.8e9, nu=0.342, alpha=8.6e-6,
        sigma_y=880e6, sigma_uts=950e6, sigma_e=510e6, an2_max_in2rpm2=4.5e10,
        derate=[(293, 1.00), (473, 0.82), (573, 0.72), (673, 0.62)],
        T_max_service_K=673.0, T_max_flame_wall_K=673.0,
        lpbf_t_min_mm=0.40, emissivity_wall=0.50,
        k_thermal_W_mK=[(293, 6.7), (573, 10.6), (773, 12.6), (1073, 17.5)],
        note="el mejor comportado en LPBF; riesgo de fuego de Ti en llama"),
    "Inconel-718": dict(
        rho=8190.0, E=200.0e9, nu=0.29, alpha=13.0e-6,
        sigma_y=1034e6, sigma_uts=1240e6, sigma_e=550e6, an2_max_in2rpm2=4.0e10,
        derate=[(293, 1.00), (673, 0.94), (923, 0.88), (1023, 0.70)],
        T_max_service_K=923.0, T_max_flame_wall_K=923.0,
        lpbf_t_min_mm=0.40, emissivity_wall=0.80,
        k_thermal_W_mK=[(293, 11.4), (473, 14.7), (673, 18.3), (873, 21.8),
                        (1073, 25.2)],
        note="PROHIBIDO en pared caliente (γ″ sobre-envejece a >650 °C); "
             "el más caracterizado en LPBF — ideal para la CARCASA fría"),
    "17-4PH-H900": dict(
        rho=7750.0, E=196.0e9, nu=0.27, alpha=10.8e-6,
        sigma_y=1170e6, sigma_uts=1310e6, sigma_e=620e6, an2_max_in2rpm2=3.5e10,
        derate=[(293, 1.00), (573, 0.85), (673, 0.75)],
        T_max_service_K=616.0, T_max_flame_wall_K=616.0,
        lpbf_t_min_mm=0.50, emissivity_wall=0.75,
        k_thermal_W_mK=[(293, 18.0), (473, 21.0), (673, 23.0)],
        note="acero PH; carcasa fría barata, nunca pared caliente"),
    # --- Superaleaciones de liner (nuevas en Phy-CB) ----------------------
    "Inconel-625": dict(
        rho=8440.0, E=207.5e9, nu=0.278, alpha=12.8e-6,
        sigma_y=414e6, sigma_uts=827e6, sigma_e=340e6, an2_max_in2rpm2=3.8e10,
        # k(T) mezcla YS a T (Special Metals, recocido) con el límite por
        # FLUENCIA a 100 h por encima de ~1000 K: por eso cae más rápido que
        # la YS pura. ⚠️ recalibrar con curvas de Larson-Miller reales.
        derate=[(293, 1.00), (811, 0.85), (1033, 0.62), (1144, 0.38),
                (1253, 0.20)],
        T_max_service_K=1253.0, T_max_flame_wall_K=1253.0,
        lpbf_t_min_mm=0.50, emissivity_wall=0.85,
        k_thermal_W_mK=[(293, 9.8), (473, 12.5), (673, 15.7), (873, 18.7),
                        (1073, 22.8), (1173, 25.2)],
        note="DEFAULT de liner: «polvo preferido para piezas de combustor»; "
             "AMS7000; usado por Bath/HiETA (EstadoDelArte §4.1)"),
    "Hastelloy-X": dict(
        rho=8220.0, E=197.0e9, nu=0.32, alpha=13.9e-6,
        sigma_y=372e6, sigma_uts=772e6, sigma_e=300e6, an2_max_in2rpm2=3.5e10,
        derate=[(293, 1.00), (811, 0.72), (1033, 0.58), (1200, 0.35),
                (1423, 0.10)],
        T_max_service_K=1173.0, T_max_flame_wall_K=1423.0,
        # ⚠️ 0.8 mm y NO 0.5: al bajar el espesor aumentan porosidad y
        # fisuras en HX-LPBF (baño inestable + tensión residual; las fisuras
        # nuclean en la transición columnar→equiaxial). EstadoDelArte §4.2 —
        # ~0.5 mm es la frontera problemática documentada.
        lpbf_t_min_mm=0.80, emissivity_wall=0.85,
        k_thermal_W_mK=[(293, 9.1), (473, 13.4), (673, 17.0), (873, 21.0),
                        (1073, 24.4), (1273, 27.2)],
        note="estándar industrial de liners (~900–1150 °C), usado por Sierra; "
             "⚠️ fisuración en pared fina LPBF (EstadoDelArte §4.2)"),
    "Haynes-230": dict(
        rho=8970.0, E=211.0e9, nu=0.31, alpha=12.9e-6,
        sigma_y=390e6, sigma_uts=860e6, sigma_e=340e6, an2_max_in2rpm2=3.6e10,
        derate=[(293, 1.00), (811, 0.73), (1033, 0.62), (1144, 0.42),
                (1373, 0.12)],
        T_max_service_K=1223.0, T_max_flame_wall_K=1373.0,
        lpbf_t_min_mm=0.60, emissivity_wall=0.87,
        k_thermal_W_mK=[(293, 8.9), (473, 13.3), (673, 17.1), (873, 20.9),
                        (1073, 24.4), (1273, 27.9)],
        note="alternativa a HX con mejor estabilidad térmica; parámetros "
             "LPBF menos maduros (EstadoDelArte §4.1)"),
    "Haynes-282": dict(
        rho=8270.0, E=217.0e9, nu=0.31, alpha=12.9e-6,
        sigma_y=717e6, sigma_uts=1160e6, sigma_e=480e6, an2_max_in2rpm2=4.0e10,
        derate=[(293, 1.00), (811, 0.88), (1033, 0.80), (1144, 0.55),
                (1273, 0.25)],
        T_max_service_K=1198.0, T_max_flame_wall_K=1273.0,
        # γ′ ⇒ más propenso a fisurar en LPBF que 625/230, mucho menos que
        # CM247LC. 0.7 mm es criterio propio conservador. ⚠️ sin verificar.
        lpbf_t_min_mm=0.70, emissivity_wall=0.85,
        k_thermal_W_mK=[(293, 8.7), (473, 12.6), (673, 16.2), (873, 19.9),
                        (1073, 23.6)],
        note="γ′ envejecible: la más resistente en caliente de la lista; "
             "buena para transiciones/soportes calientes cargados"),
}
# sigma_e: límite de resistencia a fatiga (Goodman). Valores handbook típicos
# (MMPDS/ASM): Ti/Ni/acero a 1e7-1e8 ciclos (≈0.40-0.55·sigma_uts); las
# aleaciones de Al no tienen límite real — valor a 5e8 ciclos. Sin derating
# térmico en este nivel L0s (igual que Phy-AC).

# --- Regla dura de pared caliente (WAVE0 §5, test T5 y T23) ---------------
# Material → T de metal [K] por encima de la cual queda PROHIBIDO en cualquier
# superficie expuesta a llama. IN718 sobre-envejece el γ″ a >650 °C ≈ 923 K
# (EstadoDelArte §4.1, Ciencia §12). La regla es dura: por encima del umbral
# g_struct[4] > 0 SIEMPRE, independientemente del resto de márgenes.
HOT_WALL_BANNED = {"Inconel-718": 923.0}

# ==========================================================================
# 1.1 Tunables del modelo (SCREAMING_CASE con procedencia — contrato §10.7)
# ==========================================================================
N_STRUCT_CONSTRAINTS = 5   # longitud CONGELADA de g_struct (contrato §5.2)

G_DEGENERATE = 10.0        # violación grande y continua ante entrada
#                            degenerada (contrato §5.1 / CONTEXTO §2.2)

T_WALL_MARGIN_K = 50.0     # margen de MODELO sobre T_max_flame_wall_K
#                            (WAVE2 §3.1). Procedencia: no es un margen de
#                            norma sino la incertidumbre declarada del
#                            térmico 1D (Ciencia §5: coeficientes 0.020 y
#                            0.29 ⚠️ sin verificar). ⚠️ RECALIBRAR en WAVE7
#                            con termopares del banco: si el térmico 1D
#                            resulta fiel a ±20 K, este margen baja.

HOOP_SAFETY_FACTOR = 1.50  # factor de seguridad sobre la tensión de aro
#                            frente a la fluencia derateada. Procedencia:
#                            práctica de recipientes a presión (ASME VIII
#                            Div.1 usa ~3.5 sobre UTS ≈ 1.5-1.6 sobre YS en
#                            aceros dúctiles). ⚠️ criterio propio para una
#                            pieza LPBF sin código aplicable.

SIGMA_TH_UTIL = 1.00       # utilización admisible de k(T)·σ_y por la tensión
#                            térmica de gradiente. σ_th es una tensión
#                            SECUNDARIA autolimitada: ASME III NB-3222.2
#                            admitiría hasta 3·Sm por shakedown, así que
#                            1.00 es CONSERVADOR y deliberado en v1.

LINER_T_METAL_FRAC = 1.00  # T_metal del liner = T_w2 + frac·(T_w1 − T_w2).
#                            1.00 ⇒ cara caliente (conservador). Phy-AC usa
#                            el mismo patrón con T_METAL_FRAC.

CASING_DT_K = 25.0         # sobrecalentamiento de la carcasa sobre T3 por
#                            radiación del liner. Criterio propio ⚠️: el
#                            térmico 1D de WAVE1 (Ciencia §5, término R2) lo
#                            sustituirá cuando exporte T_casing_K al record.

CASING_T_OVER_T_LINER = 2.0   # espesor de carcasa / espesor de liner cuando
#                            el record no trae `t_casing_mm`. La carcasa es
#                            el recipiente a presión y el liner no: 2× es el
#                            orden de la práctica. ⚠️ provisional hasta que
#                            WAVE3 dimensione la carcasa explícitamente.

# --- Derating fuera de tabla ---------------------------------------------
# `np.interp` ya hace CLAMP plano en los extremos (nunca extrapola). Phy-CB
# añade encima un decaimiento exponencial continuo por ENCIMA de
# T_max_service_K, porque el clamp plano de Phy-AC deja k=0.25 para el
# Al-2618 a 1200 K (donde está líquido) y eso es una mentira peligrosa cuando
# el optimizador explora materiales fuera de rango.
K_BEYOND_SERVICE_DECAY_K = 150.0   # constante del decaimiento exp. [K].
#                            Criterio propio ⚠️: elegida para que 150 K por
#                            encima del servicio quede ~37 % de la k tabulada.
K_DERATE_MIN = 0.01        # suelo de k(T): evita división por cero y mantiene
#                            g continuo y finito.

# --- Proxy LCF arranque-parada (MÉTRICA, no restricción) ------------------
# Coffin-Manson simplificado, rama plástica:  Δε_p/2 = ε_f′·(2N_f)^c
LCF_EPS_F = 0.35           # ε_f′, coeficiente de ductilidad a fatiga.
#                            Handbook para superaleaciones Ni recocidas
#                            (≈ ductilidad real a rotura). ⚠️ sin verificar.
LCF_EXP_C = -0.55          # exponente de Coffin (banda clásica −0.5…−0.7).
LCF_CONSTRAINT_FACTOR = 0.50   # fracción de la dilatación térmica que queda
#                            REALMENTE restringida en el liner (no está
#                            empotrado: se expande axialmente y solo los
#                            anclajes/domo restringen). ⚠️ criterio propio;
#                            es el parámetro MÁS sospechoso de este proxy —
#                            un FEA de WAVE7 lo mueve fácilmente ×2.
LCF_T_COLD_K = 293.15      # temperatura de arranque en frío del ciclo.
LCF_N_TARGET_CYCLES = 1000.0   # objetivo de vida a LCF que se REPORTA como
#                            referencia (no entra a g en v1).
LCF_N_MAX_CYCLES = 1.0e9   # saturación de N_f. `inf` NO es JSON válido y el
#                            record acaba serializado (contrato §9).

# --- Defaults de lectura defensiva del record ----------------------------
# El record real de physics_core aún no existe cuando se escribe este módulo
# (contrato §4). Cada default es un valor de la clase microturbina
# (EstadoDelArte §7) y se DECLARA en `record_defaults_used` de la salida.
DEFAULT_T_W1_K = 1000.0        # T pared liner objetivo, clase 0.1-5 kW
DEFAULT_DT_WALL_K = 15.0       # ΔT a través de 0.8 mm de IN625 a ~200 kW/m²
DEFAULT_T_WALL_FRAC_T04 = 0.85  # T_w1 ≈ 0.85·T04 si solo hay T04
DEFAULT_T_LINER_MM = 0.80
DEFAULT_D_L_MM = 80.0
DEFAULT_D_CASING_OVER_D_L = 1.30
DEFAULT_P3_PA = 250_000.0
DEFAULT_T3_K = 430.0
DEFAULT_DP_OVER_P = 0.035      # Δp/p 2-4 % en la clase microturbina

# --- Índices del vector θ (contrato §1, CONGELADOS) -----------------------
# Se definen aquí para no acoplar la capa 1s a physics_core (que puede no
# existir todavía). `verify_theta_layout()` los contrasta si está disponible.
IDX_T_LINER_MM = 14
IDX_P3_PA = 16
IDX_T3_K = 17
IDX_T4_K = 19
_THETA_LAYOUT = {IDX_T_LINER_MM: "t_liner_mm", IDX_P3_PA: "P3_Pa",
                 IDX_T3_K: "T3_K", IDX_T4_K: "T4_K"}


# ==========================================================================
# 2. Utilidades de material
# ==========================================================================
def list_materials() -> list[str]:
    """Nombres de la biblioteca, en orden de declaración."""
    return list(MATERIALS)


def hot_wall_materials() -> list[str]:
    """Materiales aptos como pared de llama a ≥1200 K de metal.

    Filtro puramente informativo (reporte y wizard de la CLI): un material
    entra si su `T_max_flame_wall_K` supera 1200 K y no está en
    `HOT_WALL_BANNED` por debajo de ese umbral.
    """
    out = []
    for name, m in MATERIALS.items():
        t_ban = HOT_WALL_BANNED.get(name, math.inf)
        if float(m["T_max_flame_wall_K"]) >= 1200.0 and t_ban >= 1200.0:
            out.append(name)
    return out


def _interp_table(pts, T_K: float) -> float:
    """Interpolación lineal con CLAMP plano en ambos extremos.

    `np.interp` no extrapola: por debajo del primer punto devuelve el primer
    valor y por encima del último, el último. Es exactamente lo que pide el
    contrato («sin extrapolación loca»). Los puntos se ordenan de forma
    defensiva por si una tabla llega desordenada.
    """
    arr = np.asarray(pts, dtype=float)
    if arr.ndim != 2 or arr.shape[0] < 1:
        return float("nan")
    order = np.argsort(arr[:, 0])
    Ts, ks = arr[order, 0], arr[order, 1]
    if not math.isfinite(T_K):
        return float(ks[-1])      # T degenerada ⇒ el extremo más penalizador
    return float(np.interp(T_K, Ts, ks))


def material_at_T(name: str, T_K: float) -> dict:
    """Propiedades del material con sigma_y/sigma_uts derateadas a `T_K`.

    Derating: k(T) piecewise-lineal de la tabla `derate` con CLAMP en los
    extremos (`np.interp` no extrapola). Phy-CB añade sobre Phy-AC un
    decaimiento exponencial continuo por encima de `T_max_service_K` —
    `k *= exp(−(T − T_service)/K_BEYOND_SERVICE_DECAY_K)` — porque el clamp
    plano sobreestima groseramente un material explorado fuera de su rango
    (Al a 1200 K). El decaimiento vale 1.0 justo en T_service, así que k(T)
    es continua. Suelo `K_DERATE_MIN` para que nada divida por cero.

    Añade también `k_thermal_W_mK_T` (conductividad interpolada, la consume
    el térmico 1D de WAVE1).

    Lanza ValueError si el material no existe — mismo contrato que Phy-AC.
    `evaluate_structural` NUNCA propaga esa excepción: resuelve el nombre
    defensivamente antes de llamar aquí.
    """
    if name not in MATERIALS:
        raise ValueError(f"material desconocido '{name}' — disponibles: "
                         f"{', '.join(MATERIALS)}")
    m = dict(MATERIALS[name])
    k = _interp_table(m["derate"], T_K)
    T_serv = float(m["T_max_service_K"])
    if math.isfinite(T_K) and T_K > T_serv:
        k *= math.exp(-(T_K - T_serv) / K_BEYOND_SERVICE_DECAY_K)
    k = float(min(max(k, K_DERATE_MIN), 1.0))
    m["name"] = name
    m["T_eval_K"] = float(T_K) if math.isfinite(T_K) else float("nan")
    m["derate_factor"] = k
    m["sigma_y_T"] = m["sigma_y"] * k
    m["sigma_uts_T"] = m["sigma_uts"] * k
    m["k_thermal_W_mK_T"] = _interp_table(m["k_thermal_W_mK"], T_K)
    return m


def is_hot_wall_allowed(material: str, T_wall_K: float) -> tuple[bool, float]:
    """¿Puede `material` ver llama a `T_wall_K`? → (permitido, violación).

    La violación es CONTINUA y sirve directa como `g_struct[4]`:
      * material prohibido: `T_wall/T_ban − 1` (crece suave al calentarse).
      * material no listado: `−1.0` (holgura constante; el material no es una
        variable continua del optimizador, así que no hay gradiente que dar).
    """
    t_ban = HOT_WALL_BANNED.get(material)
    if t_ban is None:
        return True, -1.0
    t_ban = max(float(t_ban), 1e-6)
    T = float(T_wall_K) if math.isfinite(T_wall_K) else t_ban * 10.0
    viol = T / t_ban - 1.0
    return bool(viol <= 0.0), float(viol)


def verify_theta_layout() -> dict:
    """Contrasta los índices de θ locales contra `physics_core.VAR_NAMES`.

    Import perezoso y VERIFICADO (`_import_sibling`): si el `physics_core` de
    Phy-CB aún no existe (WAVE1 sin aterrizar) devuelve `available=False` sin
    lanzar, y jamás se compara contra el θ de 15 posiciones de Phy-AC. Lo usan
    el smoke de `__main__` y el test T5 para detectar deriva del ABI
    (contrato §1).
    """
    mod = _import_sibling("physics_core")
    names = getattr(mod, "VAR_NAMES", None) if mod is not None else None
    if not isinstance(names, (list, tuple)):
        return dict(available=False, ok=True, mismatches=[])
    bad = []
    for i, expected in _THETA_LAYOUT.items():
        got = names[i] if i < len(names) else "<fuera de rango>"
        if got != expected:
            bad.append(f"idx {i}: contrato='{expected}' physics_core='{got}'")
    return dict(available=True, ok=not bad, mismatches=bad)


# ==========================================================================
# 3. Lectura defensiva del record y de θ
# ==========================================================================
_ABSENT = "absent"    # la clave no viene (record viejo, θ corto, WAVE1 sin
#                       aterrizar) ⇒ lectura DEFENSIVA: se usa el default y
#                       se DECLARA en `record_defaults_used`.
_BAD = "bad"          # la clave viene pero es NaN/inf/≤0 ⇒ entrada
#                       DEGENERADA: contrato §5.1 ⇒ todos los g a G_DEGENERATE.
_OK = "ok"


def _num(value, positive: bool = True) -> tuple[float, str]:
    """float(value) clasificado → (valor, estado ∈ {_OK, _ABSENT, _BAD}).

    La distinción es la clave del contrato y NO debe colapsarse:

      * `None` / clave ausente         → `_ABSENT`: lectura defensiva legítima
        (el record real de physics_core aún no existe y los artefactos JSON
        envejecen respecto al código — trampa CONTEXTO §10.3).
      * presente pero NaN/±inf/≤0      → `_BAD`: entrada degenerada. El
        contrato §5.1 exige violación grande y continua, no un default
        silencioso que disfrace basura de diseño factible.
    """
    if value is None:
        return math.nan, _ABSENT
    try:
        v = float(value)
    except (TypeError, ValueError):
        return math.nan, _BAD
    if not math.isfinite(v) or (positive and v <= 0.0):
        return math.nan, _BAD
    return v, _OK


def _theta_num(theta, idx: int, positive: bool = True) -> tuple[float, str]:
    """θ[idx] clasificado. θ corto/None ⇒ `_ABSENT` (lo padearía `pad_theta`);
    θ[idx] presente pero NaN/inf/≤0 ⇒ `_BAD` (degenerado)."""
    try:
        raw = theta[idx]
    except Exception:
        return math.nan, _ABSENT
    return _num(raw, positive)


class _Reader:
    """Lector defensivo del par (θ, record) que separa ausente de degenerado.

    Acumula en `defaults` las claves que hubo que inventar (van al reporte) y
    en `bad` las que llegaron degeneradas (disparan el vector de violación).
    """

    def __init__(self, theta, record):
        self.theta = theta
        self.rec = record if isinstance(record, dict) else {}
        self.defaults: list[str] = []
        self.bad: list[str] = []

    def rec_key(self, key: str, default: float, positive: bool = True):
        """Valor del record; default declarado si falta; None si degenerado."""
        v, st = _num(self.rec.get(key), positive)
        if st is _BAD:
            self.bad.append(key)
            return None
        if st is _ABSENT:
            self.defaults.append(key)
            return float(default)
        return v

    def rec_opt(self, key: str, positive: bool = True):
        """Clave OPCIONAL: devuelve el valor o None. Un valor degenerado
        aquí NO envenena el resultado si existe otra fuente (se registra
        solo si acaba usándose)."""
        v, st = _num(self.rec.get(key), positive)
        return v if st is _OK else None

    def theta_first(self, idx: int, key: str, default: float,
                    positive: bool = True):
        """θ manda (es la variable de diseño); el record es el respaldo."""
        v, st = _theta_num(self.theta, idx, positive)
        if st is _BAD:
            self.bad.append(f"theta[{idx}] ({key})")
            return None
        if st is _OK:
            return v
        return self.rec_key(key, default, positive)


def _read_inputs(theta, record) -> dict:
    """Extrae del par (θ, record) todo lo que consume la capa 1s.

    Prioridad: θ manda en las variables de diseño (t_liner, punto de
    operación); el record manda en lo derivado (T de pared, diámetros, Δp).
    Cada default aplicado se apunta en `defaults` para que el reporte declare
    qué se inventó — honestidad calibrada, no números silenciosos. Cada valor
    degenerado se apunta en `bad` y el llamador devuelve g = G_DEGENERATE.
    """
    r = _Reader(theta, record)

    # --- punto de operación: θ primero, record como respaldo --------------
    P3 = r.theta_first(IDX_P3_PA, "P3_Pa", DEFAULT_P3_PA)
    T3 = r.theta_first(IDX_T3_K, "T3_K", DEFAULT_T3_K)

    # --- espesores --------------------------------------------------------
    t_liner = r.theta_first(IDX_T_LINER_MM, "t_liner_mm", DEFAULT_T_LINER_MM)
    t_casing = r.rec_key("t_casing_mm",
                         CASING_T_OVER_T_LINER * (t_liner or
                                                  DEFAULT_T_LINER_MM))

    # --- temperaturas de pared (térmico 1D de WAVE1, Ciencia §5) ----------
    # Cascada de fuentes: T_w1_K → T_wall_max_K → 0.85·T04 → default de clase.
    # Solo se consulta la siguiente si la anterior está AUSENTE; una presente
    # y degenerada corta la cascada (es basura, no falta de información).
    T_w1 = None
    for key in ("T_w1_K", "T_wall_max_K"):
        T_w1, st = _num(r.rec.get(key))
        if st is _BAD:
            r.bad.append(key)
            T_w1 = None
            break
        if st is _OK:
            break
        T_w1 = None
    else:
        # Ninguna de las dos claves está: se deriva de T04 (record o θ[19]).
        T04, st = _num(r.rec.get("T04_K"))
        if st is not _OK:
            T04, st = _theta_num(theta, IDX_T4_K)
        r.defaults.append("T_w1_K")
        T_w1 = DEFAULT_T_WALL_FRAC_T04 * T04 if st is _OK else DEFAULT_T_W1_K

    T_w2 = None
    if T_w1 is not None:
        T_w2 = r.rec_key("T_w2_K", max(T_w1 - DEFAULT_DT_WALL_K, 1.0))

    # --- geometría --------------------------------------------------------
    D_L = r.rec_key("D_L_mm", DEFAULT_D_L_MM)
    D_cas = r.rec_key("D_casing_mm",
                      DEFAULT_D_CASING_OVER_D_L * (D_L or DEFAULT_D_L_MM))

    # --- cargas de presión ------------------------------------------------
    # Δp del liner: dp_liner_Pa → dp_Pa → dp_over_p·P3 → default de clase.
    dp_liner = r.rec_opt("dp_liner_Pa") or r.rec_opt("dp_Pa")
    dp_over_p = r.rec_opt("dp_over_p")
    if dp_liner is None:
        r.defaults.append("dp_liner_Pa")
        dp_liner = (dp_over_p if dp_over_p is not None
                    else DEFAULT_DP_OVER_P) * (P3 or DEFAULT_P3_PA)

    return dict(P3_Pa=P3, T3_K=T3,
                t_liner_mm=t_liner, t_casing_mm=t_casing,
                T_w1_K=T_w1, T_w2_K=T_w2,
                D_L_mm=D_L, D_casing_mm=D_cas,
                dp_over_p=dp_over_p, dp_liner_Pa=dp_liner,
                defaults=r.defaults, bad=r.bad)


def _degenerate_result(material: str, material_casing: str,
                       reason: str) -> dict:
    """Salida canónica ante entrada degenerada: g_struct todo a G_DEGENERATE.

    Contrato §5.1/§5.2: el vector nunca lanza y nunca devuelve NaN — devuelve
    una violación grande y continua para que la dominancia de Deb siga
    ordenando individuos.
    """
    return dict(
        material=material, material_casing=material_casing,
        g_struct=[G_DEGENERATE] * N_STRUCT_CONSTRAINTS,
        feasible_struct=False,
        structural_warnings=[f"entrada degenerada: {reason}"],
        degenerate=True, degenerate_reason=reason,
        N_lcf_cycles_metric=0.0, lcf_is_constraint=False,
        source="liner1d_L0s(degenerate_input)",
    )


# ==========================================================================
# 4. Evaluación estructural-térmica completa (L0s)
# ==========================================================================
def _coffin_manson(alpha: float, dT_cycle: float) -> dict:
    """Proxy de vida a LCF de arranque-parada. **MÉTRICA, NO RESTRICCIÓN.**

    Coffin-Manson simplificado (solo rama plástica, sin término elástico de
    Basquin y sin efecto de la T media ni del tiempo de mantenimiento —
    justamente los tres efectos que dominan en un liner real):

        Δε_th = LCF_CONSTRAINT_FACTOR · α · ΔT_ciclo
        Δε_th/2 = ε_f′ · (2N_f)^c   ⇒   N_f = ½ · (Δε_th / (2·ε_f′))^(1/c)

    con ΔT_ciclo = T_w1 − LCF_T_COLD_K (parada en frío → punto de diseño).

    Precisión esperada: un factor 3-5 en N_f, en el lado optimista. Por eso
    NO entra en `g_struct` en v1 (WAVE2 §3.3) — mismo criterio que el cribado
    de flutter de Phy-AC. Se reporta para que el reporte pueda decir «esta
    cámara es de banco, no de 3000 arranques».
    """
    a = abs(float(alpha))
    dT = max(float(dT_cycle), 0.0)
    d_eps = LCF_CONSTRAINT_FACTOR * a * dT
    if d_eps <= 1e-12:
        # Sin excursión térmica no hay LCF. Se satura a LCF_N_MAX_CYCLES en
        # vez de a inf: el record acaba en JSON y `inf` no es JSON válido.
        return dict(delta_eps_th=0.0, N_cycles=LCF_N_MAX_CYCLES,
                    N_target=LCF_N_TARGET_CYCLES,
                    margin=LCF_N_MAX_CYCLES / LCF_N_TARGET_CYCLES,
                    is_constraint=False, model="coffin_manson_simplificado")
    ratio = d_eps / (2.0 * LCF_EPS_F)
    # (2N)^c = ratio ⇒ 2N = ratio^(1/c); c<0 ⇒ ratio pequeño ⇒ N grande
    two_n = ratio ** (1.0 / LCF_EXP_C)
    n_f = 0.5 * two_n
    if not math.isfinite(n_f):
        n_f = 0.0
    n_f = float(min(max(n_f, 0.0), LCF_N_MAX_CYCLES))   # nunca inf en el JSON
    return dict(delta_eps_th=float(d_eps), N_cycles=n_f,
                N_target=LCF_N_TARGET_CYCLES,
                margin=float(n_f / LCF_N_TARGET_CYCLES),
                is_constraint=False, model="coffin_manson_simplificado")


def evaluate_structural(theta, record: dict,
                        material: str = "Inconel-625",
                        material_casing: str | None = None) -> dict:
    """Record estructural-térmico del liner + carcasa. **NUNCA lanza.**

    Evalúa cinco familias de márgenes (Ciencia §12, WAVE2 §3) y devuelve
    `g_struct` con EXACTAMENTE `N_STRUCT_CONSTRAINTS = 5` floats continuos en
    el orden CONGELADO del contrato §5.2:

      0. `T_wall`  : T_w1 / (T_max_flame_wall_K − T_WALL_MARGIN_K) − 1
      1. `sigma_th`: σ_th / (SIGMA_TH_UTIL·k(T)·σ_y) − 1
      2. `hoop`    : max(util_liner, util_carcasa) − 1, cada pared derateada
                     a SU temperatura (el liner a T_w1, la carcasa a T3+ΔT)
      3. `t_wall`  : lpbf_t_min_mm / t_liner_mm − 1
      4. `banned`  : material prohibido en pared caliente (HOT_WALL_BANNED)

    `material` es la pared CALIENTE (liner) y `material_casing` la pared FRÍA
    (carcasa, default IN718): la regla de `HOT_WALL_BANNED` se aplica SOLO a
    la caliente — la carcasa vive a T3 y IN718 ahí es perfectamente legal
    (WAVE2 §3.4, test T23).

    La vida a LCF se devuelve como MÉTRICA en `N_lcf_cycles_metric` y en el
    bloque `lcf` (con `is_constraint=False`); NO entra en `g_struct` en v1.

    Dos modos de robustez que NO deben confundirse:

      * **Clave AUSENTE** (record vacío/viejo, θ corto o None, WAVE1 sin
        aterrizar) ⇒ lectura DEFENSIVA: se usa el default de clase
        microturbina, se DECLARA en `record_defaults_used` y en
        `structural_warnings`, y la evaluación continúa normal.
      * **Entrada DEGENERADA** (valor presente pero NaN/±inf/cero/negativo,
        material de pared caliente desconocido, o cualquier excepción no
        prevista) ⇒ `g_struct = [10.0]·5` con
        `source="liner1d_L0s(degenerate_input)"`. Nunca una excepción, nunca
        un default silencioso que disfrace basura de diseño factible
        (contrato §5.1).
    """
    mat_cold_name = material_casing if material_casing else "Inconel-718"
    try:
        # --- resolución defensiva de materiales (nunca lanza) -------------
        warnings: list[str] = []
        if material not in MATERIALS:
            return _degenerate_result(
                str(material), str(mat_cold_name),
                f"material de pared caliente desconocido '{material}'")
        if mat_cold_name not in MATERIALS:
            warnings.append(
                f"material de carcasa desconocido '{mat_cold_name}' — se usa "
                "'Inconel-718'")
            mat_cold_name = "Inconel-718"

        inp = _read_inputs(theta, record)

        # --- cribado de degeneración (contrato §5.1) ----------------------
        # Una clave AUSENTE se rellena con su default y se declara; una clave
        # PRESENTE pero NaN/inf/≤0 degenera el vector entero a G_DEGENERATE.
        must_be_pos = ("t_liner_mm", "t_casing_mm", "T_w1_K", "T_w2_K",
                       "D_L_mm", "D_casing_mm", "P3_Pa", "T3_K",
                       "dp_liner_Pa")
        bad = list(inp["bad"]) + [
            k for k in must_be_pos
            if inp[k] is None or not math.isfinite(inp[k]) or inp[k] <= 0.0]
        if bad:
            return _degenerate_result(
                material, mat_cold_name,
                "entradas no positivas/no finitas: "
                + ", ".join(sorted(set(bad))))

        T_w1, T_w2 = inp["T_w1_K"], inp["T_w2_K"]
        dT_wall = T_w1 - T_w2
        t_liner_m = inp["t_liner_mm"] * 1e-3
        t_casing_m = inp["t_casing_mm"] * 1e-3
        r_liner_m = 0.5 * inp["D_L_mm"] * 1e-3
        r_casing_m = 0.5 * inp["D_casing_mm"] * 1e-3

        # --- (a) TÉRMICO: margen contra el límite de pared de llama -------
        T_metal = T_w2 + LINER_T_METAL_FRAC * dT_wall
        hot = material_at_T(material, T_metal)
        T_flame_allow = float(hot["T_max_flame_wall_K"]) - T_WALL_MARGIN_K
        T_flame_allow = max(T_flame_allow, 1.0)
        g_temp = T_w1 / T_flame_allow - 1.0

        # --- (b) TENSIÓN TÉRMICA DE GRADIENTE -----------------------------
        # σ_th = E·α·ΔT_pared / (2(1−ν))  [Ciencia §12]. Placa plana con
        # gradiente lineal y expansión biaxial impedida en el plano. E se
        # toma EN FRÍO (Phy-AC no deratea E): conservador, porque E cae
        # ~20-30 % a 1000 K y σ_th es proporcional a E. Declarado.
        E = float(hot["E"])
        nu = float(hot["nu"])
        alpha = float(hot["alpha"])
        sigma_th = E * alpha * abs(dT_wall) / max(2.0 * (1.0 - nu), 1e-6)
        sigma_th_allow = max(SIGMA_TH_UTIL * float(hot["sigma_y_T"]), 1.0)
        g_sigma_th = sigma_th / sigma_th_allow - 1.0

        # --- (c) PROXY LCF (métrica, NO restricción) ----------------------
        lcf = _coffin_manson(alpha, T_w1 - LCF_T_COLD_K)

        # --- (d) PRESIÓN: hoop del liner (caliente) y de la carcasa (fría)
        # Liner: ΔP es solo la caída a través de la pared perforada, no P3 —
        # el liner NO es el recipiente a presión (vive dentro del plenum).
        sigma_hoop_liner = inp["dp_liner_Pa"] * r_liner_m / max(t_liner_m,
                                                               1e-9)
        util_liner = HOOP_SAFETY_FACTOR * sigma_hoop_liner \
            / max(float(hot["sigma_y_T"]), 1.0)

        # Carcasa: SÍ es el recipiente a presión (P3 contra el ambiente) y
        # vive a T3 + ΔT de radiación ⇒ puede ser IN718 (WAVE2 §3.4).
        T_casing = inp["T3_K"] + CASING_DT_K
        cold = material_at_T(mat_cold_name, T_casing)
        dp_casing = max(inp["P3_Pa"] - _p_atm(), 0.0)
        sigma_hoop_casing = dp_casing * r_casing_m / max(t_casing_m, 1e-9)
        util_casing = HOOP_SAFETY_FACTOR * sigma_hoop_casing \
            / max(float(cold["sigma_y_T"]), 1.0)
        g_hoop = max(util_liner, util_casing) - 1.0

        # --- (e) ESPESOR MÍNIMO LPBF del material de la pared caliente ----
        t_min = float(hot["lpbf_t_min_mm"])
        g_thick = t_min / max(inp["t_liner_mm"], 1e-6) - 1.0

        # --- (f) MATERIAL PROHIBIDO EN PARED CALIENTE ---------------------
        hot_ok, g_banned = is_hot_wall_allowed(material, T_w1)

        # --- avisos legibles ----------------------------------------------
        if inp["defaults"]:
            warnings.append(
                "claves ausentes en el record (se usaron defaults de clase "
                "microturbina): " + ", ".join(sorted(set(inp["defaults"]))))
        if not hot_ok:
            warnings.append(
                f"{material} PROHIBIDO en pared caliente por encima de "
                f"{HOT_WALL_BANNED[material]:.0f} K (T_w1 = {T_w1:.0f} K) — "
                "usar IN625/Hastelloy-X/Haynes-230 en el liner")
        if hot["derate_factor"] < 0.35:
            warnings.append(
                f"{material} derateado a {hot['derate_factor']:.0%} a "
                f"{T_metal:.0f} K — fuera de su banda útil; considerar "
                "Hastelloy-X o Haynes-230")
        if T_metal > float(hot["T_max_service_K"]):
            warnings.append(
                f"T de metal {T_metal:.0f} K por encima del servicio continuo "
                f"de {material} ({hot['T_max_service_K']:.0f} K) — vida "
                "limitada por fluencia, no por fluencia estática")
        if lcf["N_cycles"] < LCF_N_TARGET_CYCLES:
            warnings.append(
                f"proxy LCF: {lcf['N_cycles']:.0f} ciclos arranque-parada "
                f"(<{LCF_N_TARGET_CYCLES:.0f}) — MÉTRICA, no restricción; "
                "el factor de restricción es el tunable más sospechoso")
        if inp["t_liner_mm"] < t_min:
            warnings.append(
                f"espesor de liner {inp['t_liner_mm']:.2f} mm < mínimo LPBF "
                f"de {material} ({t_min:.2f} mm) — porosidad y fisuras "
                "(EstadoDelArte §4.2)")

        # --- vector g_struct <= 0 (orden CONGELADO, contrato §5.2) --------
        g_struct = [g_temp, g_sigma_th, g_hoop, g_thick, g_banned]
        g_struct = [float(g) if math.isfinite(g) else G_DEGENERATE
                    for g in g_struct]

        return dict(
            # --- identidad -----------------------------------------------
            material=material, material_casing=mat_cold_name,
            # --- térmico --------------------------------------------------
            T_w1_K=float(T_w1), T_w2_K=float(T_w2),
            dT_wall_K=float(dT_wall), T_metal_K=float(T_metal),
            T_casing_K=float(T_casing),
            T_max_flame_wall_K=float(hot["T_max_flame_wall_K"]),
            T_max_service_K=float(hot["T_max_service_K"]),
            T_wall_allow_K=float(T_flame_allow),
            T_wall_margin_K=float(T_flame_allow - T_w1),
            derate_factor=float(hot["derate_factor"]),
            derate_factor_casing=float(cold["derate_factor"]),
            k_thermal_W_mK=float(hot["k_thermal_W_mK_T"]),
            emissivity_wall=float(hot["emissivity_wall"]),
            # --- tensiones [MPa en la interfaz, SI internamente] ----------
            sigma_th_MPa=float(sigma_th / 1e6),
            sigma_th_allow_MPa=float(sigma_th_allow / 1e6),
            sigma_th_ratio=float(sigma_th / sigma_th_allow),
            sigma_y_T_MPa=float(hot["sigma_y_T"] / 1e6),
            sigma_uts_T_MPa=float(hot["sigma_uts_T"] / 1e6),
            sigma_y_T_casing_MPa=float(cold["sigma_y_T"] / 1e6),
            sigma_hoop_liner_MPa=float(sigma_hoop_liner / 1e6),
            sigma_hoop_casing_MPa=float(sigma_hoop_casing / 1e6),
            hoop_util_liner=float(util_liner),
            hoop_util_casing=float(util_casing),
            dp_liner_Pa=float(inp["dp_liner_Pa"]),
            dp_casing_Pa=float(dp_casing),
            # --- geometría e imprimibilidad -------------------------------
            t_liner_mm=float(inp["t_liner_mm"]),
            t_casing_mm=float(inp["t_casing_mm"]),
            lpbf_t_min_mm=float(t_min),
            t_liner_over_t_min=float(inp["t_liner_mm"] / max(t_min, 1e-6)),
            D_L_mm=float(inp["D_L_mm"]), D_casing_mm=float(inp["D_casing_mm"]),
            # --- LCF: MÉTRICA REPORTADA, NO RESTRICCIÓN (v1) --------------
            N_lcf_cycles_metric=float(lcf["N_cycles"]),
            lcf=lcf, lcf_is_constraint=False,
            # --- regla de pared caliente ----------------------------------
            hot_wall_ok=bool(hot_ok),
            # `None` (→ JSON null) y NO inf cuando el material no está vetado:
            # `inf` no es JSON válido y el record se serializa (contrato §9).
            # El consumidor C# lee esta clave defensivamente (§9, Validate()).
            hot_wall_banned_T_K=(float(HOT_WALL_BANNED[material])
                                 if material in HOT_WALL_BANNED else None),
            # --- restricciones --------------------------------------------
            g_struct=g_struct,
            g_struct_names=["T_wall", "sigma_th", "hoop", "t_wall", "banned"],
            feasible_struct=bool(all(g <= 0.0 for g in g_struct)),
            structural_warnings=warnings,
            record_defaults_used=sorted(set(inp["defaults"])),
            degenerate=False,
            params=dict(T_wall_margin_K=T_WALL_MARGIN_K,
                        hoop_safety_factor=HOOP_SAFETY_FACTOR,
                        sigma_th_util=SIGMA_TH_UTIL,
                        liner_T_metal_frac=LINER_T_METAL_FRAC,
                        casing_dT_K=CASING_DT_K,
                        casing_t_over_t_liner=CASING_T_OVER_T_LINER,
                        lcf_eps_f=LCF_EPS_F, lcf_exp_c=LCF_EXP_C,
                        lcf_constraint_factor=LCF_CONSTRAINT_FACTOR),
            source="liner1d_L0s",
        )
    except Exception as exc:                      # red de seguridad final
        # El contrato exige que g_struct JAMÁS lance. Cualquier fallo no
        # previsto degrada a violación grande y continua, con el motivo.
        return _degenerate_result(str(material), str(mat_cold_name),
                                  f"{type(exc).__name__}: {exc}")


# ==========================================================================
# 5. Smoke test (__main__)
# ==========================================================================
def _synthetic_record() -> dict:
    """Record sintético plausible de un can de ~10 kW con jet_a.

    Valores de la clase 0.1-5 kW de la tabla de referencia (EstadoDelArte §7).
    Existe porque `physics_core.evaluate` aún no está escrito cuando se
    entrega esta capa: en cuanto WAVE1 aterrice, el smoke debe consumir el
    record real y este helper se queda solo como fixture de test.
    """
    return dict(
        schema="phycb-record-1", source="synthetic_smoke",
        # termoquímica y punto de operación
        T04_K=1150.0, P04_Pa=241_250.0, T3_K=430.0, P3_Pa=250_000.0,
        f_fuel_air=0.0165, eta_comb=0.985, dp_over_p=0.035,
        # dimensionado
        D_L_mm=78.0, D_casing_mm=104.0, D_ref_mm=92.0, L_tot_mm=118.0,
        # térmico 1D (Ciencia §5)
        T_w1_K=1020.0, T_w2_K=1002.0, T_wall_max_K=1020.0,
        eta_film=0.55, blowing_ratio_M=0.5,
        # pérdidas
        dp_liner_Pa=7_500.0, dp_Pa=8_750.0,
        # eco de θ relevante para la capa 1s
        t_liner_mm=0.80, d_eff_mm=0.60,
        # patrón y emisiones (no los consume esta capa, van por completitud)
        PF=0.42, EI_NOx=6.5, EI_CO=12.0, phi_pz=0.95, tau_ms=5.4, SMD_um=55.0,
    )


def _synthetic_theta() -> np.ndarray:
    """THETA_REF del contrato §1 (caso ancla KJ66 / REF_CB10)."""
    return np.array([
        2.0,        # arch = reverse_annular
        0.70,       # AL_over_Aref
        0.30,       # frac_pz
        0.12,       # frac_sz
        0.42,       # frac_dz
        0.70,       # swirl_SN
        8.0,        # n_holes_pz
        10.0,       # n_holes_dz
        0.60,       # d_eff_mm
        6.0,        # pitch_over_d_eff
        2.0,        # injector = vaporizer
        4.0,        # ALR
        0.75,       # L_pz_over_DL
        1.80,       # L_dz_over_DL
        0.80,       # t_liner_mm
        0.0,        # fuel = jet_a
        250_000.0,  # P3_Pa
        430.0,      # T3_K
        0.13,       # mdot_kgs
        1150.0,     # T4_K
    ])


if __name__ == "__main__":
    import sys
    import time

    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    print("=" * 78)
    print("Phy-CB · structures_core — smoke L0s (capa 1s)")
    print("=" * 78)

    _lay = verify_theta_layout()
    print(f"layout de θ vs physics_core: "
          f"{'no disponible aún (WAVE1)' if not _lay['available'] else ('OK' if _lay['ok'] else _lay['mismatches'])}")

    # --- 1. derating k(T) y conductividad a varias temperaturas -----------
    T_LIST = (300.0, 600.0, 900.0, 1100.0, 1300.0, 1500.0)
    print("\n[1/3] Derating k(T) · σ_y(T) [MPa] · k_térmica [W/m·K]")
    print(f"{'material':14s} {'T_serv':>7s} {'T_llama':>8s} {'t_min':>6s}"
          + "".join(f" {int(T):>16d}K" for T in T_LIST))
    for _name in list_materials():
        _m = MATERIALS[_name]
        _row = (f"{_name:14s} {_m['T_max_service_K']:7.0f} "
                f"{_m['T_max_flame_wall_K']:8.0f} {_m['lpbf_t_min_mm']:6.2f}")
        for _T in T_LIST:
            _p = material_at_T(_name, _T)
            _row += (f" {_p['derate_factor']:5.2f}/"
                     f"{_p['sigma_y_T'] / 1e6:5.0f}/"
                     f"{_p['k_thermal_W_mK_T']:5.1f}")
        print(_row)
    print(f"      (aptos para pared caliente ≥1200 K: "
          f"{', '.join(hot_wall_materials())})")

    # --- 2. evaluación de un record sintético con cada material -----------
    _theta = _synthetic_theta()
    _rec = _synthetic_record()
    print("\n[2/3] evaluate_structural sobre record sintético "
          f"(T_w1={_rec['T_w1_K']:.0f} K, t_liner={_rec['t_liner_mm']:.2f} mm)")
    _t0 = time.perf_counter()
    for _name in list_materials():
        _s = evaluate_structural(_theta, _rec, material=_name)
        _g = " ".join(f"{v:+6.2f}" for v in _s["g_struct"])
        print(f"{_name:14s} σth={_s['sigma_th_MPa']:6.1f} MPa "
              f"(adm {_s['sigma_th_allow_MPa']:6.1f})  "
              f"hoop_c={_s['hoop_util_casing']:.3f}  "
              f"N_LCF={_s['N_lcf_cycles_metric']:9.0f}  "
              f"g=[{_g}]  "
              f"{'OK   ' if _s['feasible_struct'] else 'VIOLA'}")
        for _w in _s["structural_warnings"]:
            print(f"{'':14s}   ⚠ {_w}")
    _dt = (time.perf_counter() - _t0) / max(len(MATERIALS), 1)
    print(f"      [{1000 * _dt:.3f} ms/eval]")

    # --- 3. reglas duras y degradación ------------------------------------
    print("\n[3/3] Reglas duras (T5/T22/T23) y degradación")
    _hot718 = evaluate_structural(_theta, _rec, material="Inconel-718")
    print(f"  T5/T23 IN718 en liner a {_rec['T_w1_K']:.0f} K → "
          f"g[4]={_hot718['g_struct'][4]:+.3f} "
          f"({'VIOLA (correcto)' if _hot718['g_struct'][4] > 0 else 'ERROR'})")
    _cold718 = evaluate_structural(_theta, _rec, material="Inconel-625",
                                   material_casing="Inconel-718")
    print(f"  T23 IN718 en CARCASA fría a {_cold718['T_casing_K']:.0f} K → "
          f"hoop_util={_cold718['hoop_util_casing']:.4f} "
          f"({'OK (correcto)' if _cold718['feasible_struct'] else 'VIOLA'})")

    _rec_iso = dict(_rec, T_w1_K=1010.0, T_w2_K=1010.0)
    _iso = evaluate_structural(_theta, _rec_iso, material="Inconel-625")
    print(f"  T22 T_w1 == T_w2 → σ_th = {_iso['sigma_th_MPa']:.6f} MPa "
          f"({'OK' if _iso['sigma_th_MPa'] == 0.0 else 'ERROR'})")
    _hi_a = evaluate_structural(_theta, _rec, material="Hastelloy-X")
    _lo_a = evaluate_structural(_theta, _rec, material="Inconel-625")
    print(f"  T22 α mayor (HX {_hi_a['sigma_th_MPa']:.1f} MPa) > "
          f"α menor (IN625 {_lo_a['sigma_th_MPa']:.1f} MPa) → "
          f"{'OK' if _hi_a['sigma_th_MPa'] > _lo_a['sigma_th_MPa'] else 'ERROR'}")

    _thin = evaluate_structural(_theta, dict(_rec, t_liner_mm=0.4),
                                material="Hastelloy-X")
    _thin_g3 = evaluate_structural(
        np.concatenate([_theta[:14], [0.4], _theta[15:]]), _rec,
        material="Hastelloy-X")["g_struct"][3]
    print(f"  T5 espesor 0.4 mm < 0.8 mm (HX) → g[3]={_thin_g3:+.3f} "
          f"({'VIOLA (correcto)' if _thin_g3 > 0 else 'ERROR'})")

    # DEGENERADO (contrato §5.1) ⇒ g todo a 10.0 · AUSENTE ⇒ default declarado
    _cases = (
        ("DEG θ con NaN", np.full(20, np.nan), _rec, "Inconel-625", True),
        ("DEG θ de ceros", np.zeros(20), _rec, "Inconel-625", True),
        ("DEG T_w1 negativa", _theta, dict(_rec, T_w1_K=-5.0),
         "Inconel-625", True),
        ("DEG T_w1 = inf", _theta, dict(_rec, T_w1_K=float("inf")),
         "Inconel-625", True),
        ("DEG D_L = 0", _theta, dict(_rec, D_L_mm=0.0), "Inconel-625", True),
        ("DEG material inexistente", _theta, _rec, "Unobtainium", True),
        ("DEF record vacío", _theta, {}, "Inconel-625", False),
        ("DEF record None", _theta, None, "Inconel-625", False),
        ("DEF θ corto (pad)", np.zeros(3), _rec, "Inconel-625", False),
        ("DEF θ None", None, _rec, "Inconel-625", False),
        ("DEF carcasa inexistente", _theta, _rec, "Inconel-625", False),
    )
    _all_ok = True
    for _tag, _th, _rc, _mat, _want_deg in _cases:
        _cas = "Kryptonite" if "carcasa" in _tag else None
        _d = evaluate_structural(_th, _rc, material=_mat, material_casing=_cas)
        _fin = all(math.isfinite(v) for v in _d["g_struct"])
        _isdeg = all(v == G_DEGENERATE for v in _d["g_struct"])
        _ok = (_fin and len(_d["g_struct"]) == N_STRUCT_CONSTRAINTS
               and _isdeg == _want_deg)
        _all_ok &= _ok
        print(f"  {_tag:26s} → {'g=10.0×5' if _isdeg else 'g evaluado'} "
              f"finito={_fin} len={len(_d['g_struct'])} "
              f"{'OK' if _ok else 'ERROR'}  ({_d['source']})")

    print(f"\nsmoke {'OK' if _all_ok else 'CON ERRORES'}")
    sys.exit(0 if _all_ok else 1)
