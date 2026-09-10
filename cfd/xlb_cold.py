#!/usr/bin/env python3
"""
QUASAR Phy-CB · cfd/xlb_cold.py — L1.5: aero-frío lattice-Boltzmann.

Qué hace
--------
Resuelve el **flujo frío isotermo** del combustor sobre una malla de voxels
(la que exporta la capa 5c en `*_field.vdb`, o una re-voxelización analítica
del contrato `phycb-combustor-1`) y devuelve el campo promediado más las
máscaras de región necesarias para medir:

* el reparto de aire REAL fila a fila (`cfd/cold_post.py`),
* la penetración de los chorros de dilución,
* la topología de recirculación que alimenta la auto-topología del CRN
  (`cfd/crn_from_field.py`).

Es la parte A de WAVE5 (la parte B, OpenFOAM, está **fuera de alcance** por
decisión de proyecto — contrato §12).

Tres backends, una sola API
---------------------------
1. **XLB** (Autodesk, Apache-2.0, Python/JAX). Todo lo específico de XLB vive
   en `_solve_xlb` y en `xlb_available()` — WAVE5 §D avisa de que su API se
   mueve rápido, así que queda encapsulado en un único adaptador. En esta
   máquina NO está instalado.
2. **`lbm_cupy`** — EL MISMO núcleo D3Q19 BGK, ejecutado sobre GPU con CuPy.
   CuPy es un reemplazo casi directo de NumPy, así que el solver se escribe
   una sola vez con un alias `xp = cupy | numpy`: **no hay dos
   implementaciones de la física que puedan divergir**. Lo que sí cambia de un
   backend a otro está aislado en cinco sitios, todos comentados:
   `_to_device` (subir máscaras e índices UNA vez, fuera del bucle),
   `_bounce_back` (índices en vez de máscara booleana: en CuPy una máscara
   relanza `nonzero()` y sincroniza en cada paso), `_update_effusion`
   (reducción segmentada con `bincount` en vez de 2·N `mean()` por franja, que
   eran 2·N sincronizaciones), `_sync` (barrera antes de cronometrar: los
   kernels de CuPy son asíncronos) y `_free_pool` (devolver la VRAM al pool).
   Ninguno toca las ecuaciones.
3. **`lbm_numpy_fallback`** — el mismo núcleo en NumPy puro. Es lo que hace
   que T41–T43 se puedan **correr de verdad** sin GPU y en CI con mallas
   mini, en vez de saltarse todos.

La etiqueta `backend` (y `source`) del resultado dice SIEMPRE cuál se usó.
Ni CuPy ni XLB son obligatorios: `backend="auto"` prueba XLB → CuPy → NumPy y
degrada **con etiqueta**, nunca con excepción (contrato §10.4).

GPU medida en esta máquina (2026-07-31)
---------------------------------------
RTX 3070 Ti, 8 GiB (6.9 GiB libres), compute 8.6, CUDA runtime 12.9,
`cupy 14.1.1`. `jax`/`xlb`/`warp`/`torch` NO están instalados. Rendimiento
MEDIDO del paso completo (D3Q19 + LES de Smagorinsky, FP32), no de un kernel
de juguete:

==============  ============  ============  ========
nodos           NumPy         CuPy          ganancia
==============  ============  ============  ========
0.016 M          2.1 ms/paso  10.4 ms/paso  ×0.2
0.108 M         12.8 ms/paso  10.6 ms/paso  ×1.2
0.965 M          148 ms/paso  18.4 ms/paso  ×8.1
3.19  M          472 ms/paso  43.8 ms/paso  ×10.8
8.00  M         1214 ms/paso   101 ms/paso  ×12.0
==============  ============  ============  ========

Es decir ~73 MLUPS (1.4e-8 s/celda/paso) frente a ~6.6 MLUPS (1.5e-7) — del
orden de la referencia pública que cita Ciencia §11.2 (1e-8 s/celda/paso en
FP32), no los ~900 MLUPS de un kernel LBM fusionado a mano: aquí cada paso son
~90 lanzamientos de kernels de NumPy vectorizado, no uno solo. Esa es la
frontera honesta de «CuPy como NumPy sobre GPU».

En el caso ancla eso permite pasar de dx = 1.23 mm (0.92 M nodos, anillo del
liner con 2.0 celdas) a **dx = 0.82 mm** (2.99 M nodos, anillo con 3.0 celdas)
dentro de un presupuesto de pared razonable. Bajar a dx = 0.60 mm cabe de
sobra en VRAM (7.5 M nodos ≈ 2.7 GiB) pero **no cabe en tiempo**: el coste va
como dx⁻⁴ y son ~39 000 pasos × 7.5 M nodos ≈ 80 min por caso.

**Determinismo (declarado, no fingido)**: el solver no usa números aleatorios,
así que a igual backend y misma malla el resultado es reproducible bit a bit.
Entre backends NO lo es: las contracciones sobre las 19 direcciones de
`_macro` y `_omega_les` son productos de matrices, y cuBLAS suma en un orden
distinto al de OpenBLAS. Medido en los casos canónicos: T41 da el mismo caudal
a 9 cifras (12.65915354 lu en ambos) y T42 el mismo C_d a 4 (0.6774); el
barrido de T43 a refine = 1 reproduce el +30.4 % de CPU en GPU.

Verificación del propio solver
------------------------------
El flujo másico entre regiones NO se mide integrando `ρu` con interpolación
(eso da errores del 2 % en un canal recto): se mide sumando las **poblaciones
que cruzan** en el paso de streaming, que es exactamente la contabilidad que
hace el solver. Con ese estimador T41 cierra a 3e-5 relativo en vez de 2e-2.
El promedio temporal se acumula sobre `g` (post-colisión, pre-streaming), que
es la magnitud que efectivamente se transporta.

Unidades (trampa clásica del LBM)
---------------------------------
Todo el interior del solver está en unidades de lattice (dx = dt = ρ₀ = 1).
La conversión se ancla en la velocidad de entrada:

    C_u  = U_in / u_lattice_in        [m/s por lu/ts]
    dt   = dx / C_u                   [s por ts]
    ν_lat= ν_fis / (C_u · dx)         [lu²/ts]
    τ    = 3·ν_lat + 0.5
    ṁ_fis= ṁ_lat · ρ₃ · C_u · dx²     [kg/s]
    Δp_fis = Δρ_lat/3 · ρ₃ · C_u²     [Pa]

`u_lattice_in` se fija en `U_LATTICE_TARGET` = 0.05 para mantener
`u_lattice < 0.1` en los chorros (WAVE5 §D) y el error de compresibilidad
(O(Ma²)) por debajo del 1 %.

Limitaciones DECLARADAS del nivel L1.5
--------------------------------------
* Flujo **frío e isotermo**: no hay expansión térmica, así que los τ por zona
  salen sesgados. `cold_post` los corrige con ρ ∝ 1/T_zona de L0 y lo dice.
* A los números de Reynolds reales (10⁴–10⁵) un BGK D3Q19 no es estable con
  la viscosidad molecular: τ se **suela** en `TAU_FLOOR` y el LES de
  Smagorinsky aporta el resto. El resultado es un campo a Re efectivo MENOR
  que el real; se reporta `Re_lattice_effective` junto al físico y
  `cfd_mesh["nu_ratio_effective"]`. **Este es el límite que manda hoy y la
  malla no lo arregla** (ver `TAU_FLOOR`): a dx = 0.61 mm en el ancla la
  viscosidad efectiva sigue siendo ×52.8 la real y el Δp/p sale ×20 el de
  diseño. La hipótesis previa de que «con dx ≤ 0.6 mm el campo sería
  cuantitativo» está por tanto **refutada por medida**: la resolución era
  necesaria, no suficiente. La trayectoria de los chorros (dominada por la
  cantidad de movimiento) sí es poco sensible; el reparto entre caminos y el
  Δp NO lo son; la capa límite del liner NO está resuelta y no se usa.
* La efusión (Ø 0.6 mm) es **irresoluble** a dx ≥ 1 mm: entra como
  transpiración distribuida gobernada por el Δp LOCAL que sí resuelve el LBM
  (ver `_effusion_sites`), no como orificios discretos. Ni siquiera a
  dx = 0.6 mm se resuelve (1 celda por orificio): el modelo se mantiene.
* El swirler entra como **taladro axial liso, sin álabes**: el contrato solo
  da `count` y `d_tip_mm`. La malla fina resuelve mejor ese taladro, pero
  seguirá sin generar swirl — es una limitación del MODELO, no de la
  resolución, y ninguna malla la arregla.
* Sin lector de OpenVDB instalado se re-voxeliza analíticamente el contrato:
  es la MISMA geometría de la que sale el `.vdb`, pero sin los detalles que
  PicoGK añade (redondeos, teardrops). Se declara en `notes`.

Dependencias: solo NumPy. XLB/JAX son opcionales y degradan con etiqueta.
"""

from __future__ import annotations

import math
import os
import sys
import time

import numpy as np

# --- TRAMPA DE ENTORNO (contrato §0.b) ------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import physics_core as pc                                        # noqa: E402

pc.assert_provenance(pc)

PROJECT = "Phy-CB"
SCHEMA = "phycb-coldfield-1"
SEED = 71                        # semilla de la familia (contrato §10.8)

# ==========================================================================
# 1. Tunables (SCREAMING_CASE + procedencia, contrato §10.7)
# ==========================================================================
#: Velocidad de lattice OBJETIVO en la entrada. WAVE5 §D exige u_lat < 0.1
#: (a Ma_lattice mayor el error de compresibilidad del BGK deja de ser
#: despreciable). 0.05 deja un factor 2 de margen para los chorros.
U_LATTICE_TARGET = 0.05
U_LATTICE_MAX = 0.10             # techo duro de aviso (WAVE5 §D)

# --------------------------------------------------------------------------
# 1.a Operador de colisión (BGK / TRT / MRT) y su suelo de τ
# --------------------------------------------------------------------------
#: Etiquetas de los tres operadores. Se usan tal cual en `collision=`, en
#: `PHYCB_LBM_COLLISION`, en `field["source"]` y en `cfd_mesh["collision"]`.
COLLISION_BGK = "bgk"
COLLISION_TRT = "trt"
COLLISION_MRT = "mrt"
COLLISIONS = (COLLISION_BGK, COLLISION_TRT, COLLISION_MRT)

#: Operador por DEFECTO. TRT (ver el barrido de estabilidad de
#: `stability_sweep`, medido en esta máquina el 2026-08-01): con el mismo
#: coste que BGK aguanta τ dos órdenes de magnitud más cerca de 0.5, que es
#: justo lo que este caso necesita. **BGK se conserva intacto y sigue siendo
#: reproducible** con `collision="bgk"` o `PHYCB_LBM_COLLISION=bgk`: los
#: números publicados de L1.5 anteriores a esta fecha se midieron con BGK.
DEFAULT_COLLISION = COLLISION_TRT

#: Parámetro MÁGICO de TRT, Λ = (τ_s − ½)(τ_a − ½). Fija τ_a una vez fijado
#: τ_s (que es quien pone la viscosidad), y con él la posición efectiva de la
#: pared de bounce-back y la estabilidad. Procedencia: Ginzburg &
#: d'Humières, *Multireflection boundary conditions for lattice Boltzmann
#: models*, Phys. Rev. E 68, 066614 (2003), §5; y Ginzburg, Verhaeghe & Luo,
#: *Two-relaxation-time lattice Boltzmann scheme*, Commun. Comput. Phys. 3
#: (2008) 427–478, tabla 1.
#:
#: * Λ = 3/16 → el bounce-back de media celda cae EXACTAMENTE a media celda
#:   de la pared, **independientemente de τ** (con BGK la pared se desplaza
#:   como (τ−½)², que es el «slip numérico» clásico). Es el valor que
#:   interesa aquí: toda la geometría del combustor son paredes de
#:   bounce-back (liner, anillo, orificios, swirler).
#: * Λ = 1/4 → máxima estabilidad lineal.
#:
#: MEDIDOS LOS DOS en `stability_sweep` (no se decidió a priori).
TRT_LAMBDA_BOUNCE_BACK = 3.0 / 16.0
TRT_LAMBDA_STABILITY = 1.0 / 4.0

#: Tasa ANTISIMÉTRICA fija, la alternativa a fijar Λ. `None` = usar Λ.
#:
#: **Por qué existe (MEDIDO, no supuesto).** Fijar Λ obliga a
#: `ω_a = 8(2−ω_s)/(8−ω_s)`, que TIENDE A CERO cuando τ_s → ½ (ω_s → 2): a
#: τ_s = 0.51 y Λ = 3/16 sale ω_a = 0.0519, es decir los modos
#: antisimétricos («fantasma») dejan de relajarse y se acumulan. El barrido
#: de `stability_sweep` sobre el canal de T41 —el operador DESNUDO, sin LES
#: que lo tape— lo dice sin ambigüedad (τ mínimo que no diverge):
#:
#:     configuración        τ mín. estable   ν_lat        ν_ef/ν_real (ancla)
#:     BGK                  0.505000         1.67e-3      ×184
#:     TRT Λ = 3/16         0.550000         1.67e-2      ×1839   ← PEOR
#:     TRT Λ = 1/4          0.550000         1.67e-2      ×1839   ← PEOR
#:     TRT ω_a = 1.00       0.505000         1.67e-3      ×184
#:     TRT ω_a = 1.50       0.501000         3.33e-4      ×36.8
#:     TRT ω_a = 1.80       0.500005         1.67e-6      ×0.18 (sin suelo)
#:     TRT ω_a = 1.85       > 0.500005       —            —
#:     TRT ω_a = 1.90       0.500020         6.67e-6      ×0.74 (sin suelo)
#:     TRT ω_a = 1.95       0.500200         6.67e-5      ×7.4
#:     MRT                  0.500500         1.67e-4      ×18.4
#:
#: Es decir: **los dos Λ de la literatura empeoran la estabilidad** en este
#: límite —son los valores correctos para EXACTITUD de pared en flujo de
#: Stokes (Ginzburg & d'Humières 2003 §5), no para viscosidad evanescente— y
#: lo que estabiliza es que la tasa libre se quede LEJOS de 0 y de 2.
#: ω_a = 1.80 es el óptimo medido: por encima (1.85, 1.90, 1.95) la tasa se
#: acerca demasiado al límite ω = 2 y por debajo (1.50, 1.00) los modos
#: fantasma se sub-relajan. Coincide con la lectura de MRT, cuyos modos
#: fantasma llevan tasas fijas de 1.2 a 1.98 y por eso aguanta ×10 más que
#: BGK sin ser TRT.
#:
#: Con `TRT_OMEGA_A` fijo, Λ pasa a ser una magnitud DERIVADA
#: (`Λ = (τ_s−½)(1/ω_a−½)`, que decrece con τ_s) y se reporta como tal en
#: `cfd_mesh["trt_lambda"]`. ⚠️ tunable numérico, no físico.
TRT_OMEGA_A = 1.80

#: Λ por defecto. Se deja el de la pared exacta por documentación, pero con
#: `TRT_OMEGA_A` distinto de `None` **manda ω_a** (ver arriba).
TRT_LAMBDA = TRT_LAMBDA_BOUNCE_BACK

#: Tasas de relajación de los momentos NO hidrodinámicos del MRT D3Q19, en el
#: orden de `_mrt_matrices`. Procedencia: d'Humières, Ginzburg, Krafczyk,
#: Lallemand & Luo, *Multiple-relaxation-time lattice Boltzmann models in
#: three dimensions*, Phil. Trans. R. Soc. Lond. A 360 (2002) 437–451, §4
#: (juego «óptimo» de la tabla de estabilidad). Los momentos conservados
#: (ρ, j_x, j_y, j_z) llevan s = 0 y los de cortadura s = 1/τ (los pone el
#: solver, no esta tabla). ⚠️ tunable numérico, no físico.
MRT_S_E = 1.19                   # m1  · energía (viscosidad de volumen)
MRT_S_EPS = 1.40                 # m2  · energía²
MRT_S_Q = 1.20                   # m4, m6, m8   · flujo de energía
MRT_S_PI = 1.40                  # m10, m12     · momentos de 4.º orden
MRT_S_M = 1.98                   # m16, m17, m18 · modos «fantasma»

#: Índices de los momentos de CORTADURA del D3Q19 (los que fijan la
#: viscosidad y por tanto los que el LES modula célula a célula).
MRT_SHEAR_IDX = (9, 11, 13, 14, 15)

#: Suelos de τ, uno POR OPERADOR. ⚠️ tunables numéricos, no físicos: subirlos
#: baja el Re efectivo del cálculo.
#:
#: **CRITERIO (el mismo para los tres, para que comparar signifique algo):**
#: el τ mínimo con el que el operador NO DIVERGE en el canal de T41 —el caso
#: DESNUDO, con el LES apagado— multiplicando después el exceso sobre ½ por
#: un margen de seguridad `TAU_FLOOR_SAFETY` = 2. Se elige el canal sin LES a
#: propósito: con LES el orificio de T42 aguanta τ = 0.500005 **con los tres
#: operadores** (medido), porque la viscosidad turbulenta de Smagorinsky tapa
#: la inestabilidad — y un suelo que depende de que un modelo de submalla
#: esté encendido no es un suelo, es una casualidad. La EXACTITUD no entra en
#: el criterio porque en el canal satura en τ ≈ 0.51 para los tres por igual:
#: por debajo el caso deja de ser estacionario y lo que se degrada es el
#: cierre del promedio de 200 pasos, no el operador (ver `stability_sweep`,
#: que devuelve `floor_suggested` y `floor_accurate` por separado).
#:
#: Medido en esta máquina el 2026-08-01 (tabla completa en `TRT_OMEGA_A`):
#:
#:     operador          τ mín. estable   suelo = ½ + 2(τ−½)   ν_ef/ν_real
#:     BGK               0.505000         (0.510000)           ×184
#:     TRT ω_a = 1.80    0.500005         0.500010             SIN SUELO
#:     MRT               0.500500         0.501000             ×36.8
#:
#: La columna ν_ef/ν_real es la del CASO ANCLA, cuya τ molecular es
#: 0.5000272: el suelo de TRT (0.500010) queda POR DEBAJO, así que el ancla
#: pasa a correr con la viscosidad molecular REAL y el cociente vale 1.000.
#: Ese es el resultado del encargo.
#:
#: **BGK conserva 0.5015 EXACTO, que NO es lo que dice su barrido** (pediría
#: 0.510 sin LES, y aguantaría 0.50001 con él). Es una decisión declarada de
#: reproducibilidad: todos los números de L1.5 publicados antes del
#: 2026-08-01 se midieron con BGK a 0.5015 y tienen que poder re-medirse
#: (`collision="bgk"` o `PHYCB_LBM_COLLISION=bgk`). Léase como lo que es: el
#: valor histórico es inseguro sin LES y conservador con él.
TAU_FLOOR_BGK = 0.5015
TAU_FLOOR_TRT = 0.500010
TAU_FLOOR_MRT = 0.501000

#: Alias retro-compatible. Sigue siendo el suelo de BGK, que es el operador
#: con el que se midió todo lo publicado antes del 2026-08-01.
TAU_FLOOR = TAU_FLOOR_BGK


def resolve_collision(collision: str | None = None) -> str:
    """Operador de colisión efectivo. `PHYCB_LBM_COLLISION` manda.

    Mismo patrón que `resolve_backend`: la variable de entorno tiene
    prioridad sobre el argumento (es el mando para re-medir con BGK un número
    publicado sin tocar código). Un nombre desconocido cae en el defecto
    **sin excepción** — la etiqueta del resultado dice cuál salió."""
    env = os.environ.get("PHYCB_LBM_COLLISION", "").strip().lower()
    c = env or (collision if collision is not None else DEFAULT_COLLISION)
    c = str(c or "").strip().lower()
    return c if c in COLLISIONS else DEFAULT_COLLISION


def tau_floor(collision: str | None = None) -> float:
    """Suelo de τ del operador indicado (ver `TAU_FLOOR_*`)."""
    return {COLLISION_BGK: TAU_FLOOR_BGK,
            COLLISION_TRT: TAU_FLOOR_TRT,
            COLLISION_MRT: TAU_FLOOR_MRT}[resolve_collision(collision)]


def collision_label(collision: str | None = None,
                    trt_lambda: float | None = TRT_LAMBDA,
                    trt_omega_a: float | None = TRT_OMEGA_A) -> str:
    """Etiqueta legible del operador para `source` / `cfd_mesh["scheme"]`."""
    c = resolve_collision(collision)
    if c == COLLISION_TRT:
        if trt_omega_a is not None:
            return f"D3Q19-TRT(omega_a={float(trt_omega_a):.6g})"
        if trt_lambda is None:
            return "D3Q19-TRT(tau_a=tau_s)"
        return f"D3Q19-TRT(Lambda={float(trt_lambda):.6g})"
    return f"D3Q19-{c.upper()}"


#: Constante de Smagorinsky del LES. Banda de literatura 0.10–0.20 para LBM;
#: 0.16 es el valor de referencia de Hou et al. ⚠️ recalibrar si se cambia
#: el operador de colisión.
CS_SMAGORINSKY = 0.16

#: Presupuesto de nodos de la malla AUTOMÁTICA del caso real EN CPU. Con el
#: fallback de NumPy (~10 MLUPS medidos en esta máquina) 250 k nodos × ~6 000
#: pasos ≈ 2.5 min.
LBM_MAX_NODES = 260_000

#: Techo DURO de nodos en GPU. El límite real lo pone la VRAM libre medida
#: (`gpu_node_budget`), pero un techo absoluto evita que una geometría rara
#: pida una malla que no terminaría nunca aunque quepa: a 500 MLUPS, 12 M
#: nodos × 20 000 pasos son 8 minutos por caso, y el deadline de `solve_cold`
#: los cortaría a mitad. ⚠️ tunable de coste, no de física.
LBM_MAX_NODES_GPU = 12_000_000

#: Nodos por debajo de los cuales la GPU NO compensa. Cada paso lanza ~90
#: kernels y cada lanzamiento cuesta ~25 µs de latencia: por debajo de este
#: tamaño el kernel termina antes de que la CPU consiga encolar el siguiente y
#: NumPy gana. Medido en esta máquina (paso completo, D3Q19+LES):
#:     0.016 M nodos → NumPy 2.1 ms · CuPy 10.4 ms   (CuPy ×0.2)
#:     0.087 M nodos → NumPy 12.0 ms · CuPy 10.6 ms  (CuPy ×1.1)
#:     0.108 M nodos → NumPy 12.8 ms · CuPy 10.6 ms  (CuPy ×1.2)
#:     0.965 M nodos → NumPy 148 ms · CuPy 18.4 ms   (CuPy ×8.1)
#:     3.19  M nodos → NumPy 472 ms · CuPy 43.8 ms   (CuPy ×10.8)
#: ⚠️ tunable de rendimiento, no de física: cambiarlo no cambia ningún número,
#: solo quién lo calcula (y el resultado NO es bit-idéntico entre backends).
GPU_MIN_NODES = 60_000

#: Fracción de la VRAM LIBRE que `gpu_node_budget` se permite reservar. El
#: resto es margen: el driver, el contexto de CUDA y los temporales de los
#: kernels de reducción también comen. Medido: con 0.55 el caso ancla a
#: dx = 0.6 mm (7.5 M nodos, 2.6 GiB) deja > 4 GiB libres.
GPU_VRAM_FRACTION = 0.55

DX_MIN_MM, DX_MAX_MM = 0.5, 4.0  # banda de voxel admisible (WAVE5 §A.2.1)

LBM_STEPS_MAX = 30_000           # bucle ACOTADO (contrato §10.5)

#: Techo de iteraciones cuando el backend es GPU. El número de pasos que
#: tarda el transitorio escala como **1/dx** (misma velocidad de lattice,
#: dominio dx veces más fino), y 30 000 se calibró para el dx que alcanzaba
#: la CPU (1.23 mm en el ancla, convergido en 17 000 pasos). MEDIDO el
#: 2026-07-31: a dx = 0.82 mm el mismo caso **agota los 30 000 pasos sin
#: converger** y devuelve un transitorio (Δp/p = 0.75 frente al 0.03 de
#: diseño — un número sin sentido físico que además contamina T44g). Con el
#: techo de CPU, afinar la malla en GPU EMPEORA el resultado en vez de
#: mejorarlo: se paga la resolución y no se llega al estacionario.
#: 60 000 = 2× el de CPU, que es justo el factor 1/dx del salto 1.23 → 0.6 mm.
#: Sigue siendo un bucle ACOTADO; el `deadline` de pared manda igualmente.
#: ⚠️ tunable numérico: subirlo no cambia la física, cambia cuánto se deja
#: converger.
LBM_STEPS_MAX_GPU = 60_000
LBM_AVG_STEPS = 400              # ventana de promedio estadístico
LBM_CHECK_EVERY = 200            # cada cuántos pasos se testea convergencia
LBM_CONV_TOL = 1.0e-5            # cambio relativo de <|u|> entre chequeos
LBM_TIMEOUT_S = float(os.environ.get("PHYCB_L15_TIMEOUT", "900"))

#: Refresco del modelo de efusión (Δp local → ṁ). Cada paso sería ruido puro.
EFFUSION_EVERY = 25
EFFUSION_RELAX = 0.30            # sub-relajación del caudal de efusión
EFFUSION_MAX_FRAC = 0.60         # cota de seguridad: ṁ_efusión ≤ 0.6·ṁ₃
#: C_d de la efusión si `physics_core` no está disponible. Procedencia:
#: tabla `data/lefebvre/cd_holes.csv` (`effusion_cylindrical`, 0.65) ×
#: `physics_core.CD_EFFUSION_INCLINED` (0.38) × rugosidad AM de Ciencia §4.1
#: (≈0.89 a d = 0.6 mm) = 0.221. RESPALDO ETIQUETADO: el valor vivo lo da
#: `physics_core.effusion_discharge_coefficient` (fuente única §12.b).
EFFUSION_CD_FALLBACK = 0.221

#: Aire: viscosidad de Sutherland. MISMAS constantes que `crn_core._mu_air`
#: (µ_ref de `physics_core.MU_AIR` a 293.15 K).
MU_AIR_REF = 1.81e-5
T_MU_REF = 293.15
SUTHERLAND_S = 110.4

#: Estado del último `solve_cold` (patrón de `crn_core.LAST_STATUS`).
LAST_STATUS = "not_run"

#: Etiquetas de backend. Se usan tal cual en `field["backend"]`,
#: `field["source"]` y `cfd_mesh["backend"]` (contrato §3: la etiqueta dice
#: SIEMPRE por qué camino salió el número).
BACKEND_XLB = "lbm_xlb"
BACKEND_CUPY = "lbm_cupy"
BACKEND_NUMPY = "lbm_numpy_fallback"


# ==========================================================================
# 1.b Resolución de backend (import PEREZOSO, degradación etiquetada)
# ==========================================================================
#: Memo del sondeo de CuPy: `(modulo_o_None, motivo)`. Se calcula UNA vez.
_CUPY_PROBE: tuple | None = None


def cupy_module():
    """Módulo CuPy **operativo**, o `None`. Import PEREZOSO (contrato §10.4).

    No basta con que `import cupy` funcione: CuPy se importa sin GPU y solo
    revienta al primer kernel. Aquí se ejecuta un kernel diminuto de verdad
    (suma de 8 floats) y se sincroniza, así que un driver roto, una GPU
    ocupada por otro proceso o un CUDA mal instalado se detectan AQUÍ y no a
    mitad de un solve de diez minutos. Nunca lanza."""
    global _CUPY_PROBE
    if _CUPY_PROBE is not None:
        return _CUPY_PROBE[0]
    if os.environ.get("PHYCB_NO_CUPY", "").strip().lower() in ("1", "yes",
                                                              "true", "si"):
        _CUPY_PROBE = (None, "desactivado por PHYCB_NO_CUPY")
        return None
    try:
        import cupy as _cp                              # noqa: PLC0415
        a = _cp.arange(8, dtype=_cp.float32)
        v = float((a * a).sum())
        _cp.cuda.runtime.deviceSynchronize()
        if not math.isclose(v, 140.0, rel_tol=1e-6):
            raise RuntimeError(f"kernel de sondeo dio {v}, esperaba 140")
        _CUPY_PROBE = (_cp, "ok")
        return _cp
    except Exception as exc:
        _CUPY_PROBE = (None, f"{type(exc).__name__}: {exc}")
        return None


def cupy_reason() -> str:
    """Motivo por el que CuPy está (o no está) disponible. Para las notas."""
    cupy_module()
    return _CUPY_PROBE[1] if _CUPY_PROBE else "no sondeado"


def cupy_available() -> bool:
    """¿Hay GPU con CuPy operativo? Único sitio donde se comprueba."""
    return cupy_module() is not None


def gpu_info() -> dict:
    """Ficha de la GPU (o `{}`): nombre, compute capability y VRAM libre."""
    cp = cupy_module()
    if cp is None:
        return {}
    try:
        free, total = cp.cuda.runtime.memGetInfo()
        dev = cp.cuda.Device()
        props = cp.cuda.runtime.getDeviceProperties(dev.id)
        name = props["name"]
        if isinstance(name, bytes):
            name = name.decode("utf-8", "replace")
        return dict(name=str(name), compute_capability=str(dev.compute_capability),
                    vram_free_GiB=float(free) / 2 ** 30,
                    vram_total_GiB=float(total) / 2 ** 30,
                    cupy_version=str(cp.__version__),
                    runtime_version=int(cp.cuda.runtime.runtimeGetVersion()))
    except Exception:
        return {}


def solver_bytes_per_node(dtype=np.float32, acc_dtype=np.float64,
                          collision: str | None = COLLISION_BGK) -> float:
    """Bytes de VRAM por nodo que consume `LBMSolver`.

    Cuenta lo que vive a la vez DENTRO del bucle:

    * `f` y `g` — 19 poblaciones cada uno (2·19·w);
    * el acumulador del promedio temporal — 19·a;
    * `_mom` (ρ y las tres componentes de u en un buffer contiguo) — 4·w;
    * `_usq`, `_tmp`, `_Q` — 3·w;
    * `_q6` (las seis componentes del tensor del LES) — 6·w;
    * `solid_idx` — 4 B por nodo en el peor caso (todo sólido);
    * `_scr` — TERCER bloque de 19 poblaciones, **solo con TRT/MRT**: los dos
      operadores necesitan leer f^neq entero mientras escriben el
      post-colisión, y aliasar entrada y salida daría basura (ver
      `_collide_trt`). Con BGK no se reserva y el consumo es EXACTAMENTE el
      de antes, que es lo que mantiene reproducible la malla automática de
      los números publicados.

    Con FP32, acumulador FP64 y BGK son 360 B/nodo (284 con acumulador FP32);
    con TRT/MRT, 436. Verificado contra
    `cupy.get_default_memory_pool().used_bytes()`."""
    w = float(np.dtype(dtype).itemsize)
    a = float(np.dtype(acc_dtype).itemsize)
    extra = 0.0 if resolve_collision(collision) == COLLISION_BGK else 19.0 * w
    return 19.0 * (2.0 * w + a) + 13.0 * w + 4.0 + extra


def gpu_node_budget(dtype=np.float32, acc_dtype=np.float64,
                    fraction: float = GPU_VRAM_FRACTION,
                    collision: str | None = COLLISION_BGK) -> int:
    """Nodos que caben en la VRAM LIBRE medida, con margen. 0 si no hay GPU.

    Se mide la memoria libre REAL en el momento de la llamada (no la total):
    si otro proceso tiene la GPU medio ocupada, el presupuesto baja solo y la
    malla sale más gruesa **con aviso**, en vez de reventar a mitad."""
    cp = cupy_module()
    if cp is None:
        return 0
    try:
        free, _ = cp.cuda.runtime.memGetInfo()
    except Exception:
        return 0
    per = solver_bytes_per_node(dtype, acc_dtype, collision)
    n = int(max(float(free) * float(fraction), 0.0) / max(per, 1.0))
    return int(min(n, LBM_MAX_NODES_GPU))


def resolve_backend(backend: str = "auto") -> tuple[str, object, str]:
    """`(etiqueta, módulo_array, motivo)` del backend que se va a usar.

    `"auto"` prueba **XLB → CuPy → NumPy** y devuelve el primero que exista.
    `"cupy"`/`"numpy"`/`"xlb"` fuerzan uno; si el forzado no está, se degrada
    igualmente a NumPy **con el motivo** (contrato §10.4: una dependencia
    ausente nunca es una excepción). `"auto_no_xlb"` salta el primer eslabón:
    lo usa `solve_cold` cuando XLB está instalado pero su adaptador (que
    necesita la malla ya construida) todavía no ha corrido. La variable de
    entorno `PHYCB_LBM_BACKEND` tiene prioridad sobre el argumento — es el
    mando para reproducir en CPU un número medido en GPU.

    OJO: el módulo devuelto para `BACKEND_XLB` es `numpy`; XLB no expone un
    módulo tipo NumPy y su camino no pasa por `LBMSolver`."""
    env = os.environ.get("PHYCB_LBM_BACKEND", "").strip().lower()
    if env:
        backend = env
    b = str(backend or "auto").strip().lower()
    if b in ("auto", "xlb") and XLB_AVAILABLE:
        return BACKEND_XLB, np, "xlb+jax instalados"
    if b == "xlb":
        return BACKEND_NUMPY, np, "xlb/jax no instalados → NumPy"
    if b == "auto_no_xlb":
        b = "auto"
    if b in ("auto", "cupy", "gpu"):
        cp = cupy_module()
        if cp is not None:
            return BACKEND_CUPY, cp, cupy_reason()
        if b in ("cupy", "gpu"):
            return BACKEND_NUMPY, np, f"CuPy no operativo ({cupy_reason()}) → NumPy"
        return BACKEND_NUMPY, np, f"sin XLB y sin CuPy ({cupy_reason()}) → NumPy"
    return BACKEND_NUMPY, np, "NumPy pedido explícitamente"

# ==========================================================================
# 2. Retículo D3Q19
# ==========================================================================
#: Direcciones del D3Q19. El ORDEN es interno pero se usa en `OPP` y en el
#: streaming: no reordenar sin tocar ambos.
CI: tuple[tuple[int, int, int], ...] = (
    (0, 0, 0),
    (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
    (1, 1, 0), (-1, -1, 0), (1, -1, 0), (-1, 1, 0),
    (1, 0, 1), (-1, 0, -1), (1, 0, -1), (-1, 0, 1),
    (0, 1, 1), (0, -1, -1), (0, 1, -1), (0, -1, 1))
WI: tuple[float, ...] = (1.0 / 3.0,) + (1.0 / 18.0,) * 6 + (1.0 / 36.0,) * 12
OPP: tuple[int, ...] = tuple(CI.index((-c[0], -c[1], -c[2])) for c in CI)
CS2 = 1.0 / 3.0                  # velocidad del sonido² en lattice

#: Los seis pares independientes del tensor de no-equilibrio Q_ab (simétrico).
QQ_PAIRS: tuple[tuple[int, int], ...] = ((0, 0), (1, 1), (2, 2),
                                         (0, 1), (0, 2), (1, 2))
#: Peso de cada par en la traza de Q²: 1 en la diagonal, 2 fuera (Q_ab = Q_ba).
QQ_WEIGHT: tuple[float, ...] = (1.0, 1.0, 1.0, 2.0, 2.0, 2.0)


def _moment_matrices(dtype):
    """Matrices que convierten poblaciones en momentos, de una sola pasada.

    Todo el núcleo del LBM son **contracciones sobre las 19 direcciones**. Si
    se escriben como bucles de Python, cada paso lanza ~1 200 operaciones
    diminutas: en NumPy eso se paga en ancho de banda y en CuPy además en
    ~25 µs de latencia POR LANZAMIENTO (medido: 28 ms/paso de puro
    lanzamiento, más que el propio cálculo). Escritas como un producto de
    matrices, son un único kernel.

    * `M4` (4, 19): `[ρ, ρu_x, ρu_y, ρu_z] = M4 · f`
    * `C3` (19, 3): `3·c_i·u = C3 · u`
    * `M6` (6, 19): `Q_ab = M6 · f^neq` para los seis pares de `QQ_PAIRS`
    """
    m4 = np.empty((4, 19), dtype=dtype)
    m4[0] = 1.0
    for k in range(3):
        m4[1 + k] = [float(c[k]) for c in CI]
    c3 = np.array([[3.0 * c[k] for k in range(3)] for c in CI], dtype=dtype)
    m6 = np.array([[float(c[a] * c[b]) for c in CI] for a, b in QQ_PAIRS],
                  dtype=dtype)
    return m4, c3, m6


#: Índices de los NUEVE pares de direcciones opuestas del D3Q19, sin contar
#: la partícula en reposo. En el orden de `CI` los pares son CONSECUTIVOS
#: —(1,2), (3,4), … (17,18)— y eso NO es casualidad ni se puede romper: es lo
#: que permite ver `f[1:]` como un bloque `(9, 2, N)` y hacer la
#: descomposición simétrica/antisimétrica de TRT **sin ningún reordenamiento
#: ni copia**, que es de dónde sale que TRT cueste lo mismo que BGK. Si
#: alguien reordena `CI`, esta comprobación salta en el import.
_PAIRS_ADJACENT = all(OPP[i] == i + 1 and OPP[i + 1] == i
                      for i in range(1, 19, 2)) and OPP[0] == 0
if not _PAIRS_ADJACENT:                                   # pragma: no cover
    raise RuntimeError(
        "el orden de CI ha cambiado: los pares opuestos del D3Q19 ya no son "
        "consecutivos y la descomposición de TRT/MRT por vistas (9, 2, N) "
        "dejaría de ser válida (ver `_PAIRS_ADJACENT`)")


def _mrt_matrices(dtype):
    """`(M, M⁻¹, ‖m_k‖²)` de la base de momentos ORTOGONAL del D3Q19.

    Procedencia: d'Humières, Ginzburg, Krafczyk, Lallemand & Luo,
    *Multiple-relaxation-time lattice Boltzmann models in three dimensions*,
    Phil. Trans. R. Soc. Lond. A **360** (2002) 437–451, ecuaciones (7)–(9).
    Los 19 momentos, en orden, con `c² = c_x²+c_y²+c_z²`:

    ====  =========  ======================================  ========
    k     nombre     polinomio en c                          ‖m_k‖²
    ====  =========  ======================================  ========
    0     ρ          1                                       19
    1     e          19c² − 30                               2394
    2     ε          (21c⁴ − 53c² + 24)/2                    252
    3     j_x        c_x                                     10
    4     q_x        (5c² − 9) c_x                           40
    5     j_y        c_y                                     10
    6     q_y        (5c² − 9) c_y                           40
    7     j_z        c_z                                     10
    8     q_z        (5c² − 9) c_z                           40
    9     3p_xx      3c_x² − c²                              36
    10    3π_xx      (3c² − 5)(3c_x² − c²)                    72
    11    p_ww       c_y² − c_z²                             12
    12    π_ww       (3c² − 5)(c_y² − c_z²)                   24
    13    p_xy       c_x c_y                                 4
    14    p_yz       c_y c_z                                 4
    15    p_xz       c_x c_z                                 4
    16    m_x        (c_y² − c_z²) c_x                       8
    17    m_y        (c_z² − c_x²) c_y                       8
    18    m_z        (c_x² − c_y²) c_z                       8
    ====  =========  ======================================  ========

    La base es ortogonal con el producto escalar EUCLÍDEO (`M·Mᵀ` es
    diagonal, verificado aquí en tiempo de ejecución), así que
    `M⁻¹ = Mᵀ·diag(1/‖m_k‖²)` es exacta y no hace falta invertir nada. Las
    normas salen exactamente las de la tabla publicada, lo que es una
    comprobación independiente de que el orden de `CI` está bien mapeado.

    Se construye en FP64 y se convierte al final: `M` tiene entradas hasta
    ±30 y la matriz de colisión `M⁻¹SM` se ensambla a partir de ella."""
    c = np.array(CI, dtype=np.float64)
    cx, cy, cz = c[:, 0], c[:, 1], c[:, 2]
    c2 = cx * cx + cy * cy + cz * cz
    one = np.ones(19, dtype=np.float64)
    M = np.array([
        one,
        19.0 * c2 - 30.0,
        (21.0 * c2 * c2 - 53.0 * c2 + 24.0) / 2.0,
        cx, (5.0 * c2 - 9.0) * cx,
        cy, (5.0 * c2 - 9.0) * cy,
        cz, (5.0 * c2 - 9.0) * cz,
        3.0 * cx * cx - c2, (3.0 * c2 - 5.0) * (3.0 * cx * cx - c2),
        cy * cy - cz * cz, (3.0 * c2 - 5.0) * (cy * cy - cz * cz),
        cx * cy, cy * cz, cx * cz,
        (cy * cy - cz * cz) * cx, (cz * cz - cx * cx) * cy,
        (cx * cx - cy * cy) * cz], dtype=np.float64)
    gram = M @ M.T
    nrm = np.diag(gram).copy()
    off = float(np.abs(gram - np.diag(nrm)).max())
    if off > 1e-9:                                        # pragma: no cover
        raise RuntimeError(
            f"la base de momentos MRT no es ortogonal ({off:g})")
    Minv = M.T / nrm
    return M.astype(dtype), Minv.astype(dtype), nrm


def _mrt_rate_vector(tau: float) -> np.ndarray:
    """Vector de 19 tasas `s_k` del MRT. Los de cortadura llevan 1/τ.

    Los momentos conservados (ρ, j) llevan s = 0: su no-equilibrio es cero en
    aritmética exacta y así lo que quede de ruido no se realimenta."""
    s = np.zeros(19, dtype=np.float64)
    s[1] = MRT_S_E
    s[2] = MRT_S_EPS
    s[4] = s[6] = s[8] = MRT_S_Q
    s[10] = s[12] = MRT_S_PI
    s[16] = s[17] = s[18] = MRT_S_M
    for k in MRT_SHEAR_IDX:
        s[k] = 1.0 / float(tau)
    return s


def _collision_matrix(collision: str, tau: float, trt_lambda, dtype,
                      trt_omega_a=None):
    """Matriz 19×19 `A` del operador, para el caso de tasas UNIFORMES.

    El post-colisión es siempre `f ← f − A·f^neq`:

    * BGK: `A = (1/τ)·I` (no se usa esta vía; BGK conserva su camino escalar
      intacto y bit a bit);
    * TRT: `A = a·I + b·P` con `P` la permutación que intercambia cada par de
      direcciones opuestas, `a = (ω_s+ω_a)/2` y `b = (ω_s−ω_a)/2`. Sale de
      sustituir `f^± = (f_i ± f_ī)/2` en
      `f_i ← f_i − (f⁺−f^eq⁺)/τ_s − (f⁻−f^eq⁻)/τ_a`;
    * MRT: `A = M⁻¹·diag(s)·M`.

    Se ensambla en FP64 y se **limpia el redondeo** (`|A_ij| < 1e-13·max|A|`
    → 0) antes de convertir: sin eso, `M⁻¹·diag(1/τ)·M` deja fuera de la
    diagonal residuos de ~1e-16 y la identidad algebraica «MRT con todas las
    tasas iguales ≡ BGK» dejaría de verificarse a precisión de máquina por un
    artefacto del ensamblado, no por la física."""
    if collision == COLLISION_TRT:
        ws, wa = _trt_rates(tau, trt_lambda, trt_omega_a)
        A = float((ws + wa) / 2.0) * np.eye(19, dtype=np.float64)
        b = float((ws - wa) / 2.0)
        for i in range(19):
            A[i, OPP[i]] += b
    elif collision == COLLISION_MRT:
        M, Minv, _ = _mrt_matrices(np.float64)
        A = Minv @ (_mrt_rate_vector(tau)[:, None] * M)
    else:
        A = (1.0 / float(tau)) * np.eye(19, dtype=np.float64)
    A[np.abs(A) < 1e-13 * max(float(np.abs(A).max()), 1e-300)] = 0.0
    return A.astype(dtype)


def _trt_rates(tau_s, trt_lambda, trt_omega_a=None):
    """`(ω_s, ω_a)` de TRT. Tres modos, en orden de prioridad:

    1. `trt_omega_a` dado ⇒ **ω_a FIJO** (Λ queda derivado y variable). Es el
       modo de producción a viscosidad evanescente: ver `TRT_OMEGA_A`.
    2. `trt_lambda` dado ⇒ Λ fijo: `τ_a = ½ + Λ/(τ_s − ½)`, es decir
       `Λ = (τ_s−½)(τ_a−½)`, la definición de Ginzburg & d'Humières.
    3. ninguno de los dos ⇒ `τ_a = τ_s`, **TRT degenerado a BGK**: es el caso
       que VERIFICA la descomposición (`collision_equivalence`), no un modo
       de producción.

    Funciona con `tau_s` escalar o con un campo (el τ efectivo del LES)."""
    ws = 1.0 / tau_s
    if trt_omega_a is not None:
        return ws, float(trt_omega_a)
    if trt_lambda is None:
        return ws, ws
    d = tau_s - 0.5
    return ws, d / (0.5 * d + float(trt_lambda))


def trt_lambda_effective(tau_s: float, trt_lambda=TRT_LAMBDA,
                         trt_omega_a=TRT_OMEGA_A) -> float:
    """Λ = (τ_s−½)(τ_a−½) que de verdad se está usando. Para el reporte."""
    _, wa = _trt_rates(float(tau_s), trt_lambda, trt_omega_a)
    return float((float(tau_s) - 0.5) * (1.0 / float(wa) - 0.5))


def _slices(n: int, s: int):
    """Rebanadas (destino, origen) de un desplazamiento periódico de paso s."""
    if s == 0:
        return ((slice(None), slice(None)),)
    if s > 0:
        return ((slice(s, None), slice(0, n - s)),
                (slice(0, s), slice(n - s, None)))
    return ((slice(0, n + s), slice(-s, None)),
            (slice(n + s, None), slice(0, -s)))


def stream_into(dst, src, c, shape):
    """`dst[x] = src[x − c]` con envolvente periódica, sin temporales.

    El contorno exterior del dominio es SIEMPRE sólido (capa de guarda), así
    que la envolvente periódica solo conecta sólido con sólido y no filtra
    masa. Esa capa de guarda es además lo que permite medir el flujo por
    cortes de semiespacio sin que la costura periódica los contamine."""
    for d0, s0 in _slices(shape[0], c[0]):
        for d1, s1 in _slices(shape[1], c[1]):
            for d2, s2 in _slices(shape[2], c[2]):
                dst[d0, d1, d2] = src[s0, s1, s2]
    return dst


# ==========================================================================
# 3. Núcleo D3Q19 (BGK / TRT / MRT) en NumPy
# ==========================================================================
class LBMSolver:
    """D3Q19 con colisión BGK, TRT o MRT (+LES Smagorinsky) sobre NumPy o CuPy.

    UNA sola implementación para los dos backends: todo el álgebra se escribe
    contra el alias `xp`, que es `numpy` o `cupy`. No hay dos kernels que
    puedan divergir; lo que cambia de un backend a otro está aislado en
    `_to_device`, `_bounce_back`, `_update_effusion` y `_sync`.

    Los tres operadores comparten TODO menos la línea de la colisión
    (`_collide_*`): mismos momentos, mismo equilibrio, mismo LES, mismo
    bounce-back, mismo streaming. `collision="bgk"` recorre exactamente el
    mismo camino de código que antes de que existieran TRT y MRT —misma
    aritmética, mismo orden de operaciones— y por tanto sigue siendo
    reproducible bit a bit.

    Parámetros
    ----------
    solid : ndarray bool (nx, ny, nz)
        Máscara de sólido (SIEMPRE de NumPy: la voxelización es de host). El
        contorno exterior DEBE ser sólido.
    patches : list[dict]
        Condiciones de contorno. Cada una:
        `dict(kind="velocity"|"massflux"|"pressure", idx=..., nb=...,
        u=(ux,uy,uz), rho=...)` con `idx` índices planos de los nodos del
        parche y `nb` los de su vecino fluido (extrapolación de no-equilibrio
        de Guo). Los índices llegan de host y se suben a la GPU aquí.
    tau : float
        Tiempo de relajación de lattice (τ > 0.5). Con TRT es τ_s (el que
        fija la viscosidad); con MRT, el de los momentos de cortadura.
    collision : {"bgk", "trt", "mrt"}
        Operador de colisión. `None` → `resolve_collision()`
        (`PHYCB_LBM_COLLISION` o `DEFAULT_COLLISION`).
    trt_lambda : float | None
        Parámetro mágico Λ de TRT (ver `TRT_LAMBDA`). `None` ⇒ τ_a = τ_s, que
        degenera TRT en BGK — es el caso de VERIFICACIÓN, no de producción.
    effusion : dict | None
        Modelo de transpiración distribuida (ver `_effusion_sites`).
    xp : module | None
        `numpy` (defecto) o `cupy`. Quien decide es `resolve_backend`.
    dtype : np.float32 | np.float64
        Precisión de trabajo. FP32 por defecto en AMBOS backends: es lo que
        pide Ciencia §11.2 (el FP64 de una GPU de consumo va a 1/64 del FP32)
        y lo que ya usaba el camino NumPy. FP64 disponible por parámetro.
    acc_dtype : np.float64 | np.float32
        Precisión del acumulador del promedio temporal. FP64 por defecto
        (400 sumas en FP32 arrastran ~2e-6 relativo y el cierre de masa de
        T41 se pide a 1e-3); en GPU cuesta 152 B/nodo, así que
        `solve_cold` lo baja a FP32 **con aviso** si no cabe.

    **Nunca lanza por dependencias**; sí lanza `FloatingPointError` si la
    simulación diverge — quien llama lo traduce en degradación etiquetada.
    """

    def __init__(self, solid, patches, tau, *, dtype=np.float32,
                 cs_smago=0.0, effusion=None, xp=None, acc_dtype=np.float64,
                 collision=None, trt_lambda=TRT_LAMBDA,
                 trt_omega_a=TRT_OMEGA_A):
        self.xp = xp if xp is not None else np
        self.on_gpu = self.xp is not np
        self.dt = dtype
        self.acc_dt = acc_dtype
        self.shape = tuple(solid.shape)
        self.solid = solid                       # host: lo usan las medidas
        self.patches = [self._patch_to_device(p) for p in (patches or ())]
        self.tau = dtype(tau)
        self.cs = float(cs_smago)
        self.effusion = effusion
        self.collision = resolve_collision(collision)
        self.trt_lambda = None if trt_lambda is None else float(trt_lambda)
        self.trt_omega_a = None if trt_omega_a is None else float(trt_omega_a)
        self.collision_label = collision_label(self.collision, self.trt_lambda,
                                               self.trt_omega_a)
        xp_ = self.xp
        n = self.shape
        #  Bounce-back por ÍNDICES y no por máscara booleana: en CuPy un
        #  `a[:, mask_bool]` relanza un `nonzero()` en cada paso (y sincroniza);
        #  con el índice precalculado son 19 gather+scatter y ningún sync. En
        #  NumPy además evita el `f[list(OPP)]` que copiaba 19·N flotantes por
        #  paso. Resultado bit-idéntico en ambos casos.
        #  int32 y no intp: el índice se lee 38 veces por paso en el
        #  bounce-back y a 8 B/entrada pesa más que las propias poblaciones.
        n_tot = int(np.prod(self.shape))
        idx_dt = np.int32 if n_tot < 2 ** 31 - 1 else np.int64
        self.solid_idx = self._to_device(
            np.flatnonzero(solid.reshape(-1)).astype(idx_dt))
        self.n_solid = int(self.solid_idx.size)
        self.f = xp_.empty((19,) + n, dtype=dtype)
        self.g = xp_.empty((19,) + n, dtype=dtype)
        #  ρ y u comparten UN buffer contiguo (4, nx, ny, nz): así `_macro`
        #  es un solo producto de matrices `M4 · f` que escribe los cuatro
        #  momentos de una vez, en vez de 60 sumas parciales.
        self._mom = xp_.empty((4,) + n, dtype=dtype)
        self.rho = self._mom[0]
        self.u = self._mom[1:]
        self.rho.fill(1.0)
        self.u.fill(0.0)
        m4, c3, m6 = _moment_matrices(dtype)
        self._M4 = self._to_device(m4)
        self._C3 = self._to_device(c3)
        self._M6 = self._to_device(m6) if self.cs > 0.0 else None
        #  w_i/2: la mitad se compensa con la identidad
        #  `f^eq = ½ w ρ [(3c·u + 1)² + 1 − 3u²]`, que ahorra dos pasadas.
        self._Whalf = self._to_device(
            (0.5 * np.array(WI, dtype=dtype)).reshape(19, 1))
        self._W6 = self._to_device(
            np.array(QQ_WEIGHT, dtype=dtype).reshape(6, 1)) \
            if self.cs > 0.0 else None
        self._usq = xp_.empty(n, dtype=dtype)
        self._tmp = xp_.empty(n, dtype=dtype)
        self._q6 = xp_.empty((6, n_tot), dtype=dtype) if self.cs > 0.0 else None
        self._Q = xp_.empty(n, dtype=dtype) if self.cs > 0.0 else None
        self._collide_setup()
        self._eff_m = None               # caudal de efusión por franja [lu]
        self._eff_w = None               # incremento de masa por celda [lu]
        self.eff_total_lat = 0.0
        self.eff_demand_lat = 0.0        # ṁ que pide la ley de orificio
        self.eff_cap_lat = 0.0           # cota de seguridad vigente
        self.eff_cap_hits = 0            # veces que la cota MORDIÓ
        self.eff_cap_updates = 0
        self._eff_prepare()
        self._feq(self.f, self.rho, self.u)

    # -- puente host ↔ device ---------------------------------------------
    def _to_device(self, a):
        """Sube un array de host al backend. En NumPy es la identidad.

        Se llama SOLO en el constructor y en `_eff_prepare`: dentro del bucle
        de pasos no hay ni una transferencia host↔device (era el error clásico
        que anula la ganancia de la GPU)."""
        if not self.on_gpu:
            return a
        return self.xp.asarray(a)

    def _patch_to_device(self, p):
        """Copia del parche con sus índices y velocidades ya en el device."""
        q = dict(p)
        q["idx"] = self._to_device(np.asarray(p["idx"], dtype=np.intp))
        q["nb"] = self._to_device(np.asarray(p["nb"], dtype=np.intp))
        if "u" in q and q["u"] is not None:
            q["u"] = tuple(self._to_device(np.asarray(v, dtype=self.dt))
                           for v in q["u"])
        return q

    def _sync(self):
        """Barrera de device. Obligatoria ANTES de cronometrar: los kernels
        de CuPy son asíncronos y sin esto se mide la latencia de encolado, no
        el cálculo. En NumPy no hace nada."""
        if self.on_gpu:
            self.xp.cuda.runtime.deviceSynchronize()

    # -- equilibrio ------------------------------------------------------
    def _feq(self, out, rho, u):
        """`out := f^eq(ρ, u)` con la expansión de segundo orden estándar.

        Se usa la identidad algebraica

            f^eq_i = w_i ρ (1 − 3/2 u² + 3c_i·u + 9/2 (c_i·u)²)
                   = ½ w_i ρ [ (3c_i·u + 1)² + 1 − 3u² ]

        que permite calcular las 19 poblaciones con **cinco pasadas sobre el
        bloque (19, N)** en vez de nueve por dirección (171 en total). El
        `3c_i·u` sale de un único producto de matrices `C3 · u`. Es idéntica
        término a término a la expansión anterior; lo que cambia es el orden
        de las sumas (diferencias del orden del último bit)."""
        xp, dt = self.xp, self.dt
        ov = out.reshape(19, -1)
        xp.matmul(self._C3, u.reshape(3, -1), out=ov)   # ov := 3·c_i·u
        usq, tmp = self._usq, self._tmp
        xp.multiply(u[0], u[0], out=usq)
        xp.multiply(u[1], u[1], out=tmp)
        usq += tmp
        xp.multiply(u[2], u[2], out=tmp)
        usq += tmp
        usq *= dt(3.0)
        xp.subtract(dt(1.0), usq, out=usq)              # usq := 1 − 3u²
        ov += dt(1.0)
        xp.multiply(ov, ov, out=ov)                     # (3c·u + 1)²
        ov += usq.reshape(1, -1)
        ov *= rho.reshape(1, -1)
        ov *= self._Whalf                               # ½·w_i
        return out

    def _feq_flat(self, rho, u):
        """`f^eq` sobre un subconjunto plano de nodos (parches).

        Misma identidad que `_feq`. `u` puede ser un array (3, n) o una tupla
        de tres arrays (lo que produce `_patch`)."""
        xp, dt = self.xp, self.dt
        U = u if getattr(u, "ndim", 0) == 2 else xp.stack(tuple(u))
        out = xp.matmul(self._C3, U)                    # (19, n)
        usq = dt(1.0) - dt(3.0) * (U[0] * U[0] + U[1] * U[1] + U[2] * U[2])
        out += dt(1.0)
        xp.multiply(out, out, out=out)
        out += usq.reshape(1, -1)
        out *= rho.reshape(1, -1)
        out *= self._Whalf
        return out

    # -- momentos y contorno ---------------------------------------------
    def _macro(self):
        """ρ y u de un solo producto de matrices `M4 · f`.

        ⚠️ ESTA es la contracción que NO es bit-idéntica entre backends: el
        GEMM de cuBLAS suma en un orden distinto al de OpenBLAS. Declarado en
        el docstring del módulo; la diferencia observada es ≲1e-5 relativo."""
        xp = self.xp
        xp.matmul(self._M4, self.f.reshape(19, -1),
                  out=self._mom.reshape(4, -1))
        xp.maximum(self.rho, self.dt(1e-6), out=self.rho)
        self.u /= self.rho                       # (3, N) / (N) por difusión

    def _apply_bc(self):
        """Guo: `f_b = f^eq(ρ_b, u_b) + [f_f − f^eq(ρ_f, u_f)]` (2.º orden)."""
        xp, dt = self.xp, self.dt
        fv = self.f.reshape(19, -1)
        rv = self.rho.reshape(-1)
        uv = self.u.reshape(3, -1)
        for p in self.patches:
            idx, nb = p["idx"], p["nb"]
            if idx.size == 0:
                continue
            rho_nb = rv[nb]
            un = uv[:, nb]
            fneq = fv[:, nb] - self._feq_flat(rho_nb, un)
            if p["kind"] in ("velocity", "massflux"):
                ub = p["u"]
                if p["kind"] == "massflux":
                    #  BC de FLUJO MÁSICO: lo que se impone es ρ·u, no u. Con
                    #  una BC de velocidad pura el caudal inyectado es ρ_b·u
                    #  y ρ_b sube con la presión del anillo — en el ancla eso
                    #  metía un 21 % de aire de más. Dividir por ρ del vecino
                    #  deja ρ·u ≡ el objetivo por construcción.
                    ub = tuple(v / xp.maximum(rho_nb, dt(1e-6)) for v in ub)
                fv[:, idx] = self._feq_flat(rho_nb, ub) + fneq
                rv[idx] = rho_nb
                for k in range(3):
                    uv[k, idx] = ub[k]
            else:
                rb = xp.full(int(idx.size), dt(p["rho"]), dtype=dt)
                fv[:, idx] = self._feq_flat(rb, un) + fneq
                rv[idx] = rb
                for k in range(3):
                    uv[k, idx] = un[k]

    # -- efusión (transpiración distribuida) ------------------------------
    def _eff_prepare(self):
        """Precalcula la contabilidad de la efusión en forma VECTORIAL.

        El código original recorría las franjas una a una haciendo
        `rho[cold].mean()` por franja. En NumPy es aceptable; en GPU son 2·N
        reducciones diminutas **y 2·N sincronizaciones** cada `EFFUSION_EVERY`
        pasos — mata el backend. Aquí se sustituye por: una sola recolección
        `rho[idx]`, un solo `bincount` segmentado y UNA transferencia de
        2·n_franjas flotantes al host. Mismo modelo, misma física.

        Efecto colateral (declarado): el `bincount` acumula en FP64 mientras
        que el `.mean()` por franja acumulaba en la precisión de trabajo, así
        que los ṁ de efusión cambian en el último dígito respecto a la versión
        anterior del módulo. Es más preciso, no menos."""
        eff = self.effusion
        if not eff:
            return
        n = len(eff["slabs"])
        idx = np.asarray(eff["idx"], dtype=np.intp)
        #  `fv[i, idx] += ...` solo es un scatter-add correcto si los índices
        #  son ÚNICOS (con repetidos, NumPy y CuPy se quedan con una sola
        #  escritura y habría que pasar por `cupyx.scatter_add`/`np.add.at`).
        #  Lo son por construcción — `cold` y `gas` son disjuntos y cada franja
        #  vive en un plano z distinto —, pero se comprueba UNA vez aquí en
        #  vez de confiar en ello.
        if np.unique(idx).size != idx.size:             # pragma: no cover
            raise ValueError("índices de efusión repetidos: haría falta "
                             "scatter-add (cupyx.scatter_add / np.add.at)")
        seg = np.empty(idx.size, dtype=np.intp)
        counts = np.empty(2 * n, dtype=np.float64)
        for s, sl in enumerate(eff["slabs"]):
            seg[sl["slice_cold"]] = 2 * s
            seg[sl["slice_gas"]] = 2 * s + 1
            counts[2 * s] = float(sl["n_cold"])
            counts[2 * s + 1] = float(sl["n_gas"])
        self._eff_idx = self._to_device(idx)
        self._eff_seg = self._to_device(seg)
        self._eff_counts = counts
        self._eff_A_lat = np.array([sl["A_lat"] for sl in eff["slabs"]],
                                   dtype=np.float64)
        self._eff_nseg = 2 * n

    def _update_effusion(self):
        """ṁ de cada franja de efusión desde el Δp LOCAL que resuelve el LBM.

        Ley de orificio de Ciencia §4.1 aplicada franja a franja:
        ṁ = C_d·A·√(2ρ Δp). El Δp es el que hay entre el anillo y el liner en
        ESA cota axial, así que el reparto de la efusión a lo largo del liner
        es un RESULTADO del campo, no un dato. El vector de incrementos por
        celda se precalcula UNA vez cada `EFFUSION_EVERY` pasos para que
        aplicarlo cueste 19 indexados y no 19·n_franjas."""
        eff = self.effusion
        if not eff:
            return
        xp = self.xp
        rv = self.rho.reshape(-1)
        sums = xp.bincount(self._eff_seg, weights=rv[self._eff_idx],
                           minlength=self._eff_nseg)
        #  ÚNICA transferencia device→host del bucle: 2·n_franjas flotantes
        #  (unos cientos de bytes), una vez cada EFFUSION_EVERY pasos.
        sums_h = np.asarray(sums.get() if self.on_gpu else sums,
                            dtype=np.float64)
        mean = sums_h / np.maximum(self._eff_counts, 1.0)
        dp = np.maximum((mean[0::2] - mean[1::2]) * CS2, 0.0)
        m = eff["cd"] * self._eff_A_lat * np.sqrt(2.0 * dp)
        cap = EFFUSION_MAX_FRAC * eff["mdot_cap_lat"]
        tot = float(m.sum())
        #  Si la cota de seguridad MUERDE, el ṁ de efusión ya no lo fija la
        #  ley de orificio sino el recorte: el número deja de ser una medida.
        #  Se cuenta y se reporta (`eff_cap_hits`) en vez de dejarlo mudo.
        self.eff_demand_lat = tot
        self.eff_cap_lat = float(cap)
        if tot > cap > 0.0:
            m *= cap / tot
            self.eff_cap_hits += 1
        self.eff_cap_updates += 1
        self._eff_m = m if self._eff_m is None else (
            (1.0 - EFFUSION_RELAX) * self._eff_m + EFFUSION_RELAX * m)
        coeff = np.empty(self._eff_nseg, dtype=np.float64)
        coeff[0::2] = -self._eff_m / np.maximum(self._eff_counts[0::2], 1.0)
        coeff[1::2] = self._eff_m / np.maximum(self._eff_counts[1::2], 1.0)
        self._eff_w = self._to_device(coeff.astype(self.dt))[self._eff_seg]
        self.eff_total_lat = float(np.sum(self._eff_m))

    def _apply_effusion(self):
        """Aplica el trasvase: sumidero isótropo en el anillo, fuente isótropa
        en el liner. La cantidad de movimiento del chorro de efusión NO se
        modela (a dx ≥ 1 mm un orificio de 0.6 mm no existe): es película
        pegada a la pared y su empuje es despreciable frente al de los
        chorros de dilución. **Limitación declarada.**"""
        if self.effusion is None or self._eff_m is None:
            return
        fv = self.f.reshape(19, -1)
        idx, w = self._eff_idx, self._eff_w
        for i in range(19):
            fv[i, idx] += self.dt(WI[i]) * w

    # -- paso -------------------------------------------------------------
    def _bounce_back(self, gv, fv):
        """Bounce-back de media celda: el nodo sólido devuelve lo que le llegó.

        Se hace población a población sobre el ÍNDICE de nodos sólidos. La
        versión con máscara booleana (`gv[:, solid] = fv[OPP][:, solid]`)
        materializaba una copia de 19·N flotantes por paso en NumPy y, en
        CuPy, además volvía a resolver la máscara en cada paso. El resultado
        numérico es idéntico: mismas parejas (i, OPP[i]) y mismos nodos."""
        si = self.solid_idx
        if si.size == 0:
            return
        for i in range(19):
            gv[i, si] = fv[OPP[i], si]

    def step(self, n_step: int = 0):
        xp = self.xp
        f, g = self.f, self.g
        if self.effusion is not None:
            #  ANTES de los momentos: así el trasvase entra en ρ y u de este
            #  mismo paso y basta UN cálculo de momentos por paso.
            if n_step % EFFUSION_EVERY == 0:
                self._update_effusion()
            self._apply_effusion()
        self._macro()
        self._apply_bc()
        #  BGK escribe el no-equilibrio en `g` (y colisiona ahí mismo, in
        #  situ, sin un tercer bloque de 19·N); TRT y MRT necesitan LEER f^neq
        #  entero mientras ESCRIBEN el post-colisión, así que su f^neq va al
        #  buffer auxiliar `_scr`. Es la única diferencia estructural.
        nq = g if self._scr is None else self._scr
        self._feq(nq, self.rho, self.u)              # nq := f^eq
        xp.subtract(f, nq, out=nq)                   # nq := f^neq
        omega = self._omega_les(nq) if self.cs > 0.0 else self._omega0
        self._collide(nq, omega)                     # g := post-colisión
        self._bounce_back(g.reshape(19, -1), f.reshape(19, -1))
        for i, c in enumerate(CI):
            if c == (0, 0, 0):
                xp.copyto(f[i], g[i])
            else:
                stream_into(f[i], g[i], c, self.shape)

    # -- colisión ---------------------------------------------------------
    def _collide_setup(self):
        """Reserva y precalcula lo que necesita el operador de colisión.

        Con BGK no reserva NADA (`_scr = None`) y `step` recorre exactamente
        el mismo camino que antes de que existieran TRT y MRT.

        Con tasas UNIFORMES (sin LES) la colisión entera es una sola matriz
        19×19 `A` —`f ← f − A·f^neq`—, así que TRT y MRT se reducen a un
        único GEMM `(19,19)·(19,N)` que mueve exactamente la misma memoria
        que el `g *= ω` de BGK: **coste idéntico**. Con LES la tasa de
        cortadura cambia célula a célula y `A` deja de ser constante; ahí TRT
        usa la descomposición por pares (tres pasadas) y MRT los dos
        productos por `M` y `M⁻¹` (que es su coste inherente)."""
        xp, dt = self.xp, self.dt
        self._omega0 = dt(1.0) / self.tau
        self._A = self._M = self._Minv = self._Sconst = None
        if self.collision == COLLISION_BGK:
            self._scr = None
            return
        self._scr = xp.empty((19,) + self.shape, dtype=self.dt)
        if self.cs <= 0.0:                    # tasas uniformes ⇒ una matriz
            self._A = self._to_device(_collision_matrix(
                self.collision, float(self.tau), self.trt_lambda, self.dt,
                self.trt_omega_a))
            return
        if self.collision == COLLISION_MRT:
            M, Minv, _ = _mrt_matrices(self.dt)
            self._M = self._to_device(M)
            self._Minv = self._to_device(Minv)
            s = _mrt_rate_vector(float(self.tau))
            s[list(MRT_SHEAR_IDX)] = 1.0     # los pone el LES por celda
            self._Sconst = self._to_device(s.astype(self.dt).reshape(19, 1))

    def _collide(self, nq, omega):
        """`g := f − A·f^neq`. Despacha al operador activo."""
        if self.collision == COLLISION_BGK:
            return self._collide_bgk(nq, omega)
        if self._A is not None:
            return self._collide_matrix(nq)
        if self.collision == COLLISION_TRT:
            return self._collide_trt(nq, omega)
        return self._collide_mrt(nq, omega)

    def _collide_bgk(self, g, omega):
        """BGK de un solo tiempo de relajación. **Intacto desde el origen.**

        `g` ES el buffer de no-equilibrio y el de salida a la vez: se
        multiplica in situ por ω y se resta de f. Dos pasadas sobre 19·N."""
        g *= omega
        self.xp.subtract(self.f, g, out=g)           # g := f + ω(f^eq − f)
        return g

    def _collide_matrix(self, nq):
        """Colisión de tasas UNIFORMES: un único GEMM `A·f^neq`.

        Vale para TRT (`A = a·I + b·P`) y para MRT (`A = M⁻¹·diag(s)·M`) cuando
        no hay LES. Mueve 19·N de lectura y 19·N de escritura, igual que el
        `g *= ω` de BGK, y los 361·N FMA del GEMM caen muy por debajo del
        techo de cómputo (intensidad aritmética ≈ 2.4 flop/B frente a los
        ~33 flop/B del punto de equilibrio de la GPU): el coste es el mismo."""
        xp = self.xp
        g = self.g
        xp.matmul(self._A, nq.reshape(19, -1), out=g.reshape(19, -1))
        xp.subtract(self.f, g, out=g)
        return g

    def _pair_views(self, a):
        """`(vista (9,2,N), vista con el par INTERCAMBIADO)` de `a[1:]`.

        Los pares de direcciones opuestas del D3Q19 son consecutivos en `CI`
        (comprobado en el import, `_PAIRS_ADJACENT`), así que `a[1:]` —que es
        contiguo— se reinterpreta como `(9, 2, N)` sin copiar y el
        intercambio `i ↔ ī` es un simple `[:, ::-1]`, una vista de paso
        negativo. Ni permutaciones, ni `take`, ni un solo byte movido."""
        v = a[1:].reshape(9, 2, -1)
        return v, v[:, ::-1]

    def _collide_trt(self, nq, omega):
        """TRT con LES: τ_s lo pone el LES célula a célula y τ_a sale de Λ.

        Descomposición simétrica/antisimétrica sobre los pares opuestos:

            f⁺ = (f_i + f_ī)/2      f⁻ = (f_i − f_ī)/2   (ídem f^eq)
            f_i ← f_i − (f⁺−f^eq⁺)/τ_s − (f⁻−f^eq⁻)/τ_a

        que, sustituyendo y agrupando, es exactamente

            f_i ← f_i − a·f^neq_i − b·f^neq_ī
            a = (ω_s+ω_a)/2      b = (ω_s−ω_a)/2

        Es decir: NO hay que construir f⁺ y f⁻ explícitamente (serían dos
        bloques de 19·N más y cuatro pasadas de memoria). Tres pasadas sobre
        el bloque de pares y se acabó.

        τ_a se deduce de Λ = (τ_s−½)(τ_a−½) ⇒ ω_a = (τ_s−½)/(½(τ_s−½)+Λ),
        con τ_s = el τ EFECTIVO del LES (τ molecular + τ turbulento): así el
        Λ pedido se respeta en cada celda, que es lo que mantiene la pared de
        bounce-back en su sitio también donde el LES está activo."""
        xp, dt = self.xp, self.dt
        f, g, scr = self.f, self.g, nq
        #  ω_s es `omega` (campo) y el τ_s EFECTIVO quedó en `self._Q`.
        tmp, usq = self._tmp, self._usq
        xp.multiply(scr[0], omega, out=g[0])         # partícula en reposo: ω_s
        if self.trt_omega_a is not None:
            #  ω_a FIJO: no depende del τ del LES, así que es un escalar y
            #  el cálculo de a y b se queda en dos pasadas de tamaño N.
            wa = dt(self.trt_omega_a)
            xp.add(omega, wa, out=tmp)
            tmp *= dt(0.5)                           # tmp := a
            xp.subtract(omega, wa, out=omega)
            omega *= dt(0.5)                         # omega := b
            return self._trt_mix(scr, tmp, omega)
        if self.trt_lambda is None:
            wa = omega                               # τ_a = τ_s ⇒ TRT ≡ BGK
        else:
            xp.subtract(self._Q, dt(0.5), out=tmp)   # tmp := τ_s − ½
            xp.multiply(tmp, dt(0.5), out=usq)
            usq += dt(self.trt_lambda)
            xp.divide(tmp, usq, out=usq)             # usq := ω_a
            wa = usq
        xp.add(omega, wa, out=tmp)
        tmp *= dt(0.5)                               # tmp := a
        if wa is omega:
            #  b ≡ 0 (τ_a = τ_s): se calcula el cero explícitamente para que
            #  el camino de código sea EL MISMO que con b ≠ 0 y la
            #  verificación de equivalencia con BGK no se apoye en un atajo.
            xp.multiply(omega, dt(0.0), out=usq)
            b = usq
        else:
            xp.subtract(omega, wa, out=omega)
            omega *= dt(0.5)                         # omega := b
            b = omega
        return self._trt_mix(scr, tmp, b)

    def _trt_mix(self, scr, a, b):
        """`g := f − a·f^neq_i − b·f^neq_ī` sobre los pares opuestos.

        Cuatro pasadas sobre el bloque de 19·N (BGK necesita dos): es el
        MÍNIMO alcanzable con primitivas de array cuando `a` y `b` son campos
        —lo son en cuanto el LES está activo—, porque `x = f − a·s − b·s̄` son
        cuatro operandos y ni NumPy ni CuPy tienen un `axpy` fusionado.
        Perfilado en esta máquina (RTX 3070 Ti, 6.48 M nodos, FP32):
        2.51 + 2.51 + 2.51 + 2.65 = 10.2 ms frente a los 5.3 ms de BGK, con
        el ancho de banda a ~466 GB/s de los 608 de pico. Sin LES las tasas
        son uniformes y la colisión entera es UN GEMM (`_collide_matrix`),
        que cuesta lo mismo que BGK."""
        xp = self.xp
        sp, spw = self._pair_views(scr)
        gp, _ = self._pair_views(self.g)
        xp.multiply(sp, a.reshape(1, 1, -1), out=gp)     # gp := a·f^neq
        sp *= b.reshape(1, 1, -1)                        # sp := b·f^neq
        gp += spw                                        # + del par opuesto
        xp.subtract(self.f, self.g, out=self.g)
        return self.g

    def _collide_mrt(self, nq, omega):
        """MRT con LES: `f ← f − M⁻¹·S·M·f^neq`, con S diagonal.

        Dos productos matriciales 19×19 por celda (ida a espacio de momentos
        y vuelta) más la relajación diagonal. Los momentos de cortadura
        (`MRT_SHEAR_IDX`) llevan la ω del LES célula a célula; el resto, las
        tasas fijas de `MRT_S_*`. Ver `_mrt_matrices` para la matriz y su
        fuente."""
        xp = self.xp
        f, g, scr = self.f, self.g, nq
        gv, sv = g.reshape(19, -1), scr.reshape(19, -1)
        xp.matmul(self._M, sv, out=gv)               # gv := m^neq
        gv *= self._Sconst                    # tasas fijas (1 en cortadura)
        om = omega.reshape(-1)
        for k in MRT_SHEAR_IDX:
            gv[k] *= om
        xp.matmul(self._Minv, gv, out=sv)            # sv := M⁻¹·S·m^neq
        xp.subtract(f, scr, out=g)
        return g

    def _omega_les(self, fneq):
        """ω efectiva del LES de Smagorinsky sobre el tensor de no-equilibrio.

        τ_t = ½(√(τ² + 18√2·C_s²·|Q|/ρ) − τ) con
        Q_ij = Σ_a c_ai c_aj (f_a − f_a^eq)  [Hou et al. 1996].

        Las seis componentes independientes de Q salen de UN producto de
        matrices `M6 · f^neq` (antes: seis bucles de 19 sumas parciales,
        ~110 operaciones sobre el campo completo)."""
        xp, dt = self.xp, self.dt
        Q, q6 = self._Q, self._q6
        xp.matmul(self._M6, fneq.reshape(19, -1), out=q6)
        xp.multiply(q6, q6, out=q6)
        q6 *= self._W6                       # ×2 fuera de la diagonal
        xp.sum(q6, axis=0, out=Q.reshape(-1))
        xp.sqrt(Q, out=Q)
        Q *= dt(18.0 * math.sqrt(2.0) * self.cs ** 2)
        Q /= self.rho
        Q += self.tau * self.tau
        xp.sqrt(Q, out=Q)
        Q += self.tau
        Q *= dt(0.5)
        return xp.reciprocal(Q)

    # -- bucle ------------------------------------------------------------
    def run(self, steps_max=LBM_STEPS_MAX, avg=LBM_AVG_STEPS,
            check_every=LBM_CHECK_EVERY, tol=LBM_CONV_TOL,
            deadline=None, log=None):
        """Avanza hasta el estacionario estadístico y promedia `avg` pasos.

        Bucle ACOTADO (contrato §10.5): `steps_max` iteraciones como techo,
        criterio de convergencia sobre `<|u|>` y `deadline` de pared duro.
        Devuelve el promedio de `g` (post-colisión), que es lo que transporta
        el streaming y por tanto lo que da flujos EXACTOS.

        `self.gbar` es SIEMPRE un array de **NumPy en host**, venga de donde
        venga el cálculo: todo el post-proceso (`region_flux`, `cold_post`)
        trabaja con máscaras booleanas de host, y esa frontera única evita
        tener que duplicarlo. La transferencia ocurre una sola vez, al final,
        fuera del bucle."""
        xp = self.xp
        acc = xp.zeros((19,) + self.shape, dtype=self.acc_dt)
        n_avg = 0
        prev = None
        n_trans = steps_max
        converged = False
        self._sync()                     # nada pendiente antes de cronometrar
        t0 = time.monotonic()
        hist = []
        for s in range(int(steps_max)):
            self.step(s)
            if (s + 1) % check_every == 0:
                #  `float()` de un array 0-d de CuPy SINCRONIZA: es el único
                #  punto de sincronización del transitorio, y ocurre una vez
                #  cada `check_every` pasos, no cada paso.
                cur = float(xp.abs(self.u).mean())
                hist.append(cur)
                if not math.isfinite(cur):
                    raise FloatingPointError(
                        f"LBM divergido en el paso {s + 1} (τ={float(self.tau):.5f}, "
                        "baja u_lattice o sube TAU_FLOOR)")
                if prev is not None and abs(cur - prev) <= tol * max(cur, 1e-30):
                    n_trans, converged = s + 1, True
                    break
                prev = cur
            if deadline is not None and time.monotonic() > deadline:
                n_trans = s + 1
                break
        for _ in range(int(avg)):
            self.step(n_trans + _)
            acc += self.g
            n_avg += 1
            if deadline is not None and time.monotonic() > deadline:
                break
        self._sync()                     # el cronómetro mide cálculo, no cola
        el = time.monotonic() - t0
        tot = n_trans + n_avg
        acc /= self.acc_dt(max(n_avg, 1))            # in-place: no duplica VRAM
        self.gbar = np.asarray(acc.get() if self.on_gpu else acc,
                               dtype=np.float64)
        del acc
        self._free_pool()
        self.stats = dict(
            steps=int(tot), steps_transient=int(n_trans), steps_avg=int(n_avg),
            converged=bool(converged), wall_s=float(el),
            mlups=float(tot * self.solid.size / 1e6 / max(el, 1e-9)),
            s_per_cell_step=float(el / max(tot * self.solid.size, 1)),
            backend=BACKEND_CUPY if self.on_gpu else BACKEND_NUMPY,
            collision=self.collision, scheme=self.collision_label,
            trt_lambda=self.trt_lambda, tau=float(self.tau),
            dtype=str(np.dtype(self.dt)), acc_dtype=str(np.dtype(self.acc_dt)),
            u_mean_history=[float(x) for x in hist[-12:]])
        if log:
            log(f"    LBM {tot} pasos ({n_trans} transitorio), "
                f"{self.solid.size} nodos, {el:.1f} s, "
                f"{self.stats['mlups']:.2f} MLUPS "
                f"({self.stats['s_per_cell_step']:.3e} s/celda/paso, "
                f"{self.stats['backend']}, {self.collision_label}, "
                f"tau={float(self.tau):.6f}), "
                f"{'convergido' if converged else 'NO convergido'}")
        return self.gbar

    def _free_pool(self):
        """Suelta los buffers del solver y vacía el pool de CuPy.

        Sin esto, el pool de memoria de CuPy se queda con los ~2.5 GiB del
        caso ancla hasta que muere el proceso, y una segunda llamada a
        `solve_cold` (o el propio post-proceso) se queda sin VRAM.

        **Consecuencia: `run()` es de un solo uso.** Tras devolver `gbar` el
        solver ya no puede avanzar más pasos; lo que sobrevive es `gbar`,
        `stats`, `solid` y `eff_total_lat`, que es todo lo que consume el
        post-proceso."""
        for a in ("f", "g", "rho", "u", "_mom", "_tmp", "_usq", "_q6", "_Q",
                  "_M4", "_C3", "_M6", "_W6", "_Whalf",
                  "_scr", "_A", "_M", "_Minv", "_Sconst",
                  "_eff_w", "_eff_idx", "_eff_seg", "solid_idx"):
            setattr(self, a, None)
        self.patches = []
        if self.on_gpu:
            try:
                self.xp.get_default_memory_pool().free_all_blocks()
            except Exception:
                pass


# ==========================================================================
# 4. Herramientas de medida sobre el campo promediado
# ==========================================================================
def macro_from_g(gbar):
    """(ρ, j) macroscópicos del promedio de poblaciones. `j = ρu`."""
    rho = gbar.sum(0)
    j = np.zeros((3,) + rho.shape, dtype=gbar.dtype)
    for k in range(3):
        for i, c in enumerate(CI):
            if c[k] == 1:
                j[k] += gbar[i]
            elif c[k] == -1:
                j[k] -= gbar[i]
    return rho, j


def region_flux(gbar, A, B, shape) -> float:
    """Flujo másico **exacto** (lattice) de la región A a la B.

    Suma las poblaciones que el streaming lleva de A a B menos las que lleva
    de B a A. Es la misma contabilidad del solver ⇒ conserva masa a precisión
    de máquina (T41 cierra a ~3e-5 frente al 2e-2 de interpolar ρu)."""
    tot = 0.0
    tmp = np.empty_like(A)
    for i, c in enumerate(CI):
        if c == (0, 0, 0):
            continue
        neg = (-c[0], -c[1], -c[2])
        stream_into(tmp, B, neg, shape)             # tmp[x] ⇔ x+c ∈ B
        tot += float(gbar[i][A & tmp].sum())
        stream_into(tmp, A, neg, shape)             # tmp[x] ⇔ x+c ∈ A
        tot -= float(gbar[i][B & tmp].sum())
    return tot


def halfspace(fluid, axis: int, k: int):
    """Corte de semiespacio: (aguas arriba, aguas abajo) del índice `k`."""
    a = np.zeros_like(fluid)
    b = np.zeros_like(fluid)
    sa = [slice(None)] * 3
    sb = [slice(None)] * 3
    sa[axis] = slice(0, k)
    sb[axis] = slice(k, None)
    a[tuple(sa)] = True
    b[tuple(sb)] = True
    return a & fluid, b & fluid


def flat_idx(mask, off=None):
    """Índices planos de `mask`; con `off`, los de sus vecinos desplazados."""
    if off is None:
        return np.flatnonzero(mask.reshape(-1))
    i3 = np.array(np.nonzero(mask)) + np.asarray(off, dtype=np.intp)[:, None]
    return np.ravel_multi_index(i3, mask.shape)


# ==========================================================================
# 5. Conversión de unidades
# ==========================================================================
def mu_air(T_K: float) -> float:
    """µ del aire por Sutherland [Pa·s]. Mismas constantes que `crn_core`."""
    T = max(float(T_K), 100.0)
    return MU_AIR_REF * (T / T_MU_REF) ** 1.5 * (T_MU_REF + SUTHERLAND_S) \
        / (T + SUTHERLAND_S)


def lattice_units(dx_m: float, U_in: float, rho3: float, T3: float,
                  u_lat: float = U_LATTICE_TARGET,
                  collision: str | None = COLLISION_BGK) -> dict:
    """Factores de conversión lattice ↔ SI anclados en la velocidad de entrada.

    Devuelve también `tau_molecular` (el que pide la viscosidad real) y `tau`
    (el que se USA, suelado en el `TAU_FLOOR_*` **del operador**): la
    diferencia entre ambos es la honestidad del nivel. El defecto es
    `COLLISION_BGK` a propósito — quien no pase operador obtiene los mismos
    números que antes."""
    U_in = max(float(U_in), 1e-9)
    dx_m = max(float(dx_m), 1e-9)
    C_u = U_in / max(u_lat, 1e-9)
    dt_s = dx_m / C_u
    nu_phys = mu_air(T3) / max(rho3, 1e-9)
    nu_lat = nu_phys / (C_u * dx_m)
    tau_mol = 3.0 * nu_lat + 0.5
    floor = tau_floor(collision)
    tau = max(tau_mol, floor)
    nu_used = (tau - 0.5) / 3.0
    return dict(
        dx_m=float(dx_m), dt_s=float(dt_s), C_u=float(C_u),
        u_lat_in=float(u_lat), rho_ref_kgm3=float(rho3),
        nu_phys_m2s=float(nu_phys), nu_lat=float(nu_lat),
        nu_lat_used=float(nu_used),
        collision=resolve_collision(collision), tau_floor=float(floor),
        tau_molecular=float(tau_mol), tau=float(tau),
        tau_floored=bool(tau > tau_mol + 1e-15),
        mdot_scale=float(rho3 * C_u * dx_m * dx_m),   # ṁ_lat → kg/s
        p_scale=float(rho3 * C_u * C_u),              # Δρ/3 → Pa
        u_scale=float(C_u))                           # u_lat → m/s


# ==========================================================================
# 6. Casos canónicos (verificación T41–T43)
# ==========================================================================
def _empty_box(nx, ny, nz):
    """Caja con capa de guarda sólida en las seis caras."""
    solid = np.ones((nx, ny, nz), dtype=bool)
    solid[1:-1, 1:-1, 1:-1] = False
    return solid


def _patch(mask, kind, off, u=None, rho=None, dtype=np.float32):
    """Parche de contorno. `kind`: `"velocity"` (u impuesta), `"massflux"`
    (ρ·u impuesta — lo que hay que usar cuando el caudal es el DATO) o
    `"pressure"` (ρ impuesta). `off` es el desplazamiento al vecino fluido."""
    n = int(mask.sum())
    p = dict(kind=kind, idx=flat_idx(mask), nb=flat_idx(mask, off))
    if kind in ("velocity", "massflux"):
        p["u"] = tuple(np.full(n, dtype(v), dtype=dtype) for v in u)
    else:
        p["rho"] = float(rho)
    return p


#: Reintentos del solve en GPU antes de degradar a NumPy. Uno: si el driver
#: devuelve un error PEGAJOSO (`CUDA_ERROR_ILLEGAL_ADDRESS` y compañía) el
#: contexto ya está muerto y reintentar en el mismo proceso rara vez salva
#: nada; pero un fallo transitorio por CONTENCIÓN (otro proceso saturando la
#: tarjeta — medido de verdad en esta máquina el 2026-07-31, con tres jobs
#: CUDA a la vez) sí se recupera. ⚠️ tunable de robustez, no de física.
GPU_RETRIES = 1


def _run_solver(build, tag, *, steps_max, avg, deadline=None, log=None,
                allow_cpu_fallback=True, notes=None):
    """Construye y corre el solver, degradando **con etiqueta** si la GPU falla.

    `build(xp)` devuelve un `LBMSolver` recién construido sobre ese módulo.

    Contrato §10.4: una dependencia que no está —o una GPU que se cae a mitad,
    que para el caso es lo mismo— NUNCA es una excepción hacia arriba. Aquí se
    reintenta una vez en GPU y, si sigue fallando, se rehace en NumPy con la
    etiqueta `…(gpu_failed)` y una nota que dice qué reventó. `FloatingPointError`
    NO se atrapa: esa es divergencia numérica, un resultado del cálculo, y quien
    llama ya la traduce en `LAST_STATUS = "diverged"`.

    Devuelve `(gbar, sim, tag_real)`."""
    xp = cupy_module() if tag == BACKEND_CUPY else np
    attempts = (GPU_RETRIES + 1) if tag == BACKEND_CUPY else 1
    last = None
    for k in range(attempts):
        sim = build(xp)
        try:
            gbar = sim.run(steps_max=steps_max, avg=avg, deadline=deadline,
                           log=log)
            return gbar, sim, tag
        except FloatingPointError:
            raise
        except Exception as exc:                      # fallo del backend
            last = exc
            if log:
                log(f"    backend {tag} falló ({type(exc).__name__}: "
                    f"{str(exc)[:120]}) — intento {k + 1}/{attempts}")
            try:
                sim._free_pool()
            except Exception:
                pass
    msg = (f"el backend `{tag}` falló ({type(last).__name__}: "
           f"{str(last)[:160]}) tras {attempts} intento(s)")
    if not allow_cpu_fallback:
        raise RuntimeError(msg)
    if notes is not None:
        notes.append(msg + " — se REHACE el cálculo en NumPy y se etiqueta "
                     "`(gpu_failed)`. El número que sale es válido; lo que "
                     "cambia es quién lo calculó y cuánto tardó.")
    if log:
        log(f"    {msg} → NumPy")
    sim = build(np)
    gbar = sim.run(steps_max=steps_max, avg=avg, deadline=deadline, log=log)
    return gbar, sim, f"{BACKEND_NUMPY}(gpu_failed)"


def _solver_kwargs(backend, dtype=np.float32, acc_dtype=np.float64,
                   n_nodes=None):
    """`(kwargs para LBMSolver, etiqueta)` resueltos desde el nombre de backend.

    Si se pasa `n_nodes` y la malla es más pequeña que `GPU_MIN_NODES`, se
    devuelve NumPy AUNQUE haya GPU: por debajo de ese tamaño la GPU pierde
    (ver la tabla medida en el tunable). La etiqueta devuelta refleja quién
    calculó de verdad, que es lo que exige el contrato §3."""
    tag, xp, _ = resolve_backend(backend)
    if tag == BACKEND_XLB:              # sin adaptador todavía: cae a NumPy
        tag, xp = BACKEND_NUMPY, np
    if tag == BACKEND_CUPY and n_nodes is not None \
            and int(n_nodes) < GPU_MIN_NODES \
            and str(backend).strip().lower() in ("auto", ""):
        tag, xp = BACKEND_NUMPY, np
    return dict(xp=xp, dtype=dtype, acc_dtype=acc_dtype), tag


def canonical_channel(n=22, nz=64, u_lat=U_LATTICE_TARGET, tau=0.6,
                      steps=LBM_STEPS_MAX, avg=300, backend="auto",
                      dtype=np.float32, collision=None,
                      trt_lambda=TRT_LAMBDA, trt_omega_a=TRT_OMEGA_A,
                      cs=0.0, log=None) -> dict:
    """**T41** — canal recto: entrada de velocidad, salida de presión.

    Verifica la conservación de masa del solver: el caudal medido en un corte
    junto a la entrada debe coincidir con el de un corte junto a la salida."""
    solid = _empty_box(n, n, nz)
    fluid = ~solid
    inlet = np.zeros_like(solid)
    inlet[1:-1, 1:-1, 1] = True
    outlet = np.zeros_like(solid)
    outlet[1:-1, 1:-1, -2] = True
    patches = [_patch(inlet, "velocity", (0, 0, 1), u=(0.0, 0.0, u_lat),
                      dtype=dtype),
               _patch(outlet, "pressure", (0, 0, -1), rho=1.0, dtype=dtype)]
    kw, tag = _solver_kwargs(backend, dtype, n_nodes=solid.size)
    gbar, sim, tag = _run_solver(
        lambda xp: LBMSolver(solid, patches, tau, cs_smago=cs,
                             collision=collision, trt_lambda=trt_lambda,
                             trt_omega_a=trt_omega_a,
                             **dict(kw, xp=xp)), tag,
        steps_max=steps, avg=avg, log=log)
    a, b = halfspace(fluid, 2, 2)
    m_in = region_flux(gbar, a, b, solid.shape)
    a, b = halfspace(fluid, 2, nz - 2)
    m_out = region_flux(gbar, a, b, solid.shape)
    a, b = halfspace(fluid, 2, nz // 2)
    m_mid = region_flux(gbar, a, b, solid.shape)
    return dict(mdot_in_lat=m_in, mdot_out_lat=m_out, mdot_mid_lat=m_mid,
                mdot_plug_lat=u_lat * (n - 2) ** 2,
                rel_error=abs(m_in - m_out) / max(abs(m_in), 1e-30),
                tau=float(tau), nu_lat=(float(tau) - 0.5) / 3.0,
                collision=sim.collision, scheme=sim.collision_label,
                stats=sim.stats, backend=tag)


def canonical_orifice(n=44, nz=56, z_plate=18, side=12, thickness=2,
                      shape="square", drho=0.015, tau=TAU_FLOOR,
                      cs=CS_SMAGORINSKY, steps=4000, avg=400,
                      backend="auto", dtype=np.float32, collision=None,
                      trt_lambda=TRT_LAMBDA, trt_omega_a=TRT_OMEGA_A,
                      log=None) -> dict:
    """**T42** — placa con orificio de canto vivo, C_d medido.

    Dos plenos a presión impuesta separados por una placa perforada. Se mide
    ṁ por el orificio y Δp entre plenos y se cierra la ley de Ciencia §4.1
    `ṁ = C_d·A·√(2ρΔp)`.

    Elecciones DECLARADAS del caso (con el estudio que las respalda, medido
    en esta máquina con este mismo código):

    ==========  =========  =====  ===============================
    lado (lu)   espesor    C_d    comentario
    ==========  =========  =====  ===============================
    12          1          0.78   membrana de espesor nulo, malla basta
    20          1          0.71   ídem, mejor resuelta
    12          2          0.68   **caso por defecto**: t/d = 0.17
    20          3          0.71   t/d = 0.15, malla fina
    ==========  =========  =====  ===============================

    * `thickness = 2` (t/d = 1/6) porque un orificio de liner **no es** una
      membrana de espesor nulo: en el ancla t/d = 0.8/5.76 = 0.14. Ese es el
      «orificio plano de canto vivo» al que Ciencia §4.1 le asigna
      C_d ≈ 0.60–0.65.
    * `shape="square"`: en una malla cartesiana el área abierta de un
      cuadrado es EXACTA, mientras que la de un círculo dentado es ambigua a
      ±(1 + 2/d). Con `shape="circle"` se devuelve además `Cd_geom` (área del
      círculo ideal) para cuantificar ese sesgo. El C_d de un orificio
      cuadrado de canto vivo comparte banda con el circular.
    * Con espesor nulo y sólo 12 lu el C_d medido sube a 0.78: la vena
      contracta no está resuelta. Es EL sesgo de discretización del nivel y
      hay que tenerlo presente al leer los splits del caso real.
    """
    solid = _empty_box(n, n, nz)
    ii = np.arange(n) - (n - 1) / 2.0
    x, y = ii[:, None], ii[None, :]
    if shape == "square":
        open_ = (np.abs(x) < side / 2.0) & (np.abs(y) < side / 2.0)
        A_geom = float(side * side)
    else:
        open_ = (x * x + y * y) < (side / 2.0) ** 2
        A_geom = float(math.pi * side * side / 4.0)
    for kk in range(z_plate, z_plate + max(int(thickness), 1)):
        solid[:, :, kk] = True
        solid[:, :, kk][open_] = False
    solid[0] = solid[-1] = True
    solid[:, 0] = solid[:, -1] = True
    solid[:, :, 0] = solid[:, :, -1] = True
    fluid = ~solid
    A_vox = float((~solid[:, :, z_plate]).sum())
    inlet = np.zeros_like(solid)
    inlet[1:-1, 1:-1, 1] = True
    outlet = np.zeros_like(solid)
    outlet[1:-1, 1:-1, -2] = True
    patches = [_patch(inlet, "pressure", (0, 0, 1), rho=1.0 + drho,
                      dtype=dtype),
               _patch(outlet, "pressure", (0, 0, -1), rho=1.0, dtype=dtype)]
    kw, tag = _solver_kwargs(backend, dtype, n_nodes=solid.size)
    gbar, sim, tag = _run_solver(
        lambda xp: LBMSolver(solid, patches, tau, cs_smago=cs,
                             collision=collision, trt_lambda=trt_lambda,
                             trt_omega_a=trt_omega_a,
                             **dict(kw, xp=xp)), tag,
        steps_max=steps, avg=avg, log=log)
    rho, _ = macro_from_g(gbar)
    a, b = halfspace(fluid, 2, z_plate)
    m = region_flux(gbar, a, b, solid.shape)

    def p_at(k):
        sel = fluid[:, :, k]
        return float(rho[:, :, k][sel].mean()) * CS2

    z2 = z_plate + max(int(thickness), 1)
    p_up = float(np.mean([p_at(k) for k in range(z_plate - 8, z_plate - 4)]))
    p_dn = float(np.mean([p_at(k) for k in range(z2 + 12, z2 + 18)]))
    A_duct = float(fluid[:, :, z_plate - 8].sum())
    u_up = m / max(A_duct, 1.0)
    dp = (p_up - p_dn) + 0.5 * u_up * u_up          # total aguas arriba
    u_hole = m / max(A_vox, 1.0)
    nu = (float(sim.tau) - 0.5) / 3.0
    return dict(
        Cd=m / max(A_vox * math.sqrt(2.0 * max(dp, 1e-30)), 1e-30),
        Cd_geom=m / max(A_geom * math.sqrt(2.0 * max(dp, 1e-30)), 1e-30),
        mdot_lat=m, A_vox_lat=A_vox, A_geom_lat=A_geom, dp_lat=dp,
        u_hole_lat=u_hole, Re_d=u_hole * side / max(nu, 1e-30),
        area_ratio=A_vox / max(A_duct, 1.0), shape=shape,
        side_lat=int(side), thickness_lat=int(thickness),
        t_over_d=float(thickness) / max(float(side), 1e-9),
        tau=float(tau), nu_lat=nu,
        collision=sim.collision, scheme=sim.collision_label,
        stats=sim.stats, backend=tag)


#: Pasadas del flujo cruzado por el dominio antes de promediar en T43. Con
#: `u_c = u_jet/√J`, a J = 36 el flujo cruzado va a 0.015 lu/ts y una pasada
#: cuesta 5 000 pasos: con 6 000 pasos el chorro **aún no ha desarrollado el
#: penacho** y Y/d sale un 39 % corto (medido). Con 4 pasadas cae a −19 %.
JICF_FLOWTHROUGHS = 4.0

#: Techo de iteraciones PROPIO del caso T43. El caso necesita
#: `JICF_FLOWTHROUGHS · nx / u_c` pasos y a refine = 2, J = 36 eso son 40 533:
#: recortarlos con el `LBM_STEPS_MAX` genérico entregaría 3 pasadas en vez de
#: 4 y sesgaría Y/d a la baja **en silencio**, que es justo el efecto que el
#: propio caso documenta (−39 % con 1 pasada). El bucle sigue ACOTADO
#: (contrato §10.5), solo que en el número que la física pide.
JICF_STEPS_MAX = 60_000


#: Estación de lectura de la penetración, en la coordenada de SEMEJANZA del
#: chorro en flujo cruzado `x/(r·d)` con `r = u_j/u_c = √J`.
#: **Derivación A PRIORI, no ajustada a los resultados**: la correlación
#: clásica de trayectoria (Margason) es `y/(r d) = A [x/(r d)]^m` con
#: A ≈ 1.6 y m ≈ ⅓; la de Ciencia §4.2 es `Y/d = 1.15 √J = 1.15 r`. Igualando
#: ambas: `1.6·s₀^⅓ = 1.15` ⇒ `s₀ = (1.15/1.6)³ = 0.371`. Es decir, la
#: estación a la que la trayectoria clásica vale exactamente la penetración
#: de Lefebvre. Cambiarla obliga a rehacer T43.
JICF_STATION_OVER_RD = 0.371


#: Refinamiento por defecto del caso T43. Todas las longitudes del caso (Ø del
#: chorro, dominio, posición del orificio) se multiplican por este entero, así
#: que los nodos crecen como r³ y los pasos como r: el coste va como r⁴.
#:
#: **Es 1 en LOS DOS backends, a propósito.** La GPU hace r = 2 asumible
#: (0.70 M nodos, ~2 min por J frente a ~2 h en CPU) y el estudio de
#: refinamiento SE HIZO — resultado medido el 2026-07-31:
#:
#:     refine=1 (d= 5 lu):  J=4 +30.4 %  ·  J=16 +13 %   ·  J=36  −4 %
#:     refine=2 (d=10 lu):  J=4  +4.3 %  ·  J=16 −54.3 % ·  J=36 −20.3 %
#:
#: Es decir: **el caso NO está convergido en malla**. Refinar arregla el
#: chorro débil (que era el fallo declarado) y rompe el intermedio. El
#: acuerdo previo a J = 16 y 36 era en parte cancelación de dos errores.
#: Poner r = 2 por defecto en GPU haría que el VEREDICTO de un test de
#: verificación dependiera de qué máquina lo corre, que es un defecto peor
#: que el que arregla. Se deja el valor calibrado, se declara la no
#: convergencia, y `refine=` queda expuesto para rehacer el estudio.
JICF_REFINE_GPU = 1
JICF_REFINE_CPU = 1


def canonical_jicf(J=16.0, d=5, nx=76, ny=52, nz=22, u_jet=0.09,
                   x_jet=14, tau=TAU_FLOOR, cs=CS_SMAGORINSKY,
                   steps=None, avg=400, backend="auto", dtype=np.float32,
                   refine=None, collision=None, trt_lambda=TRT_LAMBDA,
                   trt_omega_a=TRT_OMEGA_A,
                   station_over_rd=JICF_STATION_OVER_RD, log=None) -> dict:
    """**T43** — chorro en flujo cruzado: trayectoria y penetración Y/d.

    Canal con pared inferior en `y = 1`, orificio circular de Ø `d` que
    inyecta normal a la pared. Se fija la velocidad del CHORRO (`u_jet`, por
    debajo de `U_LATTICE_MAX`) y se deriva la del flujo cruzado
    `u_c = u_jet/√J` — así el punto más rápido de la malla es el mismo para
    los tres J y el error de compresibilidad no cambia con el barrido
    (misma densidad ⇒ `J = ρ_j u_j²/ρ_g u_g²`). Envergadura (z) periódica.

    La correlación de Ciencia §4.2 `Y_max/d = 1.15·√J` **no especifica la
    estación axial** a la que se mide Y, y la trayectoria de un chorro en
    flujo cruzado sigue creciendo como x^⅓: sin fijar estación la comparación
    no significa nada. Peor: a estación FIJA en x/d la traza medida escala
    como J^0.25 (medido aquí: 3.2 · 4.6 · 5.6 para J = 4 · 16 · 36), así que
    ninguna estación en x/d puede reproducir un √J.

    La estación se declara por tanto en la **coordenada de semejanza** del
    chorro en flujo cruzado, `x/(r·d)` con `r = u_j/u_c = √J`, que es la
    variable en la que colapsan las trayectorias de la literatura
    (`y/(r d) = A [x/(r d)]^⅓`). En esa variable, a estación fija,
    `Y/d = A·r·s₀^⅓ ∝ √J` — exactamente la forma de Lefebvre. Es decir: la
    estación de semejanza no es un ajuste para aprobar el test, es la ÚNICA
    coordenada en la que la correlación que se está verificando tiene
    sentido dimensional.

    Y se lee como mediana de la trayectoria en una ventana alrededor de esa
    estación, sobre el plano central de envergadura, con el máximo de |u|
    como traza. Se devuelve la trayectoria completa (`traj`) para que la
    elección sea auditable.

    ESTUDIO DE CONVERGENCIA EN MALLA (medido el 2026-07-31 con
    `JICF_STATION_OVER_RD = 0.371` y 4 pasadas de flujo cruzado; el
    refinamiento en GPU es lo que lo ha hecho asumible):

    ========  ======  =========  ===========  ===========  =========
    refine    d (lu)  J = 4      J = 16       J = 36       nodos
    ========  ======  =========  ===========  ===========  =========
    1         5       **+30.4 %**  +13 %        −4 %       0.087 M
    2         10      +4.3 %     **−54.3 %**  −20.3 %      0.696 M
    ========  ======  =========  ===========  ===========  =========

    Lectura honesta: **el caso NO está convergido en malla**. La hipótesis
    declarada («el chorro DÉBIL se sale porque a d = 5 lu su núcleo tiene ~20
    celdas y el LBM lo mantiene erguido») se CONFIRMA — J = 4 pasa de +30.4 %
    a +4.3 % — pero al mismo tiempo J = 16 se desploma a −54.3 %. O sea que
    el acuerdo previo a J = 16 y 36 era en buena parte **cancelación de dos
    errores**, y la lectura de Y/d por argmax de |u| en el campo promediado
    deja de ser estable cuando la malla resuelve la estructura del chorro.
    Arreglar T43 de verdad exige revisar el ESTIMADOR de trayectoria (traza
    por centroide de la cantidad de movimiento en vez de por máximo), no
    solo la malla. No se ha movido la estación para taparlo (con s₀ ≈ 0.25
    los tres caerían dentro ya a d = 5 lu, pero ese valor no lo respalda
    ninguna correlación).

    `refine` multiplica TODAS las longitudes del caso; el coste va como
    refine⁴ (refine³ nodos × refine pasos). Ver `JICF_REFINE_GPU` para por qué
    el defecto es 1 en los dos backends."""
    if refine is None:
        refine = JICF_REFINE_GPU if resolve_backend(backend)[0] == BACKEND_CUPY \
            else JICF_REFINE_CPU
    refine = max(int(refine), 1)
    d, nx, ny, nz, x_jet = (d * refine, nx * refine, ny * refine,
                            nz * refine, x_jet * refine)
    solid = np.zeros((nx, ny, nz), dtype=bool)
    solid[0] = solid[-1] = True            # guardas en x (entrada/salida)
    solid[:, 0] = solid[:, -1] = True      # paredes inferior y superior
    fluid = ~solid
    zc = nz // 2
    xx = (np.arange(nx) - x_jet)[:, None]
    zz = (np.arange(nz) - zc)[None, :]
    hole = (xx * xx + zz * zz) < (d / 2.0) ** 2
    jet = np.zeros_like(solid)
    jet[:, 1, :] = hole
    jet &= ~solid
    inlet = np.zeros_like(solid)
    inlet[1, 1:-1, :] = True
    inlet &= ~solid
    outlet = np.zeros_like(solid)
    outlet[-2, 1:-1, :] = True
    outlet &= ~solid
    u_j = float(u_jet)
    u_c = u_j / math.sqrt(max(J, 1e-9))
    if steps is None:                    # presupuesto POR PASADAS, no fijo
        steps = int(min(JICF_FLOWTHROUGHS * nx / max(u_c, 1e-9),
                        JICF_STEPS_MAX))
    patches = [_patch(inlet, "velocity", (1, 0, 0), u=(u_c, 0.0, 0.0),
                      dtype=dtype),
               _patch(outlet, "pressure", (-1, 0, 0), rho=1.0, dtype=dtype),
               _patch(jet, "velocity", (0, 1, 0), u=(0.0, u_j, 0.0),
                      dtype=dtype)]
    kw, tag = _solver_kwargs(backend, dtype, n_nodes=solid.size)
    gbar, sim, tag = _run_solver(
        lambda xp: LBMSolver(solid, patches, tau, cs_smago=cs,
                             collision=collision, trt_lambda=trt_lambda,
                             trt_omega_a=trt_omega_a,
                             **dict(kw, xp=xp)), tag,
        steps_max=steps, avg=avg, log=log)
    rho, j = macro_from_g(gbar)
    #  Traza del chorro = locus del máximo de |u| en el plano central de
    #  envergadura (definición clásica de trayectoria de chorro en flujo
    #  cruzado). `u_j/u_c = √J ≥ 2` en todo el barrido, así que el máximo es
    #  el núcleo del chorro y no el flujo cruzado.
    umag = np.sqrt(j[0] ** 2 + j[1] ** 2 + j[2] ** 2) / np.maximum(rho, 1e-9)
    traj = []
    for ix in range(x_jet, min(nx - 3, x_jet + 12 * d)):
        col = umag[ix, 1:ny - 3, zc]
        iy = int(np.argmax(col)) + 1
        traj.append((float(ix - x_jet) / d, float(iy - 1) / d,
                     float(col.max())))
    #  Y se lee como MEDIANA de la trayectoria en una ventana centrada en la
    #  estación de SEMEJANZA: un solo punto es ruidoso en una malla de d = 5 lu.
    st = float(station_over_rd) * math.sqrt(max(J, 1e-9))
    win = [yod for xod, yod, _ in traj if 0.7 * st <= xod <= 1.3 * st + 0.2]
    Y_over_d = float(np.median(win)) if win else 0.0
    Y_max_over_d = max((t[1] for t in traj), default=0.0)
    return dict(J=float(J), d_lat=int(d), u_c=float(u_c), u_j=float(u_j),
                refine=int(refine), shape=(int(nx), int(ny), int(nz)),
                n_nodes=int(nx * ny * nz),
                Y_over_d=float(Y_over_d), Y_max_over_d=float(Y_max_over_d),
                Y_over_d_lefebvre=1.15 * math.sqrt(max(J, 0.0)),
                station_over_rd=float(station_over_rd),
                station_over_d=float(st), steps_planned=int(steps),
                tau=float(tau), collision=sim.collision,
                scheme=sim.collision_label,
                traj=traj, stats=sim.stats, backend=tag)


# ==========================================================================
# 6.b Verificación de los operadores: equivalencia, barrido de τ y coste
# ==========================================================================
def _equiv_box(n=14, nz=26):
    """Caja pequeña CON un obstáculo, para que haya gradientes de verdad.

    Un canal vacío relajaría casi al equilibrio y una equivalencia medida ahí
    no probaría nada: el término que distingue TRT/MRT de BGK es justo el
    no-equilibrio."""
    solid = _empty_box(n, n, nz)
    c0, c1 = n // 2 - 1, n // 2 + 2
    solid[c0:c1, c0:c1, nz // 2:nz // 2 + 2] = True
    inl = np.zeros_like(solid)
    inl[1:-1, 1:-1, 1] = True
    inl &= ~solid
    out = np.zeros_like(solid)
    out[1:-1, 1:-1, -2] = True
    out &= ~solid
    u_in = (0.0, 0.0, U_LATTICE_TARGET)
    patches = [_patch(inl, "velocity", (0, 0, 1), u=u_in, dtype=np.float32),
               _patch(out, "pressure", (0, 0, -1), rho=1.0, dtype=np.float32)]
    return solid, patches


def collision_equivalence(steps=120, tau=0.55, cs=CS_SMAGORINSKY,
                          dtype=np.float64, backend="numpy") -> dict:
    """**Prueba de que la descomposición está bien hecha**, no de física.

    Dos identidades algebraicas exactas:

    1. **TRT con τ_a = τ_s ≡ BGK.** Sustituyendo τ_a = τ_s en
       `f_i ← f_i − (f⁺−f^eq⁺)/τ_s − (f⁻−f^eq⁻)/τ_a` y usando
       `f⁺+f⁻ = f_i`, queda `f_i ← f_i − (f_i−f^eq_i)/τ`, que ES BGK. Si la
       descomposición simétrica/antisimétrica tuviera un par mal emparejado,
       un signo cambiado o un factor ½ de más, esta igualdad se rompería.
    2. **MRT con todas las tasas iguales a 1/τ ≡ BGK**, porque
       `M⁻¹(1/τ)I·M = (1/τ)I`. Verifica la matriz de momentos y su inversa.

    Se corren `steps` pasos completos (colisión + contorno + bounce-back +
    streaming) sobre la misma caja con obstáculo y se compara el campo `f`
    ENTERO, no un escalar agregado.

    Devuelve las diferencias máximas absoluta y relativa de cada caso, y si
    fue bit a bit. Se hace en FP64 y con LES apagado Y encendido: el camino
    del LES es distinto (tasas por celda), así que hay que probar los dos.
    Para MRT con LES la equivalencia solo tiene sentido si TODOS los momentos
    no conservados llevan la ω del LES — es lo que hace el parámetro interno
    `_shear`, no una configuración de producción."""
    solid, patches = _equiv_box()
    _, xp, _ = resolve_backend(backend)

    def _run(collision, lam, cs_, shear=None, wa=None):
        old = globals()["MRT_SHEAR_IDX"]
        if shear is not None:
            globals()["MRT_SHEAR_IDX"] = shear
        try:
            s = LBMSolver(solid, patches, tau, cs_smago=cs_, xp=xp,
                          dtype=dtype, collision=collision, trt_lambda=lam,
                          trt_omega_a=wa)
            for k in range(int(steps)):
                s.step(k)
            f = np.asarray(s.f.get() if s.on_gpu else s.f, dtype=np.float64)
            s._free_pool()
            return f
        finally:
            globals()["MRT_SHEAR_IDX"] = old

    #  TODOS los momentos no conservados con la ω del LES: es la ÚNICA
    #  configuración en la que «MRT ≡ BGK» es cierto también con LES.
    all_moments = tuple(k for k in range(19) if k not in (0, 3, 5, 7))
    rows = []
    for cs_ in (0.0, float(cs)):
        ref = _run(COLLISION_BGK, None, cs_)
        scale = max(float(np.abs(ref).max()), 1e-30)
        trt = _run(COLLISION_TRT, None, cs_)          # τ_a = τ_s
        d = float(np.abs(ref - trt).max())
        rows.append(dict(case="TRT(tau_a=tau_s)", les=cs_ > 0.0, max_abs=d,
                         max_rel=d / scale,
                         bit_exact=bool(np.array_equal(ref, trt))))
        s_old = (MRT_S_E, MRT_S_EPS, MRT_S_Q, MRT_S_PI, MRT_S_M)
        try:
            #  todas las tasas del MRT = 1/τ
            for nm in ("MRT_S_E", "MRT_S_EPS", "MRT_S_Q", "MRT_S_PI",
                       "MRT_S_M"):
                globals()[nm] = 1.0 / float(tau)
            mrt = _run(COLLISION_MRT, None, cs_,
                       shear=(all_moments if cs_ > 0.0 else None))
        finally:
            (globals()["MRT_S_E"], globals()["MRT_S_EPS"],
             globals()["MRT_S_Q"], globals()["MRT_S_PI"],
             globals()["MRT_S_M"]) = s_old
        d = float(np.abs(ref - mrt).max())
        rows.append(dict(case="MRT(s_k=1/tau)", les=cs_ > 0.0, max_abs=d,
                         max_rel=d / scale,
                         bit_exact=bool(np.array_equal(ref, mrt))))
    return dict(tau=float(tau), steps=int(steps), dtype=str(np.dtype(dtype)),
                backend=backend, rows=rows,
                ok=all(r["max_rel"] < 1e-12 for r in rows))


#: τ del barrido de estabilidad, de holgado a pegado a ½. La rejilla es
#: geométrica en (τ − ½) porque lo que importa es la VISCOSIDAD, que es
#: proporcional a esa diferencia, no τ en sí.
SWEEP_TAUS = (0.6, 0.55, 0.51, 0.505, 0.5015, 0.501, 0.5005, 0.5002,
              0.5001, 0.50005, 0.50002, 0.500005)

#: Margen de seguridad sobre (τ_min − ½) medido, para fijar `TAU_FLOOR_*`.
#: ×2 porque el barrido se hace en dos casos canónicos pequeños y el ancla
#: tiene gradientes peores (chorros, esquinas vivas, efusión).
TAU_FLOOR_SAFETY = 2.0


#: Configuraciones del barrido. `(operador, Λ, ω_a, etiqueta)`; `Λ` y `ω_a`
#: son excluyentes (manda `ω_a` si no es `None`). Se prueban LOS DOS Λ de la
#: literatura que pedía el encargo Y un barrido de ω_a fijo, porque el
#: resultado del primero obligó a mirar el segundo (ver `TRT_OMEGA_A`).
SWEEP_CONFIGS = (
    (COLLISION_BGK, None, None, "bgk"),
    (COLLISION_TRT, TRT_LAMBDA_BOUNCE_BACK, None, "trt L=3/16"),
    (COLLISION_TRT, TRT_LAMBDA_STABILITY, None, "trt L=1/4"),
    (COLLISION_TRT, None, 1.00, "trt wa=1.00"),
    (COLLISION_TRT, None, 1.50, "trt wa=1.50"),
    (COLLISION_TRT, None, 1.80, "trt wa=1.80"),
    (COLLISION_TRT, None, 1.90, "trt wa=1.90"),
    (COLLISION_TRT, None, 1.95, "trt wa=1.95"),
    (COLLISION_MRT, None, None, "mrt"),
)


def _sweep_channel(collision, lam, wa, tau, backend, steps, avg, log=None):
    """Un punto del barrido sobre el canal de T41 (SIN LES: operador desnudo).

    Referencias analíticas del caso:

    * **cierre de masa** — ṁ(entrada) = ṁ(salida) es una identidad exacta del
      estacionario; el estimador por poblaciones que cruzan la mide sin error
      de interpolación, así que cualquier desvío es error del cálculo.
    * **tapón `u·A`** — en el límite no viscoso (ν → 0) el perfil de un
      conducto con bounce-back de media celda tiende al tapón
      `u_lat·(n−2)²`. `mdot/mdot_plug` mide por tanto el déficit de capa
      límite, que es la huella directa de la viscosidad EFECTIVA: si el
      operador mete viscosidad numérica de más, este número baja."""
    try:
        r = canonical_channel(n=18, nz=48, tau=tau, steps=steps, avg=avg,
                              backend=backend, collision=collision,
                              trt_lambda=lam, trt_omega_a=wa, cs=0.0, log=log)
    except FloatingPointError as exc:
        return dict(diverged=True, why=str(exc)[:80])
    ok = (math.isfinite(r["mdot_in_lat"]) and math.isfinite(r["rel_error"])
          and abs(r["mdot_in_lat"]) > 1e-12)
    return dict(diverged=not ok, converged=bool(r["stats"]["converged"]),
                mass_closure=float(r["rel_error"]),
                mdot_over_plug=float(r["mdot_in_lat"] / r["mdot_plug_lat"]),
                steps=int(r["stats"]["steps"]),
                mlups=float(r["stats"]["mlups"]))


def _sweep_orifice(collision, lam, wa, tau, backend, steps, avg, log=None):
    """Un punto del barrido sobre el orificio de T42 (CON LES).

    Referencia: la banda de C_d de canto vivo de Ciencia §4.1 (0.60–0.65,
    umbral del test 0.55–0.75). Es el caso sensible al tratamiento de pared,
    que es justo lo que cambia Λ = 3/16."""
    try:
        r = canonical_orifice(tau=tau, steps=steps, avg=avg, backend=backend,
                              collision=collision, trt_lambda=lam,
                              trt_omega_a=wa, log=log)
    except FloatingPointError as exc:
        return dict(diverged=True, why=str(exc)[:80])
    ok = math.isfinite(r["Cd"]) and 0.0 < r["Cd"] < 5.0
    return dict(diverged=not ok, converged=bool(r["stats"]["converged"]),
                Cd=float(r["Cd"]), Re_d=float(r["Re_d"]),
                u_hole=float(r["u_hole_lat"]),
                steps=int(r["stats"]["steps"]),
                mlups=float(r["stats"]["mlups"]))


def stability_sweep(case="channel", taus=SWEEP_TAUS, backend="auto",
                    steps=8000, avg=200, nu_target=None, configs=None,
                    log=None) -> dict:
    """**La evidencia que fija `TAU_FLOOR_*`.** Baja τ hacia ½ por operador.

    Para cada configuración recorre `taus` de mayor a menor y anota DOS
    veredictos, que no son el mismo y conviene no mezclar:

    * `stable` — el cálculo termina con números finitos (no `NaN`, no
      `FloatingPointError`). Es lo que fija hasta dónde llega el operador.
    * `ok` — además el observable ANALÍTICO del caso se mantiene: cierre de
      masa < 1e-3 en el canal (ṁ_entrada = ṁ_salida es una identidad exacta
      del estacionario), o C_d dentro de la banda de canto vivo de Ciencia
      §4.1 en el orificio.

    La convergencia al estacionario (`converged`) se anota pero **no** entra
    en el veredicto: a τ → ½ el canal deja de ser estacionario de verdad
    —Re_celda = u/ν pasa de 10⁴— y exigir `<|u|>` constante a 1e-5 mediría
    la unsteadiness física, no la estabilidad del operador.

    `nu_target` (opcional) es la viscosidad de lattice REAL del caso que se
    quiere reproducir — en el ancla `units["nu_lat"]` = 9.06e-6 — y sirve
    para tabular ν_efectiva/ν_real, que es la magnitud que de verdad importa:
    τ no significa nada por sí solo.

    Bucle ACOTADO por construcción (la lista de τ es finita) y cada punto
    lleva su propio techo de pasos. En cuanto una configuración diverge dos
    veces seguidas se abandona: bajar más τ no la va a resucitar."""
    runner = _sweep_channel if case == "channel" else _sweep_orifice
    if case != "channel":
        steps, avg = min(steps, 4000), max(avg, 400)
    if configs is None:
        configs = SWEEP_CONFIGS
    rows, best, best_stable = [], {}, {}
    for coll, lam, wa, key in configs:
        n_div = 0
        for tau in taus:
            r = runner(coll, lam, wa, float(tau), backend, steps, avg)
            nu = (float(tau) - 0.5) / 3.0
            r.update(collision=coll, key=key, tau=float(tau), nu_lat=nu,
                     lam_eff=(trt_lambda_effective(tau, lam, wa)
                              if coll == COLLISION_TRT else float("nan")),
                     nu_ratio=(nu / nu_target) if nu_target else float("nan"))
            r["stable"] = not r["diverged"]
            if case == "channel":
                r["ok"] = bool(r["stable"]
                               and r.get("mass_closure", 1.0) < 1.0e-3)
                obs = (f"cierre {r.get('mass_closure', float('nan')):.2e} · "
                       f"m/plug {r.get('mdot_over_plug', float('nan')):.4f} · "
                       f"{'conv' if r.get('converged') else 'NOconv'}")
            else:
                r["ok"] = bool(r["stable"] and 0.55 < r.get("Cd", 0.0) < 0.75)
                obs = f"C_d {r.get('Cd', float('nan')):.4f}"
            rows.append(r)
            if log:
                v = "OK   " if r["ok"] else (
                    "estable" if r["stable"] else "DIVERGE")
                log(f"    {key:<16s} tau={tau:.6f}  {v}  {obs}")
            if r["ok"]:
                best[key] = r
            if r["stable"]:
                best_stable[key] = r
                n_div = 0
            else:
                n_div += 1
                if n_div >= 2:
                    break
    return dict(
        case=case, rows=rows, best=best, best_stable=best_stable,
        taus=[float(t) for t in taus], nu_target=nu_target,
        #  El SUELO se fija por ESTABILIDAD, que es para lo que existe un
        #  suelo. El veredicto de exactitud se devuelve aparte porque en el
        #  canal satura en τ ≈ 0.51 para los TRES operadores por igual: por
        #  debajo el caso deja de ser estacionario (Re_celda > 10⁴) y lo que
        #  se degrada es el cierre del PROMEDIO de 200 pasos, no el operador.
        floor_suggested={k: 0.5 + TAU_FLOOR_SAFETY * (v["tau"] - 0.5)
                         for k, v in best_stable.items()},
        floor_accurate={k: 0.5 + TAU_FLOOR_SAFETY * (v["tau"] - 0.5)
                        for k, v in best.items()})


def collision_cost(n=180, nz=200, tau=0.505, steps=60, backend="auto",
                   cs=CS_SMAGORINSKY, dtype=np.float32, log=None) -> dict:
    """Coste MEDIDO de los tres operadores sobre la MISMA malla y el mismo τ.

    Devuelve MLUPS y s/celda/paso por operador, y el sobrecoste frente a BGK.
    Se cronometra con `_sync` (los kernels de CuPy son asíncronos) y tras
    ocho pasos de calentamiento, que es lo que tarda el pool de memoria en
    estabilizarse."""
    solid = _empty_box(n, n, nz)
    solid[n // 3:n // 3 + 6, :, nz // 2:nz // 2 + 6] = True
    inl = np.zeros_like(solid)
    inl[1:-1, 1:-1, 1] = True
    inl &= ~solid
    out = np.zeros_like(solid)
    out[1:-1, 1:-1, -2] = True
    out &= ~solid
    u_in = (0.0, 0.0, U_LATTICE_TARGET)
    patches = [_patch(inl, "velocity", (0, 0, 1), u=u_in, dtype=dtype),
               _patch(out, "pressure", (0, 0, -1), rho=1.0, dtype=dtype)]
    kw, tag = _solver_kwargs(backend, dtype, n_nodes=solid.size)
    res, base = {}, None
    for coll in COLLISIONS:
        s = LBMSolver(solid, patches, tau, cs_smago=cs, collision=coll, **kw)
        for k in range(8):
            s.step(k)
        s._sync()
        t0 = time.perf_counter()
        for k in range(int(steps)):
            s.step(k)
        s._sync()
        el = time.perf_counter() - t0
        s._free_pool()
        m = solid.size * steps / 1e6 / max(el, 1e-12)
        base = m if base is None else base
        res[coll] = dict(mlups=float(m),
                         s_per_cell_step=float(el / (steps * solid.size)),
                         cost_vs_bgk=float(base / m))
        if log:
            log(f"    {coll}: {m:.1f} MLUPS  "
                f"{res[coll]['s_per_cell_step']:.3e} s/celda/paso  "
                f"×{res[coll]['cost_vs_bgk']:.3f} vs BGK")
    return dict(backend=tag, n_nodes=int(solid.size), steps=int(steps),
                tau=float(tau), cs=float(cs), les=bool(cs > 0.0), ops=res)


# ==========================================================================
# 7. Geometría: lectura del .vdb o re-voxelización analítica
# ==========================================================================
def openvdb_module():
    """Módulo de OpenVDB o `None`. Dependencia OPCIONAL (contrato §10.4)."""
    for name in ("pyopenvdb", "openvdb"):
        try:
            return __import__(name)
        except Exception:
            continue
    return None


def read_vdb_field(path: str):
    """Lee un `*_field.vdb` de la capa 5c. `None` si no hay lector o archivo.

    El `.vdb` de PicoGK es un SDF de banda estrecha en **mm**. Sin
    `pyopenvdb` instalado (el caso de esta máquina) se devuelve `None` y
    `solve_cold` re-voxeliza analíticamente el contrato — misma geometría,
    sin los detalles que PicoGK añade. Se declara en `notes`."""
    vdb = openvdb_module()
    if vdb is None or not path or not os.path.isfile(path):
        return None
    try:
        grids = vdb.readAllGridMetadata(path)
        name = grids[0].name if grids else None
        grid = vdb.read(path, name) if name else None
        if grid is None:
            return None
        acc = grid.getAccessor()
        return dict(grid=grid, accessor=acc, name=name,
                    voxel_size_mm=float(grid.transform.voxelSize()[0]))
    except Exception:
        return None


def _profile_arrays(profile):
    """(z, r_in, r_out) del perfil del liner como arrays ordenados por z."""
    p = np.array([[float(r[0]), float(r[1]), float(r[2])] for r in profile
                  if isinstance(r, (list, tuple)) and len(r) >= 3],
                 dtype=float)
    if p.size == 0:
        raise ValueError("perfil de liner vacío en el contrato")
    return p[np.argsort(p[:, 0])]


#: Celdas MÍNIMAS a través del anillo de refrigeración. Con menos de dos, el
#: sellado del liner (o el propio bounce-back) estrangula el paso más
#: estrecho del combustor y el reparto de aire medido no vale nada —
#: verificado: a dx = 3 mm el ancla pasaba 0.0006 kg/s en vez de 0.13.
ANNULUS_MIN_CELLS = 2.0
#: Celdas mínimas a través del Ø del orificio más pequeño con caudal.
HOLE_MIN_CELLS = 3.0


def _geometry_scales(contract) -> dict:
    """Longitudes que la malla TIENE que resolver, sacadas del contrato."""
    liner = contract.get("liner") or {}
    prof = np.array([[float(r[0]), float(r[1]), float(r[2])]
                     for r in (liner.get("profile") or [])
                     if isinstance(r, (list, tuple)) and len(r) >= 3],
                    dtype=float) if liner.get("profile") else np.zeros((1, 3))
    t_liner = float(liner.get("t_wall_mm") or 0.8)
    r_lo = float(prof[:, 2].max()) if prof.size else 0.0
    cas = contract.get("casing") or {}
    cprof = np.array(cas.get("profile") or [[0.0, 1.2 * max(r_lo, 1.0)]],
                     dtype=float)
    r_cas = float(np.max(cprof[:, 1]))
    gap = max(r_cas - (r_lo + t_liner), 0.0)
    for fp in (contract.get("flow_path") or []):
        if isinstance(fp, dict) and str(fp.get("id", "")).startswith("anillo"):
            gap = max(float(fp.get("r_out_mm", r_cas))
                      - float(fp.get("r_in_mm", r_lo + t_liner)), 0.0)
            break
    ds = [float(r.get("d_mm", 0.0) or 0.0)
          for r in (contract.get("hole_rows") or [])
          if isinstance(r, dict) and r.get("flow", True)
          and float(r.get("d_mm", 0.0) or 0.0) > 0.0]
    return dict(r_casing_mm=r_cas, annulus_gap_mm=gap,
                d_hole_min_mm=min(ds) if ds else 0.0,
                z0_mm=float(np.min(cprof[:, 0])),
                z1_mm=float(np.max(cprof[:, 0])))


#: Celdas OBJETIVO a través del anillo cuando hay presupuesto para pedirlas.
#: `ANNULUS_MIN_CELLS = 2` es el mínimo por debajo del cual el resultado es
#: FALSO (el liner se estrangula); con 2 celdas el campo sirve para topología
#: pero no es cuantitativo — es exactamente el diagnóstico de T44. Con GPU sí
#: hay presupuesto para pedir 4, que es donde un canal empieza a tener perfil.
#: ⚠️ tunable de resolución: subirlo multiplica los nodos por (4/2)³ = 8.
ANNULUS_TARGET_CELLS = 4.0


def auto_dx_mm(contract, max_nodes=LBM_MAX_NODES,
               annulus_cells=ANNULUS_MIN_CELLS) -> float:
    """dx exigido por la GEOMETRÍA, acotado por la banda admisible.

    WAVE5 §A.2.1 pide 0.5–1 mm. El fallback de NumPy no puede con los 12 M de
    nodos que salen a 0.5 mm en el ancla, pero **el presupuesto no puede
    ganarle a la física**: si el anillo de refrigeración (2.5 mm en el ancla)
    no se resuelve con al menos `annulus_cells` celdas, el resultado no es
    «grueso», es falso. Se devuelve el dx que la geometría exige; `solve_cold`
    declara cuántos nodos salen y si se pasa del presupuesto.

    Con backend de GPU, `solve_cold` llama con `max_nodes` sacado de la VRAM
    libre REAL y `annulus_cells = ANNULUS_TARGET_CELLS`: el dx no lo elige el
    presupuesto, lo elige la geometría, y el presupuesto solo dice si cabe."""
    g = _geometry_scales(contract)
    L = max(g["z1_mm"] - g["z0_mm"], 1.0)
    D = max(2.0 * g["r_casing_mm"], 1.0)
    #  n_nodos ≈ (D/dx + 2)²·(L/dx + 2) ⇒ dx ≈ (D²L/N)^(1/3)
    dx_budget = (D * D * L / max(float(max_nodes), 1.0)) ** (1.0 / 3.0)
    dx = dx_budget
    if g["annulus_gap_mm"] > 0.0:
        dx = min(dx, g["annulus_gap_mm"] / max(float(annulus_cells), 1e-9))
    if g["d_hole_min_mm"] > 0.0:
        dx = min(dx, g["d_hole_min_mm"] / HOLE_MIN_CELLS)
    return float(min(max(dx, DX_MIN_MM), DX_MAX_MM))


def _nodes_at_dx(contract, dx_mm: float) -> int:
    """Nodos que saldrían de `voxelize_contract` a ese dx (sin voxelizar)."""
    g = _geometry_scales(contract)
    dx = max(float(dx_mm), 1e-9)
    nx = int(math.ceil(2.0 * g["r_casing_mm"] / dx)) + 2
    nz = int(math.ceil((g["z1_mm"] - g["z0_mm"]) / dx)) + 2
    return int(nx) * int(nx) * int(nz)


#: Rendimiento MEDIDO del paso completo en esta GPU (RTX 3070 Ti, FP32,
#: D3Q19 + LES + efusión), en segundos por celda y paso. 72.9 MLUPS a 3.2 M
#: nodos ⇒ 1.37e-8; se toma 1.6e-8 para incluir la efusión y no prometer de
#: más. Compárese con la referencia pública que cita Ciencia §11.2 (1e-8
#: s/celda/paso en FP32). ⚠️ recalibrar si se cambia de GPU.
LBM_GPU_S_PER_CELL_STEP = 1.0e-8

#: Sobrecoste del paso por operador de colisión, MEDIDO con `collision_cost`
#: en esta máquina (RTX 3070 Ti, 6.48 M nodos, FP32, LES activo):
#: BGK 213.5 · TRT 178.7 · MRT 190.7 MLUPS. Solo entra en la ESTIMACIÓN de
#: pared de `gpu_dx_mm` (qué malla cabe en el presupuesto); no toca ningún
#: número físico. Sin LES los tres cuestan lo mismo (×1.005) porque la
#: colisión entera se reduce a un GEMM 19×19. ⚠️ recalibrar si se cambia de
#: GPU o de operador.
LBM_COLLISION_COST = {COLLISION_BGK: 1.00, COLLISION_TRT: 1.20,
                      COLLISION_MRT: 1.12}
#: Ídem para el camino NumPy: 6.6 MLUPS medidos ⇒ 1.5e-7.
LBM_CPU_S_PER_CELL_STEP = 1.5e-7

#: Pasadas del dominio hasta el estacionario estadístico, para ESTIMAR el
#: coste antes de correr. Con 5 pasadas la estimación da 19 000 pasos para el
#: ancla a dx = 1.23 mm y el solver convergió en 17 000: buena aproximación.
#: El número de pasos escala como 1/dx (misma velocidad de lattice, dominio
#: dx veces más fino), así que afinar la malla cuesta dx⁻⁴.
LBM_FLOWTHROUGHS_EST = 5.0

#: Fracción del presupuesto de pared que se deja gastar al transitorio. El
#: resto es para el promedio, el post-proceso y el margen del modelo.
WALL_BUDGET_FRACTION = 0.75


def _steps_estimate(contract, dx_mm: float,
                    u_lat: float = U_LATTICE_TARGET) -> int:
    """Pasos estimados hasta el estacionario a ese dx (ver los tunables)."""
    g = _geometry_scales(contract)
    nz = max((g["z1_mm"] - g["z0_mm"]) / max(float(dx_mm), 1e-9), 1.0)
    return int(LBM_FLOWTHROUGHS_EST * nz / max(u_lat, 1e-9))


def gpu_dx_mm(contract, dtype=np.float32, acc_dtype=np.float64,
              fraction: float = GPU_VRAM_FRACTION,
              wall_budget_s: float | None = None,
              steps_max: int = LBM_STEPS_MAX,
              collision: str | None = COLLISION_BGK,
              ) -> tuple[float, int, list]:
    """dx del caso real EN GPU: el que pide la geometría, si cabe.

    Devuelve `(dx_mm, max_nodes, notas)`. Se pide el dx que da
    `ANNULUS_TARGET_CELLS` celdas en el anillo y luego se **engorda en pasos
    del 6 %** hasta que pase los tres filtros, DECLARANDO cuál mordió:

    1. **VRAM** — se mide la libre AHORA (`gpu_node_budget`). Reventar a
       mitad de un solve de minutos es peor que un campo más grueso y avisado.
    2. **Pasos** — el bucle está acotado en `steps_max` (contrato §10.5) y
       los pasos necesarios crecen como 1/dx: pedir una malla cuyo transitorio
       no cabe en el techo de iteraciones es pedir un resultado a medio
       converger, y eso ya lo teníamos.
    3. **Pared** — el coste va como dx⁻⁴ (dx⁻³ nodos × dx⁻¹ pasos). Con el
       rendimiento MEDIDO de esta GPU se estima el tiempo antes de arrancar y
       se coge la malla más fina que cabe en el presupuesto.

    El filtro 3 es el que manda casi siempre: en el ancla la VRAM daría para
    dx = 0.54 mm (10.5 M nodos) pero eso son ~39 000 pasos y ~85 min.
    """
    notes: list[str] = []
    cost = LBM_COLLISION_COST.get(resolve_collision(collision), 1.0)
    budget = gpu_node_budget(dtype, acc_dtype, fraction, collision)
    if budget <= 0:
        return auto_dx_mm(contract), LBM_MAX_NODES, ["sin GPU: presupuesto de CPU"]
    dx0 = auto_dx_mm(contract, max_nodes=budget,
                     annulus_cells=ANNULUS_TARGET_CELLS)
    g = _geometry_scales(contract)
    wall_cap = (WALL_BUDGET_FRACTION * float(wall_budget_s)
                if wall_budget_s else float("inf"))

    def _fits(dx):
        n = _nodes_at_dx(contract, dx)
        st = _steps_estimate(contract, dx)
        wall = st * n * LBM_GPU_S_PER_CELL_STEP * cost
        return (n <= budget, st <= steps_max, wall <= wall_cap, n, st, wall)

    dx = dx0
    grew = 0
    while grew < 96 and dx < DX_MAX_MM:
        ok_v, ok_s, ok_w, n, st, wall = _fits(dx)
        if ok_v and ok_s and ok_w:
            break
        dx = min(dx * 1.06, DX_MAX_MM)
        grew += 1
    ok_v, ok_s, ok_w, n, st, wall = _fits(dx)
    if grew:
        why = []
        if not _fits(dx / 1.06)[0]:
            why.append(f"VRAM ({budget / 1e6:.1f} M nodos al {100 * fraction:.0f} % "
                       f"de {gpu_info().get('vram_free_GiB', float('nan')):.2f} GiB libres)")
        if not _fits(dx / 1.06)[1]:
            why.append(f"techo de pasos ({steps_max})")
        if not _fits(dx / 1.06)[2]:
            why.append(f"presupuesto de pared ({wall_cap:.0f} s)")
        notes.append(
            f"malla ENGORDADA de {dx0:.3f} a {dx:.3f} mm — manda "
            + " y ".join(why) + f". Queda en {n / 1e6:.2f} M nodos, "
            f"~{st} pasos, ~{wall:.0f} s estimados, y el anillo del liner "
            f"con {g['annulus_gap_mm'] / dx:.1f} celdas (el objetivo eran "
            f"{ANNULUS_TARGET_CELLS:.0f}, el mínimo para que el número "
            f"signifique algo es {ANNULUS_MIN_CELLS:.0f}).")
    else:
        notes.append(
            f"malla a dx = {dx:.3f} mm ({n / 1e6:.2f} M nodos, ~{st} pasos, "
            f"~{wall:.0f} s estimados): el anillo del liner se resuelve con "
            f"{g['annulus_gap_mm'] / dx:.1f} celdas y el orificio más pequeño "
            f"con {g['d_hole_min_mm'] / dx:.1f}.")
    return float(dx), int(budget), notes


def voxelize_contract(contract: dict, dx_mm: float, log=None) -> dict:
    """Re-voxeliza el contrato `phycb-combustor-1` a una malla cartesiana.

    Construye, en este orden:

    1. **Gas del liner** — `r_in(z) ≤ r ≤ r_out(z)` del perfil, más el casquete
       del domo (`liner.dome.bulge_mm`, esférico: `r ≤ √(r_lo²−s²)`).
    2. **Lado frío** — el anillo (`flow_path[].anillo_*`) y el pleno del domo.
    3. **Sellado** — toda celda fría con vecina de gas pasa a sólido. A dx ≥ 1
       mm la pared del liner (0.8 mm) es MÁS FINA que un voxel: sin este paso
       el liner tendría fugas y el reparto de aire medido sería basura. Es la
       regla estándar de CFD sobre voxels; engorda la pared a ≥ 1 voxel y se
       declara.
    4. **Orificios** — cada agujero de cada `hole_rows[]` con `flow=True` se
       taladra como cilindro radial de Ø `d_mm`; cada fila recibe su propia
       etiqueta de región para poder integrarle el caudal.
    5. **Swirlers** — `swirler.count` taladros axiales de Ø `d_tip_mm` al radio
       del anillo de inyectores.
    6. **Efusión** — no se resuelve (Ø 0.6 mm): se prepara la lista de franjas
       de transpiración con su área abierta.

    Devuelve el «dominio» que consume `LBMSolver`. **Nunca lanza** por datos
    incompletos del contrato: cae a respaldos y lo anota en `notes`.
    """
    notes: list[str] = []
    dx = float(dx_mm)
    arch = str(contract.get("arch", "can"))
    liner = contract.get("liner") or {}
    prof = _profile_arrays(liner.get("profile") or [])
    t_liner = max(float(liner.get("t_wall_mm") or 0.8), 1e-3)
    dome = liner.get("dome") or {}
    bulge = max(float(dome.get("bulge_mm") or 0.0), 0.0)
    z_dome = float(dome.get("z_mm") or 0.0)
    cas = contract.get("casing") or {}
    cprof = np.array(cas.get("profile") or [[prof[0, 0], 1.2 * prof[:, 2].max()],
                                            [prof[-1, 0], 1.2 * prof[:, 2].max()]],
                     dtype=float)
    r_cas = float(np.max(cprof[:, 1]))
    z_min = float(min(np.min(cprof[:, 0]), prof[0, 0]))
    z_max = float(max(np.max(cprof[:, 0]), prof[-1, 0]))
    reverse = (arch == "reverse_annular")
    if bulge > 0.0:
        z_min = min(z_min, z_dome - bulge - dx) if not reverse else z_min
        z_max = max(z_max, z_dome + bulge + dx) if reverse else z_max

    # --- malla (con capa de guarda sólida en las seis caras) --------------
    nx = ny = int(math.ceil(2.0 * r_cas / dx)) + 2
    nz = int(math.ceil((z_max - z_min) / dx)) + 2
    shape = (nx, ny, nz)
    xs = (np.arange(nx) - (nx - 1) / 2.0) * dx
    ys = (np.arange(ny) - (ny - 1) / 2.0) * dx
    zs = z_min + (np.arange(nz) - 0.5) * dx
    RR = np.sqrt(xs[:, None] ** 2 + ys[None, :] ** 2)          # (nx, ny)
    PHI = np.arctan2(np.broadcast_to(ys[None, :], (nx, ny)),
                     np.broadcast_to(xs[:, None], (nx, ny)))

    r_in = np.interp(zs, prof[:, 0], prof[:, 1], left=np.nan, right=np.nan)
    r_out = np.interp(zs, prof[:, 0], prof[:, 2], left=np.nan, right=np.nan)
    in_prof = np.isfinite(r_in)
    r_li = float(prof[:, 1].max())
    r_lo = float(prof[:, 2].max())

    gas = np.zeros(shape, dtype=bool)
    for k in range(nz):
        if in_prof[k]:
            gas[:, :, k] = (RR >= r_in[k] - 1e-9) & (RR <= r_out[k] + 1e-9)
        elif bulge > 0.0:
            s = (zs[k] - z_dome) if reverse else (z_dome - zs[k])
            if 0.0 <= s <= bulge:
                r_cap = math.sqrt(max(r_lo * r_lo - s * s, 0.0))
                lo = r_li if r_li > 0.0 else 0.0
                gas[:, :, k] = (RR >= lo - 1e-9) & (RR <= r_cap)

    # --- lado frío: anillo (del flow_path) + pleno del domo ---------------
    r_an_in, z_an0, z_an1 = r_lo + t_liner, z_min, z_max
    for fp in (contract.get("flow_path") or []):
        if not isinstance(fp, dict):
            continue
        if str(fp.get("id", "")).startswith("anillo"):
            r_an_in = float(fp.get("r_in_mm", r_an_in))
            z_an0 = min(float(fp.get("z0_mm", z_min)),
                        float(fp.get("z1_mm", z_max)))
            z_an1 = max(float(fp.get("z0_mm", z_min)),
                        float(fp.get("z1_mm", z_max)))
            break
    cold = np.zeros(shape, dtype=bool)
    inside = RR <= r_cas
    for k in range(nz):
        z = zs[k]
        if z_an0 - 1e-9 <= z <= z_an1 + 1e-9:
            #  El anillo sigue el perfil LOCAL del liner, no un radio fijo.
            #
            #  BUG CORREGIDO (2026-08-01). Aquí iba `r_an_in`, que es
            #  `max(r_out) + t_liner` = 38.30 mm en el ancla: el radio del
            #  liner en su parte MÁS ANCHA, aplicado a TODOS los z. Donde el
            #  liner aún es estrecho (en el ancla, r_out = 28.71 mm en z ≈ 0)
            #  el hueco entre su pared real y esos 38.30 mm no quedaba ni en
            #  `gas` ni en `cold`, así que `solid = ~(gas|cold)` lo rellenaba
            #  de SÓLIDO FALSO. Medido: el parche de entrada daba 605 mm²
            #  frente a los 2622 mm² que el contrato declara en esa estación
            #  —×4.3—, y con ello la velocidad de entrada y la cabeza
            #  dinámica salían ×4.3 y ×18.6. Era la causa dominante de que el
            #  Δp/p medido saliera ×16 el de diseño.
            #
            #  No era un problema de resolución: el área convergía a 605 mm²
            #  entre dx = 1.6 y 0.35 mm (±3 % con 4.6× de refinamiento), que
            #  es justo la firma de un error de MODELO y no de malla.
            r_an_k = (r_out[k] + t_liner) if in_prof[k] else r_an_in
            cold[:, :, k] |= inside & (RR >= r_an_k - 1e-9)
        beyond = (z > z_dome) if reverse else (z < z_dome)
        if beyond:
            cold[:, :, k] |= inside              # pleno/giro del domo
    cold &= ~gas

    # --- capa de guarda y sólido ------------------------------------------
    solid = ~(gas | cold)
    solid[0] = solid[-1] = True
    solid[:, 0] = solid[:, -1] = True
    solid[:, :, 0] = solid[:, :, -1] = True
    gas &= ~solid
    cold &= ~solid

    # --- SELLADO del liner (pared más fina que un voxel) ------------------
    #  Se sella SACRIFICANDO CELDAS DE GAS, no de aire frío: el anillo de
    #  refrigeración es el paso MÁS ESTRECHO del combustor (2–3 mm en el
    #  ancla) y comérselo dejaría al liner sin alimentación — un error del
    #  100 %, no del 10 %. El liner es 4–8 veces más ancho y aguanta perder
    #  una capa. Se declara el volumen sacrificado.
    n_seal = 0
    seal_loss_frac = 0.0
    for i, c in enumerate(CI):
        if c == (0, 0, 0):
            continue
        tmp = np.empty_like(cold)
        stream_into(tmp, cold, (-c[0], -c[1], -c[2]), shape)
        leak = gas & tmp
        if leak.any():
            n_seal += int(leak.sum())
            gas &= ~leak
            solid |= leak
    if n_seal:
        seal_loss_frac = n_seal / max(int(gas.sum()) + n_seal, 1)
        notes.append(
            f"sellado del liner: {n_seal} celdas de GAS adyacentes al anillo "
            f"pasadas a sólido ({100.0 * seal_loss_frac:.1f} "
            f"% del volumen del liner; t_liner = {t_liner:.2f} mm < dx = "
            f"{dx:.2f} mm). Sin este paso el liner tendría fugas y los splits "
            "medidos serían basura; se sacrifica gas y no anillo porque el "
            "anillo es el paso más estrecho.")

    # --- taladros de las filas de orificios --------------------------------
    regions: dict[str, np.ndarray] = {}
    rows_meta: list[dict] = []
    XX = xs[:, None, None]
    YY = ys[None, :, None]
    ZZ = zs[None, None, :]
    for row in (contract.get("hole_rows") or []):
        if not isinstance(row, dict) or not row.get("flow", True):
            continue
        n_h = int(row.get("n", 0) or 0)
        d_h = float(row.get("d_mm", 0.0) or 0.0)
        if n_h <= 0 or d_h <= 0.0:
            continue
        z_h = float(row.get("z_mm", 0.0))
        r_h = float(row.get("r_mm", r_lo))
        phi0 = math.radians(float(row.get("phi0_deg", 0.0)))
        rid = str(row.get("id") or f"{row.get('zone', 'row')}")
        m_row = np.zeros(shape, dtype=bool)
        for kk in range(n_h):
            phi = phi0 + 2.0 * math.pi * kk / n_h
            ex, ey = math.cos(phi), math.sin(phi)
            #  distancia al eje del taladro (radial, en el plano z = z_h)
            dxp = XX - r_h * ex
            dyp = YY - r_h * ey
            dzp = ZZ - z_h
            along = dxp * ex + dyp * ey
            perp2 = dxp * dxp + dyp * dyp - along * along + dzp * dzp
            m_row |= (perp2 <= (0.5 * d_h) ** 2) & \
                     (np.abs(along) <= 0.5 * (t_liner + 2.0 * dx))
        m_row &= solid                       # solo se abre la PARED
        if m_row.any():
            solid &= ~m_row
            regions[f"hole:{rid}"] = m_row
        #  Área abierta REAL en la malla: se cuentan los voxels del taladro
        #  por corona radial de espesor dx y se toma la corona MÁS poblada —
        #  la sección del cilindro es constante, así que las coronas con
        #  menos son las recortadas por el `&= solid` de los extremos.
        #  Sirve para cuantificar el sesgo de discretización frente al
        #  `A_row_mm2` analítico del contrato (T44 lo reporta).
        n_vox = int(m_row.sum())
        if n_vox:
            rv = np.broadcast_to(RR[:, :, None], shape)[m_row]
            bins = np.floor((rv - float(rv.min())) / dx).astype(int)
            A_vox = float(np.bincount(bins).max()) * dx * dx
        else:
            A_vox = 0.0
        rows_meta.append(dict(
            id=rid, zone=str(row.get("zone", "")), n=n_h, d_mm=d_h,
            z_mm=z_h, r_mm=r_h, phi0_deg=float(row.get("phi0_deg", 0.0)),
            A_row_mm2=float(row.get("A_row_mm2", 0.0) or 0.0),
            A_vox_mm2=float(A_vox),
            n_vox=n_vox, Cd_contract=float(row.get("Cd", 0.0) or 0.0),
            J_contract=float(row.get("J", 0.0) or 0.0),
            mdot_contract_kgs=float(row.get("mdot_kgs", 0.0) or 0.0)))

    # --- swirlers (taladros axiales a través del domo) ---------------------
    sw = contract.get("swirler") or {}
    inj = contract.get("injectors") or {}
    n_sw = int(sw.get("count", 0) or 0)
    d_sw = float(sw.get("d_tip_mm", 0.0) or 0.0)
    r_ring = float(inj.get("ring_r_mm", 0.0) or 0.0) or 0.5 * (r_li + r_lo)
    m_sw = np.zeros(shape, dtype=bool)
    if n_sw > 0 and d_sw > 0.0:
        phi0 = math.radians(float(inj.get("phi0_deg", 0.0) or 0.0))
        for kk in range(n_sw):
            phi = phi0 + 2.0 * math.pi * kk / n_sw
            cx, cy = r_ring * math.cos(phi), r_ring * math.sin(phi)
            rad2 = (XX - cx) ** 2 + (YY - cy) ** 2
            axial = (ZZ >= z_dome) if reverse else (ZZ <= z_dome)
            m_sw |= (rad2 <= (0.5 * d_sw) ** 2) & axial
        m_sw &= solid
        if m_sw.any():
            solid &= ~m_sw
            regions["swirler"] = m_sw
    if n_sw > 0 and not m_sw.any():
        notes.append("los swirlers no abrieron ninguna celda: a este dx el "
                     "Ø de punta no llega a un voxel — el aire del domo entra "
                     "solo por la efusión y los orificios")

    gas &= ~solid
    cold &= ~solid
    regions["liner"] = gas
    regions["annulus"] = cold
    fluid = ~solid

    # --- parches de entrada y salida ---------------------------------------
    inlet = np.zeros(shape, dtype=bool)
    outlet = np.zeros(shape, dtype=bool)
    #  ENTRADA: el anillo empieza SIEMPRE en el z mínimo del dominio (el
    #  perfil de carcasa arranca ahí en las tres arquitecturas), así que el
    #  parche es el corte del ANILLO —no del pleno del domo, que a ese z no
    #  existe— en k = 1. Filtrar por `r ≥ r_an_in` es lo que evita capturar
    #  medio combustor cuando el domo cae del mismo lado.
    k_cold = 1 if abs(z_an0 - z_min) <= abs(z_an1 - z_max) else nz - 2
    #  El filtro radial usa el radio LOCAL del liner, igual que la
    #  construcción de `cold` (ver el bug corregido allí): con el `r_an_in`
    #  global el parche se quedaba en 605 mm² en vez de los 2622 mm² que el
    #  contrato declara en esa estación, porque en z ≈ 0 el liner todavía es
    #  estrecho. `cold` ya excluye gas y sólido, así que aquí basta con no
    #  volver a recortar por debajo del liner local.
    r_an_k = ((r_out[k_cold] + t_liner) if in_prof[k_cold] else r_an_in)
    inlet[:, :, k_cold] = cold[:, :, k_cold] & (RR >= r_an_k - 1e-9)
    #  SALIDA: la corona de gas en el extremo del liner (el que tenga gas).
    k_gas = max((1, nz - 2), key=lambda k: int(gas[:, :, k].sum()))
    outlet[:, :, k_gas] = gas[:, :, k_gas]
    off_in = (0, 0, 1) if k_cold == 1 else (0, 0, -1)
    off_out = (0, 0, 1) if k_gas == 1 else (0, 0, -1)
    regions["inlet"] = inlet
    regions["outlet"] = outlet

    A_inlet_m2 = float(inlet.sum()) * (dx * 1e-3) ** 2
    A_outlet_m2 = float(outlet.sum()) * (dx * 1e-3) ** 2
    if A_inlet_m2 <= 0.0 or A_outlet_m2 <= 0.0:
        notes.append("¡el dominio voxelizado no tiene entrada o salida! "
                     "revisa el contrato o baja dx")

    dom = dict(
        schema=SCHEMA, shape=shape, dx_mm=dx, solid=solid, fluid=fluid,
        regions=regions, rows=rows_meta, notes=notes,
        origin_mm=(float(xs[0]), float(ys[0]), float(z_min)),
        coords_mm=dict(x=xs, y=ys, z=zs), RR_mm=RR, PHI=PHI,
        r_casing_mm=r_cas, r_liner_in_mm=r_li, r_liner_out_mm=r_lo,
        t_liner_mm=t_liner, arch=arch, reverse=reverse,
        A_inlet_m2=A_inlet_m2, A_outlet_m2=A_outlet_m2,
        inlet_offset=off_in, outlet_offset=off_out,
        n_fluid=int(fluid.sum()), n_solid=int(solid.sum()),
        n_gas=int(gas.sum()), n_cold=int(cold.sum()),
        n_seal=int(n_seal), seal_loss_frac=float(seal_loss_frac),
        z_dome_mm=z_dome, bulge_mm=bulge, source="voxelize_contract")
    if log:
        log(f"    malla {nx}×{ny}×{nz} = {nx * ny * nz} nodos "
            f"(dx = {dx:.2f} mm), fluido {dom['n_fluid']} "
            f"(liner {dom['n_gas']}, anillo {dom['n_cold']})")
    return dom


def effusion_cd(contract: dict) -> float:
    """C_d de efusión que usa este módulo — **la misma función que la red 1D**.

    Delega en `physics_core.effusion_discharge_coefficient` (fuente única
    §12.b) con el Ø de efusión que declara el propio contrato. Existe como
    función PÚBLICA y barata para que el test de regresión T49 pueda
    comparar este número con el de `crn_core.effusion_cd` sin voxelizar nada
    ni lanzar el LBM — que es justo lo que faltaba cuando los dos módulos
    divergieron un 11.8 %.

    Nunca lanza: si `physics_core` no resuelve, devuelve el respaldo
    ETIQUETADO `EFFUSION_CD_FALLBACK`.
    """
    eff = contract.get("effusion") if isinstance(contract, dict) else None
    d_mm = 0.6
    if isinstance(eff, dict):
        try:
            d_mm = float(eff.get("d_mm") or 0.6)
        except Exception:
            d_mm = 0.6
    try:
        return float(pc.effusion_discharge_coefficient(1e-3 * max(d_mm, 0.05)))
    except Exception:
        return float(EFFUSION_CD_FALLBACK)


def _effusion_sites(dom, contract, units, mdot_cap_lat):
    """Franjas axiales de transpiración que modelan la efusión.

    A dx ≥ 1 mm un orificio de efusión de 0.6 mm no existe en la malla. Se
    reparte el área abierta total (`effusion.open_area_mm2`) entre franjas de
    una celda de espesor axial dentro de las `effusion.zones[]`, cada una con
    su pareja de nodos «frío junto a la pared» / «gas junto a la pared». El
    caudal de cada franja lo fija el Δp LOCAL del propio LBM.

    FUENTE ÚNICA DE LA EFUSIÓN (arreglado el 2026-07-31)
    ----------------------------------------------------
    Aquí no se calcula ni un área ni un C_d de efusión: los dos salen de
    `physics_core` §12.b, que es el ÚNICO sitio donde viven.

    * área abierta → `contract["effusion"]["open_area_mm2"]`, que
      `geometry_generator` integra sobre la polilínea REAL y **contrasta**
      contra `physics_core.effusion_wall_area_m2` antes de escribirla
      (`contract["validation"]["effusion"]`); la red 1D lee la misma cifra
      vía `record["A_effusion_m2"]`.
    * C_d → `effusion_cd(contract)` = tabla × `CD_EFFUSION_INCLINED` ×
      rugosidad AM (Ciencia §4.1), exactamente el mismo número que usa
      `crn_core.effusion_cd`.

    Antes de esa fecha este módulo usaba 854.1 mm² con C_d = 0.2470 y la red
    1D 621.4 mm² con C_d = 0.2210: ×1.536 de conductancia C_d·A al MISMO Δp
    para el MISMO agujero. Lo cubre el test de regresión T49."""
    eff = contract.get("effusion") or {}
    A_open_mm2 = float(eff.get("open_area_mm2", 0.0) or 0.0)
    zones = eff.get("zones") or []
    if A_open_mm2 <= 0.0 or not zones:
        return None
    dx = dom["dx_mm"]
    zs = dom["coords_mm"]["z"]
    RR = dom["RR_mm"]
    gas, cold = dom["regions"]["liner"], dom["regions"]["annulus"]
    r_lo = dom["r_liner_out_mm"]
    band = 2.0 * dx
    ks: list[int] = []
    for z in zones:
        z0 = float(z.get("z0_mm", 0.0))
        z1 = float(z.get("z1_mm", 0.0))
        lo, hi = min(z0, z1), max(z0, z1)
        ks += [k for k in range(1, len(zs) - 1) if lo <= zs[k] <= hi]
    ks = sorted(set(ks))
    slabs = []
    chunks = []
    pos = 0
    for k in ks:
        near = (RR >= r_lo - band) & (RR <= r_lo + band + dom["t_liner_mm"])
        c = cold[:, :, k] & near
        g = gas[:, :, k] & near
        if not c.any() or not g.any():
            continue
        ic = _flat_slice(dom["shape"], c, k)
        ig = _flat_slice(dom["shape"], g, k)
        slabs.append(dict(k=k, cold=ic, gas=ig,
                          n_cold=int(ic.size), n_gas=int(ig.size),
                          slice_cold=slice(pos, pos + ic.size),
                          slice_gas=slice(pos + ic.size,
                                          pos + ic.size + ig.size)))
        chunks += [ic, ig]
        pos += ic.size + ig.size
    if not slabs:
        return None
    A_lat_tot = A_open_mm2 / (dx * dx)
    for sl in slabs:
        sl["A_lat"] = A_lat_tot / len(slabs)
    cd = effusion_cd(contract)
    return dict(slabs=slabs, cd=float(cd), mdot_cap_lat=float(mdot_cap_lat),
                A_open_mm2=A_open_mm2, n_slabs=len(slabs),
                idx=np.concatenate(chunks))


def _flat_slice(shape, mask2d, k):
    """Índices planos del plano `z = k` marcado por `mask2d` (nx, ny)."""
    full = np.zeros(shape, dtype=bool)
    full[:, :, k] = mask2d
    return np.flatnonzero(full.reshape(-1))


# ==========================================================================
# 8. Punto de entrada de la fidelidad L1.5
# ==========================================================================
def xlb_available() -> bool:
    """¿Están XLB y JAX instalados? Único sitio donde se comprueba."""
    try:
        import jax                                    # noqa: F401,PLC0415
        import xlb                                    # noqa: F401,PLC0415
        return True
    except Exception:
        return False


XLB_AVAILABLE = xlb_available()


def _solve_xlb(dom, units, contract, deadline, log):
    """Adaptador de XLB — ENCAPSULADO AQUÍ Y SOLO AQUÍ (WAVE5 §D: su API se
    mueve rápido). En esta máquina no hay ni `jax` ni `xlb`, así que devuelve
    `None` y `solve_cold` cae al fallback de NumPy con etiqueta honesta.

    Cuando se instale, este adaptador debe: construir un `xlb.Grid` con la
    misma máscara `dom["solid"]`, aplicar las mismas BC (velocidad de
    entrada, presión de salida, bounce-back), correr con LES Smagorinsky y
    devolver `gbar` con la MISMA convención (promedio post-colisión) para que
    `region_flux` siga siendo exacto."""
    if not XLB_AVAILABLE:
        return None
    try:                                              # pragma: no cover
        import xlb                                    # noqa: F401,PLC0415
        raise NotImplementedError(
            "adaptador XLB pendiente de pinear versión (WAVE5 §D)")
    except Exception:
        return None


def solve_cold(contract: dict, *, dx_mm: float | None = None,
               max_nodes: int | None = None,
               steps: int = LBM_STEPS_MAX, avg: int = LBM_AVG_STEPS,
               timeout_s: float | None = None, backend: str = "auto",
               smagorinsky: bool = True, vdb_path: str | None = None,
               effusion: bool = True, precision: str = "fp32",
               collision: str | None = None, trt_lambda=TRT_LAMBDA,
               trt_omega_a=TRT_OMEGA_A, log=None) -> dict | None:
    """Contrato de geometría → campo frío promediado. `None` si no se pudo.

    Parámetros
    ----------
    contract : dict
        `phycb-combustor-1` de `geometry_generator.build_contract`.
    dx_mm : float | None
        Voxel. `None` = el que exige la geometría dentro del presupuesto
        (`auto_dx_mm` en CPU, `gpu_dx_mm` en GPU).
    max_nodes : int | None
        Techo de nodos. `None` = automático: `LBM_MAX_NODES` en CPU, y en GPU
        lo que quepa en la **VRAM libre medida** (`gpu_node_budget`).
    backend : {"auto", "xlb", "cupy", "numpy"}
        `"auto"` prueba XLB → CuPy → NumPy y ETIQUETA cuál salió en
        `field["backend"]`, `field["source"]` y `cfd_mesh["backend"]`. La
        variable de entorno `PHYCB_LBM_BACKEND` manda sobre este argumento.
    collision : {"bgk", "trt", "mrt"} | None
        Operador de colisión. `None` = `resolve_collision()`, es decir
        `PHYCB_LBM_COLLISION` si está y si no `DEFAULT_COLLISION`. El
        operador ACTIVO se reporta en `field["source"]`,
        `field["collision"]` y `cfd_mesh["scheme"]`/`["collision"]`, y es
        quien elige el `TAU_FLOOR_*` que se aplica.
    trt_lambda : float | None
        Λ de TRT (ver `TRT_LAMBDA`). Se ignora con BGK y MRT.
    precision : {"fp32", "fp64"}
        Precisión de trabajo del solver. FP32 por defecto (Ciencia §11.2: el
        FP64 de una GPU de consumo va a 1/64 del FP32). El acumulador del
        promedio temporal va SIEMPRE en FP64 salvo que no quepa en VRAM, en
        cuyo caso baja a FP32 **con nota**.
    vdb_path : str | None
        `*_field.vdb` de la capa 5c. Si no está o no hay lector de OpenVDB,
        se re-voxeliza analíticamente el contrato (se declara en `notes`).

    **Nunca lanza.** Cualquier fallo devuelve `None` y deja el motivo en
    `LAST_STATUS` (`"ok"·"degenerate"·"diverged"·"timeout"·"no_geometry"`).
    """
    global LAST_STATUS
    LAST_STATUS = "not_run"
    if not isinstance(contract, dict):
        LAST_STATUS = "degenerate"
        return None
    t_ini = time.monotonic()
    deadline = t_ini + float(LBM_TIMEOUT_S if timeout_s is None else timeout_s)
    notes: list[str] = []
    dtype = np.float64 if str(precision).lower() in ("fp64", "float64",
                                                     "double") else np.float32
    try:
        bc = contract.get("cfd_boundary_conditions") or {}
        P3 = float(bc.get("P3_Pa", 0.0) or 0.0)
        T3 = float(bc.get("T3_K", 0.0) or 0.0)
        mdot = float(bc.get("mdot_air_kgs", 0.0) or 0.0)
        rho3 = float(bc.get("rho3_kgm3", 0.0) or 0.0) or \
            (P3 / max(pc.RGAS * max(T3, 1.0), 1e-9))
        if not (P3 > 0.0 and T3 > 0.0 and mdot > 0.0 and rho3 > 0.0):
            LAST_STATUS = "degenerate"
            return None

        vdb = read_vdb_field(vdb_path) if vdb_path else None
        if vdb is None:
            notes.append(
                "campo de voxels re-VOXELIZADO analíticamente desde el "
                "contrato: no hay `*_field.vdb` legible (pyopenvdb no está "
                "instalado en esta máquina). Es la misma geometría de la que "
                "sale el .vdb, sin los redondeos y teardrops que añade PicoGK.")
        else:
            notes.append(f"campo de voxels leído de {os.path.basename(vdb_path)} "
                         f"(voxel {vdb['voxel_size_mm']:.3f} mm); la máscara se "
                         "reconstruye igualmente del contrato para conservar "
                         "las etiquetas de región (fila a fila).")
        # --- backend: XLB → CuPy → NumPy, con etiqueta -------------------
        #  El adaptador de XLB necesita `dom` y `units`, así que su intento
        #  vive más abajo; aquí solo se decide QUÉ módulo de arrays usa el
        #  núcleo propio, porque de eso depende el tamaño de malla que se
        #  puede pedir. `_solver_kwargs` ya mapea XLB → NumPy.
        try_xlb = str(backend).strip().lower() in ("auto", "xlb") \
            and XLB_AVAILABLE
        tag, xp, why = resolve_backend(backend)
        if tag == BACKEND_XLB:
            tag, xp, why = resolve_backend("auto_no_xlb")
        on_gpu = (tag == BACKEND_CUPY)
        acc_dtype = np.float64
        coll = resolve_collision(collision)
        coll_label = collision_label(coll, trt_lambda, trt_omega_a)

        # --- malla: la geometría manda, la VRAM dice si cabe --------------
        if dx_mm:
            dx = float(dx_mm)
            budget = int(max_nodes) if max_nodes else (
                gpu_node_budget(dtype, acc_dtype, collision=coll)
                if on_gpu else LBM_MAX_NODES)
        elif on_gpu:
            #  El techo de iteraciones sube en GPU: el transitorio necesita
            #  ~1/dx pasos y con el techo de CPU una malla fina se queda a
            #  medio converger (medido: 30 000 pasos insuficientes a
            #  dx = 0.82 mm en el ancla).
            if steps == LBM_STEPS_MAX:
                steps = LBM_STEPS_MAX_GPU
            dx, budget, gnotes = gpu_dx_mm(
                contract, dtype, acc_dtype,
                wall_budget_s=max(deadline - time.monotonic(), 1.0),
                steps_max=int(steps), collision=coll)
            notes += gnotes
            if max_nodes:
                budget = int(max_nodes)
                dx = max(dx, auto_dx_mm(contract, max_nodes=budget,
                                        annulus_cells=ANNULUS_TARGET_CELLS))
        else:
            budget = int(max_nodes) if max_nodes else LBM_MAX_NODES
            dx = auto_dx_mm(contract, budget)
        max_nodes = int(budget)

        if on_gpu:
            gi = gpu_info()
            need = _nodes_at_dx(contract, dx) * solver_bytes_per_node(
                dtype, acc_dtype, coll)
            free_b = gi.get("vram_free_GiB", 0.0) * 2 ** 30
            if need > GPU_VRAM_FRACTION * free_b:
                #  Último recurso ANTES de degradar la malla: bajar el
                #  acumulador a FP32. Cuesta 152 B/nodo menos (un 45 % del
                #  total) y el error del promedio sube de ~1e-7 a ~2e-6
                #  relativo, que sigue una década por debajo del 1e-3 que
                #  pide el cierre de masa de T41/T44e.
                acc_dtype = np.float32
                notes.append(
                    "acumulador del promedio temporal degradado a FP32 para "
                    "caber en VRAM: el error de redondeo del promedio sube a "
                    "~2e-6 relativo (el cierre de masa se pide a 1e-3).")
            notes.append(
                f"backend GPU `{tag}` — {gi.get('name', '?')} "
                f"({gi.get('vram_free_GiB', float('nan')):.2f} GiB libres de "
                f"{gi.get('vram_total_GiB', float('nan')):.2f}, compute "
                f"{gi.get('compute_capability', '?')}, cupy "
                f"{gi.get('cupy_version', '?')}); trabajo en "
                f"{np.dtype(dtype).name}, acumulador {np.dtype(acc_dtype).name}. "
                "La reducción de momentos en GPU NO es bit-idéntica a la de "
                "CPU (orden de suma distinto): a igual backend el resultado "
                "sí es reproducible.")
        else:
            notes.append(f"backend CPU `{tag}` — {why}.")

        #  Guarda de HONESTIDAD, no de estabilidad: por debajo de
        #  `ANNULUS_MIN_CELLS` el paso más estrecho del combustor no está
        #  resuelto y el reparto de aire medido no es «grueso», es FALSO. No
        #  se aborta (L1.5 sigue sirviendo para topología), se declara.
        _gs = _geometry_scales(contract)
        _ann_cells = _gs["annulus_gap_mm"] / max(dx, 1e-9)
        if _ann_cells < ANNULUS_MIN_CELLS:
            notes.append(
                f"⚠️ el anillo de refrigeración ({_gs['annulus_gap_mm']:.2f} mm) "
                f"queda con {_ann_cells:.2f} celdas a dx = {dx:.3f} mm, por "
                f"debajo del mínimo declarado ({ANNULUS_MIN_CELLS:.0f}): el "
                "campo sirve para TOPOLOGÍA pero el reparto de aire medido NO "
                "es utilizable. Sube el presupuesto de pared "
                "(`PHYCB_L15_TIMEOUT`) o fija `dx_mm` a mano.")
        dom = voxelize_contract(contract, dx, log=log)
        notes += dom["notes"]
        n_nodes = int(np.prod(dom["shape"]))
        if n_nodes > max_nodes:
            g = _geometry_scales(contract)
            notes.append(
                f"la malla ({n_nodes} nodos a dx = {dx:.3f} mm) supera el "
                f"presupuesto de {max_nodes}: la geometría manda sobre el "
                f"presupuesto porque el anillo de refrigeración mide "
                f"{g['annulus_gap_mm']:.2f} mm y el orificio más pequeño "
                f"{g['d_hole_min_mm']:.2f} mm. Con el fallback de NumPy esto "
                "son minutos-decenas de minutos de CPU; en GPU con CuPy son "
                "segundos-minutos.")
        if dom["A_inlet_m2"] <= 0.0 or dom["n_fluid"] < 1000:
            LAST_STATUS = "no_geometry"
            return None

        U_in = mdot / (rho3 * dom["A_inlet_m2"])
        units = lattice_units(dx * 1e-3, U_in, rho3, T3, collision=coll)
        if units["tau_floored"]:
            notes.append(
                f"τ suelado en {units['tau']:.6f} (el molecular sería "
                f"{units['tau_molecular']:.6f}, suelo de `{coll_label}` "
                f"{units['tau_floor']:.6f}): a Re real el operador no aguanta "
                f"en esta malla. Re efectivo del cálculo ≈ "
                f"{U_in * (2e-3 * dom['r_casing_mm']) / (units['nu_lat_used'] * units['C_u'] * units['dx_m']):.0f} "
                "frente al físico "
                f"{U_in * (2e-3 * dom['r_casing_mm']) / units['nu_phys_m2s']:.0f}. "
                "Los splits (C_d·A y Δp) y la trayectoria de los chorros "
                "(cantidad de movimiento) son poco sensibles; la capa límite "
                "NO está resuelta y no se usa.")

        inlet, outlet = dom["regions"]["inlet"], dom["regions"]["outlet"]
        sign = 1.0 if dom["inlet_offset"][2] > 0 else -1.0
        patches = [
            #  El caudal de aire es el DATO del punto de operación ⇒ BC de
            #  flujo másico, no de velocidad (ver `_apply_bc`).
            _patch(inlet, "massflux", dom["inlet_offset"],
                   u=(0.0, 0.0, sign * units["u_lat_in"]), dtype=dtype),
            _patch(outlet, "pressure", dom["outlet_offset"], rho=1.0,
                   dtype=dtype)]
        mdot_cap_lat = units["u_lat_in"] * float(inlet.sum())
        eff = _effusion_sites(dom, contract, units, mdot_cap_lat) \
            if effusion else None
        if eff is None:
            notes.append("sin modelo de efusión: desactivado, o el contrato no "
                         "declara área abierta, o la malla no resuelve el "
                         "hueco anillo↔liner en las zonas de efusión")
        else:
            notes.append(
                f"efusión modelada como transpiración distribuida en "
                f"{eff['n_slabs']} franjas (C_d = {eff['cd']:.3f}, "
                f"A = {eff['A_open_mm2']:.1f} mm²) gobernada por el Δp LOCAL "
                "del LBM; su cantidad de movimiento NO se modela.")

        if try_xlb:                                   # pragma: no cover
            out = _solve_xlb(dom, units, contract, deadline, log)
            if out is not None:
                LAST_STATUS = "ok"
                return out
            if str(backend).strip().lower() == "xlb":
                LAST_STATUS = "xlb_unavailable"
                return None
            notes.append("el adaptador de XLB no devolvió campo (WAVE5 §D): "
                         f"se usa el núcleo propio con `{tag}`.")
        elif tag == BACKEND_NUMPY:
            notes.append(
                "XLB/JAX no están instalados y no hay GPU con CuPy operativo "
                f"({cupy_reason()}): se usa el D3Q19 `{coll_label}` de "
                "NumPy (`lbm_numpy_fallback`).")

        if on_gpu and n_nodes < GPU_MIN_NODES:
            #  Por debajo de este tamaño la GPU pierde contra NumPy (latencia
            #  de lanzamiento > cálculo). Se degrada CON etiqueta.
            tag, xp, on_gpu = BACKEND_NUMPY, np, False
            notes.append(
                f"malla de {n_nodes} nodos por debajo de GPU_MIN_NODES "
                f"({GPU_MIN_NODES}): se calcula en NumPy, que a este tamaño "
                "es más rápido que la GPU.")
        def _build(_xp):
            return LBMSolver(dom["solid"], patches, units["tau"],
                             cs_smago=CS_SMAGORINSKY if smagorinsky else 0.0,
                             effusion=eff, xp=_xp, dtype=dtype,
                             acc_dtype=acc_dtype, collision=coll,
                             trt_lambda=trt_lambda,
                             trt_omega_a=trt_omega_a)
        try:
            #  La caída a NumPy solo se ofrece si la malla es asumible en CPU:
            #  rehacer 3 M nodos × 28 000 pasos en NumPy son ~35 h, y colgarse
            #  35 h es peor degradación que devolver `None` con el motivo.
            cpu_ok = n_nodes <= 8 * LBM_MAX_NODES
            gbar, sim, tag = _run_solver(
                _build, tag, steps_max=steps, avg=avg, deadline=deadline,
                log=log, allow_cpu_fallback=cpu_ok, notes=notes)
            on_gpu = tag == BACKEND_CUPY
        except FloatingPointError as exc:
            LAST_STATUS = "diverged"
            if log:
                log(f"    LBM divergido: {exc}")
            return None
        except Exception as exc:
            LAST_STATUS = f"gpu_error:{type(exc).__name__}"
            if log:
                log(f"    LBM abortado: {type(exc).__name__}: {exc}")
            return None
        rho, j = macro_from_g(gbar)

        fl = {}
        sh = dom["shape"]
        gas, cold = dom["regions"]["liner"], dom["regions"]["annulus"]
        fl["inlet"] = region_flux(gbar, inlet, cold & ~inlet, sh)
        fl["outlet"] = region_flux(gbar, gas & ~outlet, outlet, sh)
        for name, mask in dom["regions"].items():
            if name.startswith("hole:") or name == "swirler":
                fl[name] = region_flux(gbar, mask, gas, sh)
        fl["effusion"] = float(getattr(sim, "eff_total_lat", 0.0))
        if getattr(sim, "eff_cap_hits", 0):
            notes.append(
                f"⚠️ la COTA de seguridad de la efusión mordió en "
                f"{sim.eff_cap_hits}/{sim.eff_cap_updates} refrescos: la ley "
                f"de orificio pedía {sim.eff_demand_lat:.3f} lu y se recortó a "
                f"{sim.eff_cap_lat:.3f} ({100 * EFFUSION_MAX_FRAC:.0f} % del "
                "caudal de entrada). Mientras la cota muerda, `splits.cool` NO "
                "es una medida del campo sino el propio recorte, y la "
                "comparación con la red 1D no dice nada sobre la malla. Es un "
                "problema del MODELO de efusión (C_d·A frente al Δp local "
                "resuelto), no de resolución.")

        #  `source` lleva backend Y operador: es la etiqueta que dice por
        #  qué camino salió el número (contrato §3), y desde que hay tres
        #  operadores el backend solo ya no lo identifica.
        field = dict(
            schema=SCHEMA, ok=True,
            backend=tag, source=f"{tag}+{coll}", backend_reason=why,
            collision=coll, collision_label=coll_label,
            gpu=gpu_info() if on_gpu else {},
            dom=dom, units=units, rho=rho, j=j, gbar=gbar, flux_lat=fl,
            stats=sim.stats, notes=notes,
            mdot_ref_kgs=float(mdot), rho3_kgm3=float(rho3),
            P3_Pa=float(P3), T3_K=float(T3), U_in_ms=float(U_in),
            wall_s=float(time.monotonic() - t_ini),
            cfd_mesh=dict(
                nx=sh[0], ny=sh[1], nz=sh[2], dx_mm=dx,
                n_nodes=int(sh[0] * sh[1] * sh[2]), n_fluid=dom["n_fluid"],
                n_liner=dom["n_gas"], n_annulus=dom["n_cold"],
                A_inlet_mm2=dom["A_inlet_m2"] * 1e6,
                A_outlet_mm2=dom["A_outlet_m2"] * 1e6,
                annulus_gap_mm=_geometry_scales(contract)["annulus_gap_mm"],
                annulus_cells=_geometry_scales(contract)["annulus_gap_mm"] / dx,
                d_hole_min_cells=_geometry_scales(contract)["d_hole_min_mm"] / dx,
                backend=tag, scheme=coll_label, collision=coll,
                trt_lambda=(None if coll != COLLISION_TRT else
                            trt_lambda_effective(units["tau"], trt_lambda,
                                                 trt_omega_a)),
                trt_omega_a=(float(trt_omega_a)
                             if coll == COLLISION_TRT
                             and trt_omega_a is not None else None),
                tau_floor=units["tau_floor"],
                precision=str(np.dtype(dtype)),
                acc_precision=str(np.dtype(acc_dtype)),
                les="Smagorinsky" if smagorinsky else "none",
                tau=units["tau"], tau_molecular=units["tau_molecular"],
                tau_floored=bool(units["tau_floored"]),
                nu_ratio_effective=float(units["nu_lat_used"]
                                         / max(units["nu_lat"], 1e-30)),
                u_lattice_in=units["u_lat_in"],
                steps=sim.stats["steps"], converged=sim.stats["converged"],
                mlups=sim.stats["mlups"],
                s_per_cell_step=sim.stats["s_per_cell_step"],
                wall_s=sim.stats["wall_s"],
                effusion_capped=bool(getattr(sim, "eff_cap_hits", 0)),
                effusion_cap_hits=int(getattr(sim, "eff_cap_hits", 0)),
                effusion_demand_lat=float(getattr(sim, "eff_demand_lat", 0.0)),
                effusion_cap_lat=float(getattr(sim, "eff_cap_lat", 0.0)),
                seal_loss_frac=float(dom.get("seal_loss_frac", 0.0))))
        if not sim.stats["converged"]:
            notes.append("el LBM NO alcanzó el criterio de convergencia "
                         f"(<|u|> con tolerancia {LBM_CONV_TOL:g}) en "
                         f"{sim.stats['steps']} pasos: los números medidos son "
                         "de un transitorio promediado, trátalos como "
                         "indicativos.")
        LAST_STATUS = "ok" if sim.stats["converged"] else "not_converged"
        return field
    except Exception as exc:                          # red de seguridad
        LAST_STATUS = f"error:{type(exc).__name__}"
        if log:
            log(f"    solve_cold falló: {type(exc).__name__}: {exc}")
        return None


# ==========================================================================
# 9. CLI de comprobación
# ==========================================================================
def main(argv=None) -> int:
    """Smoke del solver frío sobre el caso ancla (o los casos canónicos)."""
    import argparse
    ap = argparse.ArgumentParser(
        description="Phy-CB L1.5 cold LBM smoke test")
    ap.add_argument("--case", default="canonical",
                    choices=("canonical", "anchor"))
    ap.add_argument("--dx-mm", type=float, default=None)
    ap.add_argument("--max-nodes", type=int, default=None)
    ap.add_argument("--backend", default="auto",
                    choices=("auto", "xlb", "cupy", "numpy"))
    ap.add_argument("--precision", default="fp32", choices=("fp32", "fp64"))
    args = ap.parse_args(argv)
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    print(f"XLB disponible : {'sí' if XLB_AVAILABLE else 'NO'}")
    gi = gpu_info()
    print(f"CuPy/GPU       : {gi.get('name', 'NO') if gi else 'NO'} "
          f"({cupy_reason()})"
          + (f" · {gi['vram_free_GiB']:.2f}/{gi['vram_total_GiB']:.2f} GiB "
             f"libres · cupy {gi['cupy_version']}" if gi else ""))
    tag, _, why = resolve_backend(args.backend)
    print(f"backend preferido: {tag} ({why}) — por debajo de "
          f"{GPU_MIN_NODES} nodos se calcula igualmente en NumPy; la etiqueta "
          "del resultado dice quién lo hizo DE VERDAD")
    if args.case == "canonical":
        r = canonical_channel(backend=args.backend, log=print)
        print(f"T41 canal [{r['backend']}]: ṁ_in={r['mdot_in_lat']:.6f} "
              f"ṁ_out={r['mdot_out_lat']:.6f} rel={r['rel_error']:.3e}")
        return 0
    import geometry_generator as gg
    pc.assert_provenance(pc, gg)
    rec = pc.evaluate(pc.THETA_REF, pc.Fidelity.L0, use_cache=False)
    con = gg.build_contract(pc.THETA_REF, rec)
    f = solve_cold(con, dx_mm=args.dx_mm, max_nodes=args.max_nodes,
                   backend=args.backend, precision=args.precision, log=print)
    if f is None:
        print(f"solve_cold devolvió None ({LAST_STATUS})")
        return 1
    sc = f["units"]["mdot_scale"]
    print(f"backend usado = {f['backend']} · malla "
          f"{f['cfd_mesh']['n_nodes']} nodos a dx = "
          f"{f['cfd_mesh']['dx_mm']:.3f} mm · {f['cfd_mesh']['mlups']:.1f} "
          f"MLUPS · {f['cfd_mesh']['wall_s']:.1f} s")
    print(f"ṁ entrada = {f['flux_lat']['inlet'] * sc:.6f} kg/s "
          f"(referencia {f['mdot_ref_kgs']:.6f})")
    for k, v in sorted(f["flux_lat"].items()):
        print(f"  {k:<18s} {v * sc:+.6f} kg/s")
    for n in f["notes"]:
        print(f"  · {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
