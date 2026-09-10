"""
QUASAR Phy-CB · cfd/ibrans/turbulence.py
========================================
**Hito M2 de WAVE6 — IMPLEMENTADO** (transporte completo). k-ε *realizable*
(Shih et al. 1995) con funciones de pared sobre |φ| (frontera inmersa) y
sobre paredes del dominio, y corrección de curvatura Spalart–Shur
simplificada. Verificado por la escalera V4–V5
(`cfd/tests/test_ibrans_v4_v5.py`); los números medidos están en
`ESTADO.md`.

Qué resuelve
------------
Transporte de k y ε en forma advectiva (la correcta a ρ variable), con el
MISMO kernel MUSCL+TVD y la MISMA velocidad de cara de Rhie–Chow que el
momento (`solver.advect_diffuse_scalar`)::

    Dk/Dt = ∇·[(ν + ν_t/σ_k)∇k] + P_k − ε
    Dε/Dt = ∇·[(ν + ν_t/σ_ε)∇ε] + C₁·S·ε·f_r − C₂·ε²/(k + √(νε))

    C₁ = max(0.43, η/(η+5)),  η = S·k/ε,  S = √(2·S_ij S_ij)
    ν_t = C_μ(S,Ω,W)·k²/ε        (C_μ variable de Shih, con A_s COMPLETO)
    P_k = ν_t·S²·f_r,  limitada a P_k ≤ PROD_LIMIT_RATIO·ε

Integración: los términos de transporte son explícitos (el dt acústico de
la pseudo-compresibilidad hace el CFL advectivo ≈ 0.04, sobra estabilidad)
y los términos fuente van **point-implicit** (el sumidero en el
denominador), que es lo que evita que la rigidez de la ecuación de ε mate
el paso explícito.

Funciones de pared: dos escalas (u* de k) + ley de Spalding
-----------------------------------------------------------
⚠️ DESVIACIÓN DECLARADA respecto del plan de ESTADO.md §7 (tarea 2.2), que
preveía `u_tau_spalding` puro. El Spalding puro obtiene u_τ de u_p, y en
separación/readherencia u_p → 0 ⇒ u_τ → 0 ⇒ ε_pared → 0 ⇒ ν_t = C_μk²/ε
explota — **exactamente en el punto de readherencia que V4 mide**. Se usa
en su lugar la formulación de dos escalas de Launder–Spalding: la escala de
velocidad es u* = C_μ^¼·√k (finita en la separación porque k no se anula),
y el perfil sigue siendo LA LEY DE SPALDING, invertida por bisección
acotada (`u_plus_of_ystar`) — válida de la subcapa a la banda log sin
ramas::

    u* = C_μ^¼·√k,   y* = u*·y/ν,   u⁺(y*) por Spalding
    τ_w/ρ  = u*·u_p / u⁺(y*)        (→ ν·u_p/y en el límite laminar)
    ε_w    = u*³/(κ·y)              (fija en la celda de pared; finita)
    P_w    = (τ_w/ρ)·u*/(κ·y)       (equilibrio; → 0 en la separación)
    ν_wall = u*·y/u⁺(y*) − ν        (≥ 0; → 0 en el límite laminar)

`ν_wall` es la viscosidad efectiva que hace que el flujo DISCRETO de
momento a través de la cara de pared (perfil lineal del ghost-cell, tanto
del IB como del contorno del dominio) reproduzca τ_w — es decir, la ley de
pared entra al momento por la viscosidad de cara, no cambiando el estencil.
Con IB la distancia es y = |φ| **gratis** (el argumento estructural de la
arquitectura, Ciencia §11.1). Riesgo declarado (Ciencia §11.3 #3): las
funciones de pared sobre IB son imprecisas en la capa límite térmica — por
eso L2a NO promete temperatura de metal.

Corrección de curvatura (riesgo #2 del sistema)
-----------------------------------------------
Spalart–Shur **simplificada** (declarado): f_r = (1+c_r1)·2r*/(1+r*) − c_r1
con r* = S/Ω, recortada a [0, FR_MAX]. El término r̃ (derivada lagrangiana
del tensor de deformación) se omite: exigiría transportar DS_ij/Dt. Con
r̃ = 0: cizalla pura (S=Ω) ⇒ f_r = 1 (sin efecto); rotación de sólido
rígido (S→0) ⇒ f_r → 0 (mata la producción en el núcleo del vórtice, que
es lo que preserva la recirculación del swirler — el motivo de que WAVE6 §0
la exija «desde v1»); deformación irrotacional ⇒ f_r → FR_MAX.

API pública (la firma de M1 se conserva; lo nuevo es aditivo)
-------------------------------------------------------------
    available() -> bool                     (True: transporte operativo)
    STATUS : dict
    KAPPA, E_WALL, C_MU, C2_EPS, SIGMA_K, SIGMA_EPS, A0_REALIZABLE
    u_tau_log_law / u_tau_spalding / spalding_yplus / yplus   [M1, intactas]
    u_plus_of_ystar(ystar, ...) -> u+                         [M2]
    c_mu_realizable(S, Omega, k, eps, w_star=None) -> C_mu    [M2: A_s full]
    curvature_correction(k, eps, S, Omega) -> f_r             [M2]
    nu_t(k, eps, c_mu) -> ndarray
    KEpsilonRealizable(mesh, cfg)
        .attach(solver)          ← el enganche M2 (una línea)
        .advance(solver, dt)     ← lo llama solver.step() vía solver.physics
        .initialize() .rhs() .step() .apply_wall()   (compat M1)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# --- constantes de pared (estándar de la literatura) ---------------------
KAPPA = 0.41           # constante de von Kármán
E_WALL = 9.793         # rugosidad lisa (valor de OpenFOAM/Fluent)
YPLUS_LAM = 11.225     # cruce subcapa/log

# --- k-ε realizable (Shih, Liou, Shabbir, Yang & Zhu, 1995) --------------
C_MU = 0.09            # valor de referencia; en realizable C_μ es variable
C2_EPS = 1.9
SIGMA_K = 1.0
SIGMA_EPS = 1.2
A0_REALIZABLE = 4.04

#: Banda objetivo de y⁺ en el liner (WAVE6 §2). Fuera de ella = warning.
YPLUS_TARGET = (30.0, 150.0)

#: Iteraciones FIJAS de las inversiones de Spalding (contrato §10.5).
SPALDING_ITERS = 30

#: Limitador de producción P_k ≤ ratio·ε. ⚠️ TUNABLE (contrato §10.7):
#: 10.0 es el valor estándar de los códigos industriales (Fluent usa 10);
#: evita que un transitorio de arranque dispare k antes de que ε responda.
PROD_LIMIT_RATIO = 10.0

#: Corrección de curvatura: c_r1 y recorte de f_r. ⚠️ TUNABLES (contrato
#: §10.7): c_r1 = 1.0 y clip [0, 1.25] son los valores de Spalart–Shur
#: (1997) tal como los adopta la variante SST-RC de Smirnov & Menter
#: (2009). Recalibrar sólo si V9 (PRECCINSTA) lo exige.
CR1_CURV = 1.0
FR_MAX = 1.25

STATUS = {
    "wall_functions": ("M2 · IMPLEMENTADO (dos escalas u*=C_mu^1/4·sqrt(k) "
                       "+ ley de Spalding invertida; sobre |phi| en el IB y "
                       "sobre las paredes del dominio)"),
    "yplus": "M2 · IMPLEMENTADO",
    "c_mu_realizable": "M2 · IMPLEMENTADO (Shih 1995 con A_s COMPLETO via W)",
    "nu_t": "M2 · IMPLEMENTADO (acoplado al transporte)",
    "k_eps_transport": ("M2 · IMPLEMENTADO (adveccion MUSCL+TVD con la uf de "
                        "Rhie-Chow del momento; fuentes point-implicit)"),
    "curvature_correction": ("M2 · IMPLEMENTADO (Spalart-Shur simplificada "
                             "con r_tilde=0 DECLARADO; riesgo #2)"),
    "ib_first_cell": ("M2 · IMPLEMENTADO (banda fluida adyacente al ghost: "
                      "eps fija, P de equilibrio, nu_wall efectiva)"),
    "energy_variable_density": "M2 · IMPLEMENTADO en energy.py",
}


def available() -> bool:
    """¿Está el modelo de turbulencia operativo? **Sí** (M2). Lo consulta
    `__init__.solve()`; el producto L2a completo sigue necesitando M3+M4."""
    return True


# ==========================================================================
# 1. Funciones de pared
# ==========================================================================
def yplus(u_tau, y, nu) -> np.ndarray:
    """y⁺ = u_τ·y/ν. Con IB, `y = |φ|` sale gratis de la malla."""
    return np.asarray(u_tau) * np.asarray(y) / max(float(nu), 1e-30)


def u_tau_log_law(u_p, y_p, nu, kappa: float = KAPPA, e: float = E_WALL,
                  iters: int = SPALDING_ITERS):
    """u_τ a partir de la velocidad tangencial `u_p` a distancia `y_p`,
    con la ley logarítmica u⁺ = ln(E·y⁺)/κ y **punto fijo acotado** (sin
    `while` abiertos).

    Bajo y⁺ = 11.225 se usa la rama laminar u⁺ = y⁺ (u_τ = √(ν·u_p/y_p)),
    que es el tratamiento estándar de dos ramas.
    """
    u_p = np.abs(np.asarray(u_p, dtype=float))
    y_p = np.maximum(np.asarray(y_p, dtype=float), 1e-30)
    nu = max(float(nu), 1e-30)
    ut = np.sqrt(nu * u_p / y_p)                       # arranque laminar
    for _ in range(int(iters)):
        yp = np.maximum(ut * y_p / nu, 1e-12)
        up = np.where(yp > YPLUS_LAM, np.log(e * yp) / kappa, yp)
        ut = np.where(up > 1e-12, u_p / np.maximum(up, 1e-12), ut)
        ut = np.maximum(ut, 1e-30)
    return ut


def spalding_yplus(u_plus, kappa: float = KAPPA, e: float = E_WALL, xp=np):
    """Ley única de **Spalding (1961)**, válida de la subcapa a la banda
    logarítmica sin ramas::

        y⁺ = u⁺ + (1/E)·[e^{κu⁺} − 1 − κu⁺ − (κu⁺)²/2 − (κu⁺)³/6]

    Monótona creciente en u⁺, que es lo que permite invertirla por
    bisección con garantía. `xp` = numpy o cupy (misma fórmula).
    """
    up = xp.maximum(xp.asarray(u_plus), 0.0)
    ku = xp.minimum(kappa * up, 300.0)          # tope anti-overflow
    return up + (xp.exp(ku) - 1.0 - ku - ku ** 2 / 2.0 - ku ** 3 / 6.0) / e


def u_tau_spalding(u_p, y_p, nu, kappa: float = KAPPA, e: float = E_WALL,
                   iters: int = SPALDING_ITERS, xp=np):
    """u_τ invirtiendo la ley de Spalding por **bisección acotada**.

    Se usa bisección y no Newton A PROPÓSITO: el residuo tiene una
    exponencial que hace divergir a Newton desde un arranque pobre (se
    verificó: con u⁺ ≈ 100 el paso de Newton se colapsa y 30 iteraciones
    no llegan a la raíz). `g(u_τ) = y⁺_Spalding(u_p/u_τ) − y_p·u_τ/ν` es
    **estrictamente decreciente**, así que la bisección converge siempre y
    en un número FIJO de pasos (contrato §10.5: sin `while` abiertos).

    En M2 esta función queda como **diagnóstico** (y⁺ del reporte) y juez
    de verificación; el acoplamiento usa `u_plus_of_ystar` (dos escalas,
    ver docstring del módulo).
    """
    u_p = xp.abs(xp.asarray(u_p, dtype=float))
    y_p = xp.maximum(xp.asarray(y_p, dtype=float), 1e-30)
    nu = max(float(nu), 1e-30)
    # Cota superior GARANTIZADA: con u_τ = 2·max(u_p, ν/y_p) se tiene
    # u⁺ ≤ 0.5 (⇒ y⁺_Spalding ≤ 0.5) y y_p·u_τ/ν ≥ 2, luego g < 0.
    # Cota inferior: 0 (con u⁺ → ∞, g → +∞).
    hi = 2.0 * xp.maximum(u_p, nu / y_p) + 1e-300
    lo = xp.zeros_like(hi)
    n = max(int(iters), 60)
    for _ in range(n):
        mid = 0.5 * (lo + hi)
        g = spalding_yplus(u_p / xp.maximum(mid, 1e-300), kappa, e, xp) \
            - y_p * mid / nu
        lo = xp.where(g > 0.0, mid, lo)
        hi = xp.where(g > 0.0, hi, mid)
    return 0.5 * (lo + hi)


def u_plus_of_ystar(ystar, kappa: float = KAPPA, e: float = E_WALL,
                    iters: int = SPALDING_ITERS, xp=np):
    """u⁺(y*) invirtiendo la ley de Spalding por **bisección acotada**
    (contrato §10.5: pasos fijos). Es la inversa «en el otro sentido» que
    `u_tau_spalding`: aquí la incógnita es u⁺ dado y*.

    Cotas garantizadas: y⁺_Spalding(u⁺) ≥ u⁺ ⇒ la raíz está en [0, y*].
    Límites correctos: y* pequeño ⇒ u⁺ ≈ y* (subcapa); y* grande ⇒ banda
    logarítmica. Monótona ⇒ converge siempre.
    """
    ys = xp.maximum(xp.asarray(ystar), 1e-12)
    lo = xp.zeros_like(ys)
    hi = ys * 1.0
    n = max(int(iters), 60)
    for _ in range(n):
        mid = 0.5 * (lo + hi)
        g = spalding_yplus(mid, kappa, e, xp) - ys
        lo = xp.where(g < 0.0, mid, lo)
        hi = xp.where(g < 0.0, hi, mid)
    return 0.5 * (lo + hi)


# ==========================================================================
# 2. Cierre realizable — fórmulas
# ==========================================================================
def c_mu_realizable(s_mag, omega_mag, k, eps, a0: float = A0_REALIZABLE,
                    w_star=None, xp=np):
    """C_μ variable del k-ε *realizable* (Shih et al. 1995)::

        C_μ = 1 / (A0 + A_s·U*·k/ε)
        U*  = √(S_ij S_ij + Ω_ij Ω_ij)
        A_s = √6·cos(φ),  φ = (1/3)·acos(√6·W),  W = S_ij S_jk S_ki / S̃³

    Convención de entrada: `s_mag = √(2·S_ij S_ij)` y
    `omega_mag = √(2·Ω_ij Ω_ij)` (las magnitudes estándar que usan la
    producción y C₁), de modo que U* = √((s_mag² + omega_mag²)/2).

    `w_star = W` activa el A_s COMPLETO (M2, con el tensor disponible);
    con `w_star=None` se congela A_s en su valor de cizalla pura
    (W = 0 ⇒ φ = π/6 ⇒ A_s = √6·cos(π/6)), que en la banda logarítmica
    recupera C_μ ≈ 0.09 — comprobado en el test M2d.
    """
    s_mag = xp.asarray(s_mag, dtype=float)
    omega_mag = xp.asarray(omega_mag, dtype=float)
    k = xp.maximum(xp.asarray(k, dtype=float), 1e-12)
    eps = xp.maximum(xp.asarray(eps, dtype=float), 1e-12)
    u_star = xp.sqrt(0.5 * (s_mag ** 2 + omega_mag ** 2))
    if w_star is None:
        a_s = math.sqrt(6.0) * math.cos(math.pi / 6.0)
    else:
        w = xp.clip(math.sqrt(6.0) * xp.asarray(w_star, dtype=float),
                    -1.0, 1.0)
        a_s = math.sqrt(6.0) * xp.cos(xp.arccos(w) / 3.0)
    return 1.0 / (a0 + a_s * u_star * k / eps)


def nu_t(k, eps, c_mu=C_MU, xp=np):
    """ν_t = C_μ k²/ε (acoplada al transporte desde M2)."""
    k = xp.maximum(xp.asarray(k, dtype=float), 0.0)
    eps = xp.maximum(xp.asarray(eps, dtype=float), 1e-12)
    return xp.asarray(c_mu) * k ** 2 / eps


def curvature_correction(k, eps, s_mag, omega_mag, radius=None,
                         c_r1: float = CR1_CURV, fr_max: float = FR_MAX,
                         xp=np):
    """Corrección de curvatura **Spalart–Shur simplificada** (riesgo #2)::

        f_r = clip( (1+c_r1)·2r*/(1+r*) − c_r1 , 0 , fr_max ),   r* = S/Ω

    Simplificación DECLARADA: el término r̃ de Spalart–Shur (derivada
    lagrangiana de S_ij) se omite (r̃ = 0). Comportamiento: cizalla pura
    (r* = 1) ⇒ f_r = 1; rotación de sólido rígido (r* → 0) ⇒ f_r → 0, que
    apaga la producción en el núcleo del vórtice y preserva la
    recirculación del swirler. `k`, `eps` y `radius` se conservan en la
    firma (congelada en M1) pero esta forma no los necesita.
    """
    s = xp.asarray(s_mag, dtype=float)
    o = xp.maximum(xp.asarray(omega_mag, dtype=float), 1e-30)
    r = s / o
    fr = (1.0 + c_r1) * (2.0 * r / (1.0 + r)) - c_r1
    return xp.clip(fr, 0.0, fr_max)


@dataclass
class TurbulenceConfig:
    """Parámetros del modelo (la firma de M1 se conserva; lo nuevo tiene
    default y es aditivo)."""
    model: str = "k-eps-realizable"
    wall_function: str = "spalding-two-scale"
    curvature_correction: bool = True
    intensity: float = 0.05          # I de entrada
    length_scale_m: float = 0.007    # ℓ = 0.07·D_h (geometry_generator)
    k_floor: float = 1e-10
    eps_floor: float = 1e-10
    nu_t_max_ratio: float = 1.0e5    # limitador de ν_t/ν
    sigma_k: float = SIGMA_K
    sigma_eps: float = SIGMA_EPS
    c2: float = C2_EPS
    prod_limit: float = PROD_LIMIT_RATIO


class KEpsilonRealizable:
    """k-ε realizable ACOPLADO (M2). Enganche en dos líneas::

        tur = KEpsilonRealizable(mesh).attach(solver)
        info = solver.run()          # solver.step() avanza el modelo solo

    `attach` registra el modelo en `solver.physics`, apunta
    `solver.nu_field` a la ν efectiva (ν + ν_t) y deriva los contornos de
    k y ε de los del solver (`inlet` puede traer `k=`/`eps=` en su spec;
    si no, se usan I y ℓ de `TurbulenceConfig`). El IB no necesita nada:
    la banda de pared se detecta con las máscaras de la malla y la
    distancia es |φ|.
    """

    name = "k-eps-realizable"

    def __init__(self, mesh, cfg: TurbulenceConfig | None = None):
        self.mesh = mesh
        self.cfg = cfg or TurbulenceConfig()
        self.k = np.zeros(mesh.shape_tot)
        self.eps = np.zeros(mesh.shape_tot)
        self.nu_t_field = np.zeros(mesh.shape_tot)
        self.nu_eff = None
        self.solver = None
        self.status = "detached"
        self._max_diff = 0.0
        self._band = None
        self._wall_planes: list[dict] = []
        self._last_ystar = None

    # -- valores de entrada ------------------------------------------------
    def inlet_values(self, u_ref: float) -> tuple[float, float]:
        """(k, ε) de entrada desde I y ℓ (los que escribe
        `geometry_generator.cfd_boundary_conditions`)."""
        i = float(self.cfg.intensity)
        ell = max(float(self.cfg.length_scale_m), 1e-9)
        k0 = 1.5 * (i * float(u_ref)) ** 2
        eps0 = (C_MU ** 0.75) * k0 ** 1.5 / ell
        return max(k0, self.cfg.k_floor), max(eps0, self.cfg.eps_floor)

    def initialize(self, u_ref: float = 1.0) -> None:
        """Campos iniciales uniformes (compat M1). `attach` ya lo hace."""
        k0, e0 = self.inlet_values(u_ref)
        self.k[...] = k0
        self.eps[...] = e0
        self.nu_t_field[...] = min(C_MU * k0 * k0 / e0,
                                   self.cfg.nu_t_max_ratio * 1e-5)

    # -- enganche M2 -------------------------------------------------------
    def attach(self, solver) -> "KEpsilonRealizable":
        """Registra el modelo en el solver: crea k, ε y ν_t en el backend
        del solver, deriva los contornos y engancha `solver.nu_field`."""
        from .mesh import _dilate
        from .solver import scalar_bc_spec

        xp, m = solver.xp, solver.mesh
        dt_ = solver.dtype
        self.solver = solver
        k0, e0 = self.inlet_values(solver.cfg.u_ref)
        nut0 = min(C_MU * k0 * k0 / max(e0, 1e-30),
                   self.cfg.nu_t_max_ratio * solver.cfg.nu)
        self.k = xp.full(m.shape_tot, dt_(k0), dtype=dt_)
        self.eps = xp.full(m.shape_tot, dt_(e0), dtype=dt_)
        self.nu_t_field = xp.full(m.shape_tot, dt_(nut0), dtype=dt_)
        self.nu_eff = xp.full(m.shape_tot, dt_(solver.cfg.nu + nut0),
                              dtype=dt_)
        self.k_bc = scalar_bc_spec(solver, "k", k0)
        self.e_bc = scalar_bc_spec(solver, "eps", e0)
        # spec «neutro» para ν_t: periódico donde toque, Neumann en el resto
        self.n_bc = {fk: (("periodic", None) if sp[0] == "periodic"
                          else ("neumann", None))
                     for fk, sp in self.k_bc.items()}
        # banda del IB: celdas FLUIDAS del INTERIOR adyacentes a un
        # fantasma; su distancia a pared es |φ| exacta (el argumento de la
        # arquitectura). Se restringe al interior para que los índices
        # `tot` (shape_tot) e `int` (shape) sean 1:1 en orden C.
        self._band = None
        if m.phi is not None and m.n_ghost > 0:
            imask = np.zeros(m.shape_tot, dtype=bool)
            imask[m.interior] = True
            band = m.fluid & _dilate(m.ghost, 1) & imask
            tot = np.flatnonzero(band.reshape(-1))
            if tot.size:
                iidx = np.flatnonzero(band[m.interior].reshape(-1))
                y = np.abs(m.phi).reshape(-1)[tot]
                nrm = [n.reshape(-1)[tot] for n in m.normals]
                self._band = dict(
                    tot=xp.asarray(tot), int=xp.asarray(iidx),
                    y=xp.asarray(y, dtype=dt_),
                    n=[xp.asarray(v, dtype=dt_) for v in nrm])
        # paredes del dominio: primera celda a y = dx/2. Los planos sobre
        # los arrays CON fantasmas recortan los ejes transversales al
        # interior, para casar con los planos de los arrays interiores.
        self._wall_planes = []
        ng = m.ng
        for d in range(m.ndim):
            n = m.shape[d]
            for side, im, ii in (("lo", ng, 0), ("hi", ng + n - 1, n - 1)):
                spec = solver.bcs[f"{'xyz'[d]}{side}"]
                if spec["kind"] != "wall":
                    continue
                pm = [slice(ng, ng + m.shape[e]) for e in range(m.ndim)]
                pm[d] = im
                pin = [slice(None)] * m.ndim
                pin[d] = ii
                uw = spec.get("u")
                if uw is not None:
                    # sólo velocidades de pared ESCALARES; un perfil por
                    # celda en una pared (raro) se trata como pared quieta
                    # para la ley de pared — declarado
                    uw = [float(v) if np.ndim(v) == 0 else 0.0 for v in uw]
                self._wall_planes.append(dict(
                    axis=d, plane_m=tuple(pm), plane_int=tuple(pin), uw=uw))
        solver.physics.append(self)
        solver.nu_field = self.nu_eff
        self._fill_bcs(solver)
        self._max_diff = float(solver.cfg.nu + nut0)
        self.status = "attached"
        return self

    def max_diffusivity(self) -> float:
        """Difusividad máxima CACHEADA del último paso (σ_k = 1 ⇒ la
        difusividad de k es la efectiva del momento). Sin reducción aquí."""
        return self._max_diff

    # -- contornos ---------------------------------------------------------
    def _fill_bcs(self, solver) -> None:
        from .solver import fill_scalar_bc
        m = solver.mesh
        fill_scalar_bc(solver, self.k, self.k_bc)
        fill_scalar_bc(solver, self.eps, self.e_bc)
        fill_scalar_bc(solver, self.nu_t_field, self.n_bc)
        if m.phi is not None:
            # IB: ∂k/∂n = ∂ε/∂n = 0 con wall functions; ν_t del fantasma =
            # el de la imagen (≈ ν_wall de la banda) para que la cara corte
            # vea la viscosidad de la ley de pared
            m.apply_neumann(self.k)
            m.apply_neumann(self.eps)
            m.apply_neumann(self.nu_t_field)
            m.zero_deep_solid(self.k, self.cfg.k_floor)
            m.zero_deep_solid(self.eps, self.cfg.eps_floor)
            m.zero_deep_solid(self.nu_t_field, 0.0)
        self.nu_eff[...] = solver.dtype(solver.cfg.nu) + self.nu_t_field

    # -- ley de pared ------------------------------------------------------
    def _wall_values(self, k_w, u_p, y, nu, xp):
        """Valores de pared de dos escalas (docstring del módulo).
        Devuelve `(eps_w, prod_w, nu_wall, ystar)`."""
        y = xp.maximum(y, 1e-12)
        us = (C_MU ** 0.25) * xp.sqrt(xp.maximum(k_w, 0.0))
        ys = xp.maximum(us * y / nu, 1e-8)
        up = xp.maximum(u_plus_of_ystar(ys, xp=xp), 1e-8)
        tau = us * u_p / up                       # τ_w/ρ
        eps_w = us ** 3 / (KAPPA * y)
        prod_w = tau * us / (KAPPA * y)
        nu_wall = xp.maximum(us * y / up - nu, 0.0)
        return eps_w, prod_w, nu_wall, ys

    # -- un paso del modelo ------------------------------------------------
    def advance(self, solver, dt: float) -> bool:
        """Avanza k y ε un paso `dt` (lo llama `solver.step()` tras el RK
        del momento — acoplamiento segregado). Nunca lanza hacia el bucle:
        un NaN aquí acaba en los residuales del solver, que ya lo cazan."""
        from .solver import advect_diffuse_scalar

        xp, m, cfg = solver.xp, solver.mesh, self.cfg
        dt_ = solver.dtype
        I = m.interior
        nd = m.ndim
        nu = float(solver.cfg.nu)
        inv2dx = 0.5 / m.dx
        # 1. tensor de gradientes en el interior (fantasmas ya rellenos)
        gu = [[(m.view(solver.u[c], d, 1) - m.view(solver.u[c], d, -1))
               * dt_(inv2dx) for d in range(nd)] for c in range(nd)]
        s2 = None
        o2 = None
        for i in range(nd):
            for j in range(nd):
                sij = 0.5 * (gu[i][j] + gu[j][i])
                oij = 0.5 * (gu[i][j] - gu[j][i])
                s2 = sij * sij if s2 is None else s2 + sij * sij
                o2 = oij * oij if o2 is None else o2 + oij * oij
        s2 = 2.0 * s2                       # 2·S_ij S_ij
        o2 = 2.0 * o2
        s_mag = xp.sqrt(s2)
        o_mag = xp.sqrt(o2)
        # W = S_ij S_jk S_ki / (S_ij S_ij)^{3/2} para el A_s completo
        ss = None
        for i in range(nd):
            for j in range(nd):
                for kx in range(nd):
                    t = (0.5 * (gu[i][j] + gu[j][i])
                         * 0.5 * (gu[j][kx] + gu[kx][j])
                         * 0.5 * (gu[kx][i] + gu[i][kx]))
                    ss = t if ss is None else ss + t
        w = ss / xp.maximum((0.5 * s2) ** 1.5, 1e-30)
        k_i = self.k[I]
        e_i = self.eps[I]
        kk = xp.maximum(k_i, dt_(cfg.k_floor))
        ee = xp.maximum(e_i, dt_(cfg.eps_floor))
        c_mu = c_mu_realizable(s_mag, o_mag, kk, ee, w_star=w, xp=xp)
        if cfg.curvature_correction:
            f_r = curvature_correction(kk, ee, s_mag, o_mag, xp=xp)
        else:
            f_r = 1.0
        # 2. producción (con curvatura y limitador)
        prod = self.nu_t_field[I] * s2
        prod = xp.minimum(prod, dt_(cfg.prod_limit) * ee) * f_r
        # 3. ley de pared: banda IB + paredes del dominio (sobrescriben P y
        #    fijan ε; ν_wall entra al momento por la viscosidad de cara)
        wall_fix = []                      # (idx_o_plane, eps_w, nu_wall)
        ys_all = []
        if self._band is not None:
            b = self._band
            kw = self.k.reshape(-1)[b["tot"]]
            uvec = [solver.u[c].reshape(-1)[b["tot"]] for c in range(nd)]
            un = None
            for c in range(nd):
                t = uvec[c] * b["n"][c]
                un = t if un is None else un + t
            up2 = None
            for c in range(nd):
                t = (uvec[c] - un * b["n"][c]) ** 2
                up2 = t if up2 is None else up2 + t
            u_p = xp.sqrt(up2)
            eps_w, prod_w, nu_w, ys = self._wall_values(
                kw, u_p, b["y"], nu, xp)
            prod.reshape(-1)[b["int"]] = prod_w
            wall_fix.append((("flat", b["int"]), eps_w, nu_w))
            ys_all.append(ys)
        y_dom = dt_(0.5 * m.dx)
        for wp in self._wall_planes:
            d = wp["axis"]
            kw = self.k[wp["plane_m"]]
            up2 = None
            for c in range(nd):
                if c == d:
                    continue
                v = solver.u[c][wp["plane_m"]]
                if wp["uw"] is not None:
                    v = v - dt_(wp["uw"][c])
                up2 = v * v if up2 is None else up2 + v * v
            u_p = xp.sqrt(up2) if up2 is not None else xp.zeros_like(kw)
            eps_w, prod_w, nu_w, ys = self._wall_values(
                kw, u_p, y_dom, nu, xp)
            prod[wp["plane_int"]] = prod_w
            wall_fix.append((("plane", wp["plane_int"]), eps_w, nu_w))
            ys_all.append(ys)
        # 4. transporte + fuentes point-implicit
        uf = solver.face_velocities(dt)
        g_k = self.nu_eff if cfg.sigma_k == 1.0 else \
            (dt_(nu) + self.nu_t_field / dt_(cfg.sigma_k))
        g_e = dt_(nu) + self.nu_t_field / dt_(cfg.sigma_eps)
        rhs_k, _ = advect_diffuse_scalar(solver, self.k, g_k, uf)
        rhs_e, _ = advect_diffuse_scalar(solver, self.eps, g_e, uf)
        eta = s_mag * kk / ee
        c1 = xp.maximum(0.43, eta / (eta + 5.0))
        k_new = (k_i + dt_(dt) * (rhs_k + prod)) / (1.0 + dt_(dt) * ee / kk)
        e_new = (e_i + dt_(dt) * (rhs_e + c1 * s_mag * ee * f_r)) \
            / (1.0 + dt_(dt) * dt_(cfg.c2) * ee
               / (kk + xp.sqrt(dt_(nu) * ee)))
        fm = solver._fmask
        k_new = k_i + (k_new - k_i) * fm
        e_new = e_i + (e_new - e_i) * fm
        k_new = xp.maximum(k_new, dt_(cfg.k_floor))
        e_new = xp.maximum(e_new, dt_(cfg.eps_floor))
        # 5. ε fija en las celdas de pared (equilibrio de la ley de pared)
        for where, eps_w, _ in wall_fix:
            if where[0] == "flat":
                e_new.reshape(-1)[where[1]] = eps_w
            else:
                e_new[where[1]] = eps_w
        self.k[I] = k_new
        self.eps[I] = e_new
        # 6. ν_t nueva + sobrescritura de pared + contornos
        nut = c_mu * k_new * k_new / xp.maximum(e_new, dt_(cfg.eps_floor))
        nut = xp.minimum(nut, dt_(cfg.nu_t_max_ratio * nu)) * fm
        for where, _, nu_w in wall_fix:
            if where[0] == "flat":
                nut.reshape(-1)[where[1]] = nu_w
            else:
                nut[where[1]] = nu_w
        self.nu_t_field[I] = nut
        self._fill_bcs(solver)
        # una sola reducción por paso (float cacheado para time_step)
        self._max_diff = float(xp.max(self.nu_eff))
        if solver.be.is_gpu:
            solver._sync_calls += 1
        self._last_ystar = xp.concatenate(ys_all) if ys_all else None
        return True

    # -- compat M1 (firma congelada) ---------------------------------------
    def rhs(self, solver, dt: float):
        """Compat M1: hoy el camino canónico es `advance` (vía
        `solver.physics`). Devuelve `(None, None)` sin tocar nada."""
        return None, None

    def step(self, solver, dt: float) -> bool:
        """Compat M1: alias de `advance` (True si el modelo está
        enganchado; False si no — nunca lanza)."""
        if self.status != "attached":
            return False
        return self.advance(solver, dt)

    def apply_wall(self, solver) -> bool:
        """Compat M1: el tratamiento de pared vive dentro de `advance`."""
        return self.status == "attached"

    # -- diagnóstico -------------------------------------------------------
    def ystar_stats(self) -> dict:
        """Estadística del y* de la ÚLTIMA actualización de pared (banda
        IB + paredes del dominio). Para el histograma del reporte."""
        if self._last_ystar is None:
            return dict(ok=False, reason="sin celdas de pared")
        from .backend import to_numpy
        ys = np.asarray(to_numpy(self._last_ystar), dtype=float)
        inb = float(np.count_nonzero(
            (ys >= YPLUS_TARGET[0]) & (ys <= YPLUS_TARGET[1])) / ys.size)
        return dict(ok=True, n=int(ys.size), ystar_min=float(ys.min()),
                    ystar_mean=float(ys.mean()), ystar_max=float(ys.max()),
                    frac_in_target=inb, target=list(YPLUS_TARGET))

    def summary(self) -> dict:
        return dict(model=self.cfg.model, status=self.status,
                    available=available(), detail=STATUS,
                    wall_function=self.cfg.wall_function,
                    curvature_correction=bool(self.cfg.curvature_correction),
                    yplus_target=list(YPLUS_TARGET),
                    ystar=self.ystar_stats() if self.status == "attached"
                    else None)
