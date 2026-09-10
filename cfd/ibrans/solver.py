"""
QUASAR Phy-CB · cfd/ibrans/solver.py
====================================
**Hito M1** de WAVE6: Navier–Stokes incompresible (ρ constante) resuelto por
**pseudo-compresibilidad explícita** sobre malla cartesiana con frontera
inmersa por SDF. Sin Poisson implícito — que es exactamente donde mueren los
solvers caseros (Ciencia §11.2, decisión CONGELADA de WAVE6 §0).

Ecuaciones resueltas
--------------------
Método de compresibilidad artificial (Chorin 1967) en forma **evolutiva**
(«weakly compressible»), que sirve a la vez para régimen estacionario
(V1, V2) y para régimen no estacionario (V3, desprendimiento de vórtices):

    ∂u/∂t + ∇·(u u) = −(1/ρ)∇p + ∇·(ν ∇u) + f
    ∂p/∂t          = −ρ c² ∇·u_f

`c` es la **velocidad de sonido artificial**, fijada por
`c = mach_art⁻¹ · U_ref` con `mach_art = 0.1` por defecto ⇒ el error de
compresibilidad es O(M²) ≈ 1 %. Es un parámetro numérico, no físico, y
aparece etiquetado en el `summary()`.

Discretización (WAVE6 §1)
-------------------------
* **Volúmenes finitos colocados**, malla uniforme, 2 capas fantasma.
* Convección: reconstrucción **MUSCL de 2.º orden con limitador TVD**
  (van Leer por defecto, minmod y superbee disponibles) + upwind sobre la
  velocidad de cara.
* Difusión: diferencias centradas de 2.º orden con ν de cara.
* Acoplamiento p–u: velocidad de cara de **Rhie–Chow** en la divergencia.
  Sin ella la malla colocada desarrolla el modo tablero de ajedrez (es LA
  trampa de los esquemas colocados); con ella el filtro de presión es de
  4.º orden y desaparece al converger la malla.
* Tiempo: **Runge–Kutta explícito** SSP (rk1/rk2/rk3, rk3 por defecto).
* IB: **ghost-cell con interpolación lineal a la imagen del fantasma**
  (ver `mesh.py`), Dirichlet en velocidad y Neumann en presión.

Backend: el MISMO kernel en CPU y en GPU
----------------------------------------
Todo el bucle caliente (`rhs`, `_apply`, `_blend`, `apply_bc`, el ghost-cell
del IB) está escrito contra `self.xp`, que es `numpy` en CPU y `cupy` en
GPU. No hay dos versiones del kernel que mantener sincronizadas, que era el
argumento en contra de portar a Warp o a JAX. Reglas que se respetan y que
son las que deciden si un port a GPU sirve para algo:

* **Ninguna transferencia device↔host dentro del bucle.** Campos, máscaras
  y estenciles del IB viven en el dispositivo desde el constructor
  (`mesh.to_backend`). Lo único que baja a CPU es un **escalar por paso**
  (el `u_max` del CFL) y los residuales cada `residual_every` pasos.
* **Sincronización antes de cronometrar** (`backend.Backend.sync`): los
  lanzamientos de CuPy son asíncronos y sin `sync()` se mediría el tiempo
  de encolar kernels, no el de ejecutarlos.
* Si no hay GPU o falla, `backend.get_backend()` degrada a NumPy **con
  etiqueta**; el solver no cambia ni una línea y nada lanza.

Los residuales, la media de presión y el `u_max` del CFL son REDUCCIONES:
en GPU el orden de acumulación difiere del de CPU, así que **los resultados
GPU y CPU no son bit-idénticos** aunque cada backend sea reproducible
consigo mismo con `SEED = 71`. Está declarado en `backend.py` y medido en
`ESTADO.md` §5.4.

Ganchos M2 (turbulencia + energía + ρ variable) — TODOS opt-in
---------------------------------------------------------------
El hito M2 vive en `turbulence.py` (k-ε realizable) y `energy.py` (energía
+ ρ(T) low-Mach + escalar pasivo). Este módulo sólo aporta los ganchos, y
todos son **opt-in**: sin modelos registrados la aritmética de M1 es
BIT-IDÉNTICA a la verificada por V1–V3 (lo vigila el test M2f).

* `solver.physics` — lista de modelos con `.advance(solver, dt)` y
  `.max_diffusivity()`; se avanzan UNA vez por paso, tras el RK del
  momento (acoplamiento segregado: ν_t/ρ congelados durante el subpaso).
* `solver.nu_field` — ν efectiva por celda (la escribe la turbulencia).
* `solver.rho_field` — ρ(T) por celda (la escribe la energía). Activa la
  forma ADVECTIVA del momento (Du/Dt = ∂u/∂t + u·∇u, que es la ecuación
  correcta a ρ variable tras dividir por ρ): se añade el término +u·(∇·u_f)
  que en incompresible es cero. El término de Rhie–Chow y su peso siguen
  usando `cfg.rho` (es un filtro numérico, no física — declarado).
* `solver.div_source` — q_T = ∇·(α∇T)/T de la energía: a presión
  termodinámica constante la continuidad exige ∇·u = q_T, y aquí entra en
  la ecuación de presión como ∂p/∂t = −ρc²(∇·u_f − q_T).
* `face_velocities()` — las uf de Rhie–Chow del paso, cacheadas por
  iteración para que N modelos no las recomputen N veces.
* `advect_diffuse_scalar()` / `fill_scalar_bc()` — el MISMO kernel
  MUSCL+TVD del momento aplicado a un escalar genérico (k, ε, T, Z),
  escrito contra `xp` (CPU y GPU).

Lo que M1+M2 **prometen** y lo que **no**
-----------------------------------------
PROMETEN: flujo incompresible frío con paredes inmersas (V1–V3) y, con los
modelos de M2 registrados, RANS k-ε realizable con funciones de pared y
energía low-Mach (V4–V5, `cfd/tests/test_ibrans_v4_v5.py`).
NO prometen: especies ni combustión (M3, `chem.py`), producto de combustor
(M4, `post.py`). Y en ningún caso **temperatura de metal** — eso no lo da
Phy-CB en ninguna fidelidad (contrato §12).

API pública
-----------
    SolverConfig, SolverState, IBRANSSolver
    make_bcs(**kwargs) -> dict
    advect_diffuse_scalar(solver, fld, gamma, uf) -> (rhs, dif)
    fill_scalar_bc(solver, fld, spec)
    scalar_bc_spec(solver, key, inlet_value) -> dict
    LIMITERS
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from . import backend as _bk
from .mesh import CartesianMesh

#: Semilla de la familia (contrato §10.8) — cualquier perturbación la usa.
SEED = 71

#: Limitadores TVD disponibles (`SolverConfig.limiter`).
LIMITERS = ("vanleer", "minmod", "superbee", "none")

#: Tipos de contorno del dominio cartesiano.
BC_KINDS = ("periodic", "wall", "slip", "inlet", "outlet")

#: CFL máximo comprobado ESTABLE con rk3 en esta discretización. A 0.8 el
#: acoplamiento p–u oscila: el término de Rhie–Chow escala con Δt y a CFL
#: alto sobre-amortigua hasta desestabilizar (medido 2026-07-31, cavidad
#: Re=100: residuales de 1e2–1e4 sin llegar a divergir, que es peor).
CFL_MAX_STABLE = 0.6


# ==========================================================================
# 1. Configuración
# ==========================================================================
@dataclass
class SolverConfig:
    """Parámetros del solver. Todo en SI.

    `mach_art` es el número de Mach artificial: `c = U_ref / mach_art`.
    Bajarlo reduce el error de compresibilidad (O(M²)) y encarece el paso
    de tiempo en la misma proporción — es EL compromiso del método.
    """
    rho: float = 1.0
    nu: float = 1.0e-2
    u_ref: float = 1.0
    l_ref: float = 1.0
    mach_art: float = 0.1
    cfl: float = 0.4
    rk: str = "rk3"
    limiter: str = "vanleer"
    rc_alpha: float = 1.0          # peso de Rhie–Chow (1 = estándar)
    body_force: tuple[float, ...] | None = None
    p_ref: float = 0.0
    max_iter: int = 200_000
    t_end: float = float("inf")
    tol: float = 1.0e-5
    timeout_s: float = 0.0         # 0 = sin límite de reloj
    log_every: int = 0             # 0 = silencioso
    residual_every: int = 20
    monitor_every: int = 0         # 0 = sin monitores
    checkpoint_every: int = 0
    checkpoint_path: str = ""
    precision: str = "auto"
    backend: str = "auto"
    ib_wall_u: object = None       # None | secuencia | callable(mesh)->list

    @property
    def c_art(self) -> float:
        """Velocidad de sonido artificial."""
        return float(self.u_ref) / max(float(self.mach_art), 1e-9)


@dataclass
class SolverState:
    """Estado del solver (campos CON capas fantasma)."""
    u: list                       # ndim arrays
    p: np.ndarray
    t: float = 0.0
    it: int = 0
    history: dict = field(default_factory=dict)


def make_bcs(**kwargs) -> dict:
    """Azúcar para construir el dict de contornos.

    Claves: `xlo, xhi, ylo, yhi, zlo, zhi` (o `x-`, `x+`, …). Valor = dict
    con `kind` y lo que el tipo necesite::

        make_bcs(xlo=dict(kind="inlet", u=(1.0, 0.0)),
                 xhi=dict(kind="outlet", p=0.0),
                 ylo=dict(kind="wall"), yhi=dict(kind="wall", u=(1.0, 0.0)))
    """
    ren = {"x-": "xlo", "x+": "xhi", "y-": "ylo", "y+": "yhi",
           "z-": "zlo", "z+": "zhi"}
    out = {}
    for k, v in kwargs.items():
        out[ren.get(k, k)] = v
    return out


# ==========================================================================
# 2. Limitadores TVD
# ==========================================================================
def _slope(a, b, kind: str, xp=np):
    """Pendiente limitada a partir de las diferencias hacia atrás `a` y
    hacia delante `b`. Todas son TVD y se anulan en los extremos locales.

    `xp` es el módulo de arrays (numpy o cupy): la fórmula es idéntica en
    los dos backends, sólo cambia dónde se ejecuta.
    """
    if kind == "none":
        return 0.5 * (a + b)
    if kind == "minmod":
        return xp.where(a * b > 0.0,
                        xp.sign(a) * xp.minimum(xp.abs(a), xp.abs(b)), 0.0)
    if kind == "superbee":
        s1 = xp.where(a * b > 0.0,
                      xp.sign(a) * xp.minimum(xp.abs(a), 2.0 * xp.abs(b)), 0.0)
        s2 = xp.where(a * b > 0.0,
                      xp.sign(a) * xp.minimum(2.0 * xp.abs(a), xp.abs(b)), 0.0)
        return xp.where(xp.abs(s1) > xp.abs(s2), s1, s2)
    # van Leer (media armónica): el default
    aa, ab = xp.abs(a), xp.abs(b)
    return (a * ab + aa * b) / (aa + ab + 1e-30)


# ==========================================================================
# 3. Solver
# ==========================================================================
class IBRANSSolver:
    """Solver M1. Uso típico::

        m = CartesianMesh(lo, hi, dx); m.set_sdf(phi)
        s = IBRANSSolver(m, SolverConfig(nu=0.01, u_ref=1.0),
                         bcs=make_bcs(...))
        info = s.run()
    """

    def __init__(self, mesh: CartesianMesh, cfg: SolverConfig,
                 bcs: dict | None = None, u0=None, p0=None):
        self.mesh = mesh
        self.cfg = cfg
        self.be = _bk.get_backend(cfg.backend, cfg.precision)
        req = str(cfg.backend or "auto").lower()
        # (a) Selección por TAMAÑO. Con `auto` y una malla pequeña la GPU
        # pierde: el paso lo domina el coste de lanzar kernels, no el
        # cálculo (tabla medida en `backend.GPU_MIN_CELLS`). Pedir "cupy"
        # explícitamente salta esta regla — es lo que hace la V&V forzada.
        if (self.be.is_gpu and req == "auto"
                and mesh.ncell < _bk.GPU_MIN_CELLS
                and not os.environ.get(_bk.ENV_BACKEND)):
            self.be = _bk.get_backend("numpy", cfg.precision)
            self.be.note = (
                f"auto→numpy: {mesh.ncell} celdas < {_bk.GPU_MIN_CELLS}; "
                "por debajo del cruce medido la GPU es mas lenta (coste de "
                "lanzamiento de kernels). Forzar con backend='cupy'.")
        # (b) Guardián de VRAM: si el caso no cabe con margen, se degrada a
        # CPU ANTES de reservar nada, con etiqueta. Reventar por memoria a
        # mitad de un barrido no es un modo de fallo aceptable (§10.4).
        if self.be.is_gpu:
            fits, why = _bk.fits_in_vram(self.be, mesh.shape_tot, mesh.ndim)
            if not fits:
                self.be = _bk.get_backend("numpy", cfg.precision)
                self.be.note = (f"degradado a numpy por VRAM: {why}. Usa un "
                                "dx mas grueso para volver a GPU.")
        self.xp = self.be.xp
        self.dtype = self.be.dtype
        nd = mesh.ndim
        self.bcs = self._normalize_bcs(bcs or {})
        xp = self.xp
        self.u = [xp.zeros(mesh.shape_tot, dtype=self.dtype)
                  for _ in range(nd)]
        self.p = xp.full(mesh.shape_tot, cfg.p_ref, dtype=self.dtype)
        if u0 is not None:
            for c in range(nd):
                self.u[c][...] = self._up(u0[c])
        if p0 is not None:
            self.p[...] = self._up(p0)
        # el IB vive en el dispositivo: sin esto cada ghost-cell exigiría
        # bajar el campo a CPU y volver a subirlo (ver mesh.to_backend)
        mesh.to_backend(self.be)
        self.nu_field = None            # M2: viscosidad efectiva por celda
        self.rho_field = None           # M2: rho(T) por celda (energy.py)
        self.div_source = None          # M2: q_T = div(alpha grad T)/T
        self.div_extra: list = []       # M4: fuentes de masa por celda
        self.force_field = None         # M4: fuerza por celda (ndim arrays)
        self.physics: list = []         # M2: modelos con .advance(solver, dt)
        self._uf_cache: tuple | None = None
        self.t = 0.0
        self.it = 0
        self.dt = 0.0
        self.status = "init"
        self.monitors: dict[str, Callable] = {}
        # `t_mon` se crea AQUÍ y no en `add_monitor`: con `monitor_every > 0`
        # y ningún monitor registrado, el driver escribía en una clave
        # inexistente (KeyError). Verificado y corregido.
        self.history: dict[str, list] = {"t": [], "res_u": [], "res_div": [],
                                         "t_mon": []}
        self._fluid_int = (mesh.fluid[mesh.interior] if mesh.phi is not None
                           else np.ones(mesh.shape, dtype=bool))
        self._nfluid = max(1, int(np.count_nonzero(self._fluid_int)))
        self._fmask = xp.asarray(self._fluid_int.astype(self.dtype))
        self._wall_u = self._resolve_wall_u()
        self._t_start = 0.0
        self._steps_done = 0
        self.body_force = (np.zeros(nd) if cfg.body_force is None
                           else np.asarray(cfg.body_force, dtype=float))
        self._sync_calls = 0            # transferencias device→host contadas
        # Sin ningún contorno de presión impuesta (todo paredes/periódico) la
        # presión sólo está definida salvo constante y deriva lentamente con
        # el error de divergencia del IB: se ancla su media.
        self._pin_p = not any(s["kind"] == "outlet" for s in self.bcs.values())
        self.apply_bc()

    # -- puente host↔dispositivo -------------------------------------------
    def _up(self, a):
        """Sube `a` al backend con el dtype del solver (identidad en CPU)."""
        if self.be.is_gpu:
            return self.xp.asarray(_bk.to_numpy(a), dtype=self.dtype)
        return np.asarray(a, dtype=self.dtype)

    def _down(self, a) -> np.ndarray:
        """Baja `a` a NumPy en float64. **Cuenta la transferencia**: el
        contador `_sync_calls` es lo que demuestra que el bucle caliente no
        hace copias (ver `_info`)."""
        if self.be.is_gpu:
            self._sync_calls += 1
        return np.asarray(_bk.to_numpy(a), dtype=np.float64)

    # -- contornos ---------------------------------------------------------
    def _normalize_bcs(self, bcs: dict) -> dict:
        keys = []
        for d in range(self.mesh.ndim):
            keys += [f"{'xyz'[d]}lo", f"{'xyz'[d]}hi"]
        out = {}
        ren = {"x-": "xlo", "x+": "xhi", "y-": "ylo", "y+": "yhi",
               "z-": "zlo", "z+": "zhi"}
        src = {ren.get(k, k): v for k, v in bcs.items()}
        for k in keys:
            spec = src.get(k, {"kind": "periodic"})
            if isinstance(spec, str):
                spec = {"kind": spec}
            spec = dict(spec)
            spec.setdefault("kind", "periodic")
            if spec["kind"] not in BC_KINDS:
                spec["kind"] = "periodic"
            # una componente de `u` puede ser un ARRAY (perfil por celda
            # sobre el plano de la cara, p. ej. capa límite de entrada o
            # ranura de chorro): se sube al backend UNA vez, aquí — subirla
            # en `apply_bc` sería una copia host→device por subpaso RK
            if spec.get("u") is not None:
                spec["u"] = tuple(self._up(v) if np.ndim(v) else float(v)
                                  for v in spec["u"])
            out[k] = spec
        return out

    def _resolve_wall_u(self):
        """Velocidad de pared del IB por componente: escalar o array por
        celda fantasma (permite paredes inmersas móviles, p. ej. Couette).

        Los arrays se suben al dispositivo UNA vez, aquí: si se subieran en
        `apply_bc` habría una copia host→device por subpaso RK.
        """
        w = self.cfg.ib_wall_u
        if w is None:
            return [0.0] * self.mesh.ndim
        if callable(w):
            try:
                w = w(self.mesh)
            except Exception:
                return [0.0] * self.mesh.ndim
        return [self._up(v) if np.ndim(v) else float(v) for v in w]

    def apply_bc(self) -> None:
        """Rellena las capas fantasma del dominio y aplica el IB."""
        m, ng = self.mesh, self.mesh.ng
        nd = m.ndim
        for d in range(nd):
            n = m.shape[d]
            for side in ("lo", "hi"):
                spec = self.bcs[f"{'xyz'[d]}{side}"]
                kind = spec["kind"]
                for k in range(1, ng + 1):
                    if side == "lo":
                        gi, mi = ng - k, ng + k - 1
                        pi = ng + n - k          # periódico
                    else:
                        gi, mi = ng + n - 1 + k, ng + n - k
                        pi = ng + k - 1
                    G = self._plane(d, gi)
                    M = self._plane(d, mi)
                    P = self._plane(d, pi)
                    E = self._plane(d, ng if side == "lo" else ng + n - 1)
                    if kind == "periodic":
                        for c in range(nd):
                            self.u[c][G] = self.u[c][P]
                        self.p[G] = self.p[P]
                    elif kind in ("wall", "inlet"):
                        uw = spec.get("u")
                        for c in range(nd):
                            # escalar o perfil por celda (ya en el backend,
                            # ver _normalize_bcs)
                            wv = 0.0 if uw is None else uw[c]
                            self.u[c][G] = 2.0 * wv - self.u[c][M]
                        self.p[G] = self.p[M]
                    elif kind == "slip":
                        for c in range(nd):
                            self.u[c][G] = (-self.u[c][M] if c == d
                                            else self.u[c][M])
                        self.p[G] = self.p[M]
                    elif kind == "outlet":
                        pw = float(spec.get("p", self.cfg.p_ref))
                        for c in range(nd):
                            self.u[c][G] = self.u[c][E]
                        self.p[G] = 2.0 * pw - self.p[M]
        if m.phi is not None:
            for c in range(nd):
                m.apply_dirichlet(self.u[c], self._wall_u[c])
                m.zero_deep_solid(self.u[c], 0.0)
            m.apply_neumann(self.p)

    def _plane(self, axis: int, i: int) -> tuple:
        idx = [slice(None)] * self.mesh.ndim
        idx[axis] = i
        return tuple(idx)

    # -- núcleo numérico ---------------------------------------------------
    def _nu_faces(self, d: int):
        if self.nu_field is None:
            return self.cfg.nu
        return 0.5 * (self.mesh.fview(self.nu_field, d, 0)
                      + self.mesh.fview(self.nu_field, d, 1))

    def _face_vel(self, d: int, dt: float):
        """Velocidad de cara de Rhie–Chow en el eje `d` (mata el modo
        tablero). Devuelve `(uf, pL, pR)`; pL/pR se reutilizan en `rhs`
        para el gradiente de presión. El peso rc usa `cfg.rho` incluso con
        ρ variable: es un filtro numérico de 4.º orden, no física."""
        m, cfg = self.mesh, self.cfg
        inv_dx = 1.0 / m.dx
        pL = m.fview(self.p, d, 0)
        pR = m.fview(self.p, d, 1)
        pLL = m.fview(self.p, d, -1)
        pRR = m.fview(self.p, d, 2)
        udL = m.fview(self.u[d], d, 0)
        udR = m.fview(self.u[d], d, 1)
        rc = (pR - pL) * inv_dx - 0.25 * inv_dx * ((pR - pLL) + (pRR - pL))
        rc_w = cfg.rc_alpha * dt / cfg.rho
        return 0.5 * (udL + udR) - rc_w * rc, pL, pR

    def face_velocities(self, dt: float | None = None) -> list:
        """Las uf de Rhie–Chow del paso ACTUAL, por eje, **cacheadas por
        iteración**: los modelos de M2 (turbulencia, energía, escalares)
        las comparten en vez de recomputarlas N veces. Se invalidan solas
        al cambiar `self.it`."""
        dt = self.dt if dt is None else float(dt)
        if self._uf_cache is not None and self._uf_cache[0] == self.it:
            return self._uf_cache[1]
        ufs = [self._face_vel(d, dt)[0] for d in range(self.mesh.ndim)]
        self._uf_cache = (self.it, ufs)
        return ufs

    def rhs(self, dt: float):
        """Miembro derecho de (u, p). Devuelve `(du, dp)` del tamaño del
        interior. `dt` sólo entra en el término de Rhie–Chow.

        Con `rho_field`/`div_source` en None (M1) la aritmética es la
        verificada por V1–V3. Con ellos activos (M2): 1/ρ por celda en ∇p,
        ρc² por celda en la presión, +u·(∇·u_f) en el momento (forma
        advectiva, la correcta a ρ variable) y −q_T en la divergencia."""
        m, cfg = self.mesh, self.cfg
        xp = self.xp
        nd, dx = m.ndim, m.dx
        inv_dx = 1.0 / dx
        rho = cfg.rho
        lim = cfg.limiter
        I = m.interior
        rho_i = None if self.rho_field is None else self.rho_field[I]
        div = xp.zeros(m.shape, dtype=self.dtype)
        acc = [xp.zeros(m.shape, dtype=self.dtype) for _ in range(nd)]
        for d in range(nd):
            uf, pL, pR = self._face_vel(d, dt)
            nu_f = self._nu_faces(d)
            up = uf > 0.0
            for c in range(nd):
                aL = m.fview(self.u[c], d, 0)
                aR = m.fview(self.u[c], d, 1)
                aLL = m.fview(self.u[c], d, -1)
                aRR = m.fview(self.u[c], d, 2)
                dcen = aR - aL
                sL = _slope(aL - aLL, dcen, lim, xp)
                sR = _slope(dcen, aRR - aR, lim, xp)
                face = xp.where(up, aL + 0.5 * sL, aR - 0.5 * sR)
                flux = uf * face - nu_f * dcen * inv_dx
                acc[c] += xp.diff(flux, axis=d) * inv_dx
            # gradiente de presión (caras ⇒ diferencia centrada de 2·dx)
            gp = xp.diff(0.5 * (pL + pR), axis=d) * inv_dx
            acc[d] += (gp / rho) if rho_i is None else (gp / rho_i)
            div += xp.diff(uf, axis=d) * inv_dx
        du = []
        for c in range(nd):
            v = -acc[c]
            if rho_i is not None:
                # forma advectiva: ∇·(uu) − u(∇·u) = u·∇u
                v = v + self.u[c][I] * div
            if self.body_force[c] != 0.0:
                v = v + self.dtype(self.body_force[c])
            if self.force_field is not None:
                # fuerza volumétrica POR CELDA (M4: momento de los chorros
                # transpirados). Con `None` la aritmética es la de M1.
                v = v + self.force_field[c]
            du.append(v * self._fmask)
        dv = div if self.div_source is None else (div - self.div_source)
        for q in self.div_extra:
            # fuentes de masa POR CELDA (M4: caudal de las filas no
            # resueltas y de la efusión). Entran en la continuidad
            # low-Mach igual que la dilatación térmica: ∇·u = q, que es lo
            # que garantiza que el caudal inyectado sea EXACTAMENTE el
            # pedido — una BC de velocidad sólo entrega lo que el campo le
            # deja (medido: −17 % con U_jet real).
            dv = dv - q
        if rho_i is None:
            dp = (-self.dtype(rho * cfg.c_art ** 2) * dv) * self._fmask
        else:
            dp = (-(rho_i * self.dtype(cfg.c_art ** 2)) * dv) * self._fmask
        return du, dp

    def time_step(self) -> float:
        """Paso estable: mínimo entre el CFL acústico-convectivo y el
        límite difusivo explícito. Sin `while` abiertos (contrato §10.5).

        El máximo se reduce ENTERO en el dispositivo y sólo baja a CPU el
        escalar final: es la ÚNICA transferencia device→host por paso de
        tiempo (una, no `ndim`). Se cuenta en `_sync_calls`.
        """
        cfg, m = self.cfg, self.mesh
        xp = self.xp
        I = m.interior
        acc = None
        for c in range(m.ndim):
            v = xp.abs(self.u[c][I])
            acc = v if acc is None else xp.maximum(acc, v)
        umax = float(acc.max()) if (acc is not None and acc.size) else 0.0
        if self.be.is_gpu:
            self._sync_calls += 1
        if not math.isfinite(umax):
            umax = cfg.u_ref
        dt_c = m.dx / max(umax + cfg.c_art, 1e-12)
        nu_max = (cfg.nu if self.nu_field is None
                  else float(self.xp.max(self.nu_field)))
        # las difusividades de los modelos M2 (ν_t/σ, α_eff, D_eff) entran
        # en el límite difusivo; devuelven un float CACHEADO (sin reducción
        # ni transferencia aquí)
        for ph in self.physics:
            try:
                nu_max = max(nu_max, float(ph.max_diffusivity()))
            except Exception:
                pass
        dt_v = (m.dx ** 2) / max(2.0 * m.ndim * nu_max, 1e-30)
        return float(cfg.cfl * min(dt_c, dt_v))

    def _apply(self, du, dp, dt) -> None:
        I = self.mesh.interior
        for c in range(self.mesh.ndim):
            self.u[c][I] += self.dtype(dt) * du[c]
        self.p[I] += self.dtype(dt) * dp
        self.apply_bc()

    def _blend(self, u_old, p_old, a: float, b: float) -> None:
        """u ← a·u_old + b·u (combinación convexa de los estados SSP)."""
        I = self.mesh.interior
        for c in range(self.mesh.ndim):
            self.u[c][I] = self.dtype(a) * u_old[c] + self.dtype(b) \
                * self.u[c][I]
        self.p[I] = self.dtype(a) * p_old + self.dtype(b) * self.p[I]
        self.apply_bc()

    def step(self, dt: float | None = None) -> float:
        """Un paso de tiempo. Devuelve el `dt` usado."""
        dt = self.time_step() if dt is None else float(dt)
        self.dt = dt
        I = self.mesh.interior
        rk = self.cfg.rk
        if rk == "rk1":
            du, dp = self.rhs(dt)
            self._last_du, self._last_dp = du, dp
            self._apply(du, dp, dt)
        elif rk == "rk2":
            u0 = [self.u[c][I].copy() for c in range(self.mesh.ndim)]
            p0 = self.p[I].copy()
            du, dp = self.rhs(dt)
            self._last_du, self._last_dp = du, dp
            self._apply(du, dp, dt)
            du, dp = self.rhs(dt)
            self._apply(du, dp, dt)
            self._blend(u0, p0, 0.5, 0.5)
        else:                                   # rk3 SSP (Shu–Osher)
            u0 = [self.u[c][I].copy() for c in range(self.mesh.ndim)]
            p0 = self.p[I].copy()
            du, dp = self.rhs(dt)
            self._last_du, self._last_dp = du, dp
            self._apply(du, dp, dt)
            du, dp = self.rhs(dt)
            self._apply(du, dp, dt)
            self._blend(u0, p0, 0.75, 0.25)
            du, dp = self.rhs(dt)
            self._apply(du, dp, dt)
            self._blend(u0, p0, 1.0 / 3.0, 2.0 / 3.0)
        # modelos M2 (turbulencia, energía, escalares): acoplamiento
        # SEGREGADO — ν_t/ρ quedaron congelados durante el RK del momento y
        # se actualizan aquí, una vez por paso, con el mismo dt
        for ph in self.physics:
            ph.advance(self, dt)
        self.t += dt
        self.it += 1
        self._steps_done += 1
        return dt

    # -- residuales y diagnóstico -----------------------------------------
    def residuals(self) -> tuple[float, float]:
        """(res_u, res_div) adimensionales, ambos evaluados sobre el ÚLTIMO
        miembro derecho — es decir, sobre lo que el esquema realmente
        anula, no sobre un operador distinto.

        `res_u`  = rms(∂u/∂t)·L_ref/U_ref²   (aceleración normalizada)
        `res_div`= rms(∇·u_f)·L_ref/U_ref    con la divergencia de
                   **Rhie–Chow** (la del método), recuperada de
                   ∂p/∂t = −ρc²∇·u_f. La divergencia centrada de
                   `divergence()` NO es la que el método anula: difiere en
                   el filtro de presión de 4.º orden y se estanca en un
                   valor finito aunque la solución esté convergida.
        """
        cfg = self.cfg
        xp = self.xp
        du = getattr(self, "_last_du", None)
        dp = getattr(self, "_last_dp", None)
        if du is None or dp is None:
            return float("inf"), float("inf")
        # la SUMA se acumula en float64 aunque el campo sea fp32: si no, en
        # fp32 el residual se satura por cancelación y el criterio de
        # convergencia mediría redondeo en vez de física.
        acc = None
        for c in range(self.mesh.ndim):
            t = du[c].astype(np.float64, copy=False) ** 2
            acc = t if acc is None else acc + t
        s = float(xp.sum(acc))
        res_u = math.sqrt(s / self._nfluid) * cfg.l_ref / max(
            cfg.u_ref ** 2, 1e-30)
        c2 = cfg.rho * cfg.c_art ** 2
        dv = dp.astype(np.float64, copy=False) / max(c2, 1e-30)
        res_div = float(xp.sqrt(xp.sum(dv ** 2) / self._nfluid)) \
            * cfg.l_ref / max(cfg.u_ref, 1e-30)
        if self.be.is_gpu:
            self._sync_calls += 2
        return res_u, res_div

    def pin_pressure(self) -> None:
        """Ancla la media de la presión en el fluido a `cfg.p_ref` (sólo
        tiene sentido si NINGÚN contorno impone presión). Un desplazamiento
        uniforme de p no cambia ∇p: es cosmética numérica, no física."""
        I = self.mesh.interior
        pin = self.p[I]
        mean = float(self.xp.sum(
            pin.astype(np.float64, copy=False) * self._fmask) / self._nfluid)
        if self.be.is_gpu:
            self._sync_calls += 1
        self.p -= self.dtype(mean - self.cfg.p_ref)
        self.apply_bc()

    def divergence(self) -> np.ndarray:
        """∇·u centrada (diagnóstico; la del solver es la de Rhie–Chow).
        Devuelve SIEMPRE un array de NumPy: es post-proceso, no bucle."""
        m, xp = self.mesh, self.xp
        out = xp.zeros(m.shape, dtype=np.float64)
        for d in range(m.ndim):
            out += (m.view(self.u[d], d, 1).astype(np.float64, copy=False)
                    - m.view(self.u[d], d, -1)) / (2.0 * m.dx)
        return self._down(out) * self._fluid_int

    def add_monitor(self, name: str, fn: Callable) -> None:
        """Registra un monitor `fn(solver) -> float` muestreado cada
        `cfg.monitor_every` iteraciones (0 = nunca)."""
        self.monitors[name] = fn
        self.history.setdefault(name, [])
        self.history.setdefault("t_mon", [])

    def sample(self, x: Sequence[float], comp: int = 0) -> float:
        """Valor interpolado (multilineal) de `u[comp]` en el punto `x`.

        En GPU se baja el **bloque 2^ndim de una vez** y se interpola en
        CPU: con una lectura escalar por esquina serían 2^ndim
        transferencias por muestreo, y este método lo llaman los monitores
        de la V3 cada 20 iteraciones.
        """
        m = self.mesh
        fi = [(float(x[d]) - m.lo[d]) / m.dx + m.ng - 0.5
              for d in range(m.ndim)]
        i0 = [int(np.clip(math.floor(fi[d]), 0, m.shape_tot[d] - 2))
              for d in range(m.ndim)]
        w = [fi[d] - i0[d] for d in range(m.ndim)]
        blk = self._down(self.u[comp][tuple(slice(i0[d], i0[d] + 2)
                                            for d in range(m.ndim))])
        tot = 0.0
        for cnr in range(2 ** m.ndim):
            wt = 1.0
            idx = []
            for d in range(m.ndim):
                bit = (cnr >> d) & 1
                wt *= w[d] if bit else (1.0 - w[d])
                idx.append(bit)
            tot += wt * float(blk[tuple(idx)])
        return tot

    def checkpoint(self, path: str) -> str:
        """Vuelca el estado a `.npz`. **Nunca lanza** (devuelve "" si falla).
        Los campos bajan a CPU aquí: un `.npz` es siempre de host."""
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            np.savez_compressed(
                path, it=self.it, t=self.t, p=_bk.to_numpy(self.p),
                **{f"u{c}": _bk.to_numpy(self.u[c])
                   for c in range(self.mesh.ndim)})
            return path
        except Exception:
            return ""

    def load_checkpoint(self, path: str) -> bool:
        """Recupera un estado volcado. Devuelve False si no se pudo."""
        try:
            z = np.load(path)
            for c in range(self.mesh.ndim):
                self.u[c][...] = self._up(z[f"u{c}"])
            self.p[...] = self._up(z["p"])
            self.it = int(z["it"])
            self.t = float(z["t"])
            self.apply_bc()
            return True
        except Exception:
            return False

    # -- driver de convergencia -------------------------------------------
    def run(self, max_iter: int | None = None, t_end: float | None = None,
            tol: float | None = None, callback: Callable | None = None
            ) -> dict:
        """Bucle ACOTADO (contrato §10.5): itera hasta converger, agotar
        `max_iter`, alcanzar `t_end`, vencer el `timeout_s` o divergir.

        `status` ∈ {"converged", "max_iter", "t_end", "timeout",
        "diverged"}. **Nunca lanza.**
        """
        cfg = self.cfg
        max_iter = int(cfg.max_iter if max_iter is None else max_iter)
        t_end = float(cfg.t_end if t_end is None else t_end)
        tol = float(cfg.tol if tol is None else tol)
        self.be.sync()          # el reloj no empieza con kernels encolados
        self._t_start = time.perf_counter()
        self._steps_done = 0
        self._sync_calls = 0
        status = "max_iter"
        res_u = res_div = float("inf")
        for k in range(max_iter):
            try:
                self.step()
            except Exception as exc:            # red de seguridad
                self.status = "diverged"
                return self._info("diverged", res_u, res_div,
                                  f"{type(exc).__name__}: {exc}")
            if (k % max(1, cfg.residual_every)) == 0 or k == max_iter - 1:
                if self._pin_p:
                    self.pin_pressure()
                res_u, res_div = self.residuals()
                self.history["t"].append(self.t)
                self.history["res_u"].append(res_u)
                self.history["res_div"].append(res_div)
                if not (math.isfinite(res_u) and math.isfinite(res_div)):
                    return self._info("diverged", res_u, res_div,
                                      "residual no finito (NaN/inf)")
                if cfg.log_every and (k % cfg.log_every) == 0:
                    print(f"  it={self.it:7d} t={self.t:11.5f} "
                          f"dt={self.dt:.3e} res_u={res_u:.3e} "
                          f"res_div={res_div:.3e}", flush=True)
                # k > 0: la PRIMERA evaluación no puede declarar
                # convergencia — un estado inicial uniforme es equilibrio
                # exacto del rhs ANTES de que los modelos M2 (fuente de
                # calor, turbulencia) hayan actuado, y salir ahí es un
                # falso verde (pasó dos veces montando los tests de M2)
                if max(res_u, res_div) < tol and k > 0:
                    status = "converged"
                    break
            if cfg.monitor_every and (k % cfg.monitor_every) == 0:
                self.history["t_mon"].append(self.t)
                for nm, fn in self.monitors.items():
                    try:
                        self.history[nm].append(float(fn(self)))
                    except Exception:
                        self.history[nm].append(float("nan"))
            if cfg.checkpoint_every and cfg.checkpoint_path and \
                    (k % cfg.checkpoint_every) == 0 and k:
                self.checkpoint(cfg.checkpoint_path)
            if callback is not None and callback(self) is False:
                status = "callback"
                break
            if self.t >= t_end:
                status = "t_end"
                break
            if cfg.timeout_s and \
                    (time.perf_counter() - self._t_start) > cfg.timeout_s:
                status = "timeout"
                break
        self.status = status
        return self._info(status, res_u, res_div, "")

    def _info(self, status: str, res_u: float, res_div: float,
              note: str) -> dict:
        self.be.sync()          # sin esto se cronometraría el encolado
        el = max(time.perf_counter() - self._t_start, 1e-12)
        ncell = self.mesh.ncell
        steps = max(1, self._steps_done)
        return dict(status=status, iterations=self.it, steps=steps,
                    t=self.t, dt=self.dt, res_u=res_u, res_div=res_div,
                    wall_time_s=el, cells=ncell,
                    s_per_cell_step=el / (ncell * steps),
                    Mcell_steps_per_s=1e-6 * ncell * steps / el,
                    d2h_per_step=self._sync_calls / steps,
                    vram_used_MiB=self.be.used_bytes() / 2 ** 20,
                    note=note, **self.summary())

    def summary(self) -> dict:
        """Bloque trazable: qué se resolvió y con qué números mágicos."""
        phys = [getattr(ph, "name", type(ph).__name__) for ph in self.physics]
        return dict(solver="ibrans-M2" if phys else "ibrans-M1",
                    scheme="artificial-compressibility",
                    rk=self.cfg.rk, limiter=self.cfg.limiter,
                    mach_art=self.cfg.mach_art, c_art=self.cfg.c_art,
                    cfl=self.cfg.cfl, rho=self.cfg.rho, nu=self.cfg.nu,
                    rhie_chow=self.cfg.rc_alpha, physics=phys,
                    variable_density=self.rho_field is not None,
                    backend=self.be.summary(), mesh=self.mesh.summary(),
                    promises=(("RANS k-eps realizable + energia low-Mach "
                               "segun los modelos registrados; " if phys else
                               "laminar incompresible frio con IB; ")
                              + "sin especies (M3), sin producto de "
                              "combustor (M4); NUNCA temperatura de metal"))

    # -- utilidades de post ------------------------------------------------
    def field(self, comp: int) -> np.ndarray:
        """Componente `comp` de la velocidad en el interior (sin fantasmas),
        SIEMPRE en NumPy float64 (baja del dispositivo si hace falta)."""
        return self._down(self.u[comp][self.mesh.interior])

    def pressure(self) -> np.ndarray:
        return self._down(self.p[self.mesh.interior])

    def coords(self) -> list:
        """Coordenadas de centro de celda del interior, por eje."""
        m = self.mesh
        return [m.lo[d] + (np.arange(m.shape[d]) + 0.5) * m.dx
                for d in range(m.ndim)]


# ==========================================================================
# 4. Transporte de escalares (el kernel de M2: k, ε, T, Z)
# ==========================================================================
def advect_diffuse_scalar(solver, fld, gamma, uf_list=None):
    """RHS de un escalar en forma **advectiva** sobre el interior::

        ∂φ/∂t = −∇·(u_f φ̂) + φ·(∇·u_f) + ∇·(Γ∇φ)
               = −u·∇φ + ∇·(Γ∇φ)

    con φ̂ reconstruido por el MISMO MUSCL+TVD del momento y u_f la
    velocidad de cara de Rhie–Chow (así el escalar ve la misma advección
    que el momento, sin modo tablero). La forma advectiva es la correcta a
    ρ variable (tras dividir por ρ) y coincide con la conservativa cuando
    ∇·u = 0.

    `gamma` es la difusividad por celda (array `shape_tot`) o un escalar.
    Devuelve `(rhs, dif)` del tamaño del interior; `dif` es SOLO el término
    difusivo ∇·(Γ∇φ), que la energía necesita aparte para la fuente de
    divergencia q_T = ∇·(α∇T)/T. Escrito contra `xp`: CPU y GPU.
    """
    m, xp, cfg = solver.mesh, solver.xp, solver.cfg
    inv_dx = 1.0 / m.dx
    lim = cfg.limiter
    if uf_list is None:
        uf_list = solver.face_velocities()
    scalar_gamma = (np.ndim(gamma) == 0)
    adv = xp.zeros(m.shape, dtype=solver.dtype)
    dif = xp.zeros(m.shape, dtype=solver.dtype)
    div = xp.zeros(m.shape, dtype=solver.dtype)
    for d in range(m.ndim):
        uf = uf_list[d]
        aL = m.fview(fld, d, 0)
        aR = m.fview(fld, d, 1)
        aLL = m.fview(fld, d, -1)
        aRR = m.fview(fld, d, 2)
        dcen = aR - aL
        sL = _slope(aL - aLL, dcen, lim, xp)
        sR = _slope(dcen, aRR - aR, lim, xp)
        face = xp.where(uf > 0.0, aL + 0.5 * sL, aR - 0.5 * sR)
        adv -= xp.diff(uf * face, axis=d) * inv_dx
        if scalar_gamma:
            g_f = solver.dtype(gamma)
        else:
            g_f = 0.5 * (m.fview(gamma, d, 0) + m.fview(gamma, d, 1))
        dif += xp.diff(g_f * dcen * inv_dx, axis=d) * inv_dx
        div += xp.diff(uf, axis=d) * inv_dx
    adv += fld[m.interior] * div
    return adv + dif, dif


def scalar_bc_spec(solver, key: str, inlet_value: float,
                   default_value: float | None = None) -> dict:
    """Deriva el spec de contorno de un escalar desde los contornos del
    solver: `inlet` → Dirichlet (el valor `key` del spec de la cara si
    existe, si no `inlet_value`); `wall` → Dirichlet sólo si la cara trae
    `key` (p. ej. pared isoterma con `T=...`), si no Neumann; `outlet`/
    `slip` → Neumann; `periodic` → periódico. Devuelve
    `{clave_cara: ("dirichlet", v) | ("neumann", None) | ("periodic", None)}`.

    El valor puede ser un ARRAY por celda sobre el plano de la cara (misma
    convención que los perfiles de `u`: tamaño del plano CON fantasmas
    transversales); se sube al backend aquí, una sola vez.
    """
    out = {}
    for fk, spec in solver.bcs.items():
        kind = spec["kind"]
        if kind == "periodic":
            out[fk] = ("periodic", None)
        elif kind == "inlet" or (kind == "wall" and key in spec):
            v = spec.get(key, inlet_value)
            v = solver._up(v) if np.ndim(v) else float(v)
            out[fk] = ("dirichlet", v)
        else:
            out[fk] = ("neumann", None)
        if default_value is not None and out[fk][0] == "dirichlet" \
                and np.ndim(out[fk][1]) == 0 \
                and not math.isfinite(out[fk][1]):
            out[fk] = ("dirichlet", float(default_value))
    return out


def fill_scalar_bc(solver, fld, spec: dict) -> None:
    """Rellena las capas fantasma del DOMINIO de un campo escalar según
    `spec` (formato de `scalar_bc_spec`). El IB no se toca aquí: cada
    modelo aplica su condición inmersa (`mesh.apply_neumann` o valor de
    pared) después. Misma maquinaria de planos que `apply_bc`."""
    m, ng = solver.mesh, solver.mesh.ng
    for d in range(m.ndim):
        n = m.shape[d]
        for side in ("lo", "hi"):
            kind, val = spec.get(f"{'xyz'[d]}{side}", ("neumann", None))
            for k in range(1, ng + 1):
                if side == "lo":
                    gi, mi = ng - k, ng + k - 1
                    pi = ng + n - k
                else:
                    gi, mi = ng + n - 1 + k, ng + n - k
                    pi = ng + k - 1
                G = solver._plane(d, gi)
                M = solver._plane(d, mi)
                P = solver._plane(d, pi)
                if kind == "periodic":
                    fld[G] = fld[P]
                elif kind == "dirichlet":
                    # `val` escalar o perfil por celda del plano
                    vv = val if np.ndim(val) else solver.dtype(val)
                    fld[G] = 2.0 * vv - fld[M]
                else:
                    fld[G] = fld[M]
