"""
QUASAR Phy-CB · cfd/ibrans/energy.py
====================================
**Hito M2 de WAVE6 — ecuación de energía + ρ(T) low-Mach + escalar pasivo**
(tarea 2.4 de `ESTADO.md` §7). Verificado por los tests M2 de
`cfd/tests/test_ibrans_v4_v5.py` (expansión térmica analítica, conducción,
conservación del escalar).

Modelo low-Mach (presión termodinámica constante, WAVE6 §2)
-----------------------------------------------------------
Gas ideal a presión termodinámica p₀ fija (la acústica ya la lleva la
pseudo-compresibilidad del solver; aquí p₀ NO es la p del solver, que es
la presión mecánica)::

    DT/Dt = ∇·(α_eff ∇T) + S_T,      α_eff = ν/Pr + ν_t/Pr_t
    ρ(T)  = p₀ / (R_g · T)
    ∇·u   = q_T = (∇·(α_eff ∇T) + S_T)/T      (continuidad low-Mach)

La última línea es exacta para gas ideal a p₀ constante:
(1/ρ)Dρ/Dt = −(1/T)DT/Dt. El solver la recibe por `solver.div_source` y la
mete en su ecuación de presión (∂p/∂t = −ρc²(∇·u_f − q_T)); `ρ(T)` entra
por `solver.rho_field` (1/ρ en ∇p y forma advectiva del momento). El
término S_T es una fuente volumétrica opcional en K/s (la usará M3 para la
liberación de calor EDM; aquí sirve para el test analítico de expansión).

Lo que NO hace (declarado)
--------------------------
* Sin radiación ni CHT (WAVE6 §0): por eso L2a no promete T de metal.
* Sin entalpía de formación ni especies: eso es M3 (`chem.py`).
* μ y λ no dependen de T (ν y Pr constantes): a los ΔT del combustor esto
  sesga la difusión molecular, que en régimen turbulento está dominada por
  ν_t/Pr_t — declarado, no corregido en v1.
* Paredes ADIABÁTICAS por defecto (IB incluida). Isoterma opt-in: la cara
  del dominio con `T=...` en su spec, o `T_wall_ib_K` para la inmersa
  (lo necesita V9/PRECCINSTA, WAVE6 §4).

El escalar pasivo (fracción de mezcla Z) comparte kernel y contornos:
`inlet` con `Z=...` en el spec (0 por defecto), Neumann en el resto,
D_eff = ν/Sc + ν_t/Sc_t. Es el transporte que M3 convertirá en Y_k.

API pública
-----------
    EnergyConfig, EnergyLowMach(mesh, cfg).attach(solver, turbulence=None)
    ScalarConfig, PassiveScalar(mesh, cfg).attach(solver, turbulence=None)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Constante del aire (SI). El combustor real ajustará R_g con la mezcla
#: en M3; para M2 el aire es suficiente y está declarado.
R_AIR = 287.05

#: Prandtl turbulento estándar de los códigos RANS (0.85). ⚠️ TUNABLE
#: (contrato §10.7) — la banda de la literatura es 0.85–0.92.
PRANDTL_T = 0.85


@dataclass
class EnergyConfig:
    """Parámetros de la energía low-Mach (SI).

    `nu_power_T > 0` activa el transporte molecular variable
    ν(T) = ν₀·(T/T_ref)^n (y con él α(T) = ν(T)/Pr), **sólo en régimen
    laminar** (sin turbulencia enganchada): es lo que necesitan las llamas
    laminares de V6/V7 — α en la zona quemada es ~20× la fría y con α
    constante S_L saldría varias veces baja. Con turbulencia, ν_t domina y
    el molecular queda constante (declarado, §8.13 de ESTADO). El
    exponente ~1.7 es el ajuste estándar de aire/gases de combustión;
    V6 lo calibra contra el α(T) real de Cantera del mecanismo."""
    p_thermo_Pa: float = 101_325.0
    r_gas: float = R_AIR
    prandtl: float = 0.7
    prandtl_t: float = PRANDTL_T
    T_inlet_K: float = 300.0
    T_floor_K: float = 150.0
    T_ceiling_K: float = 3000.0
    T_wall_ib_K: float | None = None      # None = IB adiabática
    nu_power_T: float = 0.0               # 0 = transporte constante (M2)
    T_ref_transport_K: float = 300.0


class EnergyLowMach:
    """Energía + ρ(T) acoplada al solver por `solver.physics`. Uso::

        tur = KEpsilonRealizable(mesh).attach(solver)     # opcional
        en  = EnergyLowMach(mesh).attach(solver, turbulence=tur)
        solver.run()

    Registrarla DESPUÉS de la turbulencia: usa su ν_t del mismo paso.
    """

    name = "energy-lowmach"

    def __init__(self, mesh, cfg: EnergyConfig | None = None):
        self.mesh = mesh
        self.cfg = cfg or EnergyConfig()
        self.T = None
        self.rho = None
        self.turbulence = None
        self.solver = None
        self.status = "detached"
        self.source_K_s = None            # S_T opcional (K/s, shape_tot)
        self._q = None
        self._max_diff = 0.0
        #: pared transpirante del M4: (máscara por fantasma, T impuesta).
        #: El aire que entra por la pared porosa debe entrar A SU
        #: TEMPERATURA (T3), no a la del gas local — sin esto los chorros
        #: de dilución «entran calientes» y no refrigeran nada (medido:
        #: T4 salía ≈ T_ad). Ver `set_wall_dirichlet`.
        self._wall_dir = None

    def attach(self, solver, turbulence=None) -> "EnergyLowMach":
        from .solver import scalar_bc_spec
        xp, m, cfg = solver.xp, solver.mesh, self.cfg
        dt_ = solver.dtype
        self.solver = solver
        self.turbulence = turbulence
        self.T = xp.full(m.shape_tot, dt_(cfg.T_inlet_K), dtype=dt_)
        self.bc = scalar_bc_spec(solver, "T", cfg.T_inlet_K)
        self._fill_bcs(solver)
        self.rho = xp.asarray(
            dt_(cfg.p_thermo_Pa) / (dt_(cfg.r_gas) * self.T), dtype=dt_)
        self._q = xp.zeros(m.shape, dtype=dt_)
        solver.rho_field = self.rho
        solver.div_source = self._q
        solver.physics.append(self)
        self._max_diff = float(solver.cfg.nu / cfg.prandtl)
        self.status = "attached"
        return self

    def max_diffusivity(self) -> float:
        return self._max_diff

    def laminar_nu(self, solver):
        """ν laminar del caso: campo ν(T) si `nu_power_T` está activo (y
        no hay turbulencia), si no el escalar `cfg.nu`. Lo comparten el
        momento (`solver.nu_field`), α y el D de las especies (Le = 1)."""
        cfg = self.cfg
        if cfg.nu_power_T <= 0.0 or self.turbulence is not None:
            return solver.dtype(solver.cfg.nu)
        xp = solver.xp
        tr = solver.dtype(cfg.T_ref_transport_K)
        return solver.dtype(solver.cfg.nu) \
            * (xp.maximum(self.T, solver.dtype(cfg.T_floor_K)) / tr) \
            ** solver.dtype(cfg.nu_power_T)

    def _alpha(self, solver):
        """α_eff = ν(T)/Pr + ν_t/Pr_t (campo si hay turbulencia o ν(T))."""
        cfg = self.cfg
        base = self.laminar_nu(solver) / solver.dtype(cfg.prandtl)
        if self.turbulence is None:
            return base
        return base + self.turbulence.nu_t_field \
            / solver.dtype(cfg.prandtl_t)

    def set_wall_dirichlet(self, mask_per_ghost, value_K: float
                           ) -> "EnergyLowMach":
        """Impone T = `value_K` SOLO en los fantasmas de `mask` (pared
    transpirante del M4); el resto sigue con su condición global. La
        máscara se sube al backend UNA vez aquí."""
        xp = self.solver.xp
        self._wall_dir = (xp.asarray(np.asarray(mask_per_ghost, bool)),
                          float(value_K))
        return self

    def _fill_bcs(self, solver) -> None:
        from .solver import fill_scalar_bc
        m, cfg = solver.mesh, self.cfg
        fill_scalar_bc(solver, self.T, self.bc)
        if m.phi is not None:
            if cfg.T_wall_ib_K is None:
                m.apply_neumann(self.T)               # adiabática
            else:
                m.apply_dirichlet(self.T, float(cfg.T_wall_ib_K))
            if self._wall_dir is not None:            # pared transpirante
                m.apply_dirichlet(self.T, self._wall_dir[1],
                                  only=self._wall_dir[0])
            m.zero_deep_solid(self.T, cfg.T_inlet_K)

    def advance(self, solver, dt: float) -> bool:
        from .solver import advect_diffuse_scalar
        xp, m, cfg = solver.xp, solver.mesh, self.cfg
        dt_ = solver.dtype
        I = m.interior
        alpha = self._alpha(solver)
        rhs, dif = advect_diffuse_scalar(solver, self.T, alpha,
                                         solver.face_velocities(dt))
        if self.source_K_s is not None:
            s = self.source_K_s[I]
            rhs = rhs + s
            dif = dif + s
        fm = solver._fmask
        t_i = self.T[I]
        t_new = t_i + dt_(dt) * rhs * fm
        t_new = xp.clip(t_new, dt_(cfg.T_floor_K), dt_(cfg.T_ceiling_K))
        self.T[I] = t_new
        self._fill_bcs(solver)
        # ρ(T) en TODO el array (los fantasmas ya están rellenos)
        self.rho[...] = dt_(cfg.p_thermo_Pa) / (dt_(cfg.r_gas) * self.T)
        # continuidad low-Mach: ∇·u objetivo = (∇·(α∇T)+S_T)/T
        self._q[...] = (dif / xp.maximum(t_new, dt_(cfg.T_floor_K))) * fm
        if cfg.nu_power_T > 0.0 and self.turbulence is None:
            # régimen laminar con ν(T): el momento ve el campo actualizado
            solver.nu_field = self.laminar_nu(solver)
            tmax = float(xp.max(self.T))
            if solver.be.is_gpu:
                solver._sync_calls += 1
            self._max_diff = (solver.cfg.nu / cfg.prandtl) \
                * (tmax / cfg.T_ref_transport_K) ** cfg.nu_power_T
        elif self.turbulence is None:
            self._max_diff = float(solver.cfg.nu / cfg.prandtl)
        else:
            # la turbulencia ya redujo max(ν_eff) este paso: se reutiliza
            # su float cacheado en vez de pagar otra reducción
            self._max_diff = (solver.cfg.nu / cfg.prandtl
                              + max(self.turbulence.max_diffusivity()
                                    - solver.cfg.nu, 0.0) / cfg.prandtl_t)
        return True

    def summary(self) -> dict:
        return dict(model=self.name, status=self.status,
                    p_thermo_Pa=self.cfg.p_thermo_Pa,
                    r_gas=self.cfg.r_gas, prandtl=self.cfg.prandtl,
                    prandtl_t=self.cfg.prandtl_t,
                    wall_ib=("adiabatica" if self.cfg.T_wall_ib_K is None
                             else f"isoterma {self.cfg.T_wall_ib_K} K"),
                    promises=("T de gas para ranking; NUNCA temperatura de "
                              "metal (contrato §12); sin radiacion ni CHT "
                              "(WAVE6 §0)"))


@dataclass
class ScalarConfig:
    """Escalar pasivo (fracción de mezcla Z en M2; Y_k lo hereda en M3)."""
    key: str = "Z"
    inlet_value: float = 0.0
    schmidt: float = 0.7
    schmidt_t: float = 0.7
    clip: tuple[float, float] = (0.0, 1.0)


class PassiveScalar:
    """Escalar pasivo conservado (misma advección MUSCL + uf de Rhie–Chow
    que el momento). El clip a [0,1] es un limitador DECLARADO: el MUSCL
    con limitador TVD ya es casi monótono y el clip sólo recorta el
    redondeo, no la física."""

    name = "passive-scalar"

    def __init__(self, mesh, cfg: ScalarConfig | None = None):
        self.mesh = mesh
        self.cfg = cfg or ScalarConfig()
        self.Z = None
        self.turbulence = None
        self.status = "detached"
        self._max_diff = 0.0

    def attach(self, solver, turbulence=None) -> "PassiveScalar":
        from .solver import scalar_bc_spec
        xp, m, cfg = solver.xp, solver.mesh, self.cfg
        dt_ = solver.dtype
        self.turbulence = turbulence
        self.Z = xp.zeros(m.shape_tot, dtype=dt_)
        self.bc = scalar_bc_spec(solver, cfg.key, cfg.inlet_value)
        self._fill_bcs(solver)
        solver.physics.append(self)
        self._max_diff = float(solver.cfg.nu / cfg.schmidt)
        self.status = "attached"
        return self

    def max_diffusivity(self) -> float:
        return self._max_diff

    def _fill_bcs(self, solver) -> None:
        from .solver import fill_scalar_bc
        m = solver.mesh
        fill_scalar_bc(solver, self.Z, self.bc)
        if m.phi is not None:
            m.apply_neumann(self.Z)
            m.zero_deep_solid(self.Z, 0.0)

    def advance(self, solver, dt: float) -> bool:
        from .solver import advect_diffuse_scalar
        xp, m, cfg = solver.xp, solver.mesh, self.cfg
        dt_ = solver.dtype
        I = m.interior
        gamma = dt_(solver.cfg.nu / cfg.schmidt)
        if self.turbulence is not None:
            gamma = gamma + self.turbulence.nu_t_field \
                / dt_(cfg.schmidt_t)
        rhs, _ = advect_diffuse_scalar(solver, self.Z, gamma,
                                       solver.face_velocities(dt))
        z_new = self.Z[I] + dt_(dt) * rhs * solver._fmask
        self.Z[I] = xp.clip(z_new, dt_(cfg.clip[0]), dt_(cfg.clip[1]))
        self._fill_bcs(solver)
        if self.turbulence is None:
            self._max_diff = float(solver.cfg.nu / cfg.schmidt)
        else:
            self._max_diff = (solver.cfg.nu / cfg.schmidt
                              + max(self.turbulence.max_diffusivity()
                                    - solver.cfg.nu, 0.0) / cfg.schmidt_t)
        return True

    def summary(self) -> dict:
        return dict(model=self.name, key=self.cfg.key, status=self.status,
                    schmidt=self.cfg.schmidt, schmidt_t=self.cfg.schmidt_t)
