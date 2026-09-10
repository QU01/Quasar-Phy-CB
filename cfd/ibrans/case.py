"""
QUASAR Phy-CB · cfd/ibrans/case.py
==================================
**Hito M4 — el CASO del combustor completo** (WAVE6 §4, plan detallado en
`M4_PLAN.md`). Este módulo convierte el contrato `phycb-combustor-1` en un
caso reactivo turbulento LISTO PARA CORRER: SDF completo, perfiles de
entrada, transpiración de la efusión y de las filas no resueltas, fuente de
combustible y plan de encendido. `__init__.solve()` y los tests de la
escalera V9/M4 lo usan; `post.py` lee lo que este módulo deja en
`solver.m4`.

Arquitectura del caso (todas las decisiones DECLARADAS)
-------------------------------------------------------
* **Dominio**: caja cartesiana que encierra la CARCASA; las caras z caen en
  el plano del domo y en el plano 4. El volumen fluido es el SDF completo de
  `mesh.sdf_combustor_from_contract` (liner ∪ anillo ∪ filas resueltas).
* **Entrada** = la cara del domo con PERFIL POR CELDA (la técnica de la
  ranura del JICF, `run_v5_jcf.py`): anillo del swirler con
  u_θ = u_ax·tan(β) impuesta (opción A del plan: NO se resuelven álabes; da
  el swirl correcto, que es lo que el PF necesita) + anillo de la carcasa
  con el aire restante; u = 0 fuera (equivale al domo sólido).
* **Reparto objetivo por conductancias**: ṁ_i ∝ C_d·A_i del CONTRATO
  (swirler, cada fila, efusión). Es el mismo modelo de caminos en paralelo
  de la red 1D de `crn_core`, y para la efusión usa `effusion.Cd` y el área
  del contrato (contrato ENTRE CAPAS, test T49).
* **Filas de orificios**: resueltas en el SDF si d/dx ≥ 4 **y** el anillo
  cabe; si no, **fuente volumétrica de masa y momento** calibrada por
  C_d·A en las celdas fluidas contiguas a la pared (`TranspirationSource`)
  — que es lo que WAVE6 §0 pide literalmente. NO se usa una condición de
  contorno de velocidad: fija u en la pared pero el caudal lo decide el
  campo, y con la U_jet real entregaba un 17 % menos (medido 2026-08-03).
  Qué fila fue a qué camino queda declarado en `solver.m4["paths"]`.
* **Efusión**: NUNCA resuelta; misma fuente volumétrica repartida sobre la
  pared exterior del liner (`effusion.walls = ['outer']`, trampa §3 del
  plan), con U_jet ≈ 0 (es transpiración lenta, no chorro).
* **Modo «sin anillo»** (DECLARADO): si la holgura liner↔carcasa es menor
  que 2·dx —el caso del KJ66 a dx = 2 mm, ~3 mm de holgura— el anillo no
  se resuelve, el dominio es sólo el interior del liner y TODAS las filas
  van como fuente. Consecuencia que hay que decir en voz alta: en ese modo
  el reparto de aire es un **input**, no una predicción, y la pérdida de
  orificio no ocurre dentro del dominio (`post.dp_over_p` lo declara con
  `comparable_with_1d=False`). Resolverlo pide dx ≤ 1 mm.
* **Chorros con su turbulencia**: los perfiles de entrada llevan k/ε de
  orificio (I = 10 %, ℓ = 0.07·d — la lección del JICF, +26 % si no).
  Límite declarado: los chorros TRANSPIRADOS por el IB no imponen su k/ε
  (el gancho no existe); sesga la penetración de esas filas, no el reparto.
* **Combustible**: fuente volumétrica de mezcla conservativa
  (`SpeciesTransport.set_fuel_source`) en la región del inyector del
  contrato; spray no resuelto, evaporación instantánea DECLARADA.
* **Encendido**: parche en la posición de la BUJÍA del contrato (la fila
  `bujia` con `flow=false`), plan V6c (hold + rampa + sub-relajación);
  criterio `sp.anchored()`.

API pública
-----------
    CaseConfig
    build_case(contract, cfg) -> IBRANSSolver     (con `solver.m4` puesto)
    run_case(solver, ...) -> dict                 (driver con anclaje)
    TranspirationSource                           (fuente masa+momento)
    measure_splits(solver) -> dict                (balance de VC axial)
    swirl_number_measured(solver) -> dict

⚠️ `measure_splits` mide por balance de volumen de control axial, y ese
método deja de ser limpio cuando los chorros llevan momento y reorganizan
el flujo (medido: ±30 % por fila con el caudal inyectado EXACTO). El
caudal que de verdad entra cada fila es el de `TranspirationSource.rows`,
que es exacto por construcción; `measure_splits` sirve para el cierre de
masa global y para el caso con anillo resuelto, donde el reparto sí es una
predicción.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .backend import to_numpy as _to_numpy
from .chem import ChemConfig, IgnitionPlan, SpeciesTransport
from .energy import EnergyConfig, EnergyLowMach
from .mesh import mesh_combustor_from_contract
from .solver import IBRANSSolver, SolverConfig, make_bcs
from .turbulence import C_MU, KEpsilonRealizable, TurbulenceConfig

_MM = 1.0e-3

#: C_d del canal entre álabes del swirler. ⚠️ ESPEJO de `crn_core.CD_SWIRLER`
#: (0.80): si la red 1D lo cambia, hay que cambiarlo aquí — los splits
#: objetivo del caso y los de la red describirían combustores distintos.
CD_SWIRLER = 0.80

#: C_d de respaldo de la efusión si el contrato no trae `effusion.Cd`
#: (espejo del `CD_EFFUSION_INCLINED` compartido por `crn_core`).
CD_EFFUSION_FALLBACK = 0.38

#: Intensidad y escala de los CHORROS de orificio (lección del JICF,
#: ESTADO.md §11.4): I = 10 % y ℓ = 0.07·d escalando con u_chorro.
JET_INTENSITY = 0.10
JET_LENGTH_FRAC = 0.07


@dataclass
class CaseConfig:
    """Parámetros del caso M4. Los defaults son los del plan (mach_art
    0.2–0.3 en estacionario; FP64 mientras la V&V no se rehaga en FP32)."""
    dx_mm: float = 1.5
    reactive: bool = True
    ignition: bool = True
    mach_art: float = 0.25
    cfl: float = 0.4
    rk: str = "rk2"
    max_iter: int = 15_000
    timeout_s: float = 1500.0
    backend: str = "auto"
    precision: str = "auto"
    t_ad_K: float | None = None       # T_ad objetivo para calibrar c_p
    T_wall_ib_K: float | None = None  # None = adiabática (V9 usa isoterma)
    fuel_src_r_mm: float = 6.0        # radio de la región de inyección
    residual_every: int = 100
    n_hold: int = 1500
    n_ramp: int = 3000
    n_stable: int = 400
    #: iteraciones de ASENTAMIENTO tras la rampa antes de admitir la parada
    #: por anclaje: la llama ancla antes de que el CAMPO se establezca
    #: (medido: parar al anclar dejaba el plano 4 a medio llenar,
    #: cierre de masa −0.5 espurio). ~1 tiempo de tránsito.
    n_settle: int = 3000


def _f(x, d=0.0) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else float(d)
    except Exception:
        return float(d)


def _conductance_paths(contract: dict, meta: dict) -> dict:
    """Reparto objetivo de aire por caminos en paralelo, ṁ_i ∝ C_d·A_i del
    contrato (el modelo de la red 1D). Devuelve
    `{path: dict(CdA_m2, frac, mdot_kgs, kind, row)}`."""
    bc = (contract or {}).get("cfd_boundary_conditions") or {}
    mdot_air = _f(bc.get("mdot_air_kgs"), 0.1)
    sw = (contract or {}).get("swirler") or {}
    eff = (contract or {}).get("effusion") or {}
    paths: dict[str, dict] = {}
    a_sw = _f(sw.get("A_sw_total_mm2")) * _MM * _MM
    if a_sw > 0.0:
        paths["swirler"] = dict(CdA_m2=CD_SWIRLER * a_sw, kind="inlet",
                                row=None)
    for row in meta.get("rows") or []:
        if row["CdA_m2"] <= 0.0:
            continue
        kind = "resolved" if row["resolved"] else "transpired"
        paths[row["id"]] = dict(CdA_m2=row["CdA_m2"], kind=kind, row=row)
    cd_e = eff.get("Cd")
    cd_e = _f(cd_e, CD_EFFUSION_FALLBACK) if cd_e is not None \
        else CD_EFFUSION_FALLBACK
    a_open = _f(eff.get("open_area_mm2")) * _MM * _MM
    if a_open > 0.0:
        paths["effusion"] = dict(CdA_m2=cd_e * a_open, kind="transpired",
                                 row=None)
    tot = sum(p["CdA_m2"] for p in paths.values())
    for p in paths.values():
        p["frac"] = p["CdA_m2"] / max(tot, 1e-12)
        p["mdot_kgs"] = p["frac"] * mdot_air
    return paths


class TranspirationSource:
    """Filas de orificios NO resueltas y efusión como **fuente volumétrica
    de masa y de momento** en las celdas fluidas contiguas a la pared —
    literalmente lo que WAVE6 §0 pide: «si no [resuelta], fila = fuente de
    momento/masa calibrada por C_d».

    Por qué fuente y no condición de contorno de velocidad
    ------------------------------------------------------
    Una BC de velocidad fija u EN LA PARED, pero el caudal que de verdad
    cruza lo decide el campo. Con la U_jet real (65–80 m/s) el Dirichlet
    entregaba un 17 % menos de masa y los splits se iban al −47 %
    (medido el 2026-08-03; con 15 m/s el error era del 2 % y por eso no se
    había visto). La fuente entra en la continuidad low-Mach por el MISMO
    gancho que la dilatación térmica, así que el caudal es EXACTO por
    construcción y la velocidad de descarga deja de ser un compromiso::

        q_i = ṁ_i /(ρ₃·V_i)                    [1/s]    → ∇·u = q
        f_i = ṁ_i·U_jet,i/(ρ₃·V_i)·n̂           [m/s²]   → momento
        U_jet,i = ṁ_i/(ρ₃·C_d·A_geom,i)                 (descarga real)

    El aire entra además **frío y fresco**: mezcla hacia T₃ y hacia
    composición de aire con la misma tasa `q`, que es lo que hace que la
    dilución diluya y refrigere.
    """

    name = "transpiration-source"

    def __init__(self, mesh, rho3: float):
        self.mesh = mesh
        self.rho3 = float(rho3)
        self.q = None
        self.rows: list[dict] = []
        self.energy = None
        self.species = None
        self.T_in = 300.0

    def attach(self, solver, rows: list[dict], energy=None, species=None,
               T_in: float = 300.0) -> "TranspirationSource":
        """`rows` = [{id, mask(interior), mdot_kgs, u_jet_ms, nx, ny}]."""
        xp, m = solver.xp, solver.mesh
        dt_ = solver.dtype
        dv = m.dx ** m.ndim
        self.energy = energy
        self.species = species
        self.T_in = float(T_in)
        self.q = xp.zeros(m.shape, dtype=dt_)
        # densidades de fuente INDEPENDIENTES de ρ: kg/(m³·s) de masa y
        # kg/(m²·s²) de momento. La división por ρ se hace en `advance`
        # con la densidad LOCAL — ver allí por qué.
        self._sm = xp.zeros(m.shape, dtype=dt_)
        self._fx = xp.zeros(m.shape, dtype=dt_)
        self._fy = xp.zeros(m.shape, dtype=dt_)
        for r in rows:
            msk = np.asarray(r["mask"], dtype=bool)
            n = int(np.count_nonzero(msk))
            if n == 0 or r["mdot_kgs"] <= 0.0:
                continue
            s_i = r["mdot_kgs"] / (n * dv)            # kg/(m³·s)
            mk = xp.asarray(msk)
            self._sm = self._sm + mk * dt_(s_i)
            f_i = s_i * float(r["u_jet_ms"])          # kg/(m²·s²)
            self._fx = self._fx + mk * xp.asarray(r["nx"] * f_i, dtype=dt_)
            self._fy = self._fy + mk * xp.asarray(r["ny"] * f_i, dtype=dt_)
            self.rows.append(dict(id=r["id"], cells=n,
                                  q_1_s=s_i / self.rho3,
                                  u_jet_ms=float(r["u_jet_ms"]),
                                  mdot_kgs=float(r["mdot_kgs"])))
        self._fz = xp.zeros(m.shape, dtype=dt_)
        self.q = self._sm / dt_(self.rho3)
        solver.div_extra.append(self.q)
        solver.force_field = [self._fx / dt_(self.rho3),
                              self._fy / dt_(self.rho3), self._fz]
        solver.physics.append(self)
        return self

    def max_diffusivity(self) -> float:
        return 0.0

    def _rescale(self, solver) -> None:
        """Reescala la fuente con la densidad LOCAL.

        ⚠️ El error que esto corrige: en la continuidad low-Mach la fuente
        es ∇·u = q, así que el caudal MÁSICO que entra vale ρ_local·q·V.
        Con q fijado a ṁ/(ρ₃·V), en la llama —donde ρ cae a ~1/5 de ρ₃—
        entraba cinco veces menos aire del pedido: la dilución no diluía,
        T4 se iba a 1791 K (+56 %) y el cierre de masa a −42 % (medido el
        2026-08-03; en frío no se notaba porque ρ ≡ ρ₃ y el cierre daba
        +7.6 %). Con q = ṁ/(ρ_local·V) el caudal másico es el del contrato
        en frío y en caliente. Ídem el momento.
        """
        rho = solver.rho_field
        if rho is None:
            return                       # ρ constante: lo de `attach` vale
        xp = solver.xp
        r = xp.maximum(rho[solver.mesh.interior], solver.dtype(1e-6))
        self.q[...] = self._sm / r
        solver.force_field[0][...] = self._fx / r
        solver.force_field[1][...] = self._fy / r

    def advance(self, solver, dt: float) -> bool:
        """La masa y el momento los aplica el solver por sus ganchos; aquí
        se reescalan con ρ local y entra lo que viaja CON esa masa: T₃ y
        composición de aire."""
        xp, m = solver.xp, solver.mesh
        dt_ = solver.dtype
        I = m.interior
        self._rescale(solver)
        # fracción de celda renovada por paso. Se ACOTA a 1: con q del
        # orden de 4e4 1/s y dt del CFL acústico, dt·q ≈ 0.2, pero un dt
        # mayor (mach_art alto) la pasaría de 1 y la mezcla divergiría.
        fac = xp.minimum(dt_(dt) * self.q, dt_(1.0))
        if self.energy is not None and self.energy.T is not None:
            t = self.energy.T[I]
            self.energy.T[I] = t + fac * (dt_(self.T_in) - t)
        if self.species is not None:
            for sp_k, y0 in SpeciesTransport._AIR.items():
                if sp_k not in self.species.Y:
                    continue
                y = self.species.Y[sp_k][I]
                self.species.Y[sp_k][I] = y + fac * (dt_(y0) - y)
        return True

    def summary(self) -> dict:
        return dict(model=self.name, rows=self.rows,
                    note=("fuente volumetrica de masa+momento calibrada por "
                          "C_d (WAVE6 §0); caudal EXACTO por construccion"))


def _transpiration_rows(s, meta: dict, paths: dict, rho3: float,
                        diag: dict, z_eff=None) -> list[dict]:
    """Celdas FLUIDAS de inyección de cada fila transpirada y de la
    efusión, con su caudal, velocidad de descarga y normal (hacia −r, que
    es hacia el interior del liner)."""
    m = s.mesh
    itp = meta["_interp"]
    dx = m.dx
    t_w = meta["t_liner_mm"] * _MM
    xi, yi, zi = _interior_coords(s)
    ri = np.sqrt(xi ** 2 + yi ** 2)
    fl = m.fluid[m.interior]
    rout = np.interp(zi, itp["z"], itp["r_out"])
    band = fl & (ri >= rout - 2.5 * dx) & (ri < rout)
    nx = np.where(ri > 1e-12, -xi / np.maximum(ri, 1e-12), 0.0)
    ny = np.where(ri > 1e-12, -yi / np.maximum(ri, 1e-12), 0.0)
    out = []
    used = np.zeros_like(band)
    for pid, p in paths.items():
        row = p.get("row")
        if row is None or p["kind"] != "transpired":
            continue
        a_disc = row["n"] * math.pi * (0.5 * row["d_m"]) ** 2 * row["Cd"]
        u_jet = p["mdot_kgs"] / max(rho3 * a_disc, 1e-12)
        s_r = max(0.5 * row["d_m"] * math.sqrt(max(row["Cd"], 1e-3)),
                  0.9 * dx)
        r_c = row["r_m"]
        msk = np.zeros_like(band)
        for grow in range(4):                       # bucle ACOTADO
            rr = s_r * (1.0 + 0.5 * grow)
            msk = np.zeros_like(band)
            for k in range(row["n"]):
                a = row["phi0_rad"] + 2.0 * math.pi * k / row["n"]
                px, py = r_c * math.cos(a), r_c * math.sin(a)
                d2 = (xi - px) ** 2 + (yi - py) ** 2 + (zi - row["z_m"]) ** 2
                msk |= band & (d2 <= rr ** 2)
            if np.count_nonzero(msk) >= row["n"]:
                break
        if not np.any(msk):
            diag.setdefault("transpired_lost", []).append(pid)
            continue
        used |= msk
        out.append(dict(id=pid, mask=msk, mdot_kgs=p["mdot_kgs"],
                        u_jet_ms=u_jet, nx=nx, ny=ny))
        diag.setdefault("transpired_rows", []).append(
            dict(id=pid, cells=int(np.count_nonzero(msk)),
                 u_jet_ms=u_jet, mdot_kgs=p["mdot_kgs"]))
    eff = paths.get("effusion")
    if eff is not None and z_eff:
        msk = band & (zi >= z_eff[0]) & (zi <= z_eff[1]) & ~used
        if np.any(msk):
            out.append(dict(id="effusion", mask=msk,
                            mdot_kgs=eff["mdot_kgs"], u_jet_ms=0.0,
                            nx=nx, ny=ny))
            diag["effusion_cells"] = int(np.count_nonzero(msk))
    return out


def build_case(contract: dict, cfg: CaseConfig | None = None
               ) -> IBRANSSolver:
    """Contrato → caso M4 montado (malla + BCs + modelos + fuentes).

    Deja TODO lo que el post-proceso necesita en `solver.m4` (meta del SDF,
    caminos, caras, termodinámica). Lanza `ValueError` sólo ante un
    contrato sin perfil de liner — `solve()` lo captura y degrada."""
    cfg = cfg or CaseConfig()
    c = contract or {}
    bc = c.get("cfd_boundary_conditions") or {}
    P3 = _f(bc.get("P3_Pa"), 101_325.0)
    T3 = _f(bc.get("T3_K"), 300.0)
    rho3 = _f(bc.get("rho3_kgm3"), P3 / (287.05 * max(T3, 1.0)))
    mdot_air = _f(bc.get("mdot_air_kgs"), 0.1)
    mdot_fuel = _f(bc.get("mdot_fuel_kgs"), 0.0)
    fuel = str(bc.get("fuel", "jet_a"))

    m, meta = mesh_combustor_from_contract(c, dx_mm=cfg.dx_mm)
    if not meta.get("ib"):
        raise ValueError("el contrato no produjo un SDF con IB")
    axis_z = 2
    sign = float(meta["flow_sign"])
    itp = meta["_interp"]
    t_w = meta["t_liner_mm"] * _MM
    z_dome = meta["dome_z_mm"] * _MM
    paths = _conductance_paths(c, meta)
    diag: dict = {}

    # --- perfiles de la cara del domo (entrada) ---------------------------
    xg, yg = np.meshgrid(m.xc[0], m.xc[1], indexing="ij")
    rr = np.sqrt(xg ** 2 + yg ** 2)
    phig = np.arctan2(yg, xg)
    r_in_d = float(np.interp(z_dome, itp["z"], itp["r_in"]))
    r_out_d = float(np.interp(z_dome, itp["z"], itp["r_out"]))
    r_cas_d = float(np.interp(z_dome, itp["z_cas"], itp["r_cas"]))

    sw = c.get("swirler") or {}
    inj = c.get("injectors") or {}
    ring_r = _f(inj.get("ring_r_mm")) * _MM
    a_sw = _f(sw.get("A_sw_total_mm2")) * _MM * _MM
    beta = math.radians(_f(sw.get("beta_deg"), 50.0))
    if ring_r <= 1e-4:                      # can: anillo centrado en el eje
        r_lo = 0.5 * _f(sw.get("d_hub_mm")) * _MM
        r_hi = 0.5 * _f(sw.get("d_tip_mm")) * _MM
    else:                                   # anular: corona de área A_sw
        h = a_sw / max(4.0 * math.pi * ring_r, 1e-9)
        r_lo, r_hi = ring_r - h, ring_r + h
    h_min = 0.9 * m.dx                      # que la corona exista en malla
    if (r_hi - r_lo) < 2.0 * h_min:
        mid = 0.5 * (r_lo + r_hi)
        r_lo, r_hi = mid - h_min, mid + h_min
    r_lo = max(r_lo, r_in_d + 0.5 * m.dx)
    r_hi = min(r_hi, r_out_d - 0.5 * m.dx)
    sw_mask = (rr >= r_lo) & (rr <= r_hi)
    if meta.get("annulus_resolved"):
        ann_mask = (rr >= r_out_d + t_w) & (rr <= r_cas_d)
    else:               # modo sin anillo: todo el aire no-swirler entra
        ann_mask = np.zeros_like(sw_mask)      # por transpiración C_d·A

    # plano interior pegado a la cara de entrada, para áreas DISCRETAS
    k_in = m.ng if sign > 0 else m.ng + m.shape[axis_z] - 1
    fl_in = m.fluid[:, :, k_in]
    a_sw_cells = max(int(np.count_nonzero(sw_mask & fl_in)), 1) * m.dx ** 2
    a_ann_cells = max(int(np.count_nonzero(ann_mask & fl_in)), 1) * m.dx ** 2

    mdot_sw = _f((paths.get("swirler") or {}).get("mdot_kgs"))
    mdot_ann = (mdot_air - mdot_sw) if meta.get("annulus_resolved") else 0.0
    u_sw = mdot_sw / max(rho3 * a_sw_cells, 1e-12)
    u_ann = mdot_ann / max(rho3 * a_ann_cells, 1e-12)
    # u_θ se elige para que el anillo EQUIVALENTE reproduzca el SN objetivo
    # con la definición integral (SN ≈ (u_θ/u_ax)·r̄/r_punta): el anillo
    # axisimétrico no comparte geometría con el swirler real, así que
    # tan(β) NO transfiere — lo que el PF necesita es el swirl correcto
    # (opción A del plan). Sin SN en el contrato se cae a tan(β).
    sn_t = _f(sw.get("SN"))
    r_bar = 0.5 * (r_lo + r_hi)
    if sn_t > 0.0:
        u_th = sn_t * u_sw * r_hi / max(r_bar, 1e-9)
    else:
        u_th = u_sw * math.tan(beta)

    uz = np.where(sw_mask, u_sw, 0.0) + np.where(ann_mask, u_ann, 0.0)
    ux = np.where(sw_mask, -u_th * np.sin(phig), 0.0)
    uy = np.where(sw_mask, u_th * np.cos(phig), 0.0)
    uz = sign * uz

    # k/ε: anillo con la I/ℓ del contrato; corona del swirler con la
    # turbulencia DEL CHORRO (I = 10 %, ℓ = 0.07·gap) — lección del JICF
    i_ann = _f(bc.get("turbulence_intensity"), 0.05)
    ell_ann = max(_f(bc.get("turbulence_length_scale_mm"), 1.0) * _MM,
                  1e-4)
    k_ann = 1.5 * (i_ann * max(u_ann, 0.1)) ** 2
    e_ann = C_MU ** 0.75 * k_ann ** 1.5 / ell_ann
    gap_sw = max(r_hi - r_lo, m.dx)
    u_sw_tot = math.hypot(u_sw, u_th)
    k_sw = 1.5 * (JET_INTENSITY * max(u_sw_tot, 0.1)) ** 2
    e_sw = C_MU ** 0.75 * k_sw ** 1.5 / (JET_LENGTH_FRAC * gap_sw)
    k_prof = np.where(sw_mask, k_sw, k_ann)
    e_prof = np.where(sw_mask, e_sw, e_ann)

    inlet = dict(kind="inlet", u=(ux, uy, uz), k=k_prof, eps=e_prof, T=T3)
    outlet = dict(kind="outlet", p=0.0)
    bcs = make_bcs(xlo="slip", xhi="slip", ylo="slip", yhi="slip",
                   zlo=inlet if sign > 0 else outlet,
                   zhi=outlet if sign > 0 else inlet)

    u_jet_max = max([_f(p["row"]["U_jet_ms"]) if p.get("row") else 0.0
                     for p in paths.values()] + [u_sw_tot, u_ann, 1.0])
    nu_mol = 1.81e-5 * (T3 / 288.15) ** 0.7 / rho3
    d_l = _f((c.get("derived") or {}).get("D_L_mm"), 50.0) * _MM
    scfg = SolverConfig(
        rho=rho3, nu=nu_mol, u_ref=u_jet_max, l_ref=d_l,
        mach_art=cfg.mach_art, cfl=cfg.cfl, rk=cfg.rk,
        max_iter=cfg.max_iter, timeout_s=cfg.timeout_s, tol=0.0,
        residual_every=cfg.residual_every, backend=cfg.backend,
        precision=cfg.precision)
    # la pared del liner es NO DESLIZANTE: la transpiración ya NO es una
    # condición de contorno de velocidad (no entregaba el caudal), sino la
    # fuente volumétrica de `TranspirationSource`.
    s = IBRANSSolver(m, scfg, bcs)

    ell_glob = max(JET_LENGTH_FRAC * gap_sw, 1e-4)
    tur = KEpsilonRealizable(m, TurbulenceConfig(
        intensity=i_ann, length_scale_m=ell_glob)).attach(s)

    sp = en = None
    if cfg.reactive:
        # bujía del contrato → centro del parche de encendido
        buj = next((r for r in (c.get("hole_rows") or [])
                    if str(r.get("id")) == "bujia"), None)
        patch_r = max(4.0 * m.dx, 5.0 * _MM)
        if buj is not None:
            a0 = math.radians(_f(buj.get("phi0_deg")))
            r_b = _f(buj.get("r_mm")) * _MM - patch_r - m.dx
            z_b = _f(buj.get("z_mm")) * _MM
        else:
            a0 = 0.0
            r_b = 0.5 * (r_in_d + r_out_d)
            z_b = z_dome + sign * 0.5 * _f(
                (c.get("derived") or {}).get("L_pz_mm"), 20.0) * _MM
        center = (r_b * math.cos(a0), r_b * math.sin(a0), z_b)
        plan = IgnitionPlan(patch_center_m=center, patch_radius_m=patch_r,
                            n_hold=cfg.n_hold, n_ramp=cfg.n_ramp,
                            n_stable=cfg.n_stable)
        ccfg = ChemConfig(fuel=fuel, ignition=plan)
        sp = SpeciesTransport(m, ccfg)
        en = EnergyLowMach(m, EnergyConfig(
            p_thermo_Pa=P3, T_inlet_K=T3, T_wall_ib_K=cfg.T_wall_ib_K))
        if cfg.t_ad_K:
            sp.calibrate_cp(cfg.t_ad_K, sp.z_stoichiometric(), T3)
        else:
            # T_ad del propio modelo con la mezcla estequiométrica
            plan.patch_T_K = min(plan.patch_T_K, 0.95 * sp.t_adiabatic(
                sp.z_stoichiometric(), T3))
        sp.attach(s, turbulence=tur, energy=en, ignition=cfg.ignition)
        # techo de T del caso: attach lo calibró con el Y_FUEL INICIAL
        # (aire ⇒ respaldo 0.02 ⇒ techo ~1100 K, que ESTRANGULA la llama —
        # medido: no anclaba). En difusión el techo correcto es el de la
        # mezcla estequiométrica.
        t_ad_case = float(cfg.t_ad_K or sp.t_adiabatic(
            sp.z_stoichiometric(), T3))
        en.cfg.T_ceiling_K = float(plan.T_max_factor * t_ad_case)
        en.cfg.T_floor_K = max(250.0, T3 + float(plan.T_min_offset_K))
        # región de inyección: toro en el anillo de inyectores (o esfera en
        # el eje para `can`), DENTRO de la zona primaria. El vaporizador
        # real devuelve el combustible hacia el domo (codo de 180°): la
        # salida del tubo (aguas abajo de la PZ en el KJ66) NO es donde el
        # fuel entra a la zona de llama — medido: con la fuente en la
        # salida del tubo la bujía no veía combustible y no encendía.
        l_pz = _f((c.get("derived") or {}).get("L_pz_mm"), 20.0) * _MM
        zeta_tube = sign * (_f(inj.get("tube_z1_mm")) * _MM - z_dome) \
            if inj.get("tube_z1_mm") else 0.3 * l_pz
        zeta_src = min(max(zeta_tube, 3.0 * m.dx),
                       max(0.5 * l_pz, 3.0 * m.dx))
        z_src = z_dome + sign * zeta_src
        r_src = max(cfg.fuel_src_r_mm * _MM, 2.0 * m.dx)
        Xi = [m.xc[d][m.interior[d]] for d in range(3)]
        xi, yi, zi = np.meshgrid(Xi[0], Xi[1], Xi[2], indexing="ij")
        ri = np.sqrt(xi ** 2 + yi ** 2)
        fl_i = m.fluid[m.interior]
        mask = None
        for grow in range(3):                      # bucle ACOTADO
            rs = r_src * (1.0 + grow)
            if ring_r > 1e-4:
                d2 = (ri - ring_r) ** 2 + (zi - z_src) ** 2
            else:
                d2 = ri ** 2 + (zi - z_src) ** 2
            mask = (d2 <= rs ** 2) & fl_i
            if np.count_nonzero(mask) >= 8:
                break
        if mdot_fuel > 0.0:
            sp.set_fuel_source(mask, mdot_fuel)
        diag["fuel_src_cells"] = int(np.count_nonzero(mask))
        diag["ignition_center_m"] = list(center)

    zones_eff = (c.get("effusion") or {}).get("zones") or []
    z_eff_m = (sorted((_f(zones_eff[0].get("z0_mm")) * _MM,
                       _f(zones_eff[0].get("z1_mm")) * _MM))
               if zones_eff else None)
    # transpiración: fuente volumétrica de masa+momento en las celdas
    # FLUIDAS contiguas a la pared (se registra la ÚLTIMA, para que su
    # mezcla hacia T3/aire actúe sobre los campos ya actualizados)
    trs_rows = _transpiration_rows(s, meta, paths, rho3, diag, z_eff_m)
    trs = None
    if trs_rows:
        trs = TranspirationSource(m, rho3).attach(
            s, trs_rows, energy=en, species=sp, T_in=T3)
    s.apply_bc()
    s.m4 = dict(
        z_effusion_m=z_eff_m, n_settle=int(cfg.n_settle),
        meta={k: v for k, v in meta.items() if k != "_interp"},
        interp=meta["_interp"], paths=paths, diag=diag,
        axis_z=axis_z, flow_sign=sign, z_dome_m=z_dome,
        t_liner_m=t_w, P3_Pa=P3, T3_K=T3, rho3_kgm3=rho3,
        mdot_air_kgs=mdot_air, mdot_fuel_kgs=mdot_fuel, fuel=fuel,
        u_inlet=dict(u_sw_ms=u_sw, u_theta_ms=u_th, u_ann_ms=u_ann,
                     beta_deg=math.degrees(beta),
                     r_sw_mm=[r_lo / _MM, r_hi / _MM]),
        SN_target=_f(sw.get("SN")),
        L_zones_mm=dict(
            pz=_f((c.get("derived") or {}).get("L_pz_mm")),
            sz=_f((c.get("derived") or {}).get("L_sz_mm")),
            dz=_f((c.get("derived") or {}).get("L_dz_mm")),
            tot=_f((c.get("derived") or {}).get("L_tot_mm"))),
        models=dict(turbulence=tur, species=sp, energy=en,
                    transpiration=trs))
    return s


def _np64(a) -> np.ndarray:
    return np.asarray(_to_numpy(a), dtype=np.float64)


def _rho_field(s) -> np.ndarray:
    en = (s.m4["models"] or {}).get("energy")
    if en is not None and en.rho is not None:
        return _np64(en.rho[s.mesh.interior])
    return np.full(s.mesh.shape, float(s.cfg.rho))


def _interior_coords(s):
    m = s.mesh
    Xi = [m.xc[d][m.interior[d]] for d in range(3)]
    return np.meshgrid(Xi[0], Xi[1], Xi[2], indexing="ij")


def measure_splits(s) -> dict:
    """Splits de aire **MEDIDOS** en el campo (fracciones del ṁ del
    contrato), por **balance de volumen de control axial**: el caudal que
    aporta cada fila es el SALTO del flujo axial del liner entre un plano
    aguas arriba y otro aguas abajo de su ventana (menos la efusión de esa
    franja, que se resta con su valor impuesto — declarado). El flujo
    radial local cerca de la pared NO es medible fila a fila en la PZ con
    swirl (la recirculación lo contamina; medido: −41 % espurio en pz00
    con la corteza radial, con el cierre global a −2 %).

    * swirler: flujo axial por el liner antes de la primera fila.
    * efusión: residuo plano4 − swirler − Σ filas (método declarado).
    Diagnóstico en NumPy float64 (nunca en el bucle caliente)."""
    m = s.mesh
    m4 = s.m4
    itp = m4["interp"]
    sign = m4["flow_sign"]
    dx = m.dx
    rho = _rho_field(s)
    uzf = _np64(s.u[2][m.interior])
    xi, yi, zi = _interior_coords(s)
    ri = np.sqrt(xi ** 2 + yi ** 2)
    fl = m.fluid[m.interior]
    rout = np.interp(zi, itp["z"], itp["r_out"])
    rin = np.interp(zi, itp["z"], itp["r_in"])
    in_liner = fl & (ri < rout) & (ri > rin - 0.5 * dx)
    flux_ax = rho * sign * uzf * dx * dx

    def _plane(k: int) -> float:
        k = int(np.clip(k, 1, m.shape[2] - 2))
        return float(np.sum(np.where(in_liner[:, :, k], flux_ax[:, :, k],
                                     0.0)))

    def _k_of_z(z_m: float) -> int:
        return int(round((z_m - m.lo[2]) / dx - 0.5))

    # entrada (toda la cara) y plano 4 (liner)
    k0 = 1 if sign > 0 else m.shape[2] - 2
    k4 = m.shape[2] - 2 if sign > 0 else 1
    mdot_in = float(np.sum(np.where(fl[:, :, k0], flux_ax[:, :, k0], 0.0)))
    mdot_4 = _plane(k4)

    rows = sorted([p for p in m4["paths"].values() if p.get("row")],
                  key=lambda p: sign * (p["row"]["z_m"] - m4["z_dome_m"]))
    zeta_first = min([sign * (p["row"]["z_m"] - m4["z_dome_m"])
                      for p in rows]
                     + [m4["meta"]["L_tot_mm"] * _MM])
    z_sw = m4["z_dome_m"] + sign * 0.5 * zeta_first
    mdot_sw = _plane(_k_of_z(z_sw))

    # efusión por franja (se resta IMPUESTA en cada ventana de fila)
    z_eff = m4.get("z_effusion_m")
    mdot_eff_t = _f((m4["paths"].get("effusion") or {}).get("mdot_kgs"))
    len_eff = (z_eff[1] - z_eff[0]) if z_eff else 0.0

    def _eff_in(z_a: float, z_b: float) -> float:
        if not z_eff or len_eff <= 0.0 or mdot_eff_t <= 0.0:
            return 0.0
        lo, hi = sorted((z_a, z_b))
        seg = max(0.0, min(hi, z_eff[1]) - max(lo, z_eff[0]))
        return mdot_eff_t * seg / len_eff

    out: dict[str, float] = {}
    for p in rows:
        row = p["row"]
        half = 0.5 * row["d_m"] + 2.0 * dx
        z_up = row["z_m"] - sign * half        # aguas arriba del recorrido
        z_dn = row["z_m"] + sign * half        # aguas abajo
        # Φ ya está orientado con el recorrido (lleva sign·u_z): el aporte
        # de la fila es el salto aguas abajo − aguas arriba, sin más signos
        d_flux = _plane(_k_of_z(z_dn)) - _plane(_k_of_z(z_up))
        out[row["id"]] = d_flux - _eff_in(z_up, z_dn)
    out["swirler"] = mdot_sw
    if "effusion" in m4["paths"]:
        out["effusion"] = mdot_4 - mdot_sw - sum(
            v for k, v in out.items()
            if k not in ("swirler", "effusion"))
    # referencia: el ṁ de aire del CONTRATO (en modo sin anillo, parte del
    # aire entra por transpiración y no cruza la cara de entrada)
    denom = max(abs(m4["mdot_air_kgs"]), 1e-12)
    fracs = {k: v / denom for k, v in out.items()}
    target = {k: p["frac"] for k, p in m4["paths"].items()}
    return dict(splits_measured=fracs,
                mdot_paths_measured_kgs=out,
                mdot_in_measured_kgs=mdot_in,
                mdot_plane4_kgs=mdot_4,
                mass_closure_rel=(mdot_4 - denom) / denom,
                splits_target_CdA=target,
                rows_discretization=dict(
                    resolved=[p["row"]["id"] for p in m4["paths"].values()
                              if p["kind"] == "resolved"],
                    transpired=[p["row"]["id"] for p in m4["paths"].values()
                                if p["kind"] == "transpired"
                                and p.get("row")],
                    rule=m4["meta"]["rule"]))


def swirl_number_measured(s, n_plane: int = 2) -> dict:
    """Número de swirl medido en un plano del liner aguas abajo del domo:
    SN = ∫ρ u_ζ u_θ r dA / (R_tip·∫ρ u_ζ² dA) — la comprobación de cordura
    barata del plan (opción A debe reproducir `swirler.SN`)."""
    m = s.mesh
    m4 = s.m4
    sign = m4["flow_sign"]
    k = n_plane if sign > 0 else m.shape[2] - 1 - n_plane
    rho = _rho_field(s)[:, :, k]
    ux = _np64(s.u[0][m.interior])[:, :, k]
    uy = _np64(s.u[1][m.interior])[:, :, k]
    uz = _np64(s.u[2][m.interior])[:, :, k] * sign
    xi, yi, _ = _interior_coords(s)
    xi, yi = xi[:, :, k], yi[:, :, k]
    ri = np.sqrt(xi ** 2 + yi ** 2)
    itp = m4["interp"]
    zc = m.lo[2] + (k + 0.5) * m.dx
    rout = float(np.interp(zc, itp["z"], itp["r_out"]))
    fl = m.fluid[m.interior][:, :, k] & (ri < rout)
    u_th = (-yi * ux + xi * uy) / np.maximum(ri, 1e-12)
    g_ang = float(np.sum(np.where(fl, rho * uz * u_th * ri, 0.0)))
    g_ax = float(np.sum(np.where(fl, rho * uz * uz, 0.0)))
    # R de referencia: la PUNTA del swirler (la misma del SN de θ[5]),
    # no el radio del liner — comparar contra `swirler.SN` exige la
    # misma definición.
    r_tip = m4["u_inlet"]["r_sw_mm"][1] * _MM
    sn = g_ang / max(r_tip * g_ax, 1e-12)
    return dict(SN_measured=sn, SN_target=m4.get("SN_target"),
                plane_z_mm=zc / _MM, r_tip_mm=r_tip / _MM)


def run_case(s, callback=None) -> dict:
    """Driver del caso M4: corre con el criterio del plan — parada por
    `anchored()` (si hay llama), `max_iter`, `t_end` o `timeout`. El
    residual NO es el criterio (los casos turbulentos lo estancan en
    1e-4–5e-3, ciclo límite del acoplamiento segregado — medido)."""
    sp = (s.m4["models"] or {}).get("species")
    plan = sp.cfg.ignition if sp is not None else None
    settle = int(s.m4.get("n_settle", 0))

    def _cb(sv):
        if callback is not None and callback(sv) is False:
            return False
        if sp is not None and plan is not None \
                and sv.it > plan.n_hold + plan.n_ramp + settle \
                and sp.anchored():
            return False
        return True

    info = s.run(callback=_cb)
    if sp is not None:
        info["anchored"] = bool(sp.anchored())
        info["heat_release_W"] = float(sp.heat_release())
        info["fuel_injected_kg"] = float(sp.fuel_injected_kg)
        n = int(plan.n_stable) if plan is not None else 400
        w = np.asarray(sp.hrr_history[-n:], dtype=float)
        if w.size >= 2 and np.isfinite(w).all() and abs(w.mean()) > 0.0:
            info["hrr_window_mean_W"] = float(w.mean())
            info["hrr_window_osc_rel"] = float(
                np.max(np.abs(w - w.mean())) / abs(w.mean()))
    return info
