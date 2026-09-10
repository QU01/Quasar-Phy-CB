#!/usr/bin/env python3
"""
QUASAR Phy-CB · cfd/tests/test_ibrans_v1_v3.py
==============================================
Escalera de **verificación** V1–V3 del solver L2a nativo (Ciencia §13.1,
WAVE6 §1). Verifica —«¿resolvemos bien las ecuaciones?»— contra soluciones
analíticas y datos de referencia publicados; NO valida contra máquinas
(eso es `validation/`).

    V1  Poiseuille / Couette laminar ....... error < 2 % vs analítico
    V2  Cavidad de tapa deslizante ......... perfiles ±5 % vs Ghia (1982)
    V3  Cilindro en flujo cruzado con IB ... Cd ±10 %, Strouhal ±10 %

Qué corre en CI y qué NO (regla de la casa: un test que no se puede correr
se SALTA con motivo visible, jamás se convierte en un check trivial)
---------------------------------------------------------------------------
* **Siempre**: V1 (dos casos), V2 a Re = 100 con malla mini 32², V3 con
  malla mini — donde el **Strouhal** sí cumple ±10 % pero el **Cd NO**
  (a D/5 el IB no resuelve la capa límite: medido Cd ≈ 1.86 frente a
  1.35). Por eso el check de Cd de la malla mini es un check de *cordura*
  (0.8 < Cd < 3.0) declarado como tal, y el de ±10 % vive en el test lento.
* **Sólo con `PHYCB_SLOW=1`** (o `--slow`): V2 a Re = 1000 en 128² y V3 a
  D/10 en dominio 24D×16D. Ambos tienen además **script manual**:
  `run_v2_re1000.py` y `run_v3_cylinder.py`. Sus números medidos están en
  `cfd/ibrans/ESTADO.md`.

Interpretación de «±5 %» en V2 (declarada, no negociada a posteriori)
---------------------------------------------------------------------
El error se mide **en unidades de la velocidad de la tapa** (U = 1), es
decir `max|u_num − u_Ghia| ≤ 0.05`. Un error relativo punto a punto no
tiene sentido en un perfil que cruza cero. Se excluyen los dos puntos
extremos de las tablas de Ghia (y = 0 y y = 1) porque son valores de
contorno impuestos, no resultados, y la malla de centros de celda no llega
a ellos.

Backend: CPU o GPU, MISMOS umbrales
-----------------------------------
El solver tiene backend CuPy ademas de NumPy. Los casos de esta escalera
son pequenos (384-38 400 celdas), muy por debajo del cruce CPU/GPU medido
(`backend.GPU_MIN_CELLS` = 100 000 celdas), asi que con `auto` corren en
**CPU aunque haya GPU**: por debajo del cruce el paso lo domina el coste de
lanzar kernels y la GPU es mas lenta. Para comprobar que la escalera pasa
**tambien en GPU**, con los mismos umbrales y sin aflojar nada::

    PHYCB_L2A_BACKEND=cupy python cfd/tests/test_ibrans_v1_v3.py

Los resultados de esa corrida forzada estan en `cfd/ibrans/ESTADO.md` 5.4.
Los errores GPU y CPU no coinciden en el ultimo digito (las reducciones no
son bit-identicas entre backends); lo que se comprueba es que los dos caen
igual de dentro del umbral.

Doble modo::

    python cfd/tests/test_ibrans_v1_v3.py           # tabla ✓/✗/–
    python cfd/tests/test_ibrans_v1_v3.py --slow    # + los caros
    python -m pytest cfd/tests/test_ibrans_v1_v3.py -q
"""

from __future__ import annotations

import os
import sys
import time

# --- contrato §0.b: Phy-AC está instalado como paquete EDITABLE y ensombrece
# --- los módulos de Phy-CB desde cualquier cwd. Esto va SIEMPRE primero.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np                                            # noqa: E402

import physics_core as pc                                     # noqa: E402

pc.assert_provenance(pc)          # ImportError ruidoso si es el de Phy-AC

from cfd.ibrans import backend as _bk                         # noqa: E402
from cfd.ibrans import post                                   # noqa: E402
from cfd.ibrans.mesh import (CartesianMesh, sdf_cylinder,     # noqa: E402
                             sdf_slab)
from cfd.ibrans.solver import (IBRANSSolver, SolverConfig,    # noqa: E402
                               make_bcs)

SLOW = bool(os.environ.get("PHYCB_SLOW")) or "--slow" in sys.argv


# ==========================================================================
# Datos de referencia (con su cita — contrato §7: nada hard-coded sin fuente)
# ==========================================================================
#: Ghia, U., Ghia, K.N. & Shin, C.T. (1982) «High-Re solutions for
#: incompressible flow using the Navier-Stokes equations and a multigrid
#: method», J. Comput. Phys. 48(3), 387–411.
#: Tabla I — u en la vertical x = 0.5. Tabla II — v en la horizontal y = 0.5.
GHIA_Y = (1.0000, 0.9766, 0.9688, 0.9609, 0.9531, 0.8516, 0.7344, 0.6172,
          0.5000, 0.4531, 0.2813, 0.1719, 0.1016, 0.0703, 0.0625, 0.0547,
          0.0000)
GHIA_U = {
    100: (1.00000, 0.84123, 0.78871, 0.73722, 0.68717, 0.23151, 0.00332,
          -0.13641, -0.20581, -0.21090, -0.15662, -0.10150, -0.06434,
          -0.04775, -0.04192, -0.03717, 0.00000),
    1000: (1.00000, 0.65928, 0.57492, 0.51117, 0.46604, 0.33304, 0.18719,
           0.05702, -0.06080, -0.10648, -0.27805, -0.38289, -0.29730,
           -0.22220, -0.20196, -0.18109, 0.00000),
}
GHIA_X = (1.0000, 0.9688, 0.9609, 0.9531, 0.9453, 0.9063, 0.8594, 0.8047,
          0.5000, 0.2344, 0.2266, 0.1563, 0.0938, 0.0781, 0.0703, 0.0625,
          0.0000)
GHIA_V = {
    100: (0.00000, -0.05906, -0.07391, -0.08864, -0.10313, -0.16914,
          -0.22445, -0.24533, 0.05454, 0.17527, 0.17507, 0.16077, 0.12317,
          0.10890, 0.10091, 0.09233, 0.00000),
    1000: (0.00000, -0.21388, -0.27669, -0.33714, -0.39188, -0.51550,
           -0.42665, -0.31966, 0.02526, 0.32235, 0.33075, 0.37095, 0.32627,
           0.30353, 0.29012, 0.27485, 0.00000),
}

#: Cilindro circular, Re = 100 (laminar, desprendimiento periódico 2S).
#: Cd: Braza, Chassaing & Ha Minh (1986) JFM 165, 79–130 → 1.364;
#:     Liu, Zheng & Sung (1998) J. Comput. Phys. 139, 35–57 → 1.35.
#: St: Williamson (1989) JFM 206, 579–627 (experimental) → 0.164–0.166.
CYL_CD_REF = 1.35
CYL_ST_REF = 0.165

TOL_V1 = 0.02          # 2 % del valor de escala (Ciencia §13.1)
TOL_V2 = 0.05          # 5 % de U_tapa
TOL_V3 = 0.10          # 10 % (Cd y St)


# ==========================================================================
# V1 · Poiseuille y Couette laminares con paredes INMERSAS
# ==========================================================================
def case_channel(dx: float = 0.025, height: float = 1.0, nu: float = 0.1,
                 wall_offset_dx: float = 0.25, nx: int = 8,
                 drive: str = "force", u_wall: float = 1.0,
                 force: float = 1.0):
    """Canal plano con las DOS paredes inmersas y desplazadas
    `wall_offset_dx·dx` respecto de las caras de celda — a propósito: con
    las paredes alineadas el IB no se estaría probando.

    `drive="force"` ⇒ Poiseuille por fuerza volumétrica (p uniforme, la
    solución analítica es exacta). `drive="lid"` ⇒ Couette con la pared
    superior inmersa moviéndose a `u_wall`.
    """
    y0 = 0.10 + wall_offset_dx * dx
    y1 = y0 + height
    ly = y1 + 0.10
    m = CartesianMesh([0.0, 0.0], [nx * dx, ly], dx)
    m.set_sdf(sdf_slab(1, y0, y1))
    ymid = 0.5 * (y0 + y1)

    def wall_u(mm):
        xy = mm.ghost_coords()
        if xy.size == 0:
            return [0.0, 0.0]
        return [np.where(xy[:, 1] > ymid, u_wall, 0.0), 0.0]

    if drive == "force":
        u_max = force * height ** 2 / (8.0 * nu)
        cfg = SolverConfig(rho=1.0, nu=nu, u_ref=u_max, l_ref=height,
                           mach_art=0.1, cfl=0.4, rk="rk3",
                           body_force=(force, 0.0), max_iter=60_000,
                           t_end=8.0, tol=3.0e-4, residual_every=100)
    else:
        u_max = u_wall
        cfg = SolverConfig(rho=1.0, nu=nu, u_ref=u_max, l_ref=height,
                           mach_art=0.1, cfl=0.4, rk="rk3",
                           ib_wall_u=wall_u, max_iter=60_000, t_end=8.0,
                           tol=3.0e-4, residual_every=100)
    bcs = make_bcs(xlo="periodic", xhi="periodic", ylo="wall", yhi="wall")
    return IBRANSSolver(m, cfg, bcs), dict(y0=y0, y1=y1, u_max=u_max,
                                           height=height, nu=nu, force=force,
                                           u_wall=u_wall, drive=drive)


def run_v1(drive: str = "force", **kw) -> dict:
    s, ex = case_channel(drive=drive, **kw)
    info = s.run()
    u = s.field(0)
    yc = s.coords()[1]
    i = u.shape[0] // 2
    prof = u[i, :]
    mask = s.mesh.fluid[s.mesh.interior][i, :]
    yy, uu = yc[mask], prof[mask]
    if drive == "force":
        exact = (ex["force"] / (2.0 * ex["nu"])) * (yy - ex["y0"]) \
            * (ex["y1"] - yy)
    else:
        exact = ex["u_wall"] * (yy - ex["y0"]) / ex["height"]
    err = float(np.max(np.abs(uu - exact)) / ex["u_max"])
    l2 = float(np.sqrt(np.mean((uu - exact) ** 2)) / ex["u_max"])
    return dict(case=f"V1 {drive}", err_max=err, err_l2=l2,
                backend=info["backend"]["backend"],
                dtype=info["backend"]["dtype"],
                status=info["status"], iterations=info["iterations"],
                wall_time_s=info["wall_time_s"],
                s_per_cell_step=info["s_per_cell_step"],
                cells=info["cells"], ghost=s.mesh.n_ghost,
                orphan=s.mesh.n_orphan, tol=TOL_V1, ok=err < TOL_V1)


# ==========================================================================
# V2 · Cavidad de tapa deslizante (Ghia)
# ==========================================================================
def case_cavity(n: int = 32, re: float = 100.0, cfl: float = 0.5,
                tol: float = 1.0e-5, max_iter: int = 400_000):
    """Cavidad unitaria, tapa superior a U = 1. Paredes ALINEADAS con la
    malla (sin IB): aquí lo que se verifica es el acoplamiento p–u, no el
    IB — eso ya lo hicieron V1 y lo hará V3."""
    dx = 1.0 / int(n)
    m = CartesianMesh([0.0, 0.0], [1.0, 1.0], dx)
    m.set_sdf(None)
    cfg = SolverConfig(rho=1.0, nu=1.0 / float(re), u_ref=1.0, l_ref=1.0,
                       mach_art=0.1, cfl=cfl, rk="rk3", max_iter=max_iter,
                       t_end=float("inf"), tol=tol, residual_every=200)
    bcs = make_bcs(xlo="wall", xhi="wall", ylo="wall",
                   yhi=dict(kind="wall", u=(1.0, 0.0)))
    return IBRANSSolver(m, cfg, bcs)


def run_v2(n: int = 32, re: int = 100, **kw) -> dict:
    s = case_cavity(n=n, re=float(re), **kw)
    info = s.run()
    n = s.mesh.shape[0]
    u, v = s.field(0), s.field(1)
    xc, yc = s.coords()
    i = n // 2
    up = 0.5 * (u[i - 1, :] + u[i, :]) if n % 2 == 0 else u[i, :]
    vp = 0.5 * (v[:, i - 1] + v[:, i]) if n % 2 == 0 else v[:, i]
    un = np.interp(GHIA_Y, yc, up)
    vn = np.interp(GHIA_X, xc, vp)
    # se excluyen los extremos (y=0, y=1): son contorno impuesto
    eu = float(np.max(np.abs(un - np.asarray(GHIA_U[re]))[1:-1]))
    ev = float(np.max(np.abs(vn - np.asarray(GHIA_V[re]))[1:-1]))
    return dict(case=f"V2 cavidad Re={re} n={n}", err_u=eu, err_v=ev,
                backend=info["backend"]["backend"],
                dtype=info["backend"]["dtype"],
                status=info["status"], iterations=info["iterations"],
                wall_time_s=info["wall_time_s"],
                s_per_cell_step=info["s_per_cell_step"],
                cells=info["cells"], res_u=info["res_u"],
                res_div=info["res_div"], tol=TOL_V2,
                ok=max(eu, ev) < TOL_V2)


# ==========================================================================
# V3 · Cilindro en flujo cruzado con frontera inmersa
# ==========================================================================
def case_cylinder(dx: float = 0.1, xlo: float = -8.0, xhi: float = 16.0,
                  yh: float = 8.0, re: float = 100.0, d: float = 1.0,
                  u_inf: float = 1.0, t_end: float = 180.0,
                  cfl: float = 0.5, backend: str = "auto",
                  precision: str = "auto"):
    """Cilindro sólido de diámetro `d` centrado en el origen, con IB puro.

    El desprendimiento se dispara con una perturbación transversal inicial
    ACOTADA (gaussiana en la estela) para no esperar a que la asimetría
    crezca desde el redondeo — determinista, sin números aleatorios.
    """
    m = CartesianMesh([xlo, -yh], [xhi, yh], dx)
    m.set_sdf(sdf_cylinder((0.0, 0.0), 0.5 * d, axis=None, solid=True))
    cfg = SolverConfig(rho=1.0, nu=u_inf * d / float(re), u_ref=u_inf,
                       l_ref=d, mach_art=0.1, cfl=cfl, rk="rk3",
                       max_iter=2_000_000, t_end=float(t_end), tol=0.0,
                       residual_every=500, monitor_every=20,
                       backend=backend, precision=precision)
    bcs = make_bcs(xlo=dict(kind="inlet", u=(u_inf, 0.0)),
                   xhi=dict(kind="outlet", p=0.0), ylo="slip", yhi="slip")
    xg, yg = np.meshgrid(*m.xc, indexing="ij")
    u0 = [np.full(m.shape_tot, u_inf),
          0.25 * u_inf * np.exp(-((xg - 1.0) ** 2 + (yg - 0.3) ** 2) / 0.25)]
    return IBRANSSolver(m, cfg, bcs, u0=u0), dict(d=d, u_inf=u_inf,
                                                  t_end=t_end)


def run_v3(dx: float = 0.1, t_end: float = 180.0, **kw) -> dict:
    s, ex = case_cylinder(dx=dx, t_end=t_end, **kw)
    d, u_inf = ex["d"], ex["u_inf"]
    box = ((-2.0 * d, 4.0 * d), (-2.5 * d, 2.5 * d))
    samples: list = []
    s.add_monitor("v_probe", lambda sv: sv.sample((1.5 * d, 0.0), 1))

    def cb(sv):
        if sv.t > 0.35 * t_end and (sv.it % 20) == 0:
            samples.append(post.control_volume_force(sv, box))
        return True

    info = s.run(callback=cb)
    st = post.strouhal_from_signal(s.history["t_mon"], s.history["v_probe"],
                                   u_inf, d, t_skip_frac=0.5)
    fr = post.force_from_history(samples, u_inf, s.cfg.rho, d)
    yp = post.yplus_histogram(s)
    cd = float(fr.get("Cd", float("nan")))
    stv = float(st.get("St", float("nan")))
    return dict(case=f"V3 cilindro dx=D/{d / dx:.0f}", Cd=cd, St=stv,
                St_zerocross=float(st.get("St_zerocross", float("nan"))),
                Cl=float(fr.get("Cl", float("nan"))),
                err_Cd=abs(cd - CYL_CD_REF) / CYL_CD_REF,
                err_St=abs(stv - CYL_ST_REF) / CYL_ST_REF,
                backend=info["backend"]["backend"],
                dtype=info["backend"]["dtype"],
                status=info["status"], iterations=info["iterations"],
                wall_time_s=info["wall_time_s"],
                s_per_cell_step=info["s_per_cell_step"],
                cells=info["cells"], ghost=s.mesh.n_ghost,
                orphan=s.mesh.n_orphan,
                yplus_max=float(yp.get("yplus_max", float("nan"))),
                amplitude=float(st.get("amplitude", 0.0)), tol=TOL_V3)


# ==========================================================================
# Tests (pytest)
# ==========================================================================
def test_v1_poiseuille():
    """V1a · Poiseuille por fuerza volumétrica, paredes inmersas."""
    r = run_v1("force")
    assert r["err_max"] < TOL_V1, r
    assert r["orphan"] == 0, r


def test_v1_couette():
    """V1b · Couette con la pared superior INMERSA en movimiento."""
    r = run_v1("lid")
    assert r["err_max"] < TOL_V1, r


def test_v2_cavity_re100():
    """V2a · Cavidad Re = 100 en malla mini 32² vs Ghia (1982) Tablas I/II."""
    r = run_v2(n=32, re=100)
    assert r["status"] == "converged", r
    assert max(r["err_u"], r["err_v"]) < TOL_V2, r


def test_v2_cavity_re1000():
    """V2b · Cavidad Re = 1000 en 128². **LENTO** (~10 min): sólo con
    `PHYCB_SLOW=1`. Script manual: `cfd/tests/run_v2_re1000.py`."""
    if not SLOW:
        _skip("V2 Re=1000 necesita 128^2 y ~10 min de CPU; corre con "
              "PHYCB_SLOW=1 o el script run_v2_re1000.py")
        return
    r = run_v2(n=128, re=1000, tol=2.0e-5, max_iter=600_000)
    assert max(r["err_u"], r["err_v"]) < TOL_V2, r


def test_v3_cylinder_mini():
    """V3 (malla mini) · IB + fuerza por volumen de control + Strouhal.

    Con dx = D/5 el **Strouhal SÍ** cumple ±10 % (es una frecuencia global,
    poco sensible a la capa límite); el **Cd NO** — a esa resolución el IB
    sobre-predice la fuerza (medido 1.65 frente a 1.35, +22 %). Aquí el Cd
    sólo se comprueba por cordura; el ±10 % lo cumple `test_v3_cylinder_full`
    a D/10 (medido Cd = 1.396, +3.4 %).
    """
    r = run_v3(dx=0.2, xlo=-5.0, xhi=9.0, yh=5.0, t_end=60.0)
    assert r["amplitude"] > 1.0e-3, ("no hay desprendimiento medible", r)
    assert r["err_St"] < TOL_V3, r
    assert 0.8 < r["Cd"] < 3.0, ("Cd fuera de cordura", r)
    assert r["orphan"] == 0, r


def test_v3_cylinder_full():
    """V3 · Cd ±10 % y St ±10 % a dx = D/10 en dominio 24D×16D.
    **LENTO** (~15 min). Script manual: `cfd/tests/run_v3_cylinder.py`."""
    if not SLOW:
        _skip("V3 a D/10 necesita ~38k celdas x 40k pasos (~15 min); corre "
              "con PHYCB_SLOW=1 o el script run_v3_cylinder.py")
        return
    r = run_v3(dx=0.1, xlo=-8.0, xhi=16.0, yh=8.0, t_end=180.0)
    assert r["err_St"] < TOL_V3, r
    assert r["err_Cd"] < TOL_V3, r


def test_ib_wall_distance_is_free():
    """La distancia a pared del IB es |φ| EXACTA (no una aproximación
    numérica): es el argumento estructural de la arquitectura (Ciencia
    §11.1). Se comprueba contra la distancia analítica al cilindro."""
    m = CartesianMesh([-2.0, -2.0], [2.0, 2.0], 0.05)
    m.set_sdf(sdf_cylinder((0.0, 0.0), 0.5, axis=None, solid=True))
    xg, yg = np.meshgrid(*m.xc, indexing="ij")
    exact = np.abs(np.sqrt(xg ** 2 + yg ** 2) - 0.5)
    assert float(np.max(np.abs(m.wall_distance - exact))) < 1e-12
    # normales unitarias y apuntando al fluido
    nn = np.sqrt(sum(n ** 2 for n in m.normals))
    band = m.wall_distance < 0.3
    assert float(np.max(np.abs(nn[band] - 1.0))) < 1e-6
    assert m.n_ghost > 0 and m.n_orphan == 0


def test_l2a_degrades_labeled():
    """Contrato §3: `--fidelity L2A` NO puede fallar ni mentir. Con M4
    hecho, `solve()` PRODUCE o degrada ETIQUETADO — nunca lanza y nunca
    miente. Este test (que en M1–M3 aseguraba que degradaba) asegura ahora
    las dos ramas con un presupuesto mínimo: un caso de 40 iteraciones no
    puede dar producto en falso «converged», pero la cadena entera
    (contrato → SDF completo → caso → producto/None) tiene que
    funcionar y la etiqueta tiene que decir la verdad."""
    import cfd.ibrans as ib
    # presupuesto mínimo: ejercita la cadena, no la física
    sol = ib.solve(pc.THETA_REF, {}, dx_mm=3.0, timeout_s=90.0,
                   max_iter=40, reactive=False)
    if sol is None:
        assert ib.DEGRADE_TAG, "degradó sin decir por qué"
        assert ib.LAST_STATUS["status"].startswith("unavailable"), \
            ib.LAST_STATUS
    else:
        assert "cfd_mesh" in sol and "honesty" in sol, sorted(sol)
        assert ib.LAST_STATUS["status"] in ("product", "mixed_is_burned")
    # la vía physics_core: el record queda etiquetado con la verdad
    old = (ib.L2A_MAX_ITER, ib.L2A_TIMEOUT_S, ib.L2A_DX_MM)
    ib.L2A_MAX_ITER, ib.L2A_TIMEOUT_S, ib.L2A_DX_MM = 40, 90.0, 3.0
    try:
        rec = pc.evaluate(pc.THETA_REF, pc.Fidelity.L2A, use_cache=False)
    finally:
        ib.L2A_MAX_ITER, ib.L2A_TIMEOUT_S, ib.L2A_DX_MM = old
    assert rec["source"] == "ibrans_L2A" \
        or rec["source"].startswith("ibrans_L2A("), rec["source"]
    assert len(rec["g"]) == pc.N_CONSTRAINTS
    assert isinstance(rec["feasible"], bool)


def test_backend_selection_and_gpu_parity():
    """M0d · El backend se **elige, se etiqueta y nunca lanza**; y si hay
    GPU, el MISMO caso da el MISMO campo en CPU y en GPU.

    Tres cosas, en este orden:

    (a) Pedir un backend ausente o inventado degrada a NumPy con motivo
        legible (contrato §10.4: nunca una excepción por dependencia).
    (b) Con `auto` y una malla pequeña se elige CPU aunque haya GPU: por
        debajo de `backend.GPU_MIN_CELLS` la GPU es más lenta (tabla medida
        en `ESTADO.md` §5.2). Es una regla, no un accidente.
    (c) **Paridad CPU↔GPU**: 25 pasos del cilindro con IB en los dos
        backends. No se exige bit-identidad —las reducciones de CuPy
        acumulan en otro orden y eso está DECLARADO— sino que la diferencia
        sea de redondeo (< 1e-8 relativo), no de modelo. Es la prueba de
        que el port a GPU no cambió la física.
    """
    # (a) degradación etiquetada, sin excepciones
    for nm in ("warp", "jax", "backend_que_no_existe"):
        b = _bk.get_backend(nm)
        assert b.name == "numpy", (nm, b)
        assert b.note, (nm, "degradó sin decir por qué")
        assert b.is_gpu is False

    # (b) regla de tamaño
    s_small, _ = case_cylinder(dx=0.4, xlo=-3.0, xhi=5.0, yh=3.0)
    assert s_small.mesh.ncell < _bk.GPU_MIN_CELLS
    if not os.environ.get(_bk.ENV_BACKEND):
        assert s_small.be.name == "numpy", s_small.be.summary()

    # (c) paridad CPU/GPU
    if not _bk.available_backends().get("cupy"):
        _skip("no hay CuPy/GPU en esta maquina: la paridad CPU-GPU no se "
              "puede comprobar (y el solver corre en NumPy, que es el "
              "camino verificado por V1-V3)")
        return
    fields = {}
    for bk_name in ("numpy", "cupy"):
        s, _ = case_cylinder(dx=0.25, xlo=-3.0, xhi=5.0, yh=3.0,
                             backend=bk_name, precision="fp64")
        for _ in range(25):
            s.step()
        fields[bk_name] = (s.field(0), s.field(1), s.pressure(),
                           s.be.name, getattr(s, "_sync_calls", 0))
    assert fields["cupy"][3] == "cupy", ("no se pudo forzar la GPU",
                                         fields["cupy"][3])
    for k, name in enumerate(("u", "v", "p")):
        a, b = fields["numpy"][k], fields["cupy"][k]
        scale = max(float(np.max(np.abs(a))), 1e-12)
        diff = float(np.max(np.abs(a - b))) / scale
        assert diff < 1.0e-8, (f"{name}: CPU y GPU difieren {diff:.2e} "
                               "(redondeo esperado < 1e-8)")
    # el bucle no debe copiar campos: ~1 transferencia por paso (el u_max
    # del CFL) mas los residuales; si esto se dispara, hay una copia oculta
    d2h = fields["cupy"][4] / 25.0
    assert d2h < 3.0, (f"{d2h:.2f} transferencias device->host por paso: "
                       "el bucle caliente esta copiando campos")


def test_turbulence_and_chem_declare_absence():
    """M2 y M3 están IMPLEMENTADOS (`available()` es True en ambos — sus
    verificaciones viven en `test_ibrans_v4_v5.py` y `test_ibrans_v6_v8.py`)
    y las funciones puras dan números correctos. El producto L2a (M4) tiene
    su propia escalera en `test_ibrans_v9.py`."""
    from cfd.ibrans import chem, turbulence
    assert turbulence.available() is True
    assert chem.available() is True
    nu = 1.5e-5
    # (a) la rama logarítmica resuelve EXACTAMENTE u+ = ln(E·y+)/kappa
    ut = float(turbulence.u_tau_log_law(30.0, 5.0e-3, nu))
    yp = float(turbulence.yplus(ut, 5.0e-3, nu))
    assert yp > turbulence.YPLUS_LAM
    assert abs(30.0 / ut
               - np.log(turbulence.E_WALL * yp) / turbulence.KAPPA) < 1e-6
    # (b) Spalding invierte su propia ley con residuo de máquina, en las
    #     tres bandas (log, buffer y subcapa viscosa)
    for u_p, y_w in ((30.0, 5.0e-3), (1.0, 2.0e-4), (0.05, 1.0e-4)):
        us = float(turbulence.u_tau_spalding(u_p, y_w, nu))
        ypp = float(turbulence.yplus(us, y_w, nu))
        assert abs(float(turbulence.spalding_yplus(u_p / us)) - ypp) < 1e-9
    # (c) en la subcapa viscosa Spalding degenera a u+ = y+ (el residuo que
    #     queda, ~2e-5, es el término exponencial de la propia ley)
    us = float(turbulence.u_tau_spalding(0.05, 1.0e-4, nu))
    assert abs(0.05 / us - float(turbulence.yplus(us, 1.0e-4, nu))) < 5e-4
    # (d) muy dentro de la banda log, Spalding y la ley log coinciden < 3 %
    a = float(turbulence.u_tau_spalding(30.0, 5.0e-3, nu))
    b = float(turbulence.u_tau_log_law(30.0, 5.0e-3, nu))
    assert abs(a / b - 1.0) < 0.03
    # EDM: la tasa se anula sin productos (por eso un campo frío no enciende)
    r0 = float(chem.edm_rate(1.0, 1.0, 100.0, 0.05, 0.2, 0.0, 3.4))
    r1 = float(chem.edm_rate(1.0, 1.0, 100.0, 0.05, 0.2, 0.5, 3.4))
    assert r0 == 0.0 and r1 > 0.0
    assert chem.global_mechanism("h2")["n_steps"] == 0     # H2 excluido


# ==========================================================================
# Doble modo (tabla propia + pytest)
# ==========================================================================
class _Skip(Exception):
    pass


def _skip(reason: str):
    try:
        import pytest
        pytest.skip(reason)
    except ImportError:
        raise _Skip(reason)


_TESTS = [
    ("V1a", "Poiseuille laminar, paredes inmersas", test_v1_poiseuille),
    ("V1b", "Couette, pared inmersa movil", test_v1_couette),
    ("V2a", "Cavidad Ghia Re=100 (32^2)", test_v2_cavity_re100),
    ("V2b", "Cavidad Ghia Re=1000 (128^2) [lento]", test_v2_cavity_re1000),
    ("V3a", "Cilindro IB malla mini (St)", test_v3_cylinder_mini),
    ("V3b", "Cilindro IB D/10 (Cd y St) [lento]", test_v3_cylinder_full),
    ("M0a", "distancia a pared = |phi| exacta", test_ib_wall_distance_is_free),
    ("M0b", "L2A degrada etiquetado (contrato §3)", test_l2a_degrades_labeled),
    ("M0c", "M2 y M3 disponibles (funciones puras)",
     test_turbulence_and_chem_declare_absence),
    ("M0d", "backend etiquetado + paridad CPU/GPU",
     test_backend_selection_and_gpu_parity),
]


def main(argv=None) -> int:
    """Tabla ✓/✗/– con exit code 1 si algo falla (estilo `test_phycb.py`)."""
    argv = list(sys.argv[1:] if argv is None else argv)
    only = None
    for a in argv:
        if a.startswith("--only"):
            only = set(a.split("=", 1)[-1].split(","))
    fails = 0
    _be = _bk.get_backend()
    _rule = ("" if not _be.is_gpu or os.environ.get(_bk.ENV_BACKEND)
             else f" → numpy si celdas < {_bk.GPU_MIN_CELLS}")
    print(f"\nEscalera V&V del solver L2a (WAVE6 M1) · slow={SLOW} · "
          f"backend: {_be.name}/{_be.dtype.__name__}{_rule}\n"
          + "-" * 72)
    for tid, name, fn in _TESTS:
        if only and tid not in only:
            continue
        t0 = time.perf_counter()
        try:
            fn()
            mark, note = "OK  ", ""
        except _Skip as exc:
            mark, note = "SKIP", f"  ({exc})"
        except BaseException as exc:            # incluye pytest.skip.Exception
            if type(exc).__name__ == "Skipped":
                mark, note = "SKIP", f"  ({exc})"
            else:
                mark, note = "FAIL", f"  ({type(exc).__name__}: {exc})"
                fails += 1
        el = time.perf_counter() - t0
        print(f"  [{mark}] {tid:5s} {name:42s} {el:7.1f} s{note}")
    print("-" * 72)
    print(f"  {'TODO EN VERDE' if not fails else str(fails) + ' FALLO(S)'}\n")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
