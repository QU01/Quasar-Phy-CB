"""
QUASAR Phy-CB · cfd/ibrans/mesh.py
==================================
Malla cartesiana uniforme + **frontera inmersa (IB) por campo de distancia
con signo (SDF)**. Es la pieza que hace que este solver no necesite mallado
(WAVE6 §1, Ciencia §11.1):

    campo de voxels SDF de PicoGK (φ)
      → remuestreo a la malla cartesiana del solver (dx = 0.5–2 mm)
      → φ > 0 fluido · φ < 0 sólido · normales n = ∇φ/|∇φ| (analíticas)
      → celdas fantasma (ghost) en la banda sólida adyacente al fluido
      → **distancia a pared = |φ| GRATIS** en cada celda

El último punto es el argumento estructural de toda la arquitectura: en un
solver body-fitted la distancia a pared es un problema caro (Eikonal o
búsqueda de vecino más próximo) y la necesitan las funciones de pared y
todos los modelos de turbulencia. Aquí es una lectura de array.

Convenio de signos (CONGELADO, lo repite `IB_SIGN_NOTE`)
--------------------------------------------------------
    φ > 0  ⇒ FLUIDO      φ < 0  ⇒ SÓLIDO      φ = 0 ⇒ pared
    n = ∇φ/|∇φ| apunta HACIA EL FLUIDO (normal exterior al sólido)

Unidades: **SI internamente** (metros). El contrato de geometría viene en
mm y se convierte en la frontera (`sdf_from_contract`), con el sufijo `_mm`
sólo en las claves de interfaz.

Interpolación fantasma (WAVE6 §1: «interpolación lineal a la imagen»)
---------------------------------------------------------------------
Para cada celda fantasma g (sólida, con fluido a ≤ `NBAND` celdas):

    pie de pared:  x_w  = x_g + |φ_g|·n
    punto imagen:  x_ip = x_w + d_ip·n,   d_ip = max(|φ_g|, IMAGE_MIN_DX·dx)
    Dirichlet:     u_g  = u_w − (|φ_g|/d_ip)·(u_ip − u_w)
    Neumann:       p_g  = p_ip

`u_ip` sale de interpolación multilineal (2^ndim nodos). Los nodos del
estencil que caen en sólido se sustituyen por el valor de pared (Dirichlet)
o se renormalizan sobre los nodos fluidos (Neumann) — el remedio estándar
de Mittal et al. (2008) para que el estencil de la imagen no se contamine.
Empujar la imagen a ≥ 1.5·dx de la pared acota el coeficiente |φ_g|/d_ip
≤ 1 y con ello el esquema es incondicionalmente no amplificante.

CPU y GPU
---------
La **construcción** de la malla (muestreo del SDF, máscaras, normales,
estenciles de los fantasmas) se hace SIEMPRE en NumPy/CPU: ocurre una vez y
usa `argwhere`, `ravel_multi_index` y lógica de índices que no gana nada en
GPU. Lo que sí vive en el dispositivo es lo que el **bucle caliente** toca
en cada subpaso RK: los índices de las celdas fantasma, los nodos de sus
estenciles y sus pesos. `to_backend(be)` sube ese bloque y a partir de ahí
`apply_dirichlet`/`apply_neumann`/`zero_deep_solid` funcionan sobre arrays
de CuPy **sin ninguna transferencia device↔host** (medido: 0 copias por
paso). El despacho es por tipo del array que se les pasa, así que la misma
malla sirve a un solver de CPU y a uno de GPU.

Lo que este módulo **NO** hace
------------------------------
* No lee `.vdb` de OpenVDB: `sample_vdb()` lo intenta con import perezoso y
  **degrada a None con motivo** si no hay binding de Python (es el caso en
  esta máquina). El camino soportado hoy es el SDF **analítico** derivado
  del contrato `phycb-combustor-1`, que es una **aproximación declarada**
  del campo de PicoGK, no el campo mismo.
* No hace cut-cell conservativo (upgrade declarado en WAVE6 §0) ni AMR.

API pública
-----------
    FLUID, GHOST, SOLID, NBAND, IMAGE_MIN_DX, IB_SIGN_NOTE
    CartesianMesh(lo, hi, dx, ng=2)
        .set_sdf(phi) .fluid .solid .ghost .phi .wall_distance .normals
        .apply_dirichlet(field, wall_value)  .apply_neumann(field)
        .to_backend(backend)  .device_note
        .summary()
    Patch, patches_from_contract(contract)
    sdf_slab, sdf_cylinder, sdf_sphere, sdf_box, sdf_union, sdf_intersect,
    sdf_invert, sdf_annulus_profile
    sdf_from_contract(contract) -> (callable, dict)
    sdf_combustor_from_contract(contract, dx_mm) -> (callable, dict)   [M4]
    sample_vdb(path) -> (array|None, str)
    mesh_from_contract(contract, dx_mm=...) -> (CartesianMesh, dict)
    mesh_combustor_from_contract(contract, dx_mm=...) -> (mesh, dict)  [M4]
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

# --- códigos de tipo de celda (congelados) -------------------------------
FLUID, GHOST, SOLID = 0, 1, 2

#: Ancho en celdas de la banda de fantasmas. 2 porque la reconstrucción
#: MUSCL del solver lee hasta 2 celdas más allá de una cara de fluido.
NBAND = 2

#: Distancia mínima de la imagen a la pared, en unidades de dx.
#: ⚠️ TUNABLE CALIBRADO (contrato §10.7). Empujar la imagen lejos de la
#: pared estabiliza el estencil pero linealiza el perfil sobre una
#: distancia mayor de la real, y eso sesga la tensión de pared. Barrido
#: medido el 2026-07-31 sobre V1 (referencia ANALÍTICA) y V3 (D/5):
#:
#:   IMAGE_MIN_DX   err V1 Poiseuille   err V1 Couette   Cd V3 (D/5)
#:        0.05           0.76 %             0.20 %          1.649
#:        0.40           0.33 %             0.09 %          1.649
#:      **0.55**       **0.17 %**         **0.04 %**        1.651
#:        0.75           0.17 %             0.04 %          1.658
#:        1.50           —                  —               1.864
#:
#: Se fija en 0.55 porque MINIMIZA el error contra la solución analítica
#: (único juez exacto disponible) y además cae en la meseta del Cd. No se
#: calibró contra el Cd: eso sería ajustar a la respuesta.
IMAGE_MIN_DX = 0.55

#: Peso fluido mínimo del estencil de la imagen para fiarse de él.
MIN_FLUID_WEIGHT = 0.10

#: Espesor MÍNIMO de la pared del liner del caso M4, en celdas. Una pared
#: más fina que ~2 celdas no genera sólido en un IB de celda centrada y
#: queda porosa (medido en el humo M4 del 2026-08-02: el anillo se vaciaba
#: al liner a través de la pared «fantasma»). El espesor real del contrato
#: se conserva si ya es mayor.
WALL_MIN_CELLS = 2.0

IB_SIGN_NOTE = ("phi>0 fluido, phi<0 solido, n=grad(phi)/|grad(phi)| hacia "
                "el fluido; distancia a pared = |phi|")

_MM = 1.0e-3        # mm → m (las interfaces llevan sufijo _mm)


# ==========================================================================
# 1. Primitivas de SDF (todas broadcastean sobre mallas dispersas)
# ==========================================================================
def sdf_slab(axis: int, lo: float, hi: float) -> Callable:
    """Losa fluida entre `lo` y `hi` a lo largo de `axis` (paredes planas).
    Es un SDF exacto."""
    def phi(*X):
        x = X[axis]
        return np.minimum(x - lo, hi - x)
    return phi


def sdf_cylinder(center: Sequence[float], radius: float,
                 axis: int | None = None, solid: bool = True) -> Callable:
    """Cilindro (3D, eje `axis`) o círculo (2D, `axis=None`).

    `solid=True` ⇒ el cilindro es SÓLIDO (φ = r − R, fluido fuera): es el
    caso V3 del banco. `solid=False` ⇒ el cilindro es el conducto FLUIDO.
    """
    c = np.asarray(center, dtype=float)

    def phi(*X):
        acc = None
        for d, x in enumerate(X):
            if d == axis:
                continue
            t = (x - c[d]) ** 2
            acc = t if acc is None else acc + t
        r = np.sqrt(acc)
        return (r - radius) if solid else (radius - r)
    return phi


def sdf_sphere(center: Sequence[float], radius: float,
               solid: bool = True) -> Callable:
    """Esfera sólida (o cavidad esférica fluida con `solid=False`)."""
    c = np.asarray(center, dtype=float)

    def phi(*X):
        acc = None
        for d, x in enumerate(X):
            t = (x - c[d]) ** 2
            acc = t if acc is None else acc + t
        r = np.sqrt(acc)
        return (r - radius) if solid else (radius - r)
    return phi


def sdf_box(lo: Sequence[float], hi: Sequence[float],
            solid: bool = True) -> Callable:
    """Caja alineada con los ejes. SDF exacto sólo fuera de la caja; dentro
    es la distancia de Chebyshev con signo (aproximación estándar, basta
    para el IB porque sólo se usa en la banda |φ| ≲ 2·dx)."""
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)

    def phi(*X):
        inner = None
        for d, x in enumerate(X):
            t = np.minimum(x - lo[d], hi[d] - x)
            inner = t if inner is None else np.minimum(inner, t)
        return -inner if solid else inner
    return phi


def sdf_union(*fns: Callable) -> Callable:
    """Unión de regiones FLUIDAS (máximo de los φ)."""
    def phi(*X):
        out = fns[0](*X)
        for f in fns[1:]:
            out = np.maximum(out, f(*X))
        return out
    return phi


def sdf_intersect(*fns: Callable) -> Callable:
    """Intersección de regiones fluidas (mínimo de los φ)."""
    def phi(*X):
        out = fns[0](*X)
        for f in fns[1:]:
            out = np.minimum(out, f(*X))
        return out
    return phi


def sdf_invert(fn: Callable) -> Callable:
    """Intercambia fluido y sólido."""
    def phi(*X):
        return -fn(*X)
    return phi


def sdf_annulus_profile(z_mm: Sequence[float], r_in_mm: Sequence[float],
                        r_out_mm: Sequence[float], axis_z: int = 2,
                        axis_r: Sequence[int] | None = None,
                        cap_ends: bool = True) -> Callable:
    """Conducto anular de sección variable definido por un perfil
    `(z, r_in, r_out)` — exactamente el bloque `liner.profile` del contrato
    `phycb-combustor-1`. Devuelve φ > 0 DENTRO del anillo (fluido).

    `cap_ends=False` deja los extremos axiales ABIERTOS (φ depende sólo de
    r): es lo que hace falta para que los parches de entrada/salida del
    dominio cartesiano coincidan con la boca del conducto. Con
    `cap_ends=True` el volumen queda cerrado y **el fluido no puede
    moverse** (se comprobó: residual idénticamente 0).

    Aproximación declarada: la distancia se toma como el mínimo de las
    distancias radiales y axiales (no es el SDF exacto de un tronco de cono
    inclinado; el error es O(sinθ) con θ el ángulo del perfil, y sólo
    importa en la banda IB). Longitudes de ENTRADA en mm, salida en m.
    """
    z = np.asarray(z_mm, float) * _MM
    ri = np.asarray(r_in_mm, float) * _MM
    ro = np.asarray(r_out_mm, float) * _MM
    order = np.argsort(z)
    z, ri, ro = z[order], ri[order], ro[order]
    z0, z1 = float(z[0]), float(z[-1])

    def phi(*X):
        zz = X[axis_z]
        idx = [d for d in range(len(X)) if d != axis_z] \
            if axis_r is None else list(axis_r)
        acc = None
        for d in idx:
            t = X[d] ** 2
            acc = t if acc is None else acc + t
        r = np.sqrt(acc)
        zb = np.broadcast_arrays(zz, r)[0]
        rin = np.interp(np.clip(zb, z0, z1), z, ri)
        rout = np.interp(np.clip(zb, z0, z1), z, ro)
        d_rad = np.minimum(r - rin, rout - r)
        if not cap_ends:
            return d_rad
        d_ax = np.minimum(zb - z0, z1 - zb)
        return np.minimum(d_rad, d_ax)
    return phi


# ==========================================================================
# 2. Lectura del campo de voxels de PicoGK (degradación etiquetada)
# ==========================================================================
def sample_vdb(path: str) -> tuple[np.ndarray | None, str]:
    """Intenta leer un `*_field.vdb` de PicoGK. Import PEREZOSO de
    `pyopenvdb` / `openvdb`.

    Devuelve `(None, motivo)` si no hay binding — que es el caso en esta
    máquina (no existe rueda de `pyopenvdb` para Windows/CPython 3.13).
    **Nunca lanza.** El camino soportado hoy es `sdf_from_contract`.
    """
    vdb = None
    why = ""
    for mod in ("pyopenvdb", "openvdb"):
        try:
            vdb = __import__(mod)
            break
        except Exception as exc:
            why = f"{type(exc).__name__}: {exc}"
    if vdb is None:
        return None, (f"binding de OpenVDB no disponible ({why}); usa "
                      "sdf_from_contract() como aproximación declarada")
    try:
        grids = vdb.readAllGridMetadata(path)
        name = grids[0].name if grids else "phi"
        grid = vdb.read(path, name)
        arr = np.zeros(grid.evalActiveVoxelDim(), dtype=np.float64)
        grid.copyToArray(arr)
        return arr, ""
    except Exception as exc:
        return None, f"lectura de '{path}' falló: {type(exc).__name__}: {exc}"


# ==========================================================================
# 3. Parches de contorno desde el contrato
# ==========================================================================
@dataclass
class Patch:
    """Parche de contorno de una cara del dominio cartesiano.

    `kind` ∈ {"inlet", "outlet", "wall", "slip", "periodic"}.
    `u` es la velocidad impuesta (m/s, lista de ndim), `p` la presión
    estática impuesta (Pa) — cada `kind` usa lo que le toca.
    """
    name: str
    axis: int
    side: str                     # "lo" | "hi"
    kind: str
    u: tuple[float, ...] | None = None
    p: float | None = None
    meta: dict = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{'xyz'[self.axis]}{'-' if self.side == 'lo' else '+'}"


def patches_from_contract(contract: dict, axis_z: int = 2) -> list[Patch]:
    """Parches de entrada/salida derivados de
    `contract["cfd_boundary_conditions"]` (lo escribe
    `geometry_generator.cfd_boundary_conditions`).

    Convenio del contrato §9: **Z es el eje de simetría** y el gas va a +Z.
    Entrada = anillo de la carcasa (aire del difusor) en z mínimo, salida =
    presión estática del ciclo en z máximo. Lectura DEFENSIVA: un contrato
    incompleto degrada a valores coherentes y lo anota en `meta`.
    """
    bc = (contract or {}).get("cfd_boundary_conditions") or {}
    notes: list[str] = []

    def _f(key, dflt):
        try:
            v = float(bc.get(key, dflt))
            return v if math.isfinite(v) else float(dflt)
        except Exception:
            notes.append(f"{key} ausente/ilegible → {dflt}")
            return float(dflt)

    rho = _f("rho3_kgm3", 1.2)
    mdot = _f("mdot_air_kgs", 0.1)
    p_out = _f("P_out_static_Pa", 101_325.0)
    # velocidad de entrada estimada con el área del anillo del flow_path
    a_in = 0.0
    for seg in (contract or {}).get("flow_path") or []:
        if str(seg.get("id", "")).startswith("anillo"):
            try:
                a_in = float(seg.get("area_mm2") or 0.0) * _MM * _MM
            except Exception:
                a_in = 0.0
            break
    if a_in <= 0.0:
        a_in = 1.0e-3
        notes.append("area del anillo de entrada ausente → 1e-3 m²")
    u_in = mdot / max(rho * a_in, 1e-12)
    uvec = [0.0, 0.0, 0.0]
    uvec[axis_z] = u_in
    ps = [
        Patch("inlet", axis_z, "lo", "inlet", tuple(uvec), None,
              dict(mdot_kgs=mdot, rho_kgm3=rho, area_m2=a_in, notes=notes)),
        Patch("outlet", axis_z, "hi", "outlet", None, p_out,
              dict(notes=notes)),
    ]
    for d in range(3):
        if d == axis_z:
            continue
        ps.append(Patch(f"far{'xyz'[d]}lo", d, "lo", "slip"))
        ps.append(Patch(f"far{'xyz'[d]}hi", d, "hi", "slip"))
    return ps


# ==========================================================================
# 4. Malla cartesiana con IB
# ==========================================================================
def _dilate(mask: np.ndarray, n: int) -> np.ndarray:
    """Dilatación booleana de Chebyshev en `n` pasos, sin scipy y **sin
    envolver** por los bordes (np.roll daría vecindades falsas)."""
    out = mask.copy()
    nd = mask.ndim
    for _ in range(int(n)):
        nxt = out.copy()
        for d in range(nd):
            a = [slice(None)] * nd
            b = [slice(None)] * nd
            a[d], b[d] = slice(0, -1), slice(1, None)
            nxt[tuple(a)] |= out[tuple(b)]
            nxt[tuple(b)] |= out[tuple(a)]
        out = nxt
    return out


class CartesianMesh:
    """Malla cartesiana uniforme con `ng` capas fantasma por lado.

    El índice global `I` de una celda y su centro se relacionan por
    `x = lo + (I − ng + 0.5)·dx`, así que el interior son los índices
    `[ng, ng+n)` en cada eje.
    """

    def __init__(self, lo: Sequence[float], hi: Sequence[float], dx: float,
                 ng: int = NBAND):
        lo = [float(v) for v in lo]
        hi = [float(v) for v in hi]
        if len(lo) != len(hi) or len(lo) not in (2, 3):
            raise ValueError("la malla debe ser 2D o 3D y lo/hi del mismo "
                             "tamaño")
        self.ndim = len(lo)
        self.dx = float(dx)
        self.ng = int(ng)
        self.shape = tuple(max(1, int(round((hi[d] - lo[d]) / self.dx)))
                           for d in range(self.ndim))
        self.lo = tuple(lo)
        self.hi = tuple(lo[d] + self.shape[d] * self.dx
                        for d in range(self.ndim))
        self.shape_tot = tuple(n + 2 * self.ng for n in self.shape)
        self.ncell = int(np.prod(self.shape))
        #: coordenadas de centro de celda por eje, con fantasmas
        self.xc = [self.lo[d] + (np.arange(-self.ng, self.shape[d] + self.ng)
                                 + 0.5) * self.dx for d in range(self.ndim)]
        #: malla dispersa (broadcast) — memoria O(n), no O(n^ndim)
        self.X = np.meshgrid(*self.xc, indexing="ij", sparse=True)

        self.phi: np.ndarray | None = None
        self.fluid = np.ones(self.shape_tot, dtype=bool)
        self.solid = np.zeros(self.shape_tot, dtype=bool)
        self.ghost = np.zeros(self.shape_tot, dtype=bool)
        self.celltype = np.zeros(self.shape_tot, dtype=np.int8)
        self.wall_distance = np.full(self.shape_tot, np.inf)
        self.normals: list[np.ndarray] = []
        self.n_ghost = 0
        self.n_orphan = 0
        self._has_ib = False
        self._sdf_note = "sin IB (todas las celdas son fluido)"
        #: sólido profundo (no fantasma). Se precomputa una vez: calcularlo
        #: dentro de `zero_deep_solid` costaba dos operaciones booleanas
        #: sobre TODO el dominio en cada subpaso RK (medido, era caro).
        self._deep = np.zeros(self.shape_tot, dtype=bool)
        self._ibhost: dict = {}
        self._ibdev: dict | None = None
        self.device_note = "estructuras del IB en CPU (NumPy)"

    # -- vistas ------------------------------------------------------------
    @property
    def interior(self) -> tuple[slice, ...]:
        """Slice del interior (sin fantasmas)."""
        return tuple(slice(self.ng, self.ng + n) for n in self.shape)

    def view(self, a: np.ndarray, axis: int, k: int) -> np.ndarray:
        """Vista del tamaño del interior desplazada `k` celdas en `axis`."""
        idx = [slice(self.ng, self.ng + n) for n in self.shape]
        idx[axis] = slice(self.ng + k, self.ng + k + self.shape[axis])
        return a[tuple(idx)]

    def fview(self, a: np.ndarray, axis: int, k: int) -> np.ndarray:
        """Vista del tamaño «interior + 1 cara» en `axis`, desplazada `k`.

        La cara `j` (j = 0..n) separa las celdas `j−1` y `j`; `k=0` da la
        celda izquierda de cada cara y `k=1` la derecha."""
        idx = [slice(self.ng, self.ng + n) for n in self.shape]
        idx[axis] = slice(self.ng + k - 1, self.ng + k + self.shape[axis])
        return a[tuple(idx)]

    def zeros(self, dtype=np.float64) -> np.ndarray:
        return np.zeros(self.shape_tot, dtype=dtype)

    # -- SDF e IB ----------------------------------------------------------
    def set_sdf(self, phi, note: str = "") -> "CartesianMesh":
        """Muestrea el SDF en los centros de celda y construye toda la
        maquinaria de IB. `phi` puede ser un callable `φ(*X)` (broadcast
        sobre la malla dispersa) o un array ya del tamaño `shape_tot`.

        `phi=None` desactiva el IB (todas las celdas fluido) — es el caso
        de los bancos con paredes alineadas (V2 cavidad).
        """
        if phi is None:
            self._has_ib = False
            self._sdf_note = note or "sin IB (todas las celdas son fluido)"
            return self
        if callable(phi):
            val = phi(*self.X)
        else:
            val = np.asarray(phi, dtype=float)
        self.phi = np.ascontiguousarray(
            np.broadcast_to(np.asarray(val, dtype=float),
                            self.shape_tot).astype(np.float64))
        self.fluid = self.phi > 0.0
        self.solid = ~self.fluid
        self.wall_distance = np.abs(self.phi)
        # normales: ∇φ normalizado (np.gradient usa diferencias centradas en
        # el interior y de un lado en los bordes ⇒ 2.º orden salvo bordes)
        grad = np.gradient(self.phi, self.dx, edge_order=1)
        if self.ndim == 1:
            grad = [grad]
        mag = np.sqrt(sum(g * g for g in grad))
        mag = np.where(mag < 1e-30, 1.0, mag)
        self.normals = [g / mag for g in grad]
        # banda fantasma: sólido a ≤ NBAND celdas de fluido
        near = _dilate(self.fluid, self.ng)
        self.ghost = near & self.solid
        self.celltype = np.where(self.fluid, FLUID,
                                 np.where(self.ghost, GHOST, SOLID)
                                 ).astype(np.int8)
        self._deep = self.solid & ~self.ghost
        self._build_ghost_stencils()
        self._has_ib = True
        self._sdf_note = note or IB_SIGN_NOTE
        self._ibdev = None
        self.device_note = "estructuras del IB en CPU (NumPy)"
        return self

    def _build_ghost_stencils(self) -> None:
        """Precomputa, para cada celda fantasma, el punto imagen y los
        pesos de la interpolación multilineal (2^ndim nodos)."""
        idx = np.argwhere(self.ghost)                    # (Ng, ndim)
        self.n_ghost = int(idx.shape[0])
        self._g_idx = tuple(idx[:, d] for d in range(self.ndim))
        if self.n_ghost == 0:
            self._g_nodes = np.zeros((0, 1), dtype=np.intp)
            self._g_w = np.zeros((0, 1))
            self._g_wfluid = np.zeros((0, 1))
            self._g_wsum = np.zeros(0)
            self._g_ratio = np.zeros(0)
            self._g_wwall = np.zeros(0)
            self._g_wneu = np.zeros((0, 1))
            self._g_orphan = np.zeros(0, dtype=bool)
            self.n_orphan = 0
            self._pack_host()
            return
        d_g = np.abs(self.phi[self._g_idx])
        nrm = np.stack([n[self._g_idx] for n in self.normals], axis=1)
        d_ip = np.maximum(d_g, IMAGE_MIN_DX * self.dx)
        self._g_ratio = d_g / d_ip
        # coordenadas del fantasma y de su imagen
        x_g = np.stack([self.lo[d] + (idx[:, d] - self.ng + 0.5) * self.dx
                        for d in range(self.ndim)], axis=1)
        x_ip = x_g + (d_g + d_ip)[:, None] * nrm
        # índice fraccionario en el array con fantasmas
        fi = (x_ip - np.asarray(self.lo)[None, :]) / self.dx + self.ng - 0.5
        i0 = np.floor(fi).astype(np.intp)
        for d in range(self.ndim):
            i0[:, d] = np.clip(i0[:, d], 0, self.shape_tot[d] - 2)
        w = fi - i0
        w = np.clip(w, 0.0, 1.0)
        ncorner = 2 ** self.ndim
        nodes = np.zeros((self.n_ghost, ncorner), dtype=np.intp)
        wts = np.ones((self.n_ghost, ncorner))
        for c in range(ncorner):
            multi = []
            for d in range(self.ndim):
                bit = (c >> d) & 1
                multi.append(i0[:, d] + bit)
                wts[:, c] *= w[:, d] if bit else (1.0 - w[:, d])
            nodes[:, c] = np.ravel_multi_index(tuple(multi), self.shape_tot)
        isfl = self.fluid.reshape(-1)[nodes]
        self._g_nodes = nodes
        self._g_w = wts
        self._g_wfluid = np.where(isfl, wts, 0.0)
        self._g_wsum = self._g_wfluid.sum(axis=1)
        self._g_wwall = 1.0 - self._g_wsum
        self.n_orphan = int(np.count_nonzero(
            self._g_wsum < MIN_FLUID_WEIGHT))
        # renormalización segura para Neumann
        self._g_wneu = self._g_wfluid / np.where(
            self._g_wsum > MIN_FLUID_WEIGHT, self._g_wsum, 1.0)[:, None]
        self._g_orphan = self._g_wsum < MIN_FLUID_WEIGHT
        self._pack_host()

    # -- juegos de estructuras del IB (CPU / dispositivo) ------------------
    def _pack_host(self) -> None:
        """Agrupa en un dict lo que el bucle caliente del IB necesita. Los
        atributos `_g_*` se conservan como están porque `post.py` y los
        tests los leen: esto es una VISTA, no un renombrado."""
        # el sólido profundo se guarda como ÍNDICES PLANOS, no como máscara
        # booleana: indexar con un booleano obliga a CuPy a hacer un scan
        # (cumsum) en CADA llamada, y `zero_deep_solid` se llama 2·ndim
        # veces por subpaso RK. Con índices es un scatter puro.
        self._deep_idx = np.flatnonzero(self._deep.reshape(-1))
        self._ibhost = dict(
            xp=np, idx=self._g_idx, nodes=self._g_nodes,
            wfluid=self._g_wfluid, wwall=self._g_wwall,
            wneu=self._g_wneu, ratio=self._g_ratio,
            orphan=self._g_orphan, deep=self._deep_idx,
            deep_any=bool(self._deep_idx.size))

    def to_backend(self, be) -> "CartesianMesh":
        """Sube al dispositivo de `be` las estructuras del IB que el bucle
        caliente toca en cada subpaso RK.

        Sin esto, cada `apply_dirichlet` sobre un campo de CuPy exigiría
        traer el campo a CPU y devolverlo — el error clásico que convierte
        un port a GPU en algo más lento que el original. Con CPU (o sin IB)
        es un no-op. **Nunca lanza**: si la subida falla, se queda en CPU y
        lo anota en `device_note`.
        """
        if be is None or not getattr(be, "is_gpu", False) or not self._has_ib:
            # NO se borra `_ibdev`: una misma malla puede alimentar a la vez
            # a un solver de CPU y a uno de GPU, y el despacho ya elige el
            # juego correcto por el tipo del array. Borrarlo aquí dejaría al
            # solver de GPU mezclando índices de host con campos de device.
            if self._ibdev is None:
                self.device_note = "estructuras del IB en CPU (NumPy)"
            return self
        xp = be.xp
        try:
            wdt = be.dtype                 # pesos con el dtype del campo
            self._ibdev = dict(
                xp=xp,
                idx=tuple(xp.asarray(i) for i in self._g_idx),
                nodes=xp.asarray(self._g_nodes),
                wfluid=xp.asarray(self._g_wfluid, dtype=wdt),
                wwall=xp.asarray(self._g_wwall, dtype=wdt),
                wneu=xp.asarray(self._g_wneu, dtype=wdt),
                ratio=xp.asarray(self._g_ratio, dtype=wdt),
                orphan=xp.asarray(self._g_orphan),
                deep=xp.asarray(self._deep_idx),
                deep_any=bool(self._deep_idx.size))
            self.device_note = (
                f"estructuras del IB en {be.name} ({self.n_ghost} fantasmas, "
                f"pesos en {np.dtype(wdt).name}); sin transferencias "
                "device<->host en el bucle")
        except Exception as exc:
            self._ibdev = None
            self.device_note = (f"subida del IB al dispositivo fallo "
                                f"({type(exc).__name__}: {exc}) → CPU")
        return self

    def _pack_for(self, fld) -> dict:
        """Juego de estructuras (CPU o dispositivo) que corresponde a `fld`.
        El despacho es por TIPO del array, no por una bandera global: así la
        misma malla puede servir a un solver de CPU y a uno de GPU."""
        if self._ibdev is not None and not isinstance(fld, np.ndarray):
            return self._ibdev
        return self._ibhost

    def apply_dirichlet(self, fld, wall_value=0.0, only=None) -> None:
        """Impone valor de pared `wall_value` (escalar o array por
        fantasma) por extrapolación lineal desde el punto imagen.

        En los fantasmas «huérfanos» (estencil casi todo sólido, esquinas
        cóncavas) se impone directamente el valor de pared: es de 1.er
        orden pero acotado, y su número se reporta en `summary()`.

        `only` (opcional): máscara booleana POR FANTASMA — el Dirichlet se
        aplica sólo donde es True y el resto de fantasmas conserva su valor
        (lo usa el M4: la pared que TRANSPIRA impone T3 y composición de
        aire, la que no, sigue adiabática/Neumann). En GPU la máscara debe
        vivir YA en el dispositivo (se sube una vez, como los perfiles).
        Con `only=None` el comportamiento es EXACTAMENTE el de M1.

        Funciona igual con arrays de NumPy y de CuPy (ver `to_backend`).
        """
        if not self._has_ib or self.n_ghost == 0:
            return
        pk = self._pack_for(fld)
        xp = pk["xp"]
        if xp is np:
            uw = np.asarray(wall_value, dtype=float)
        elif np.ndim(wall_value) == 0:
            uw = float(wall_value)
        else:
            # `wall_value` puede llegar YA en el dispositivo (el solver lo
            # sube una vez en `_resolve_wall_u`) o en host. `xp.asarray`
            # sirve para los dos; `np.asarray` NO — sobre un array de CuPy
            # lanza a propósito. Ese fue un fallo real de la V1b.
            uw = xp.asarray(wall_value, dtype=fld.dtype)
        flat = fld.reshape(-1)
        f_ip = (flat[pk["nodes"]] * pk["wfluid"]).sum(axis=1) \
            + pk["wwall"] * uw
        val = uw - pk["ratio"] * (f_ip - uw)
        # `uw` puede ser escalar o un array por fantasma: `where` difunde en
        # los dos casos y en los dos backends.
        val = xp.where(pk["orphan"], uw, val)
        if only is None:
            fld[pk["idx"]] = val
        else:
            fld[pk["idx"]] = xp.where(only, val, fld[pk["idx"]])

    def apply_neumann(self, fld) -> None:
        """Impone gradiente normal nulo (∂f/∂n = 0 ⇒ f_g = f_imagen)."""
        if not self._has_ib or self.n_ghost == 0:
            return
        pk = self._pack_for(fld)
        xp = pk["xp"]
        flat = fld.reshape(-1)
        f_ip = (flat[pk["nodes"]] * pk["wneu"]).sum(axis=1)
        cur = fld[pk["idx"]]
        fld[pk["idx"]] = xp.where(pk["orphan"], cur, f_ip)

    def zero_deep_solid(self, fld, value: float = 0.0) -> None:
        """Pone `value` en el sólido que NO es fantasma (nunca lo lee el
        estencil del fluido; sirve para que la visualización sea limpia y
        para que no haya NaN latentes). Los índices están precomputados."""
        if not self._has_ib:
            return
        pk = self._pack_for(fld)
        if pk["deep_any"]:
            fld.reshape(-1)[pk["deep"]] = value

    def ghost_coords(self) -> np.ndarray:
        """Coordenadas (Ng, ndim) de los centros de las celdas fantasma, en
        el MISMO orden que usan `apply_dirichlet`/`apply_neumann`. Sirve
        para imponer paredes inmersas con velocidad variable (Couette,
        rotor, pared efusiva del M4)."""
        if not self._has_ib or self.n_ghost == 0:
            return np.zeros((0, self.ndim))
        return np.stack([self.lo[d] + (self._g_idx[d] - self.ng + 0.5)
                         * self.dx for d in range(self.ndim)], axis=1)

    # -- diagnóstico -------------------------------------------------------
    def summary(self) -> dict:
        """Bloque `cfd_mesh` del record (contrato §4.4)."""
        nf = int(np.count_nonzero(self.fluid[self.interior])) \
            if self._has_ib else self.ncell
        return dict(
            schema="phycb-cfdmesh-1", ndim=self.ndim,
            shape=list(self.shape), dx_mm=self.dx / _MM,
            lo_mm=[v / _MM for v in self.lo],
            hi_mm=[v / _MM for v in self.hi],
            cells_total=self.ncell, cells_fluid=nf,
            cells_ghost=int(self.n_ghost), ghost_orphan=int(self.n_orphan),
            ib=bool(self._has_ib), ghost_band=self.ng,
            wall_distance="|phi| (exacta, gratis)",
            device=self.device_note,
            sdf_note=self._sdf_note, sign_convention=IB_SIGN_NOTE)


# ==========================================================================
# 5. Del contrato de geometría a la malla
# ==========================================================================
def sdf_from_contract(contract: dict, axis_z: int = 2,
                      cap_ends: bool = False) -> tuple[Callable, dict]:
    """SDF **analítico aproximado** del volumen fluido del liner desde el
    contrato `phycb-combustor-1`.

    ⚠️ HONESTIDAD: esto NO es el campo de voxels de PicoGK. Es el conducto
    anular que define `liner.profile` (`[z_mm, r_in_mm, r_out_mm]`),
    suficiente para probar la cadena SDF→malla→solver y para el M1, pero
    sin swirler, sin orificios y sin domo. El camino canónico
    (`sample_vdb`) está escrito y degrada porque no hay binding de OpenVDB
    en esta máquina. Devuelve `(phi, meta)`; **nunca lanza**.
    """
    meta = {"source": "contract.liner.profile", "exact": False,
            "missing": ["swirler", "hole_rows", "efusion", "domo"]}
    prof = ((contract or {}).get("liner") or {}).get("profile") or []
    try:
        z = [float(p[0]) for p in prof]
        ri = [float(p[1]) for p in prof]
        ro = [float(p[2]) for p in prof]
    except Exception:
        z, ri, ro = [], [], []
    if len(z) < 2:
        meta["source"] = "fallback (perfil ausente o ilegible)"
        z, ri, ro = [0.0, 100.0], [10.0, 10.0], [30.0, 30.0]
    meta.update(z_mm=[min(z), max(z)], r_in_mm=[min(ri), max(ri)],
                r_out_mm=[min(ro), max(ro)], cap_ends=bool(cap_ends))
    return sdf_annulus_profile(z, ri, ro, axis_z=axis_z,
                               cap_ends=cap_ends), meta


def _profile_interp_mm(prof: list, col: int) -> tuple[np.ndarray, np.ndarray]:
    """Pares (z, valor) en **metros** de una polilínea del contrato, listos
    para `np.interp`. `col` = columna del valor (1 = r_in, 2 = r_out en el
    liner; 1 = r en la carcasa). Nunca lanza: un perfil ilegible degrada a
    dos puntos constantes."""
    try:
        z = np.asarray([float(p[0]) for p in prof], dtype=float)
        v = np.asarray([float(p[col]) for p in prof], dtype=float)
        order = np.argsort(z)
        z, v = z[order], v[order]
        if z.size < 2:
            raise ValueError
    except Exception:
        z = np.array([0.0, 100.0])
        v = np.array([30.0, 30.0])
    return z * _MM, v * _MM


def sdf_combustor_from_contract(contract: dict, dx_mm: float = 1.5,
                                axis_z: int = 2) -> tuple[Callable, dict]:
    """SDF analítico del volumen fluido **COMPLETO** del combustor (M4):

        interior del liner  ∪  anillo liner↔carcasa (aire de la carcasa)
                            ∪  cilindros de las filas de orificios RESUELTAS

    * El anillo exterior queda **tapado aguas abajo** (el aire sólo puede
      salir por orificios/efusión/swirler) y abierto en la cara del domo,
      que es el parche de entrada del dominio.
    * La cara del domo NO se cierra en el SDF: la cierra el parche de
      entrada con u = 0 fuera de los anillos activos (misma técnica que la
      ranura del JICF, `run_v5_jcf.py`).
    * **Regla de la wave (WAVE6 §0)**: una fila se RESUELVE en el SDF sólo
      si d/dx ≥ 4; si no, va como transpiración calibrada por C_d·A (la
      aplica `case.py` vía `ib_wall_u`). Qué fila fue a qué camino queda
      DECLARADO en `meta["rows"]`.
    * La efusión NUNCA se resuelve (Ø 0.3–0.5 mm): siempre transpiración.
    * En `reverse_annular` el recorrido real del gas hace dos giros de
      180°; aquí el conducto se trata como pasante en Z (aproximación
      DECLARADA, la misma de `cold_smoke`). `meta["flow_sign"]` dice hacia
      dónde va el gas (+1 si el domo está en z mínimo).

    Devuelve `(phi, meta)` con φ en metros. **Nunca lanza.**
    """
    c = contract or {}
    liner = (c.get("liner") or {})
    prof = liner.get("profile") or []
    z_l, r_in = _profile_interp_mm(prof, 1)
    _, r_out = _profile_interp_mm(prof, 2)
    t_geo = float(liner.get("t_wall_mm") or 0.8) * _MM
    # ⚠️ La pared REAL (0.5–1.5 mm) es SUB-MALLA a dx = 1–3 mm: sin celdas
    # sólidas dentro, el IB de celda centrada la deja porosa y el anillo se
    # comunica con el liner por toda la pared (se midió: los splits salían
    # sin sentido). Se INFLA a >= WALL_MIN_CELLS·dx para que sea estanca —
    # aproximación geométrica DECLARADA (reduce el anillo en t_eff − t_geo).
    t_w = max(t_geo, WALL_MIN_CELLS * float(dx_mm) * _MM)
    cas = (c.get("casing") or {}).get("profile") or []
    z_c, r_cas = _profile_interp_mm(cas, 1)
    if not cas:                       # respaldo: carcasa 1.25·r_out
        z_c, r_cas = z_l, r_out * 1.25 + 2.0 * t_w
    dome_z = float(((liner.get("dome") or {}).get("z_mm")
                    or z_l[0] / _MM)) * _MM
    z0, z1 = float(z_l[0]), float(z_l[-1])
    # el gas va del domo al extremo opuesto del perfil
    sign = 1.0 if abs(dome_z - z0) <= abs(dome_z - z1) else -1.0
    z_dome = z0 if sign > 0 else z1
    L_tot = z1 - z0
    dx = float(dx_mm) * _MM
    cap = 2.0 * dx                   # tapa del anillo exterior aguas abajo
    has_hub = bool(np.max(r_in) > 1e-6)
    # ¿Cabe el anillo exterior en la malla? Necesita >= 2 celdas de holgura
    # TRAS inflar la pared. Si no cabe (caso KJ66: ~3 mm de holgura), el
    # dominio es SOLO el interior del liner y TODAS las filas van como
    # transpiración C_d·A — modo DECLARADO «sin anillo». Sin esto, o el
    # anillo queda estrangulado por la pared inflada (medido: el caso
    # revienta) o la pared vuelve a ser porosa.
    gap_min = float(np.min(np.interp(z_l, z_c, r_cas) - r_out)) - t_w
    annulus_ok = gap_min >= 2.0 * dx

    rows_meta: list[dict] = []
    resolved: list[dict] = []
    for hr in (c.get("hole_rows") or []):
        if not isinstance(hr, dict) or not hr.get("flow", True):
            continue
        d_m = float(hr.get("d_mm") or 0.0) * _MM
        n = int(hr.get("n") or 0)
        if d_m <= 0.0 or n <= 0:
            continue
        is_res = annulus_ok and (d_m / dx) >= 4.0
        cda = float(hr.get("Cd") or 0.62) \
            * float(hr.get("A_row_mm2") or 0.0) * _MM * _MM
        phi0 = math.radians(float(hr.get("phi0_deg") or 0.0))
        row = dict(id=str(hr.get("id", "?")), zone=str(hr.get("zone", "?")),
                   z_m=float(hr.get("z_mm") or 0.0) * _MM,
                   r_m=float(hr.get("r_mm") or 0.0) * _MM,
                   n=n, d_m=d_m, Cd=float(hr.get("Cd") or 0.62),
                   CdA_m2=cda, phi0_rad=phi0,
                   d_over_dx=d_m / dx, resolved=bool(is_res),
                   U_jet_ms=float(hr.get("U_jet_ms") or 0.0),
                   mdot_kgs=float(hr.get("mdot_kgs") or 0.0))
        rows_meta.append(row)
        if is_res:
            resolved.append(row)

    def phi(*X):
        zz = X[axis_z]
        idx = [d for d in range(len(X)) if d != axis_z]
        acc = None
        for d in idx:
            t = X[d] ** 2
            acc = t if acc is None else acc + t
        r = np.sqrt(acc)
        zb = np.broadcast_arrays(zz, r)[0]
        zc = np.clip(zb, z0, z1)
        rin = np.interp(zc, z_l, r_in)
        rout = np.interp(zc, z_l, r_out)
        rcas = np.interp(zc, z_c, r_cas)
        # interior del liner (abierto en los dos extremos axiales)
        d_lin = rout - r
        if has_hub:
            d_lin = np.minimum(d_lin, r - rin)
        if annulus_ok:
            # anillo exterior, tapado aguas abajo (el domo lo cierra el
            # parche de entrada)
            d_ann = np.minimum(r - (rout + t_w), rcas - r)
            zeta = sign * (zb - z_dome)          # coordenada de recorrido
            d_ann = np.minimum(d_ann, (L_tot - cap) - zeta)
            out = np.maximum(d_lin, d_ann)
        else:
            out = d_lin
        # filas resueltas: n cilindros fluidos radiales que cruzan la pared
        for row in resolved:
            span_lo = row["r_m"] - 1.5 * dx
            span_hi = row["r_m"] + t_w + 1.5 * dx
            xx, yy = X[idx[0]], X[idx[1]]
            for k in range(row["n"]):
                a = row["phi0_rad"] + 2.0 * math.pi * k / row["n"]
                # distancia del punto al EJE radial del orificio
                s_perp = xx * math.sin(a) - yy * math.cos(a)
                ell = xx * math.cos(a) + yy * math.sin(a)
                d_ax2 = s_perp ** 2 + (zb - row["z_m"]) ** 2
                d_cyl = 0.5 * row["d_m"] - np.sqrt(d_ax2)
                d_cyl = np.minimum(d_cyl, ell - span_lo)
                d_cyl = np.minimum(d_cyl, span_hi - ell)
                out = np.maximum(out, d_cyl)
        return out

    meta = dict(
        source="contract completo (liner + anillo + filas resueltas)",
        exact=False, schema=str(c.get("schema", "?")),
        z_mm=[z0 / _MM, z1 / _MM], dome_z_mm=dome_z / _MM,
        flow_sign=sign, L_tot_mm=L_tot / _MM,
        t_liner_mm=t_w / _MM, t_liner_geo_mm=t_geo / _MM,
        wall_inflated=bool(t_w > t_geo * 1.001), dx_mm=float(dx_mm),
        r_in_mm=[float(np.min(r_in)) / _MM, float(np.max(r_in)) / _MM],
        r_out_mm=[float(np.min(r_out)) / _MM, float(np.max(r_out)) / _MM],
        r_casing_mm=[float(np.min(r_cas)) / _MM, float(np.max(r_cas)) / _MM],
        has_hub=has_hub, rows=rows_meta,
        annulus_resolved=bool(annulus_ok),
        annulus_gap_min_mm=gap_min / _MM,
        n_rows_resolved=len(resolved),
        n_rows_source=len(rows_meta) - len(resolved),
        rule=("fila resuelta en el SDF si d/dx >= 4 (WAVE6 §0) y el anillo "
              "cabe (holgura >= 2·dx tras inflar la pared); si no, "
              "transpiracion C_d·A; efusion SIEMPRE transpiracion"
              + ("" if annulus_ok else
                 "; ANILLO NO RESUELTO en este caso (holgura "
                 f"{gap_min / _MM:.1f} mm < 2·dx): dominio = interior del "
                 "liner, todas las filas transpiradas")),
        missing=[])
    # interpoladores que `case.py` y `post.py` reutilizan (en metros)
    meta["_interp"] = dict(z=z_l, r_in=r_in, r_out=r_out,
                           z_cas=z_c, r_cas=r_cas)
    return phi, meta


def mesh_combustor_from_contract(contract: dict, dx_mm: float = 1.5,
                                 pad_cells: int = 2, axis_z: int = 2
                                 ) -> tuple[CartesianMesh, dict]:
    """Malla cartesiana del caso M4: encierra la CARCASA (no sólo el liner)
    y muestrea el SDF completo de `sdf_combustor_from_contract`. Las caras
    z del dominio caen en el plano del domo y en el plano 4. **Nunca
    lanza** (contrato ilegible ⇒ caja por defecto anotada)."""
    phi, meta = sdf_combustor_from_contract(contract, dx_mm=dx_mm,
                                            axis_z=axis_z)
    if meta["annulus_resolved"]:
        r_max = max(meta["r_casing_mm"])
    else:                       # sin anillo: la caja abraza sólo el liner
        r_max = max(meta["r_out_mm"]) + meta["t_liner_mm"]
    z0, z1 = meta["z_mm"]
    dx = float(dx_mm) * _MM
    pad = pad_cells * dx
    lo = [-r_max * _MM - pad] * 3
    hi = [r_max * _MM + pad] * 3
    lo[axis_z] = z0 * _MM
    hi[axis_z] = z1 * _MM
    m = CartesianMesh(lo, hi, dx)
    m.set_sdf(phi, note=("SDF analitico COMPLETO del combustor (liner + "
                         "anillo + filas resueltas) — aproximacion "
                         "declarada del campo de PicoGK"))
    meta.update(m.summary())
    return m, meta


def mesh_from_contract(contract: dict, dx_mm: float = 2.0,
                       pad_cells: int = 3, axis_z: int = 2
                       ) -> tuple[CartesianMesh, dict]:
    """Malla cartesiana que encierra el liner del contrato, con el SDF ya
    muestreado. `dx_mm` en el rango 0.5–2 mm de Ciencia §11.2.

    Devuelve `(mesh, meta)` con `meta` trazable. **Nunca lanza**: un
    contrato ilegible produce una caja por defecto anotada en `meta`.

    El dominio se recorta EN Z al tramo del perfil y el SDF se genera con
    los extremos abiertos, para que los parches de entrada/salida del
    dominio cartesiano caigan sobre la boca del conducto (si no, el volumen
    fluido queda sellado y no se mueve nada).
    """
    phi, meta = sdf_from_contract(contract, axis_z=axis_z, cap_ends=False)
    z0, z1 = meta.get("z_mm", [0.0, 100.0])
    r_max = max(meta.get("r_out_mm", [30.0]))
    dx = float(dx_mm) * _MM
    pad = pad_cells * dx
    lo = [-r_max * _MM - pad] * 3
    hi = [r_max * _MM + pad] * 3
    lo[axis_z] = z0 * _MM
    hi[axis_z] = z1 * _MM
    m = CartesianMesh(lo, hi, dx)
    m.set_sdf(phi, note=("SDF analitico del liner desde el contrato "
                         "(aproximacion declarada, ver sdf_from_contract)"))
    meta.update(m.summary())
    return m, meta
