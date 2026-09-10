"""
QUASAR Phy-CB · cfd/ibrans/backend.py
=====================================
Abstracción de backend de cómputo del solver L2a: **NumPy** (obligatorio),
**CuPy** (GPU, opcional), **NVIDIA Warp** y **JAX** (opcionales). Todos con
import PEREZOSO. Ninguna ausencia lanza: si el backend pedido no está, se
degrada a NumPy y se **etiqueta** el motivo (`Backend.note`), igual que hace
el resto de la familia.

Qué hay REALMENTE en esta máquina (medido el 2026-07-31)
--------------------------------------------------------
Una tanda anterior de trabajo escribió «sin GPU» en este archivo y en
`ESTADO.md`. **Era falso.** Comprobado con `python cfd/ibrans/backend.py`:

    GPU      : NVIDIA GeForce RTX 3070 Ti, 8 GiB, compute capability 8.6
    driver   : 591.86 · CUDA toolkit 12.0 · runtime CUDA 12.9
    cupy     : 14.1.1  INSTALADO Y FUNCIONANDO      → backend GPU real
    warp/jax : NO instalados (y no se instalan: regla del proyecto)

Por eso el backend de producción de M1 **no es Warp sino CuPy**: es un
reemplazo casi directo de NumPy sobre GPU, así que el kernel del solver es
el MISMO código (`xp.` en vez de `np.`) y no hay que portar nada a un DSL.
Warp y JAX siguen listados porque la wave los nombra, pero **no tienen
kernel escrito**: pedirlos degrada a NumPy con etiqueta.

Precisión
---------
Ciencia §11.2 fija **FP32** para GPU (el FP64 de las GPU de consumo está
capado a 1/64). Aquí `precision="auto"` da **FP64 en los dos backends**, y
el motivo está medido, no supuesto:

* en CPU el FP64 es gratis;
* en GPU el FP32 es ×1.8 más rápido a partir de 1 M celdas y usa la mitad
  de VRAM (`ESTADO.md` §5.2), así que la decisión de la wave es correcta;
* **pero** en FP32 el caso exigente de la escalera (V2, cavidad de Ghia a
  Re = 1000, tolerancia de parada 2e-5) **no converge**: el residual se
  estanca en 2.75e-5 y el solver agota las 600 000 iteraciones. La
  precisión aguanta (err 0.0051 vs 0.0048 en FP64, umbral 0.05); el
  criterio de PARADA no, y ese umbral no se afloja. Medido, no supuesto:
  `ESTADO.md` §5.4.

FP64 es por tanto la única precisión en la que la escalera entera para por
convergencia. Para los casos NO estacionarios de M2/M3 —criterio de promedio
temporal, no de residual— FP32 vuelve a ser la opción sensata: cambiarlo es
`GPU_AUTO_PRECISION`, y exige rehacer la escalera.

Determinismo
------------
Con `SEED = 71` y **el mismo backend** los resultados son reproducibles.
**GPU y CPU no dan resultados bit-idénticos**: las reducciones de CuPy
(`sum`, `max`) usan un orden de acumulación por bloques distinto del de
NumPy, y en FP32 además cambia el redondeo. Es una diferencia real y aquí
se DECLARA en vez de fingir que no existe; se comprueba en su lugar que la
diferencia CPU↔GPU está en el nivel del error de discretización, no del
modelo (`ESTADO.md` §5.4).

API pública
-----------
    BACKEND_NAMES, GPU_BACKENDS, DTYPES
    available_backends() -> dict[str, bool]
    why_unavailable(name) -> str
    gpu_info() -> dict
    get_backend(name="auto", precision="auto") -> Backend
    Backend(name, xp, dtype, note, is_gpu)
        .zeros .asarray .to_numpy .sync() .free_bytes() .summary()
    to_numpy(a) -> np.ndarray        (helper libre, acepta cupy o numpy)
    field_bytes(shape_tot, ndim, dtype, n_fields) -> int
    benchmark(n=96, steps=20, ndim=3, ...) -> dict
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

#: Orden de preferencia cuando `name="auto"`. CuPy primero porque es el
#: único backend con kernel escrito Y con GPU detrás en esta máquina.
#: Warp/JAX se conservan en la lista porque la wave los nombra, pero no
#: tienen kernel: si estuvieran instalados, degradarían a NumPy con nota.
BACKEND_NAMES = ("cupy", "warp", "jax", "numpy")

#: Backends que ejecutan en GPU.
GPU_BACKENDS = ("cupy",)

#: Alias de precisión → dtype de NumPy (CuPy acepta dtypes de NumPy).
DTYPES = {"fp32": np.float32, "fp64": np.float64}

#: Nombres alternativos aceptados para la precisión. Existen porque los
#: nombres de NumPy son los que uno escribe por reflejo, y antes caían al
#: default sin avisar.
PRECISION_ALIASES = {"float32": "fp32", "single": "fp32", "f32": "fp32",
                     "float64": "fp64", "double": "fp64", "f64": "fp64"}

#: Variable de entorno que fuerza el backend (útil en CI y en los tests).
ENV_BACKEND = "PHYCB_L2A_BACKEND"
ENV_PRECISION = "PHYCB_L2A_PRECISION"

#: Precisión por defecto en GPU. FP64 y no el FP32 de Ciencia §11.2 porque
#: la escalera V1–V3 está verificada ENTERA en FP64 y sólo a medias en FP32
#: (ver docstring del módulo y `ESTADO.md` §5.4). Ponerlo en "fp32" da ×1.8
#: y media VRAM en mallas grandes, y **exige** volver a correr V2 Re=1000 y
#: V3 a D/10 con los mismos umbrales antes de dejarlo así.
GPU_AUTO_PRECISION = "fp64"

#: Celdas por debajo de las cuales `backend="auto"` se queda en CPU aunque
#: haya GPU. ⚠️ TUNABLE MEDIDO (contrato §10.7) el 2026-07-31 con
#: `cfd/tests/run_perf_budget.py`, cubo 3D con esfera inmersa, rk3, fp64:
#:
#:    celdas    NumPy/CPU s/celda/paso   CuPy/GPU s/celda/paso   gana
#:      8 000        7.37e-07                7.68e-06            CPU x10
#:     32 768        5.51e-07                1.91e-06            CPU x3.5
#:     64 000        5.91e-07                9.05e-07            CPU x1.5
#:    140 608        2.00e-06                4.02e-07            GPU x5.0
#:    262 144        1.96e-06                2.04e-07            GPU x9.6
#:    512 000        2.01e-06                1.17e-07            GPU x17
#:  1 000 000        2.04e-06                9.66e-08            GPU x21
#:
#: El cruce está entre 64 k y 140 k celdas: por debajo, el paso lo domina el
#: coste de LANZAR kernels (unos cientos por paso RK3) y la GPU pierde; por
#: encima, el trabajo por kernel tapa ese coste. Se fija en 100 000 celdas,
#: dentro de la banda del cruce. Forzar GPU igualmente: `backend="cupy"` o
#: `PHYCB_L2A_BACKEND=cupy` (la escalera V&V se corre así a propósito).
GPU_MIN_CELLS = 100_000

#: Fracción máxima de la VRAM libre que el solver se permite reservar antes
#: de degradar a CPU con aviso. ⚠️ TUNABLE (contrato §10.7): 0.60 deja aire
#: para los temporales del kernel, que no se contabilizan campo a campo.
VRAM_SAFETY_FRAC = 0.60

_PROBE_CACHE: dict[str, tuple[bool, str]] = {}
_GPU_INFO_CACHE: dict | None = None


def _probe(name: str) -> tuple[bool, str]:
    """¿Está el backend `name` USABLE? Devuelve (disponible, motivo).

    Import perezoso y cacheado. **Nunca lanza**: cualquier excepción
    (módulo ausente, DLL de CUDA rota, driver viejo, GPU ocupada) se
    convierte en un motivo legible.

    Para CuPy no basta con que importe: se comprueba que hay dispositivo y
    que un kernel trivial corre de verdad. Un `import cupy` que funciona
    con una GPU inaccesible es exactamente el falso verde que hay que cazar.
    """
    if name in _PROBE_CACHE:
        return _PROBE_CACHE[name]
    ok, why = False, ""
    if name == "numpy":
        ok, why = True, ""
    elif name == "cupy":
        try:
            import cupy as cp
            if int(cp.cuda.runtime.getDeviceCount()) < 1:
                why = "cupy importa pero no hay dispositivo CUDA visible"
            else:
                # kernel real: import + device count no prueban que ejecute
                v = float(cp.arange(8, dtype=cp.float32).sum())
                ok = (v == 28.0)
                why = "" if ok else f"kernel de prueba dio {v} (esperado 28)"
        except Exception as exc:
            why = f"{type(exc).__name__}: {exc}"
    elif name == "warp":
        try:
            import warp  # noqa: F401
            ok, why = True, ""
        except Exception as exc:
            why = f"{type(exc).__name__}: {exc}"
    elif name == "jax":
        try:
            import jax  # noqa: F401
            ok, why = True, ""
        except Exception as exc:
            why = f"{type(exc).__name__}: {exc}"
    else:
        why = f"backend desconocido: {name!r}"
    _PROBE_CACHE[name] = (ok, why)
    return ok, why


def available_backends() -> dict[str, bool]:
    """Mapa nombre → disponible. La CLI lo consulta UNA vez, al arranque."""
    return {n: _probe(n)[0] for n in BACKEND_NAMES}


def why_unavailable(name: str) -> str:
    """Motivo legible de por qué `name` no está (cadena vacía si sí está)."""
    return _probe(name)[1]


def gpu_info() -> dict:
    """Ficha de la GPU (nombre, VRAM, compute capability, CUDA). **Nunca
    lanza**: sin GPU devuelve `dict(ok=False, reason=...)`."""
    global _GPU_INFO_CACHE
    if _GPU_INFO_CACHE is not None:
        return dict(_GPU_INFO_CACHE)
    ok, why = _probe("cupy")
    if not ok:
        _GPU_INFO_CACHE = dict(ok=False, reason=why)
        return dict(_GPU_INFO_CACHE)
    try:
        import cupy as cp
        dev = cp.cuda.Device(0)
        props = cp.cuda.runtime.getDeviceProperties(0)
        free_b, total_b = cp.cuda.runtime.memGetInfo()
        nm = props.get("name", b"")
        _GPU_INFO_CACHE = dict(
            ok=True,
            name=nm.decode("utf-8", "replace") if isinstance(nm, bytes)
            else str(nm),
            compute_capability=".".join(str(dev.compute_capability)),
            vram_total_GiB=total_b / 2 ** 30, vram_free_GiB=free_b / 2 ** 30,
            cupy_version=cp.__version__,
            cuda_runtime=int(cp.cuda.runtime.runtimeGetVersion()),
            cuda_driver=int(cp.cuda.runtime.driverGetVersion()))
    except Exception as exc:
        _GPU_INFO_CACHE = dict(ok=False,
                               reason=f"{type(exc).__name__}: {exc}")
    return dict(_GPU_INFO_CACHE)


def to_numpy(a) -> np.ndarray:
    """Copia a NumPy aceptando arrays de CuPy **o** de NumPy.

    Existe porque `np.asarray(cupy_array)` NO es una conversión silenciosa:
    CuPy lanza a propósito para que nadie mueva datos device→host sin
    enterarse. Todo el post-proceso pasa por aquí.
    """
    if isinstance(a, np.ndarray):
        return a
    get = getattr(a, "get", None)
    if callable(get):
        try:
            return np.asarray(get())
        except Exception:
            pass
    return np.asarray(a)


def field_bytes(shape_tot, ndim: int, dtype, n_fields: int = 24) -> int:
    """Bytes estimados de `n_fields` campos del tamaño `shape_tot`.

    `n_fields = 24` es el conteo REAL del bucle de M1 en 3D (u:3, p:1,
    máscara:1, estado RK u0/p0:4, y ~15 temporales vivos a la vez dentro de
    `rhs`). Sirve para el guardián de VRAM, no como cifra de mérito.
    """
    itm = int(np.dtype(dtype).itemsize)
    return int(np.prod(shape_tot)) * itm * int(n_fields)


@dataclass
class Backend:
    """Backend resuelto. `xp` es el módulo tipo-NumPy con el que trabaja el
    solver: `numpy` en CPU, `cupy` en GPU. El kernel del solver es el mismo
    código en los dos casos."""

    name: str
    xp: Any
    dtype: Any
    note: str = ""
    is_gpu: bool = False
    requested: str = "auto"
    extra: dict = field(default_factory=dict)

    def zeros(self, shape):
        return self.xp.zeros(shape, dtype=self.dtype)

    def asarray(self, a):
        """Sube `a` al dispositivo con el dtype del backend."""
        if isinstance(a, np.ndarray) or np.isscalar(a) or isinstance(a, list):
            return self.xp.asarray(np.asarray(a), dtype=self.dtype)
        return self.xp.asarray(a, dtype=self.dtype)

    def to_numpy(self, a) -> np.ndarray:
        """Copia a NumPy (identidad en el backend NumPy)."""
        return to_numpy(a)

    def sync(self) -> None:
        """Espera a que el dispositivo termine. **Obligatorio antes de
        cronometrar**: los lanzamientos de CuPy son asíncronos y sin esto se
        mediría el tiempo de encolar, no el de calcular. No-op en CPU."""
        if not self.is_gpu:
            return
        try:
            self.xp.cuda.runtime.deviceSynchronize()
        except Exception:
            pass

    def free_bytes(self) -> int:
        """VRAM libre en bytes (0 en CPU o si no se puede consultar)."""
        if not self.is_gpu:
            return 0
        try:
            return int(self.xp.cuda.runtime.memGetInfo()[0])
        except Exception:
            return 0

    def used_bytes(self) -> int:
        """Bytes actualmente en el pool de memoria de CuPy (0 en CPU)."""
        if not self.is_gpu:
            return 0
        try:
            return int(self.xp.get_default_memory_pool().used_bytes())
        except Exception:
            return 0

    def summary(self) -> dict:
        """Bloque trazable para el record y para `ESTADO.md`."""
        out = dict(backend=self.name, requested=self.requested,
                   dtype=np.dtype(self.dtype).name, is_gpu=bool(self.is_gpu),
                   note=self.note, available=available_backends())
        if self.is_gpu:
            g = gpu_info()
            out["gpu"] = dict(name=g.get("name", ""),
                              vram_free_GiB=g.get("vram_free_GiB", 0.0),
                              cupy=g.get("cupy_version", ""))
            out["determinism"] = ("reproducible con el MISMO backend; las "
                                  "reducciones GPU no son bit-identicas a "
                                  "las de CPU (orden de acumulacion)")
        return out


def get_backend(name: str = "auto", precision: str = "auto") -> Backend:
    """Resuelve el backend. **Nunca lanza.**

    * `name="auto"` recorre `BACKEND_NAMES` y se queda con el primero
      USABLE; en esta máquina eso es `cupy` (GPU).
    * Pedir `warp`/`jax` **degrada a NumPy con etiqueta**: no hay kernel
      escrito para ellos (y no están instalados). Pedir `cupy` sin GPU
      degrada igualmente. Nunca una excepción por dependencia ausente.
    * `precision="auto"` ⇒ fp64 en CPU y fp64 en GPU (ver docstring del
      módulo: la V&V no converge a tolerancia en FP32).
    """
    req = str(os.environ.get(ENV_BACKEND, name) or "auto").lower()
    prec = str(os.environ.get(ENV_PRECISION, precision) or "auto").lower()
    note = ""

    if req == "auto":
        chosen = "numpy"
        for cand in BACKEND_NAMES:
            if _probe(cand)[0]:
                chosen = cand
                break
        if chosen == "numpy":
            miss = {n: why_unavailable(n) for n in ("cupy", "warp", "jax")}
            note = ("auto→numpy: " + "; ".join(
                f"{k} no disponible ({v.split(':')[0]})"
                for k, v in miss.items()))
    else:
        ok, why = _probe(req)
        chosen = req if ok else "numpy"
        if not ok:
            note = f"backend '{req}' pedido pero no disponible ({why}) → numpy"

    # Warp y JAX: aunque estuvieran instalados, el kernel del solver está
    # escrito con la API de NumPy/CuPy. Se declara para no prometer de más.
    if chosen in ("warp", "jax"):
        note = (f"'{chosen}' importable pero el solver no tiene kernel para "
                "el; el camino GPU de este proyecto es CuPy → se usa NumPy.")
        chosen = "numpy"

    is_gpu = chosen in GPU_BACKENDS
    if prec == "auto":
        prec = GPU_AUTO_PRECISION if is_gpu else "fp64"
    prec = PRECISION_ALIASES.get(prec, prec)
    if prec not in DTYPES:
        # degradación ETIQUETADA: pedir una precisión con un nombre que no
        # se reconoce daba fp64 EN SILENCIO, y un caso que se cree en fp32
        # y corre en fp64 miente sobre su coste (cazado el 2026-08-03
        # pidiendo "float32" en el caso M4 a dx = 0.8 mm).
        note = (note + "; " if note else "") + (
            f"precision '{prec}' no reconocida (validas: "
            f"{sorted(DTYPES) + sorted(PRECISION_ALIASES)}) → fp64")
        prec = "fp64"
    dtype = DTYPES[prec]

    xp = np
    if is_gpu:
        try:
            import cupy as cp
            xp = cp
        except Exception as exc:            # carrera improbable, no lanza
            note = f"cupy dejó de estar usable ({type(exc).__name__}) → numpy"
            chosen, is_gpu, xp = "numpy", False, np
    return Backend(name=chosen, xp=xp, dtype=dtype, note=note, is_gpu=is_gpu,
                   requested=req)


def fits_in_vram(be: Backend, shape_tot, ndim: int,
                 n_fields: int = 24) -> tuple[bool, str]:
    """¿Cabe el caso en la VRAM libre con el margen `VRAM_SAFETY_FRAC`?

    Devuelve `(cabe, motivo)`. En CPU siempre cabe (es cosa del SO). Es un
    guardián, no una promesa: los temporales del kernel no se cuentan uno a
    uno, por eso el margen.
    """
    if not be.is_gpu:
        return True, ""
    need = field_bytes(shape_tot, ndim, be.dtype, n_fields)
    free = be.free_bytes()
    if free <= 0:
        return True, ""
    if need > VRAM_SAFETY_FRAC * free:
        return False, (f"el caso necesita ~{need / 2 ** 20:.0f} MiB y solo "
                       f"hay {free / 2 ** 20:.0f} MiB libres de VRAM "
                       f"(margen {VRAM_SAFETY_FRAC:.0%})")
    return True, ""


# ==========================================================================
# Presupuesto de rendimiento (WAVE6 §5, Ciencia §11.2)
# ==========================================================================
#: Referencia pública citada por la wave: PID de JAX-Fluids ≈ 1e-8 s/celda/paso
#: en FP32 sobre GPU. Es la vara de medir, NO una promesa de este backend.
REF_S_PER_CELL_STEP_FP32_GPU = 1.0e-8

#: Objetivo v1 de la wave: 500k celdas × 10k iteraciones.
TARGET_CELLS = 500_000
TARGET_STEPS = 10_000


def benchmark(n: int = 96, steps: int = 20, ndim: int = 3,
              precision: str = "fp32", backend: str = "auto",
              warmup: int = 3) -> dict:
    """Kernel sintético de advección-difusión (el mismo que la wave manda
    usar para elegir backend) sobre `n**ndim` celdas.

    Mide **s/celda/paso**, que es la única cifra comparable con la
    referencia pública. Devuelve además cuánto costaría el objetivo v1
    (500k celdas × 10k pasos) a ese ritmo.

    En GPU hace `warmup` pasos sin cronometrar (la primera llamada compila
    y reserva el pool) y **sincroniza** antes y después de medir. Sin eso se
    cronometraría el encolado, no el cálculo. **Nunca lanza.**
    """
    be = get_backend(backend, precision)
    xp, dt_ = be.xp, be.dtype
    shape = (n,) * int(ndim)
    ncell = int(np.prod(shape))
    try:
        rng = np.random.default_rng(71)          # SEED de la familia
        a = xp.asarray(rng.random(shape), dtype=dt_)
        u = [xp.asarray(rng.random(shape), dtype=dt_) for _ in range(ndim)]
        nu = dt_(1.0e-3)

        def _sweep(a):
            out = a.copy()
            for d in range(ndim):
                ap = xp.roll(a, -1, axis=d)
                am = xp.roll(a, 1, axis=d)
                out -= u[d] * (ap - am) * dt_(0.5)      # advección centrada
                out += nu * (ap - dt_(2.0) * a + am)    # difusión
            return out

        for _ in range(int(max(0, warmup)) if be.is_gpu else 0):
            a = _sweep(a)
        be.sync()
        t0 = time.perf_counter()
        for _ in range(int(max(1, steps))):
            a = _sweep(a)
        be.sync()
        el = time.perf_counter() - t0
        used = be.used_bytes()
    except Exception as exc:                     # degradación, nunca excepción
        return dict(ok=False, error=f"{type(exc).__name__}: {exc}",
                    backend=be.name)
    s_cell_step = el / (ncell * max(1, steps))
    target_s = s_cell_step * TARGET_CELLS * TARGET_STEPS
    return dict(ok=True, backend=be.name, is_gpu=bool(be.is_gpu),
                dtype=np.dtype(dt_).name,
                cells=ncell, steps=int(steps), elapsed_s=el,
                s_per_cell_step=s_cell_step,
                Mcell_steps_per_s=1e-6 / s_cell_step,
                ratio_vs_ref_gpu=s_cell_step / REF_S_PER_CELL_STEP_FP32_GPU,
                target_cells=TARGET_CELLS, target_steps=TARGET_STEPS,
                target_s=target_s, target_h=target_s / 3600.0,
                vram_used_MiB=used / 2 ** 20, note=be.note)


def main(argv=None) -> int:
    """`python cfd/ibrans/backend.py` → tabla de disponibilidad + benchmark
    del kernel sintético en TODOS los backends usables."""
    import sys
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    print("Backends disponibles:", available_backends())
    for nm in BACKEND_NAMES:
        w = why_unavailable(nm)
        if w:
            print(f"  {nm}: {w}")
    g = gpu_info()
    if g.get("ok"):
        print(f"  GPU: {g['name']} · cc {g['compute_capability']} · "
              f"{g['vram_free_GiB']:.2f}/{g['vram_total_GiB']:.2f} GiB libres "
              f"· cupy {g['cupy_version']} · CUDA rt {g['cuda_runtime']}")
    print(f"\nKernel sintetico adveccion-difusion 64^3 "
          f"(referencia GPU publica {REF_S_PER_CELL_STEP_FP32_GPU:.0e} "
          "s/celda/paso)")
    for bk in ("numpy", "cupy"):
        if not _probe(bk)[0]:
            continue
        for prec in ("fp32", "fp64"):
            r = benchmark(n=64, steps=20, ndim=3, precision=prec, backend=bk)
            if r.get("ok"):
                print(f"  {bk:5s} {prec}: {r['s_per_cell_step']:.3e} "
                      f"s/celda/paso · objetivo 500k×10k = "
                      f"{r['target_h']:.3f} h · "
                      f"x{r['ratio_vs_ref_gpu']:.1f} vs referencia GPU")
            else:
                print(f"  {bk:5s} {prec}: FALLO {r.get('error')}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
