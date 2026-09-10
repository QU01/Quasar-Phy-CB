"""
QUASAR Phy-CB · neural_optimizer.py
===================================
Capas 2–4 del Computational Engineering Model: SURROGATE con incertidumbre,
BÚSQUEDA INVERSA multiobjetivo y ACTIVE LEARNING bayesiano.

PROCEDENCIA — COPIA VERBATIM de `Phy-AC/neural_optimizer.py` (estado del
repo el 2026-07-30, 812 líneas), que a su vez está portado casi literal de
Phy-CC. El motor es **agnóstico al dominio**; siguiendo WAVE0 §4 solo se ha
escrito el ADAPTADOR DE DOMINIO. Lo que se tocó, y nada más:

  (a) imports de dominio y `OUT_KEYS`  → objetivos del combustor;
  (b) `EnsembleSurrogate._targets` / transformadas log / `gate_ok`;
  (c) `DesignSpec` completo            → espec de combustor (WAVE4 §1.1);
  (d) el `perf` LCB de `NSGA2._eval_pop` (mismas líneas, claves nuevas);
  (e) nombre de la clase orquestadora y textos de log;
  (f) `scalar_fitness`, `evaluate_design`, `design`, `__main__`.

NO se tocó: MLPs, Adam, early stopping, bootstrap, quality gate (solo sus
umbrales por clave), NSGA-II, dominancia restringida de Deb, crowding,
adquisición LCB + k-means, orquestador, checkpointing, `load()`/`save()`.

DESVIACIÓN DOCUMENTADA (una sola): en `NSGA2._eval_pop`, Phy-AC copiaba del
record L0 una tupla corta `_REC0_KEYS`; aquí `perf` PARTE del record L0
completo y luego se sobreescriben las salidas con los valores LCB del
surrogate. Es la misma semántica llevada al límite (el combustor necesita
~30 claves del L0 para g_struct + g_dfam, no 7) y deja el algoritmo intacto.

Dado únicamente un DesignSpec (P3, T3, ṁ, T4, combustible, Δp máximo),
devuelve el frente de Pareto factible y el diseño óptimo VERIFICADO con la
física real — sin que el ingeniero proponga geometría.

Todo NumPy puro: corre en cualquier laptop, sin GPU.
"""

from __future__ import annotations

import json
import math
import time

import numpy as np

from physics_core import (
    Fidelity, evaluate, evaluate_batch, physics_features,
    denormalize, normalize, VAR_NAMES, NDIM, NDIM_LEGACY, N_FEATURES,
    N_CONSTRAINTS, pad_theta, FUELS, ARCH_NAMES,
)
import structures_core

SEED = 71  # semilla de la familia (contrato §10.8)
rng = np.random.default_rng(SEED)

#: salidas que aprende el ensemble (WAVE4 §1.5). Orden CONGELADO: el índice
#: 4 (EI_NOx) se aprende en log — abarca órdenes de magnitud.
OUT_KEYS = ["eta_comb", "dp_over_p", "PF", "T_wall_max_K", "EI_NOx"]

#: normalización de EI_NOx en los objetivos [g/kg de combustible]. NO es una
#: calibración física: solo pone f2 en O(1) para que el crowding de NSGA-II
#: no quede dominado por una escala. 10 g/kg es el orden típico de un
#: combustor convencional (Ciencia §8).
EI_NOX_REF = 10.0

#: peso de NOx en el fitness escalar de reporte (10 g/kg ≈ 2 puntos de η).
W_NOX_FITNESS = 0.02

#: claves del record L0 que la dominancia restringida necesita además de las
#: predicciones del surrogate. Se conserva por compatibilidad con el patrón
#: de los hermanos; el `_eval_pop` de Phy-CB copia el record entero (ver la
#: desviación documentada en el docstring del módulo).
_REC0_KEYS = ("g", "T_w1_K", "T_w2_K", "D_L_mm", "D_casing_mm", "L_tot_mm",
              "dp_liner_Pa", "phi_pz", "tau_ms")


# ==========================================================================
# 1. MLP base (NumPy, Adam, early stopping) — portado verbatim de Phy-CC
# ==========================================================================
class _Adam:
    """Adam sobre una lista de parámetros NumPy, actualizados in-place."""

    def __init__(self, params, lr=2e-3):
        self.params = list(params)
        self.lr = lr
        self.m = [np.zeros_like(p) for p in self.params]
        self.v = [np.zeros_like(p) for p in self.params]
        self.t = 0

    def step(self, grads):
        self.t += 1
        bias1 = 1.0 - 0.9 ** self.t
        bias2 = 1.0 - 0.999 ** self.t
        for p, g, m, v in zip(self.params, grads, self.m, self.v):
            m[:] = 0.9 * m + 0.1 * g
            v[:] = 0.999 * v + 0.001 * g ** 2
            p -= self.lr * (m / bias1) / (np.sqrt(v / bias2) + 1e-8)


class _MLP:
    def __init__(self, d_in, d_out, hidden=(96, 96, 64), seed=0):
        r = np.random.default_rng(seed)
        sizes = [d_in, *hidden, d_out]
        self.W, self.b = [], []
        for i in range(len(sizes) - 1):
            scale = np.sqrt(2.0 / (sizes[i] + sizes[i + 1]))
            self.W.append(r.normal(0, scale, (sizes[i], sizes[i + 1])))
            self.b.append(np.zeros(sizes[i + 1]))

    @staticmethod
    def _act(z):                       # SiLU
        s = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        return z * s

    @staticmethod
    def _act_grad(z):
        s = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        return s * (1 + z * (1 - s))

    def forward(self, X, cache=False):
        a, acts, zs = X, [X], []
        for i in range(len(self.W)):
            z = a @ self.W[i] + self.b[i]
            zs.append(z)
            a = self._act(z) if i < len(self.W) - 1 else z
            acts.append(a)
        return (a, acts, zs) if cache else a

    def _backward(self, acts, zs, out, yb, weight_decay):
        delta = 2.0 * (out - yb) / out.shape[0]
        gW = [None] * len(self.W)
        gb = [None] * len(self.b)
        for i in reversed(range(len(self.W))):
            gW[i] = acts[i].T @ delta + weight_decay * self.W[i]
            gb[i] = delta.sum(axis=0)
            if i > 0:
                delta = (delta @ self.W[i].T) * self._act_grad(zs[i - 1])
        return gW, gb

    def train(self, X, Y, Xv, Yv, epochs=600, lr=2e-3, batch=64,
              weight_decay=1e-5, patience=60, seed=0):
        r = np.random.default_rng(seed)
        opt = _Adam(self.W + self.b, lr=lr)
        best_val, best_state, since = np.inf, None, 0
        n = X.shape[0]
        for _ep in range(epochs):
            idx = r.permutation(n)
            for s in range(0, n, batch):
                bi = idx[s:s + batch]
                xb, yb = X[bi], Y[bi]
                out, acts, zs = self.forward(xb, cache=True)
                gW, gb = self._backward(acts, zs, out, yb, weight_decay)
                opt.step(gW + gb)
            val = float(np.mean((self.forward(Xv) - Yv) ** 2))
            if val < best_val - 1e-6:
                best_val, since = val, 0
                best_state = ([w.copy() for w in self.W],
                              [b.copy() for b in self.b])
            else:
                since += 1
                if since >= patience:
                    break
        if best_state:
            self.W, self.b = best_state
        return best_val


# ==========================================================================
# 2. Deep ensemble con residual learning sobre L0
# ==========================================================================
class EnsembleSurrogate:
    """K MLPs entrenados sobre bootstraps distintos. Predice el RESIDUAL
    entre la verdad física (L1/L2 calibrada) y el meanline L0 — el L0 ya
    viene embebido en los features de entrada (physics_features)."""

    def __init__(self, K=5, seed=SEED):
        self.K = K
        self.seed = int(seed)
        self.members: list[_MLP] = []
        self.y_mean = self.y_std = None
        self.metrics: dict = {}

    @staticmethod
    def _targets(records: list[dict]) -> np.ndarray:
        return np.array([[r["eta_comb"], r["dp_over_p"], r["PF"],
                          r["T_wall_max_K"], r["EI_NOx"]] for r in records])

    def fit(self, X: np.ndarray, Y: np.ndarray, log=print, rand=None):
        r = rand if rand is not None else rng
        Yt = Y.copy()
        Yt[:, 4] = np.log(np.maximum(Yt[:, 4], 1e-3))      # EI_NOx -> log
        self.y_mean = Yt.mean(axis=0)
        self.y_std = Yt.std(axis=0) + 1e-9
        Yn = (Yt - self.y_mean) / self.y_std

        n = X.shape[0]
        n_val = max(int(0.15 * n), 8)
        perm = r.permutation(n)
        vi, ti = perm[:n_val], perm[n_val:]
        Xv, Yv = X[vi], Yn[vi]

        self.members = []
        for k in range(self.K):
            boot = r.choice(ti, size=len(ti), replace=True)
            m = _MLP(X.shape[1], Yn.shape[1], seed=self.seed + 13 * k)
            m.train(X[boot], Yn[boot], Xv, Yv, seed=self.seed + 7 * k)
            self.members.append(m)

        # ---- Quality gate: R² en holdout + cobertura de incertidumbre ----
        mu, sd = self._predict_norm(Xv)
        r2 = {}
        for i, key in enumerate(OUT_KEYS):
            ss_res = np.sum((Yv[:, i] - mu[:, i]) ** 2)
            ss_tot = np.sum((Yv[:, i] - Yv[:, i].mean()) ** 2) + 1e-12
            r2[key] = float(1 - ss_res / ss_tot)
        inside = np.abs(Yv - mu) <= 2.0 * np.maximum(sd, 1e-6)
        coverage = float(inside.mean())
        self.metrics = dict(r2=r2, coverage_2sigma=coverage, n_train=n)
        log(f"      Surrogate R²: " +
            "  ".join(f"{k}={v:.3f}" for k, v in r2.items()) +
            f"  | coverage ±2σ = {coverage:.2f}")
        return self.metrics

    def gate_ok(self) -> bool:
        """Gate de calidad de la familia: R²(objetivo primario) ≥ 0.90,
        R²(secundario) ≥ 0.85, cobertura ±2σ ∈ [0.80, 1.0]. En Phy-CB el
        primario es η_comb y el secundario Δp/p (WAVE4 T39: si falla, se
        enriquecen los physics_features — NO se baja el umbral)."""
        r2 = self.metrics.get("r2", {})
        cov = self.metrics.get("coverage_2sigma", 0.0)
        return (r2.get("eta_comb", 0) >= 0.90
                and r2.get("dp_over_p", 0) >= 0.85
                and 0.80 <= cov <= 1.0)

    def _predict_norm(self, X):
        preds = np.stack([m.forward(X) for m in self.members])  # K x n x d
        return preds.mean(axis=0), preds.std(axis=0)

    def predict(self, X):
        """Devuelve (mu, sigma) en unidades físicas."""
        mu_n, sd_n = self._predict_norm(np.atleast_2d(X))
        mu = mu_n * self.y_std + self.y_mean
        sd = sd_n * self.y_std
        mu[:, 4], sd[:, 4] = np.exp(mu[:, 4]), np.exp(mu[:, 4]) * sd[:, 4]
        return mu, sd


# ==========================================================================
# 3. Especificación y objetivos
# ==========================================================================
#: aviso único (patrón «degrada una vez, en voz alta») del checker DfAM
_DFAM_WARNED = False


def _dfam_module():
    """Import perezoso del checker DfAM (WAVE3 §1.3). Devuelve None si el
    módulo aún no existe — el sistema sigue corriendo sin él, avisando una
    sola vez (invariante 3 de CONTEXTO §2)."""
    global _DFAM_WARNED
    try:
        import dfam_checker
        return dfam_checker
    except Exception:
        if not _DFAM_WARNED:
            _DFAM_WARNED = True
            print("  ⚠ dfam_checker no disponible → restricciones DfAM "
                  "desactivadas (el diseño puede no ser imprimible).")
        return None


class DesignSpec:
    """Lo único que el ingeniero escribe (WAVE4 §1.1).

    El punto de operación del combustor son cuatro números que salen del
    ciclo (o directamente de un run de Phy-CC/Phy-AC): `P3_Pa`, `T3_K`,
    `mdot_kgs` a la entrada y la `T4_K` que la turbina pide.

    `dp_max`: restricción DURA de ciclo (2–4 % en microturbina, Ciencia
    §2.1). En un combustor Δp se GASTA en mezcla hasta el máximo permitido,
    NO se minimiza — por eso entra como restricción y no como objetivo
    (decisión de WAVE4 §1.2).

    `arch`: None deja la arquitectura como variable de diseño (gate θ[0]);
    "can"/"annular"/"reverse_annular" la fija.

    `material`: aleación de la pared CALIENTE (liner). Activa las 5
    restricciones de structures_core dentro de la dominancia restringida.
    `material=None` las desactiva (solo aero — para estudios/debug).
    `material_casing` es la pared FRÍA (puede ser IN718; el liner no).

    `dfam=True` activa las 5 restricciones de fabricabilidad LPBF.

    `fixed_vars`: dict {nombre: valor} de variables de DISEÑO que el
    ingeniero fija (p. ej. {"swirl_SN": 0.8, "t_liner_mm": 1.0}); el
    optimizador explora solo el resto. El punto de operación y el
    combustible se fijan con los campos del spec, no aquí."""

    #: variables cuyo valor lo pone el spec, no `fixed_vars`
    _OP_NAMES = ("P3_Pa", "T3_K", "mdot_kgs", "T4_K", "fuel")

    def __init__(self, P3_Pa=250_000.0, T3_K=430.0, mdot_kgs=0.13,
                 T4_K=1150.0, fuel="jet_a", dp_max=0.04, arch=None,
                 material="Inconel-625", material_casing="Inconel-718",
                 PF_max=0.40, envelope_D_mm=None, envelope_L_mm=None,
                 dfam=True, fixed_vars=None):
        self.P3_Pa = float(P3_Pa)
        self.T3_K = float(T3_K)
        self.mdot_kgs = float(mdot_kgs)
        self.T4_K = float(T4_K)
        if fuel not in FUELS:
            raise ValueError(f"combustible '{fuel}' no existe — disponibles: "
                             f"{', '.join(FUELS)}")
        self.fuel = fuel
        self.dp_max = float(dp_max)
        if arch is not None and arch not in ARCH_NAMES:
            raise ValueError(f"arquitectura '{arch}' no existe — disponibles: "
                             f"{', '.join(ARCH_NAMES)} (o None = variable)")
        self.arch = arch
        if material is not None and material not in structures_core.MATERIALS:
            raise ValueError(
                f"material '{material}' no existe — disponibles: "
                f"{', '.join(structures_core.list_materials())} (o None)")
        self.material = material
        self.material_casing = material_casing
        self.PF_max = float(PF_max)
        self.envelope_D_mm = envelope_D_mm
        self.envelope_L_mm = envelope_L_mm
        self.dfam = bool(dfam)
        self.fixed_vars = {k: float(v) for k, v in (fixed_vars or {}).items()}
        design_names = [n for n in VAR_NAMES if n not in self._OP_NAMES]
        for name in self.fixed_vars:
            if name in self._OP_NAMES:
                raise ValueError(
                    f"'{name}' es punto de operación — se fija con los "
                    "campos P3_Pa/T3_K/mdot_kgs/T4_K/fuel del spec, no con "
                    "fixed_vars")
            if name not in design_names:
                raise ValueError(
                    f"variable '{name}' no existe — fijables: "
                    f"{', '.join(design_names)}")
        self._fixed_idx = [(VAR_NAMES.index(k), v)
                           for k, v in self.fixed_vars.items()]

    def fix_operating_point(self, theta: np.ndarray) -> np.ndarray:
        """Fija el punto de operación del spec, el combustible, la
        arquitectura (si el spec la impone) y las fixed_vars. TODA
        evaluación (LHS, NSGA-II, adquisición) pasa por aquí, así que fijar
        una variable colapsa esa dimensión."""
        theta = pad_theta(theta).copy()
        theta[16] = self.P3_Pa
        theta[17] = self.T3_K
        theta[18] = self.mdot_kgs
        theta[19] = self.T4_K
        theta[15] = float(list(FUELS).index(self.fuel))
        if self.arch is not None:
            theta[0] = float(ARCH_NAMES.index(self.arch))
        for idx, val in self._fixed_idx:
            theta[idx] = val
        return theta

    # ---- objetivos (a MINIMIZAR) y restricciones (g<=0) sobre un record --
    def objectives(self, perf: dict) -> np.ndarray:
        """Bi-objetivo (WAVE4 §1.2): maximizar η_comb y minimizar EI_NOx.
        Δp/p NO es objetivo — es restricción dura (`dp_max`)."""
        f1 = -perf["eta_comb"]                                    # max η
        f2 = perf["EI_NOx"] / EI_NOX_REF                          # min NOx
        return np.array([f1, f2])

    def constraints(self, perf: dict, theta: np.ndarray) -> np.ndarray:
        """g global (contrato §5.4):
        [g_aero (12) | g_spec (4) | g_struct (5) | g_dfam (5)]

        `g_spec` tiene longitud FIJA 4 aunque la envolvente no se declare
        (entonces vale −1.0, holgadamente factible): la dominancia de Deb
        compara vectores, y una longitud variable entre individuos rompería
        la comparación.

        g_struct y g_dfam son EXACTOS (se calculan sobre el record L0 real
        del individuo, ~1 ms), no predicciones del surrogate."""
        g_phys = np.asarray(perf.get("g", [0.0] * N_CONSTRAINTS), dtype=float)
        env_d = self.envelope_D_mm
        env_l = self.envelope_L_mm
        g_spec = [
            perf.get("dp_over_p", 0.0) / max(self.dp_max, 1e-9) - 1.0,
            perf.get("PF", 0.0) / max(self.PF_max, 1e-9) - 1.0,
            (perf.get("D_casing_mm", 0.0) / env_d - 1.0) if env_d else -1.0,
            (perf.get("L_tot_mm", 0.0) / env_l - 1.0) if env_l else -1.0,
        ]
        parts = [g_phys, np.array(g_spec, dtype=float)]
        if self.material is not None:
            try:
                s = structures_core.evaluate_structural(
                    theta, perf, material=self.material,
                    material_casing=self.material_casing)
                parts.append(np.asarray(s["g_struct"], dtype=float))
            except Exception:
                # theta degenerado: violación grande y continua, no crash
                parts.append(np.full(structures_core.N_STRUCT_CONSTRAINTS,
                                     10.0))
        if self.dfam:
            dc = _dfam_module()
            if dc is None:
                # módulo ausente: longitud conservada, restricción neutra
                parts.append(np.full(5, -1.0))
            else:
                try:
                    import geometry_generator as gg
                    contract = gg.build_contract(theta, perf, None, None)
                    parts.append(np.asarray(
                        dc.check(contract)["g_dfam"], dtype=float))
                except Exception:
                    parts.append(np.full(dc.N_DFAM_CONSTRAINTS, 10.0))
        return np.concatenate(parts)

    @staticmethod
    def violation(g: np.ndarray) -> float:
        return float(np.sum(np.maximum(g, 0.0)))


# ==========================================================================
# 4. NSGA-II con dominancia restringida de Deb — portado verbatim
# ==========================================================================
def _dominates(f1, v1, f2, v2) -> bool:
    """Dominancia restringida: factible >> infactible; entre infactibles
    gana la menor violación; entre factibles, Pareto clásico."""
    feas1, feas2 = v1 <= 1e-9, v2 <= 1e-9
    if feas1 and not feas2:
        return True
    if feas2 and not feas1:
        return False
    if not feas1 and not feas2:
        return v1 < v2
    return bool(np.all(f1 <= f2) and np.any(f1 < f2))


def _fast_nondominated_sort(F, V):
    n = len(F)
    S = [[] for _ in range(n)]
    n_dom = np.zeros(n, dtype=int)
    fronts = [[]]
    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if _dominates(F[p], V[p], F[q], V[q]):
                S[p].append(q)
            elif _dominates(F[q], V[q], F[p], V[p]):
                n_dom[p] += 1
        if n_dom[p] == 0:
            fronts[0].append(p)
    i = 0
    while fronts[i]:
        nxt = []
        for p in fronts[i]:
            for q in S[p]:
                n_dom[q] -= 1
                if n_dom[q] == 0:
                    nxt.append(q)
        i += 1
        fronts.append(nxt)
    return fronts[:-1]


def _crowding(F, idxs):
    m = F.shape[1]
    d = np.zeros(len(idxs))
    for j in range(m):
        order = np.argsort(F[idxs, j])
        d[order[0]] = d[order[-1]] = np.inf
        span = F[idxs[order[-1]], j] - F[idxs[order[0]], j] + 1e-12
        for k in range(1, len(idxs) - 1):
            d[order[k]] += (F[idxs[order[k + 1]], j]
                            - F[idxs[order[k - 1]], j]) / span
    return d


class NSGA2:
    """NSGA-II sobre el surrogate (objetivos LCB-conservadores)."""

    def __init__(self, spec: DesignSpec, surrogate: EnsembleSurrogate,
                 pop=96, generations=80, kappa=1.0, seed=SEED):
        self.spec, self.sur = spec, surrogate
        self.pop, self.gens, self.kappa = pop, generations, kappa
        self.rng = np.random.default_rng(seed)

    # -- evaluación barata de una población (solo surrogate + L0 features) -
    def _eval_pop(self, U):
        thetas = np.array([self.spec.fix_operating_point(denormalize(u))
                           for u in U])
        recs0 = [evaluate(t, fidelity=Fidelity.L0, use_cache=False,
                          calibrate=False)
                 for t in thetas]
        X = np.array([physics_features(t, rec)
                      for t, rec in zip(thetas, recs0)])
        mu, sd = self.sur.predict(X)
        F = np.empty((len(U), 2))
        V = np.empty(len(U))
        Sig = np.empty(len(U))
        for i, (t, m, s, rec0) in enumerate(zip(thetas, mu, sd, recs0)):
            # LCB conservador: η penalizada por σ; Δp/p, PF, T_pared y NOx
            # ENSANCHADOS por σ (todo lo que perjudica se supone peor).
            # `perf` parte del record L0 completo — ver la desviación
            # documentada en el docstring del módulo; _REC0_KEYS es el
            # subconjunto que la dominancia necesita como mínimo.
            perf = dict(rec0)
            perf.update(eta_comb=m[0] - self.kappa * s[0],
                        dp_over_p=m[1] + self.kappa * s[1],
                        PF=m[2] + self.kappa * s[2],
                        T_wall_max_K=m[3] + self.kappa * s[3],
                        EI_NOx=m[4] + self.kappa * s[4])
            F[i] = self.spec.objectives(perf)
            V[i] = self.spec.violation(self.spec.constraints(perf, t))
            Sig[i] = float(np.mean(s / (np.abs(m) + 1e-9)))
        return thetas, F, V, Sig

    def run(self):
        d = NDIM
        P = self.rng.random((self.pop, d))
        thetas, F, V, Sig = self._eval_pop(P)
        for _g in range(self.gens):
            children = np.empty_like(P)
            fronts = _fast_nondominated_sort(F, V)
            rank = np.empty(self.pop, dtype=int)
            for fi, fr in enumerate(fronts):
                rank[fr] = fi
            for i in range(self.pop):
                a, b = self.rng.integers(self.pop, size=2)
                p1 = a if (rank[a], V[a]) < (rank[b], V[b]) else b
                a, b = self.rng.integers(self.pop, size=2)
                p2 = a if (rank[a], V[a]) < (rank[b], V[b]) else b
                w = self.rng.random(d)
                child = w * P[p1] + (1 - w) * P[p2]
                mut = self.rng.random(d) < (1.0 / d) * 2.0
                child[mut] += self.rng.normal(0, 0.08, mut.sum())
                children[i] = np.clip(child, 0, 1)
            cthetas, cF, cV, cSig = self._eval_pop(children)
            allP = np.vstack([P, children])
            allF = np.vstack([F, cF])
            allV = np.concatenate([V, cV])
            allS = np.concatenate([Sig, cSig])
            allT = np.vstack([thetas, cthetas])
            fronts = _fast_nondominated_sort(allF, allV)
            keep = []
            for fr in fronts:
                if len(keep) + len(fr) <= self.pop:
                    keep.extend(fr)
                else:
                    cd = _crowding(allF, np.array(fr))
                    order = np.argsort(-cd)
                    keep.extend(np.array(fr)[order[: self.pop - len(keep)]])
                    break
            keep = np.array(keep)
            P, F, V, Sig, thetas = (allP[keep], allF[keep], allV[keep],
                                    allS[keep], allT[keep])
        fronts = _fast_nondominated_sort(F, V)
        pareto = np.array(fronts[0])
        return dict(U=P, thetas=thetas, F=F, V=V, Sig=Sig, pareto_idx=pareto)


# ==========================================================================
# 5. Adquisición: LCB optimista + diversidad k-means — portado verbatim
# ==========================================================================
def select_acquisition_batch(result: dict, spec: DesignSpec,
                             batch_size=14, explore_frac=0.3,
                             seed=SEED) -> np.ndarray:
    """Elige el batch a evaluar con la física real: mezcla de (a) frente de
    Pareto predicho (explotación) y (b) puntos de alta incertidumbre
    relativa (exploración), des-duplicados por k-means."""
    r = np.random.default_rng(seed)
    U, F, V, Sig = result["U"], result["F"], result["V"], result["Sig"]
    pareto = result["pareto_idx"]

    n_explore = max(1, int(batch_size * explore_frac))

    cand = list(pareto[np.argsort(V[pareto])])
    near = np.where(V <= np.quantile(V, 0.5))[0]
    cand_exp = list(near[np.argsort(-Sig[near])][: 4 * n_explore])

    pool_idx = list(dict.fromkeys(cand + cand_exp))   # únicos, ordenados
    pool = U[pool_idx]
    if len(pool) <= batch_size:
        return pool

    k = batch_size
    centers = pool[r.choice(len(pool), size=k, replace=False)]
    for _ in range(12):
        dist = np.linalg.norm(pool[:, None, :] - centers[None], axis=2)
        lab = dist.argmin(axis=1)
        for j in range(k):
            pts = pool[lab == j]
            if len(pts):
                centers[j] = pts.mean(axis=0)
    chosen_idx: list[int] = []
    for j in range(k):
        members = np.where(lab == j)[0]
        if len(members):
            chosen_idx.append(int(members[0]))
    for i in range(len(pool)):
        if len(chosen_idx) >= batch_size:
            break
        if i not in chosen_idx:
            chosen_idx.append(i)
    return pool[np.array(chosen_idx[:batch_size])]


# ==========================================================================
# 6. Orquestador: diseño inverso autónomo con active learning
# ==========================================================================
class AutonomousCombustorDesigner:
    def __init__(self, spec: DesignSpec, fidelity=Fidelity.L1,
                 n_workers=1, log=print, seed=SEED):
        self.spec = spec
        self.fidelity = fidelity
        self.n_workers = n_workers
        self.log = log
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.sur = EnsembleSurrogate(K=5, seed=self.seed)
        self.X: list[np.ndarray] = []     # features
        self.recs: list[dict] = []        # records físicos
        self.history: list[dict] = []

    @classmethod
    def load(cls, path: str, fidelity=Fidelity.L1, n_workers=1,
             log=print, seed: int | None = None) -> "AutonomousCombustorDesigner":
        """Reconstruye un designer desde un checkpoint de save() (warm
        start). El dataset guardado solo trae theta + salidas; el record
        completo (splits, térmico, g_struct...) se reconstruye re-evaluando el
        L0 (~1 ms/punto) y se sobreescriben las salidas con las guardadas
        (que pueden ser L1/calibradas) — la verdad del flywheel se
        conserva. `seed=None` continúa con la semilla del checkpoint."""
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        spec_kw = {k: v for k, v in payload["spec"].items()
                   if not k.startswith("_")}
        spec = DesignSpec(**spec_kw)
        d = cls(spec, fidelity=fidelity, n_workers=n_workers, log=log,
                seed=payload.get("seed", SEED) if seed is None else seed)
        n_legacy = sum(1 for row in payload.get("dataset", [])
                       if len(row["theta"]) == NDIM_LEGACY)
        if n_legacy:
            log(f"[load] checkpoint legacy: {n_legacy} θ de "
                f"{NDIM_LEGACY}-D paddeados a {NDIM}-D (pendientes 0 — "
                "bit-exacto con el espacio pre-fase-8).")
        for row in payload.get("dataset", []):
            t = pad_theta(np.asarray(row["theta"], dtype=float))
            rec = evaluate(t, fidelity=Fidelity.L0, use_cache=False,
                           calibrate=False)
            rec.update({k: float(row[k]) for k in OUT_KEYS if k in row})
            if row.get("g") is not None:
                rec["g"] = row["g"]
            if row.get("source"):
                rec["source"] = row["source"]
            rec["_theta"] = t.tolist()
            d.X.append(physics_features(t))
            d.recs.append(rec)
        d.history = payload.get("history", [])
        return d

    # ---- evaluación física + registro ------------------------------------
    def _eval_and_store(self, thetas: np.ndarray) -> list[dict]:
        thetas = np.array([self.spec.fix_operating_point(t) for t in thetas])
        recs = evaluate_batch(thetas, fidelity=self.fidelity,
                              n_workers=self.n_workers)
        for t, r in zip(thetas, recs):
            self.X.append(physics_features(t))
            r["_theta"] = t.tolist()
            self.recs.append(r)
        return recs

    @staticmethod
    def _lhs(n, d, r):
        cut = np.linspace(0, 1, n + 1)
        u = r.random((n, d))
        s = cut[:n, None] + u / n
        for j in range(d):
            s[:, j] = r.permutation(s[:, j])
        return s

    def _train(self):
        X = np.array(self.X)
        Y = EnsembleSurrogate._targets(self.recs)
        self.log(f"[surrogate] training ensemble on {len(X)} samples...")
        self.sur.fit(X, Y, log=self.log, rand=self.rng)
        if not self.sur.gate_ok():
            self.log("      ⚠ quality gate FAILED → extra round of "
                     "space-filling sampling before optimizing.")
        return self.sur.gate_ok()

    def _scalar_fitness(self, rec: dict, theta: np.ndarray) -> float:
        """Fitness escalar SOLO para reportar/elegir incumbente."""
        return scalar_fitness(self.spec, rec, theta)

    def _update_best(self, recs: list[dict], best):
        for r in recs:
            t = np.array(r["_theta"])
            fit = self._scalar_fitness(r, t)
            if best is None or fit > best[2]:
                best = (t, r, fit)
        return best

    def run(self, n_init=320, rounds=5, batch_size=14,
            checkpoint_path: str | None = None) -> dict:
        if n_init < 1:
            raise ValueError("n_init debe ser >= 1")
        t0 = time.time()
        if self.recs:
            # warm start (load()): el dataset previo sustituye al LHS
            self.log(f"[1/4] Resumed: {len(self.recs)} previous physical "
                     f"evaluations from checkpoint (LHS skipped).")
            recs0 = self.recs
        else:
            self.log(f"[1/4] Initial LHS sampling: {n_init} physical "
                     f"evaluations ({Fidelity(self.fidelity).name})...")
            U0 = self._lhs(n_init, NDIM, self.rng)
            recs0 = self._eval_and_store(
                np.array([denormalize(u) for u in U0]))

        gate = self._train()
        best = self._update_best(recs0, None)

        for rd in range(rounds):
            self.log(f"\n===== Round {rd + 1}/{rounds} =====")
            if not gate:
                extra = self._lhs(batch_size * 2, NDIM, self.rng)
                recs = self._eval_and_store(
                    np.array([denormalize(u) for u in extra]))
                best = self._update_best(recs, best)
                gate = self._train()
                continue

            self.log("[2/4] Constrained NSGA-II over surrogate...")
            ga = NSGA2(self.spec, self.sur, pop=96, generations=60,
                       seed=self.seed + rd)
            result = ga.run()
            n_feas = int(np.sum(result["V"][result["pareto_idx"]] <= 1e-9))
            self.log(f"      Predicted front: {len(result['pareto_idx'])} "
                     f"points ({n_feas} feasible).")

            self.log("[3/4] Acquisition (LCB + diversity): "
                     f"{batch_size} physical evaluations...")
            batchU = select_acquisition_batch(result, self.spec,
                                              batch_size=batch_size,
                                              seed=self.seed + 100 * rd)
            recs = self._eval_and_store(
                np.array([denormalize(u) for u in batchU]))

            scored = [(r, np.array(r["_theta"]),
                       self._scalar_fitness(r, np.array(r["_theta"])))
                      for r in recs]
            scored.sort(key=lambda z: z[2], reverse=True)
            cand_rec, cand_theta, cand_fit = scored[0]
            best = self._update_best(recs, best)
            self.history.append(dict(
                round=rd + 1, best_fitness=cand_fit,
                eta_comb=cand_rec["eta_comb"],
                dp_over_p=cand_rec["dp_over_p"], PF=cand_rec["PF"],
                EI_NOx=cand_rec["EI_NOx"],
                T_wall_max_K=cand_rec["T_wall_max_K"],
                n_evals=len(self.recs), surrogate=self.sur.metrics))
            self.log(f"      Best verified in round: "
                     f"eta={cand_rec['eta_comb']:.4f}  "
                     f"dp/p={cand_rec['dp_over_p']:.4f}  "
                     f"PF={cand_rec['PF']:.3f}  "
                     f"EI_NOx={cand_rec['EI_NOx']:.2f}  fit={cand_fit:.3f}")

            gate = self._train()
            if checkpoint_path:
                self.save(checkpoint_path)

        if checkpoint_path:
            self.save(checkpoint_path)   # siempre: aun sin gate hay dataset
        self.log(f"\n[4/4] Completed in {time.time() - t0:.1f}s · "
                 f"{len(self.recs)} total physical evaluations.")
        theta, rec, fit = best
        return dict(theta=theta, record=rec, fitness=fit,
                    history=self.history,
                    pareto_front=self._final_pareto())

    def _final_pareto(self) -> list[dict]:
        """Frente de Pareto VERIFICADO (solo evaluaciones físicas reales)."""
        F, V, keep = [], [], []
        for r in self.recs:
            t = np.array(r["_theta"])
            F.append(self.spec.objectives(r))
            V.append(self.spec.violation(self.spec.constraints(r, t)))
            keep.append(r)
        F, V = np.array(F), np.array(V)
        fronts = _fast_nondominated_sort(F, V)
        out = []
        for i in fronts[0]:
            r = keep[i]
            out.append(dict(theta=r["_theta"], eta_comb=r["eta_comb"],
                            dp_over_p=r["dp_over_p"], PF=r["PF"],
                            EI_NOx=r["EI_NOx"],
                            T_wall_max_K=r["T_wall_max_K"],
                            arch=r.get("arch"), injector=r.get("injector"),
                            feasible=bool(V[i] <= 1e-9),
                            source=r.get("source", "")))
        return out

    def save(self, path: str):
        payload = dict(
            spec={k: v for k, v in vars(self.spec).items()
                  if not k.startswith("_")},
            seed=self.seed,
            n_evals=len(self.recs), history=self.history,
            dataset=[dict(theta=r["_theta"],
                          **{k: r[k] for k in OUT_KEYS},
                          g=r.get("g"), source=r.get("source"))
                     for r in self.recs])
        with open(path, "w") as f:
            json.dump(payload, f, indent=1)


# alias con el nombre del patrón Phy-CC/Phy-AC (scripts genéricos)
AutonomousAxialDesigner = AutonomousCombustorDesigner
AutonomousCompressorDesigner = AutonomousCombustorDesigner


# ==========================================================================
# 7. API de producto
# ==========================================================================
def scalar_fitness(spec: DesignSpec, rec: dict, theta: np.ndarray) -> float:
    """Fitness escalar para reportar/elegir incumbente: η_comb con
    penalización por NOx; infactible → −10·violación (continua).

    El incumbente es el mejor fitness escalar sobre TODAS las evaluaciones
    físicas — no solo las de la última ronda (bug corregido y bloqueado por
    test en los hermanos: no regresionarlo)."""
    g = spec.constraints(rec, theta)
    v = spec.violation(g)
    if v > 1e-9:
        return -10.0 * v
    nox = float(rec.get("EI_NOx", 0.0) or 0.0)
    return rec["eta_comb"] - W_NOX_FITNESS * (nox / EI_NOX_REF)


def evaluate_design(spec: DesignSpec, theta_design,
                    fidelity=Fidelity.L0) -> dict:
    """Evalúa UN diseño dado por el ingeniero, SIN optimizar.

    `theta_design` acepta dos longitudes:
      16 — solo variables de diseño (VAR_NAMES[:16]); el punto de operación
           (P3, T3, ṁ, T4) lo pone el spec.
      20 — θ completo.
    Devuelve un dict con la misma forma que design():
    {theta, record, fitness, history, pareto_front} (frente de 1 punto),
    listo para geometría/reporte/STL."""
    theta = np.asarray(theta_design, dtype=float).ravel()
    n = theta.shape[0]
    if n == 16:
        theta = np.concatenate([theta, [spec.P3_Pa, spec.T3_K,
                                        spec.mdot_kgs, spec.T4_K]])
    elif n == NDIM_LEGACY and NDIM_LEGACY != NDIM:
        theta = pad_theta(theta)
    elif n != NDIM:
        raise ValueError(
            f"theta debe tener 16 variables de diseño o {NDIM} (θ completo) "
            f"— llegaron {n}")
    theta = spec.fix_operating_point(theta)
    rec = evaluate(theta, fidelity=fidelity)
    rec["_theta"] = theta.tolist()
    fit = scalar_fitness(spec, rec, theta)
    feasible = bool(spec.violation(spec.constraints(rec, theta)) <= 1e-9)
    front = [dict(theta=rec["_theta"], eta_comb=rec["eta_comb"],
                  dp_over_p=rec["dp_over_p"], PF=rec["PF"],
                  EI_NOx=rec["EI_NOx"], T_wall_max_K=rec["T_wall_max_K"],
                  arch=rec.get("arch"), injector=rec.get("injector"),
                  feasible=feasible, source=rec.get("source", ""))]
    return dict(theta=theta, record=rec, fitness=fit, history=[],
                pareto_front=front)


def pareto_from_checkpoint(path: str, log=lambda *a: None):
    """(spec, frente de Pareto verificado) desde un checkpoint de save().
    Cada punto trae su θ completo — apto para re-generar entregables."""
    d = AutonomousCombustorDesigner.load(path, fidelity=Fidelity.L0, log=log)
    return d.spec, d._final_pareto()


def design(spec: DesignSpec, rounds=5, n_init=320, batch_size=14,
           fidelity=Fidelity.L0, n_workers=1,
           checkpoint_path="phycb_run.json", log=print,
           seed: int | None = SEED) -> dict:
    """Punto de entrada Phy-CB: espec → diseño verificado + Pareto.

    Presupuestos por defecto: los de la familia (n_init=320, rounds=5,
    batch=14, pop=96, gens=60) — NO re-tunear sin datos (WAVE4 §1.6).
    `fidelity=L0` por defecto: L0 explora barato y L1 verifica los
    ganadores (patrón exacto del uso de L1 en Phy-AC)."""
    d = AutonomousCombustorDesigner(spec, fidelity=fidelity,
                                    n_workers=n_workers, log=log,
                                    seed=SEED if seed is None else seed)
    return d.run(n_init=n_init, rounds=rounds, batch_size=batch_size,
                 checkpoint_path=checkpoint_path)


if __name__ == "__main__":
    import sys
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    # Espec de demo: microturbina clase 10 kW (el caso ancla KJ66)
    spec = DesignSpec(P3_Pa=250_000.0, T3_K=430.0, mdot_kgs=0.13,
                      T4_K=1150.0, fuel="jet_a", dp_max=0.04,
                      arch="reverse_annular", material="Inconel-625")
    out = design(spec, rounds=2, n_init=120, batch_size=10,
                 fidelity=Fidelity.L0)

    print("\n" + "=" * 64)
    print("VERIFIED AUTONOMOUS OPTIMAL DESIGN — Quasar Phy-CB")
    print("=" * 64)
    for name, val in zip(VAR_NAMES, out["theta"]):
        print(f"  {name:18s} = {val:14.4f}")
    print("-" * 64)
    rec = out["record"]
    for k in ["arch", "injector", "fuel", "eta_comb", "dp_over_p", "PF",
              "T_wall_max_K", "EI_NOx", "EI_CO", "phi_pz", "tau_ms",
              "SMD_um", "D_casing_mm", "L_tot_mm", "feasible", "source"]:
        if k in rec:
            v = rec[k]
            print(f"  {k:14s} = {v:.4f}" if isinstance(v, float)
                  else f"  {k:14s} = {v}")
    print("-" * 64)
    feas = [p for p in out["pareto_front"] if p["feasible"]]
    print(f"Verified Pareto front: {len(out['pareto_front'])} points "
          f"({len(feas)} feasible). Checkpoint: phycb_run.json")
