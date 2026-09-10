#!/usr/bin/env python3
"""Re-congela `validation/anchors.json` (el ancla que consume `validate.py`).

`test_phycb.py --freeze-anchors` congela `anchors/REF_CB10.json`, que es OTRO
archivo: aquel guarda el record de física COMPLETO y este solo las claves que
la campaña de validación compara. Mover una obliga a mover la otra, y no había
herramienta para la segunda — de ahí este script.

    python validation/_refreeze_anchors.py            # muestra el diff
    python validation/_refreeze_anchors.py --write    # lo escribe

Moverlas exige citar en `docs/PENDIENTES_Lverificar.md` la corrección que las
movió (contrato §10.9).
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np                                                # noqa: E402
import physics_core as pc                                         # noqa: E402

pc.assert_provenance(pc)

PATH = os.path.join(HERE, "anchors.json")

NOTE = ("Anclas de regresión CONGELADAS. Moverlas exige --freeze-anchors y "
        "citar en docs/PENDIENTES_Lverificar.md la corrección que las movió "
        "(contrato §10.9).")


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    write = "--write" in argv

    with open(PATH, encoding="utf-8") as f:
        blob = json.load(f)
    anchors = blob.get("anchors", {}) or {}

    n_moved = 0
    for name, a in anchors.items():
        theta = np.asarray(a["theta"], dtype=float)
        fid = getattr(pc.Fidelity, str(a.get("fidelity", "L0")).upper())
        rec = pc.evaluate(theta, fidelity=fid, use_cache=False)
        for k, old in list(a.items()):
            if k in ("theta", "fidelity") or not isinstance(old, (int, float)):
                continue
            new = rec.get(k)
            if new is None or not isinstance(new, (int, float)):
                continue
            if abs(float(new) - float(old)) > 1e-6 * max(abs(float(old)), 1e-30):
                d = (float(new) / float(old) - 1.0) if old else float("nan")
                print(f"  {name}.{k}: {old!r} → {new!r}  (Δrel {d:+.3e})")
                a[k] = float(new)
                n_moved += 1

    if not n_moved:
        print("  nada que mover: el ancla ya está al día")
        return 0
    if not write:
        print(f"\n  {n_moved} clave(s) movidas. Repite con --write para "
              "escribirlas.")
        return 0

    blob["anchors"] = anchors
    blob["note"] = NOTE
    blob["generated_at"] = _dt.datetime.now(_dt.timezone.utc).isoformat(
        timespec="seconds")
    with open(PATH, "w", encoding="utf-8") as f:
        json.dump(blob, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print(f"\n  escrito {PATH} ({n_moved} clave(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
