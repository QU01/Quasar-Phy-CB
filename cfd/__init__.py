#!/usr/bin/env python3
"""
QUASAR Phy-CB · paquete `cfd` — capas de fidelidad por encima de L1.

Contenido (contrato §8):

* `cfd.xlb_cold`      — L1.5: aero-frío lattice-Boltzmann sobre los voxels.
* `cfd.cold_post`     — post-proceso del campo frío (splits, penetración,
                        zonas, corrección de densidad).
* `cfd.crn_from_field`— auto-topología del CRN de L1 con volúmenes y caudales
                        MEDIDOS. Es el punto de entrada que
                        `physics_core._try_higher` importa perezosamente
                        (`topology_from_field(theta, rec)`).
* `cfd.ibrans`        — L2a (WAVE6, fuera de esta wave).
* `cfd/openfoam_backend.py` — **NO EXISTE**: L2b está fuera de alcance por
  decisión de proyecto (contrato §12, máquina Windows sin WSL2/Docker).

Trampa de entorno (contrato §0.b)
---------------------------------
Phy-AC está instalado como paquete **editable**: desde cualquier cwd que no
sea el de Phy-CB, un `import physics_core` resuelve SILENCIOSAMENTE al de
Phy-AC (NDIM=15, compresor axial). Este `__init__` mete la raíz de Phy-CB al
FRENTE de `sys.path` para que los módulos de `cfd/` importen a sus hermanos
correctos, y cada uno de ellos vuelve a comprobarlo con
`physics_core.assert_provenance`. No es un adorno: es la única defensa.
"""

from __future__ import annotations

import os
import sys

#: Raíz del repo de Phy-CB (el directorio que contiene `physics_core.py`).
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
elif sys.path[0] != ROOT:
    # Estaba, pero detrás de otra ruta que podría ensombrecerlo: al frente.
    sys.path.remove(ROOT)
    sys.path.insert(0, ROOT)

for _s in (sys.stdout, sys.stderr):
    # Windows reporta cp1252 y revienta con η, τ, ✓ y flechas (trampa REAL).
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

__all__ = ["ROOT", "xlb_cold", "cold_post", "crn_from_field"]
