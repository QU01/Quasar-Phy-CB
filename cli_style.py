"""
QUASAR Phy-CB · cli_style.py
============================
Identidad de terminal SIN dependencias (el núcleo es solo-NumPy), construida
alrededor del logo QUASAR: un anillo segmentado por un corte diagonal (disco
de acreción) con núcleo circular sólido. El logo se RENDERIZA
proceduralmente (raster + medio-bloques Unicode, gradiente de paleta), no es
arte fijo: `_quasar_logo()` admite tamaño.

Todo degrada con elegancia: sin TTY, con `NO_COLOR` o con `--no-color` sale
texto plano (logo en ASCII). Habilita secuencias ANSI en consolas Windows
modernas (Virtual Terminal Processing).
"""
from __future__ import annotations

import math
import os
import shutil
import sys

# ---------------------------------------------------------------- estado
_COLOR = True


def _stream_is_tty(stream) -> bool:
    return hasattr(stream, "isatty") and stream.isatty()


def supports_color() -> bool:
    if (os.environ.get("NO_COLOR") or os.environ.get("PHYCB_NO_COLOR")
            or os.environ.get("PHYCC_NO_COLOR")):
        return False
    return _stream_is_tty(sys.stdout)


def _enable_windows_vt() -> None:
    if os.name != "nt":
        return
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        for handle_id in (-11, -12):                 # STDOUT, STDERR
            handle = kernel32.GetStdHandle(handle_id)
            mode = ctypes.c_uint32()
            if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                kernel32.SetConsoleMode(handle, mode.value | 0x0004)
    except Exception:
        pass


def init(use_color: bool | None = None) -> bool:
    """Configura color. use_color=None → autodetecta. Devuelve si quedó on."""
    global _COLOR
    _COLOR = supports_color() if use_color is None else bool(use_color)
    if _COLOR:
        _enable_windows_vt()
    return _COLOR


def width(default: int = 80) -> int:
    try:
        return max(40, min(shutil.get_terminal_size().columns, 100))
    except Exception:
        return default


# ---------------------------------------------------------------- color
_RESET = "\033[0m"

# Paleta "quasar" (256 colores): del cian del anillo exterior al violeta
# del borde interno, núcleo blanco.
PALETTE = {
    "brand":  "38;5;51",    # cian brillante (anillo)
    "brand2": "38;5;45",    # cian-azul
    "brand3": "38;5;39",    # azul cielo
    "brand4": "38;5;99",    # azul-violeta
    "accent": "38;5;141",   # violeta
    "core":   "38;5;231",   # blanco (núcleo del quasar)
    "ok":     "38;5;42",    # verde
    "warn":   "38;5;214",   # ámbar
    "err":    "38;5;203",   # rojo coral
    "dim":    "38;5;245",   # gris
    "key":    "38;5;80",    # cian apagado
    "val":    "38;5;231",   # blanco brillante
}

# gradiente angular del anillo (sentido de giro)
_RING_SHADES = ["brand", "brand2", "brand3", "brand4", "accent"]


def c(text, style: str = "", *, bold: bool = False) -> str:
    """Envuelve `text` en un estilo de la paleta (o código ANSI crudo)."""
    if not _COLOR:
        return str(text)
    code = PALETTE.get(style, style)
    parts = []
    if bold:
        parts.append("1")
    if code:
        parts.append(code)
    if not parts:
        return str(text)
    return f"\033[{';'.join(parts)}m{text}{_RESET}"


# ---------------------------------------------------------------- logo
def _quasar_raster(n: int = 22) -> list[list[int]]:
    """Raster n×n del logo: 0 vacío, 1..len(_RING_SHADES) anillo (índice de
    gradiente angular), 9 núcleo.

    Anillo entre el 64% y el 100% del radio, núcleo al 34%, y el corte del
    logo: una banda recta por el centro (dirección ~112°) borra el anillo
    en dos gajos opuestos — el núcleo no se toca. Supermuestreo 3×3 por
    píxel para suavizar el círculo en rasters pequeños.
    """
    cx = cy = (n - 1) / 2.0
    R = n / 2.0 - 0.3
    ring_in, core_r = 0.64 * R, 0.34 * R
    th = math.radians(62.0)           # dirección del corte (y hacia arriba):
    #                                   gajos en ~1:30 y ~7:30, como el logo
    ux, uy = math.cos(th), math.sin(th)
    band_hw = 0.085 * n               # semiancho del corte

    sub = (-1.0 / 3, 0.0, 1.0 / 3)
    px = [[0] * n for _ in range(n)]
    for y in range(n):
        for x in range(n):
            hits_core = hits_ring = 0
            for oy in sub:
                for ox in sub:
                    dx, dy = x + ox - cx, cy - (y + oy)   # y hacia arriba
                    r = math.hypot(dx, dy)
                    if r <= core_r:
                        hits_core += 1
                    elif ring_in <= r <= R and \
                            abs(dx * uy - dy * ux) >= band_hw:
                        hits_ring += 1
            if hits_core >= 5:
                px[y][x] = 9
            elif hits_ring >= 5:
                dx, dy = x - cx, cy - y
                ang = math.atan2(dy, dx) % (2 * math.pi)
                shade = int(ang / (2 * math.pi) * len(_RING_SHADES))
                px[y][x] = 1 + min(shade, len(_RING_SHADES) - 1)
    return px


def _quasar_logo(n: int = 22) -> list[str]:
    """Logo como líneas de terminal. Con color: medio-bloques (2 píxeles
    verticales por celda → círculo redondo) con gradiente; sin color:
    ASCII plano ('#' anillo, '@' núcleo)."""
    px = _quasar_raster(n)
    if not _COLOR:
        out = []
        for y in range(0, n, 2):
            row = "".join("@" if v == 9 else "#" if v else " "
                          for v in px[y])
            out.append(row)
        return out

    def paint(v: int, ch: str) -> str:
        style = "core" if v == 9 else _RING_SHADES[v - 1]
        return c(ch, style, bold=(v == 9))

    out = []
    for j in range(0, n, 2):
        top = px[j]
        bot = px[j + 1] if j + 1 < n else [0] * n
        line = []
        for x in range(n):
            t, b = top[x], bot[x]
            if t and b:
                line.append(paint(t, "█"))
            elif t:
                line.append(paint(t, "▀"))
            elif b:
                line.append(paint(b, "▄"))
            else:
                line.append(" ")
        out.append("".join(line))
    return out


# ---------------------------------------------------------------- banner
_VERSION = "v1.0"

# wordmark QUASAR en medio-bloques (3 filas, glifos de 4 columnas)
_WORDMARK = [
    "▄▀▀▄ █  █ ▄▀▀▄ ▄▀▀▀ ▄▀▀▄ █▀▀▄",
    "█  █ █  █ █▀▀█ ▀▀▀▄ █▀▀█ █▄▄▀",
    "▀▄▄▚ ▀▄▄▀ █  █ ▄▄▄▀ █  █ █ ▀▄",
]


def banner(n: int = 26) -> str:
    """Banner: logo procedural a la izquierda, wordmark + tagline a la
    derecha, alineados verticalmente."""
    logo = _quasar_logo(n)
    if _COLOR:
        right = [
            "",
            "",
            c(_WORDMARK[0], "val", bold=True),
            c(_WORDMARK[1], "val", bold=True),
            c(_WORDMARK[2], "val", bold=True),
            "",
            c("P h y - C B", "brand", bold=True)
            + c(f"   {_VERSION}", "dim"),
            "",
            c("autonomous printable combustor design", "dim"),
            c("spec", "key") + c(" → ", "dim")
            + c("verified geometry", "key") + c(" → ", "dim")
            + c("printable STL", "key"),
            c("physics-first · deterministic (seed 71)", "dim"),
        ]
    else:
        right = [
            "",
            "QUASAR  Phy-CB  " + _VERSION,
            "autonomous printable combustor design",
            "spec -> verified geometry -> printable STL",
            "physics-first, deterministic (seed 71)",
        ]
    lines = []
    n_rows = max(len(logo), len(right))
    for i in range(n_rows):
        left = logo[i] if i < len(logo) else " " * n
        pad = n - _vlen(left)
        r = right[i] if i < len(right) else ""
        lines.append("  " + left + " " * max(0, pad) + "   " + r)
    return "\n".join(lines)


def logo_mark() -> str:
    """Glifo mínimo del logo para usar inline (◉ = anillo + núcleo)."""
    return c("◉", "brand", bold=True) if _COLOR else "(o)"


# ---------------------------------------------------------------- layout
def rule(title: str = "", style: str = "brand") -> str:
    w = width()
    if not title:
        return c("─" * w, "dim")
    head = "─◉─" if _COLOR else "---"
    label = f" {title} "
    fill = max(0, w - len(head) - len(label))
    return c(head, style, bold=True) + c(label, style, bold=True) \
        + c("─" * fill, "dim")


def panel(title: str, rows: list[tuple[str, str]], style: str = "brand") -> str:
    """Caja redondeada con pares clave/valor y glifo del logo en el título."""
    w = min(width(), 72)
    inner = w - 2
    klen = max((len(k) for k, _ in rows), default=0)
    tl, tr, bl, br = ("╭", "╮", "╰", "╯") if _COLOR else ("+", "+", "+", "+")
    hz, vt = ("─", "│") if _COLOR else ("-", "|")
    mark = "◉ " if _COLOR else "(o) "
    out = [c(tl + hz * inner + tr, style)]
    tline = f" {mark}{title} "
    pad = inner - _vlen(tline)
    out.append(c(vt, style) + c(f" {mark}", style, bold=True)
               + c(f"{title} ", "val", bold=True)
               + " " * max(0, pad) + c(vt, style))
    out.append(c("├" + hz * inner + "┤" if _COLOR
                 else "+" + hz * inner + "+", style))
    for k, v in rows:
        visible = len(f" {k.ljust(klen)}  ") + _vlen(v)
        pad = max(0, inner - visible)
        out.append(c(vt, style) + c(f" {k.ljust(klen)}  ", "key")
                   + v + " " * pad + c(vt, style))
    out.append(c(bl + hz * inner + br, style))
    return "\n".join(out)


def _vlen(s: str) -> int:
    """Largo visible: descuenta secuencias ANSI."""
    out, i = 0, 0
    while i < len(s):
        if s[i] == "\033":
            j = s.find("m", i)
            if j == -1:
                break
            i = j + 1
        else:
            out += 1
            i += 1
    return out


# ---------------------------------------------------------------- estados
def ok(msg: str) -> str:
    return c("✓ ", "ok", bold=True) + msg


def warn(msg: str) -> str:
    return c("⚠ ", "warn", bold=True) + c(msg, "warn")


def err(msg: str) -> str:
    return c("✗ ", "err", bold=True) + c(msg, "err")


def step(msg: str) -> str:
    return c("▸ ", "brand") + msg


def kv(key: str, value: str) -> str:
    return c(f"{key}", "key") + c(" = ", "dim") + c(value, "val", bold=True)


# ---------------------------------------------------------------- progreso
def progress_bar(frac: float, label: str = "", barw: int = 28) -> str:
    """Barra con el gradiente del anillo (cian → violeta) y núcleo como
    cabeza de avance."""
    frac = max(0.0, min(1.0, frac))
    fill = int(round(frac * barw))
    if not _COLOR:
        return ("[" + "#" * fill + "-" * (barw - fill)
                + f"] {int(frac * 100):3d}%  " + label)
    segs = []
    for i in range(fill):
        shade = _RING_SHADES[min(int(i / max(barw - 1, 1)
                                     * len(_RING_SHADES)),
                                 len(_RING_SHADES) - 1)]
        segs.append(c("█", shade))
    head = c("◉", "core", bold=True) if 0 < fill < barw else ""
    rest = max(0, barw - fill - (1 if head else 0))
    return (c("▕", "dim") + "".join(segs) + head + c("░" * rest, "dim")
            + c("▏", "dim") + c(f" {int(frac * 100):3d}%  ", "val") + label)


# ---------------------------------------------------------------- Phy-CT parity
#: Key/value line with an optional unit and a dim trailing note, laid out
#: like the sibling tools (`phyct_cli.py`, `phyhx_cli.py`). It PRINTS,
#: unlike `kv()` above, which returns the inline `key = value` form both
#: are free to keep using.
def line(key: str, value="", unit: str = "", note: str = "",
         width_: int = 26) -> None:
    txt = f"  {c(str(key).ljust(width_), 'key')}"
    if value != "":
        txt += c(str(value), "val", bold=True)
    if unit:
        txt += " " + c(unit, "dim")
    if note:
        txt += "   " + c(str(note), "dim")
    print(txt)


def section(title_: str, style: str = "brand") -> None:
    """Prints a titled rule. `rule()` returns the string; the sibling
    tools call a printing `section()`, so both spellings exist."""
    print()
    print(rule(title_, style))


def title(t: str) -> None:
    print()
    print(rule(t, "accent"))


def status(ok_: bool, yes: str = "OK", no: str = "NO") -> str:
    return c(yes, "ok", bold=True) if ok_ else c(no, "warn", bold=True)


#: Traceability mark of a number, in the family's notation: [V] measured,
#: [S] secondary source, [M] own estimate, [?] unverified.
def mark(m: str) -> str:
    style = {"V": "ok", "S": "brand", "M": "warn", "?": "err"}.get(m, "dim")
    return c(f"[{m}]", style, bold=True)


def dim(t: str) -> str:
    return c(t, "dim")


def bold(t: str) -> str:
    return c(t, "val", bold=True)


#: Everything that degraded in this run, in order. `--strict` turns the
#: list into a failure at the end; without it each entry is only announced.
DEGRADATIONS: list[tuple[str, str]] = []


def degraded(what: str, why: str) -> str:
    """ANNOUNCED degradation: what could not be done is said, with the
    reason. Never silently.

    It is also RECORDED. Announcing each degradation separately is enough
    for a human watching the screen, but a CI needs ONE question at the
    end — «did this run in full?» — and `--strict` is that question.

    Returns the formatted line so callers can keep the house
    `print("  " + ui.degraded(...))` shape.
    """
    DEGRADATIONS.append((str(what), str(why)))
    return (c("[degraded] ", "warn", bold=True)
            + c(f"{what}: {why}", "warn"))


def strict_verdict() -> list[tuple[str, str]]:
    """What degraded, for `--strict` to judge. Empty ⇒ everything ran."""
    return list(DEGRADATIONS)


def reset_degradations() -> None:
    DEGRADATIONS.clear()
