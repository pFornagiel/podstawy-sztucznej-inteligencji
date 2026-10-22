"""Generuje schematy używane w notatniku lab_3.ipynb.

Uruchomienie (z katalogu lab_3_gradient):  python figures/make_figures.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

OUT = Path(__file__).parent

BLUE = "#2a78d6"     # parametry, przejście w przód
ORANGE = "#eb6834"   # gradienty, przejście wstecz
AQUA = "#1baf7a"     # wyróżnienie
GRAY = "#8a8984"     # dane (stałe)
RED = "#e34948"      # klasa 1
INK = "#0b0b0b"
MUTED = "#52514e"
MONO = "DejaVu Sans Mono"

plt.rcParams.update({"font.size": 12, "mathtext.fontset": "cm"})


def tint(hex_color, amount):
    """Rozjaśnia kolor w stronę bieli (amount=0 -> biały, 1 -> pełny kolor)."""
    r, g, b = (int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5))
    return (1 - amount + amount * r, 1 - amount + amount * g, 1 - amount + amount * b)


def canvas(width, height):
    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def save(fig, name):
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def arrow(ax, start, end, color=INK, text=None, text_offset=(0, 0.1), fontsize=11, rad=0.0, lw=1.6,
          va="bottom"):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14, color=color,
                                 lw=lw, connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2))
    if text:
        mx = (start[0] + end[0]) / 2 + text_offset[0]
        my = (start[1] + end[1]) / 2 + text_offset[1]
        ax.text(mx, my, text, ha="center", va=va, fontsize=fontsize, color=color)


def box(ax, cx, cy, text, color, w=1.0, h=0.55, fontsize=13, filled=0.18, lw=1.4, ec=None):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0.04,rounding_size=0.12",
                                fc=tint(color, filled), ec=ec or color, lw=lw))
    if text:
        ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize, color=INK)


def matrix(ax, x, y, rows, cols, color, cell=0.36, label=None, shape=None, strength=0.25, texts=None):
    """Rysuje macierz jako siatkę komórek. (x, y) to lewy górny róg."""
    for i in range(rows):
        for j in range(cols):
            ax.add_patch(Rectangle((x + j * cell, y - (i + 1) * cell), cell, cell,
                                   fc=tint(color, strength), ec="white", lw=1.5))
            if texts is not None and texts[i][j]:
                ax.text(x + (j + 0.5) * cell, y - (i + 0.5) * cell, texts[i][j],
                        ha="center", va="center", fontsize=10, color=INK)
    cx = x + cols * cell / 2
    if label:
        ax.text(cx, y + 0.1, label, ha="center", va="bottom", fontsize=13, color=INK)
    if shape:
        ax.text(cx, y - rows * cell - 0.1, shape, ha="center", va="top", fontsize=10, color=MUTED)
    return x + cols * cell, y - rows * cell


# --------------------------------------------------------------------------- 1
LOOP_GENERIC = [
    ("Inicjalizacja\nparametrów", r"$\theta$ - losowe", ""),
    ("1. Przejście\nw przód", r"$\hat y = \mathrm{model}(X;\, \theta)$", ""),
    ("2. Funkcja\nkosztu", r"$L(y,\, \hat y)$", ""),
    ("3. Gradient\nkosztu", r"$\nabla_\theta L = \frac{\partial L}{\partial \theta}$", ""),
    ("4. Aktualizacja\nparametrów", r"$\theta \leftarrow \theta - \eta\, \nabla_\theta L$", ""),
]

LOOP_LOGREG = [
    ("Inicjalizacja\nparametrów", r"$w, b$ - losowe", "Zadanie 2"),
    ("1. Przejście\nw przód", r"$\hat p = \sigma(Xw + b)$", "Zadanie 2"),
    ("2. Funkcja\nkosztu", r"$L = \mathrm{BCE}(y,\, \hat p)$", "Zadanie 3"),
    ("3. Gradient\nkosztu", r"$\frac{\partial L}{\partial w},\ \frac{\partial L}{\partial b}$", "Zadania 1 i 4"),
    ("4. Aktualizacja\nparametrów", r"$w \leftarrow w - \eta \frac{\partial L}{\partial w}$", "Zadanie 5"),
]

LOOP_TORCH = [
    ("Inicjalizacja\nparametrów", "model = Model()\nopt = optim.SGD(...)", "nn.Module"),
    ("1. Przejście\nw przód", "p_hat = model(X)", "forward()"),
    ("2. Funkcja\nkosztu", "loss = loss_fn(\n    p_hat, y)", "nn.BCELoss"),
    ("3. Gradient\nkosztu", "opt.zero_grad()\nloss.backward()", "autograd"),
    ("4. Aktualizacja\nparametrów", "opt.step()", "torch.optim"),
]


def petla_uczenia(steps, name, highlight=(), loop_label="powtórz - kolejna epoka", code=False):
    fig, ax = canvas(13.7, 4.0)
    xs = [1.3, 4.0, 6.7, 9.4, 12.1]
    yc, w, h = 2.3, 2.35, 1.9
    for i, (cx, (title, formula, tag)) in enumerate(zip(xs, steps)):
        active = i in highlight
        faded = bool(highlight) and not active
        color = ORANGE if active else (GRAY if faded else (GRAY if i == 0 else BLUE))
        box(ax, cx, yc, None, color, w=w, h=h, filled=0.22 if active else 0.12, lw=2.6 if active else 1.4)
        txt_color = MUTED if faded else INK
        ax.text(cx, yc + 0.5, title, ha="center", va="center", fontsize=14, color=txt_color, weight="bold")
        if code:
            ax.text(cx, yc - 0.2, formula, ha="center", va="center", fontsize=10.5, color=txt_color,
                    family=MONO)
        else:
            ax.text(cx, yc - 0.22, formula, ha="center", va="center", fontsize=16, color=txt_color)
        if tag:
            ax.text(cx, yc - h / 2 + 0.2, tag, ha="center", va="center", fontsize=12,
                    color=ORANGE if active else MUTED, family=MONO if code else None)
        if active:
            ax.text(cx, yc + h / 2 + 0.1, "▼ jesteś tutaj", ha="center", va="bottom", fontsize=14,
                    color=ORANGE, weight="bold")
    for a, b in zip(xs, xs[1:]):
        arrow(ax, (a + w / 2 + 0.03, yc), (b - w / 2 - 0.03, yc), lw=2)

    # pętla powrotna: z aktualizacji z powrotem do przejścia w przód
    loop_active = "loop" in highlight
    lc = ORANGE if loop_active else BLUE
    ybot = 0.3
    ax.plot([xs[4], xs[4], xs[1]], [yc - h / 2 - 0.05, ybot, ybot], color=lc, lw=2.4 if loop_active else 1.8)
    arrow(ax, (xs[1], ybot), (xs[1], yc - h / 2 - 0.05), color=lc, lw=2.4 if loop_active else 1.8)
    ax.text((xs[1] + xs[4]) / 2, ybot + 0.08, loop_label, ha="center", va="bottom", fontsize=14, color=lc,
            weight="bold" if loop_active else None)
    save(fig, name)


# --------------------------------------------------------------------------- 3
def _graf_regresji(ax, yc):
    """Wspólny graf obliczeń regresji logistycznej dla jednego punktu i. Zwraca pozycje węzłów."""
    box(ax, 0.8, yc + 1.3, r"$x_{i1}$", GRAY, w=0.8)
    box(ax, 0.8, yc + 0.45, r"$x_{i2}$", GRAY, w=0.8)
    box(ax, 0.8, yc - 0.45, r"$w_1,\, w_2$", BLUE, w=1.1)
    box(ax, 0.8, yc - 1.3, r"$b$", BLUE, w=0.8)
    box(ax, 3.6, yc, r"$z_i$", INK, w=0.8, filled=0.05)
    box(ax, 6.4, yc, r"$\hat p_i$", INK, w=0.8, filled=0.05)
    box(ax, 6.4, yc - 1.4, r"$y_i$", GRAY, w=0.8)
    box(ax, 9.2, yc, r"$\ell_i$", INK, w=0.8, filled=0.05)
    box(ax, 11.8, yc, r"$L$", ORANGE, w=0.8)
    for y0 in (yc + 1.3, yc + 0.45, yc - 0.45, yc - 1.3):
        arrow(ax, (1.4 if y0 == yc - 0.45 else 1.25, y0), (3.15, yc + (y0 - yc) * 0.15), color=INK, lw=1.2)
    arrow(ax, (6.85, yc - 1.25), (8.8, yc - 0.15), color=INK, lw=1.2)
    return {"z": 3.6, "p": 6.4, "l": 9.2, "L": 11.8}


def regresja_graf():
    fig, ax = canvas(13.2, 7.4)
    ax.set_ylim(-0.4, 7.4)
    yc = 5.4
    ax.text(0.2, 7.2, "Jeden punkt $i$: graf obliczeń", fontsize=14, color=BLUE, va="center")
    n = _graf_regresji(ax, yc)
    arrow(ax, (4.05, yc), (5.95, yc), color=BLUE, lw=2, text=r"$\sigma(z_i) = \frac{1}{1+e^{-z_i}}$",
          text_offset=(0, 0.15), fontsize=13)
    arrow(ax, (6.85, yc), (8.75, yc), color=BLUE, lw=2, text=r"$\mathrm{BCE}$", text_offset=(0, 0.15),
          fontsize=13)
    arrow(ax, (9.65, yc), (11.35, yc), color=BLUE, lw=2, text=r"średnia po $N$", text_offset=(0, 0.15),
          fontsize=12)
    ax.text(n["z"] + 0.35, yc - 0.45, r"$z_i = w_1 x_{i1} + w_2 x_{i2} + b$", ha="center", va="top",
            fontsize=12, color=BLUE)
    ax.text(n["z"] + 0.35, yc - 0.95, "logit (kombinacja liniowa)", ha="center", va="top", fontsize=10,
            color=MUTED)
    ax.text(n["p"], yc + 0.45, "prawdopodobieństwo\nklasy 1", ha="center", va="bottom", fontsize=10,
            color=MUTED)
    ax.text(n["l"], yc - 0.45, "koszt punktu", ha="center", va="top", fontsize=10, color=MUTED)
    ax.text(n["L"], yc - 0.45, "jedna liczba", ha="center", va="top", fontsize=10, color=MUTED)

    ax.plot([0.2, 13.0], [3.25, 3.25], color=tint(GRAY, 0.4), lw=1)
    ax.text(0.2, 2.95, "Cały zbiór naraz (wektoryzacja w NumPy) - kształty", fontsize=14, color=BLUE,
            va="center")
    c = 0.3
    top = 2.3
    xe, _ = matrix(ax, 0.6, top, 6, 2, GRAY, cell=c, label="$X$", shape="(N, 2)")
    ax.text(xe + 0.25, top - 3 * c, "@", ha="center", va="center", fontsize=16)
    we, _ = matrix(ax, xe + 0.5, top - 2 * c, 2, 1, BLUE, cell=c, label="$w$", shape="(2,)")
    ax.text(we + 0.25, top - 3 * c, "+", ha="center", va="center", fontsize=16)
    be, _ = matrix(ax, we + 0.5, top - 2.5 * c, 1, 1, BLUE, cell=c, label="$b$", shape="liczba")
    arrow(ax, (be + 0.15, top - 3 * c), (be + 1.0, top - 3 * c), color=BLUE)
    ze, _ = matrix(ax, be + 1.1, top, 6, 1, INK, cell=c, label="$z$", shape="(N,)", strength=0.1)
    arrow(ax, (ze + 0.15, top - 3 * c), (ze + 1.2, top - 3 * c), color=BLUE, text=r"$\sigma$")
    pe, _ = matrix(ax, ze + 1.3, top, 6, 1, INK, cell=c, label=r"$\hat p$", shape="(N,)", strength=0.1)
    ax.text(pe + 0.45, top - 3 * c, ",", ha="center", va="center", fontsize=16)
    ye, _ = matrix(ax, pe + 0.7, top, 6, 1, GRAY, cell=c, label="$y$", shape="(N,)")
    arrow(ax, (ye + 0.15, top - 3 * c), (ye + 1.2, top - 3 * c), color=BLUE, text=r"$\mathrm{BCE}$")
    le, _ = matrix(ax, ye + 1.3, top, 6, 1, INK, cell=c, label=r"$\ell$", shape="(N,)", strength=0.1)
    arrow(ax, (le + 0.15, top - 3 * c), (le + 1.2, top - 3 * c), color=BLUE, text="średnia")
    box(ax, le + 1.65, top - 3 * c, "$L$", ORANGE, w=0.6)
    ax.text(le + 1.65, top - 3 * c - 0.45, "liczba", ha="center", va="top", fontsize=10, color=MUTED)
    ax.text(6.6, -0.25, "każdy wiersz $X$ to jeden punkt, każda kolumna to jedna cecha; "
            "operacje σ i BCE działają element po elemencie",
            ha="center", va="center", fontsize=10.5, color=MUTED)
    save(fig, "regresja_graf.png")


def regresja_backward():
    fig, ax = canvas(13.2, 8.7)
    yc = 5.6
    ax.text(0.2, 8.45, "Przejście wstecz: reguła łańcuchowa = iloczyn lokalnych pochodnych",
            fontsize=14, color=ORANGE, va="center")
    _graf_regresji(ax, yc)
    for a, b in ((4.05, 5.95), (6.85, 8.75), (9.65, 11.35)):
        arrow(ax, (a, yc), (b, yc), color=tint(BLUE, 0.5), lw=1.5)
    kw = dict(ha="center", fontsize=13, color=ORANGE)
    yb = yc + 0.75
    arrow(ax, (11.6, yb), (9.4, yb), color=ORANGE, lw=1.6, rad=0.25)
    ax.text(10.5, yb + 0.45, r"$\frac{\partial L}{\partial \ell_i} = \frac{1}{N}$", va="bottom", **kw)
    arrow(ax, (9.0, yb), (6.6, yb), color=ORANGE, lw=1.6, rad=0.25)
    ax.text(7.8, yb + 0.45, r"$\frac{\partial \ell_i}{\partial \hat p_i}$", va="bottom", **kw)
    arrow(ax, (6.2, yb), (3.8, yb), color=ORANGE, lw=1.6, rad=0.25)
    ax.text(5.0, yb + 0.45, r"$\frac{\partial \hat p_i}{\partial z_i} = \sigma'(z_i)$", va="bottom", **kw)
    ax.text(7.8, yb + 1.1, "numerycznie:\ndf(koszt, p̂)", ha="center", va="bottom", fontsize=10,
            color=ORANGE, family=MONO)
    ax.text(5.0, yb + 1.1, "numerycznie:\ndf(sigmoid, z)", ha="center", va="bottom", fontsize=10,
            color=ORANGE, family=MONO)
    ax.text(10.5, yb + 1.1, "ze średniej", ha="center", va="bottom", fontsize=10, color=ORANGE)
    ax.text(2.6, yc - 1.75, r"$\frac{\partial z_i}{\partial w_j} = x_{ij}, \quad"
            r"\frac{\partial z_i}{\partial b} = 1$", **kw, va="top")
    ax.text(2.6, yc - 2.45, "wprost: z jest liniowe", ha="center", va="top", fontsize=10, color=ORANGE)

    ax.plot([0.2, 13.0], [2.75, 2.75], color=tint(GRAY, 0.4), lw=1)
    ax.text(0.2, 2.45, "Składamy wszystko dla całego zbioru", fontsize=14, color=ORANGE, va="center")
    ax.text(0.4, 1.65,
            r"$\delta_i = \frac{\partial L}{\partial z_i} = \frac{1}{N}\cdot"
            r"\frac{\partial \ell_i}{\partial \hat p_i}\cdot\frac{\partial \hat p_i}{\partial z_i}$",
            fontsize=16, va="center", color=INK)
    ax.text(0.4, 0.75, r"$\frac{\partial L}{\partial w_j} = \sum_i \delta_i\, x_{ij}"
            r"\qquad \frac{\partial L}{\partial b} = \sum_i \delta_i$", fontsize=16, va="center", color=INK)
    c = 0.3
    top = 2.1
    xe, _ = matrix(ax, 7.4, top - 0.6, 2, 6, GRAY, cell=c, label=r"$X^\top$", shape="(2, N)")
    ax.text(xe + 0.25, top - 0.6 - c, "@", ha="center", va="center", fontsize=16)
    de, _ = matrix(ax, xe + 0.5, top, 6, 1, ORANGE, cell=c, label=r"$\delta$", shape="(N,)")
    ax.text(de + 0.3, top - 0.6 - c, "=", ha="center", va="center", fontsize=16)
    matrix(ax, de + 0.6, top - 0.6, 2, 1, ORANGE, cell=c, label=r"$\frac{\partial L}{\partial w}$",
           shape="(2,) jak $w$")
    save(fig, "regresja_backward.png")


# --------------------------------------------------------------------------- 4
def granica_decyzyjna():
    w, b = np.array([1.5, -1.0]), 0.5
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1.1, 1]})

    zs = np.linspace(-6, 6, 300)
    ax1.plot(zs, 1 / (1 + np.exp(-zs)), color=INK, lw=2.5)
    ax1.axhline(0.5, color=GRAY, ls=":")
    ax1.axvline(0, color=GRAY, ls=":")
    ax1.fill_between(zs, 0, 1, where=zs >= 0, color=tint(RED, 0.12))
    ax1.fill_between(zs, 0, 1, where=zs <= 0, color=tint(BLUE, 0.12))
    ax1.text(3, 0.2, "$z > 0$\n$\\hat p > 0.5$\n→ klasa 1", ha="center", fontsize=13, color=RED)
    ax1.text(-3, 0.65, "$z < 0$\n$\\hat p < 0.5$\n→ klasa 0", ha="center", fontsize=13, color=BLUE)
    ax1.set_xlabel("logit $z = w^\\top x + b$")
    ax1.set_ylabel("$\\hat p = \\sigma(z)$")
    ax1.set_title("Funkcja sigmoidalna zamienia logit na prawdopodobieństwo", fontsize=12)
    ax1.set_ylim(0, 1)

    g = np.linspace(-3, 3, 200)
    xx, yy = np.meshgrid(g, g)
    zz = w[0] * xx + w[1] * yy + b
    cs = ax2.contourf(xx, yy, 1 / (1 + np.exp(-zz)), levels=np.linspace(0, 1, 21), cmap="coolwarm", alpha=0.5)
    ax2.contour(xx, yy, zz, levels=[-3, -1.5, 1.5, 3], colors=[MUTED], linewidths=0.8, linestyles=":")
    ax2.contour(xx, yy, zz, levels=[0], colors=INK, linewidths=2.5)
    p0 = np.array([-0.6, -0.4])  # punkt na prostej z = 0
    ax2.add_patch(FancyArrowPatch(p0, p0 + 0.8 * w, arrowstyle="-|>", mutation_scale=18, color=INK, lw=2.2))
    ax2.text(*(p0 + 0.85 * w + np.array([0.1, -0.05])), "$w$", fontsize=16)
    ax2.text(-2.8, 2.55, "granica decyzyjna:\n$w_1 x_1 + w_2 x_2 + b = 0$", fontsize=12, va="top",
             bbox=dict(fc="white", ec=GRAY, alpha=0.9))
    ax2.text(1.2, -2.6, "klasa 1", fontsize=13, color=RED, weight="bold")
    ax2.text(-2.6, 0.4, "klasa 0", fontsize=13, color=BLUE, weight="bold")
    ax2.set_xlabel("$x_1$")
    ax2.set_ylabel("$x_2$")
    ax2.set_aspect("equal")
    ax2.set_title("$w$ wyznacza kierunek prostej, $b$ jej przesunięcie;\n"
                  "kropkowane: punkty równej pewności", fontsize=12)
    fig.colorbar(cs, ax=ax2, label="$\\hat p$", ticks=[0, 0.25, 0.5, 0.75, 1])
    fig.tight_layout()
    save(fig, "granica_decyzyjna.png")


# --------------------------------------------------------------------------- 5
def mlp_architektura():
    fig, ax = canvas(13.5, 6.0)
    col = {"in": 1.3, "h": 5.3, "out": 9.3, "p": 11.9}
    ys_in = [3.9, 2.4]
    ys_h = [4.6, 3.15, 1.7]
    r = 0.42
    for yi in ys_in:
        for yh in ys_h:
            ax.plot([col["in"] + r, col["h"] - r], [yi, yh], color=tint(BLUE, 0.6), lw=1.3, zorder=0)
    for yh in ys_h:
        ax.plot([col["h"] + r, col["out"] - r], [yh, 3.15], color=tint(BLUE, 0.6), lw=1.3, zorder=0)
    for yi, lab in zip(ys_in, ("$x_1$", "$x_2$")):
        ax.add_patch(Circle((col["in"], yi), r, fc=tint(GRAY, 0.3), ec=GRAY, lw=1.5))
        ax.text(col["in"], yi, lab, ha="center", va="center", fontsize=15)
    for j, yh in enumerate(ys_h):
        ax.add_patch(Circle((col["h"], yh), r, fc=tint(AQUA, 0.3), ec=AQUA, lw=1.5))
        ax.text(col["h"], yh, f"$h_{j + 1}$", ha="center", va="center", fontsize=15)
    ax.add_patch(Circle((col["out"], 3.15), r, fc=tint(BLUE, 0.3), ec=BLUE, lw=1.5))
    ax.text(col["out"], 3.15, "$z$", ha="center", va="center", fontsize=15)
    arrow(ax, (col["out"] + r, 3.15), (col["p"] - 0.45, 3.15), lw=2)
    box(ax, col["p"], 3.15, r"$\hat p$", ORANGE, w=0.75, h=0.7, fontsize=15)

    ax.text(col["h"], 5.35, r"$h_j = \tanh(w^{(1)}_j \cdot x + b^{(1)}_j)$", ha="center", fontsize=13,
            color=AQUA)
    ax.text(col["out"] + 0.2, 4.05, r"$\hat p = \sigma(w^{(2)} \cdot h + b^{(2)})$", ha="center",
            fontsize=13, color=BLUE)

    yl = 0.75
    labels = [
        ((col["in"] + col["h"]) / 2, "nn.Linear(2, 3)", BLUE),
        (col["h"], "nn.Tanh()", AQUA),
        ((col["h"] + col["out"]) / 2 + 0.35, "nn.Linear(3, 1)", BLUE),
        ((col["out"] + col["p"]) / 2 + 0.1, "nn.Sigmoid()", ORANGE),
    ]
    for x, t, c in labels:
        ax.text(x, yl, t, ha="center", va="center", fontsize=11, color=c, family=MONO,
                bbox=dict(boxstyle="round,pad=0.3", fc=tint(c, 0.12), ec=c))
    shapes = [(col["in"], "(N, 2)"), (col["h"], "(N, 3)"), (col["out"], "(N, 1)"), (col["p"], "(N, 1)")]
    for x, t in shapes:
        ax.text(x, 0.05, t, ha="center", va="center", fontsize=10, color=MUTED)
    ax.text(col["in"], 5.35, "wejście", ha="center", fontsize=12, color=MUTED)
    ax.text(col["h"], 5.85, "warstwa ukryta", ha="center", fontsize=12, color=MUTED)
    ax.text(col["out"], 5.35, "wyjście", ha="center", fontsize=12, color=MUTED)
    save(fig, "mlp_architektura.png")


# --------------------------------------------------------------------------- 6
def spadek_gradientu_2d():
    # elongated, slightly rotated bowl L(t1, t2) with its gradient
    def loss(t1, t2):
        return 0.5 * t1**2 + 2.0 * t2**2 + 0.6 * t1 * t2

    def grad(t):
        return np.array([t[0] + 0.6 * t[1], 4.0 * t[1] + 0.6 * t[0]])

    path = [np.array([-2.6, 1.6])]
    for _ in range(25):
        path.append(path[-1] - 0.18 * grad(path[-1]))
    path = np.array(path)

    g = np.linspace(-3, 3, 200)
    T1, T2 = np.meshgrid(g, g)
    L = loss(T1, T2)

    fig = plt.figure(figsize=(13, 5.2))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.plot_surface(T1, T2, L, cmap="Blues_r", alpha=0.75, linewidth=0, rstride=4, cstride=4)
    ax1.plot(path[:, 0], path[:, 1], loss(path[:, 0], path[:, 1]) + 0.1, "o-", color=ORANGE, ms=3, lw=1.5, zorder=5)
    ax1.set_xlabel(r"$\theta_1$")
    ax1.set_ylabel(r"$\theta_2$")
    ax1.set_zlabel(r"$L(\theta_1, \theta_2)$")
    ax1.set_title("Funkcja kosztu jako powierzchnia nad przestrzenią parametrów", fontsize=12)
    ax1.view_init(elev=32, azim=-60)

    ax2 = fig.add_subplot(1, 2, 2)
    cs = ax2.contourf(T1, T2, L, levels=25, cmap="Blues_r")
    ax2.contour(T1, T2, L, levels=12, colors="white", linewidths=0.5, alpha=0.6)
    for k in (0, 2, 5):  # negative gradient at a few steps, scaled for readability
        p, d = path[k], -0.25 * grad(path[k])
        ax2.add_patch(FancyArrowPatch(p, p + d, arrowstyle="-|>", mutation_scale=16, color=RED, lw=2, zorder=4))
    ax2.plot(path[:, 0], path[:, 1], "o-", color=ORANGE, ms=4, lw=1.5, label="kolejne kroki spadku wzdłuż gradientu")
    ax2.plot([], [], color=RED, lw=2, label=r"$-\nabla_\theta L$ - kierunek kroku")
    ax2.scatter(*path[0], color=INK, s=50, zorder=5, label="start (losowe parametry)")
    ax2.scatter(0, 0, marker="*", color=AQUA, s=250, zorder=5, label="minimum")
    ax2.set_xlabel(r"$\theta_1$")
    ax2.set_ylabel(r"$\theta_2$")
    ax2.set_aspect("equal")
    ax2.set_title("Widok z góry: poziomice funkcji kosztu", fontsize=12)
    ax2.legend(loc="lower right", fontsize=9, framealpha=0.95)
    fig.colorbar(cs, ax=ax2, label=r"$L$", shrink=0.85)
    fig.tight_layout()
    save(fig, "spadek_gradientu_2d.png")


if __name__ == "__main__":
    petla_uczenia(LOOP_GENERIC, "petla_uczenia.png")
    for task, hl in {1: (3,), 2: (0, 1), 3: (2,), 4: (3,), 5: (4, "loop")}.items():
        petla_uczenia(LOOP_LOGREG, f"petla_uczenia_zad{task}.png", highlight=hl,
                      loop_label="powtórz - kolejna epoka (Zadanie 5)")
    petla_uczenia(LOOP_TORCH, "petla_uczenia_pytorch.png", code=True,
                  loop_label="powtórz - kolejna epoka (Zadanie 7)")
    regresja_graf()
    regresja_backward()
    granica_decyzyjna()
    mlp_architektura()
    spadek_gradientu_2d()
    print("Zapisano schematy w", OUT)
