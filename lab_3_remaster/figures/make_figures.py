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


# --------------------------------------------------------------------------- 7
def minibatche():
    fig, ax = canvas(13.5, 5.9)
    rows, cols, cell = 12, 5, 0.32
    top = 5.05
    batch_colors = [BLUE, ORANGE, AQUA]
    order = [7, 2, 11, 4, 0, 9, 5, 1, 10, 3, 8, 6]  # row order after shuffling

    texts = [[str(i + 1)] + [""] * (cols - 1) for i in range(rows)]
    xe, ye = matrix(ax, 0.6, top, rows, cols, GRAY, cell=cell, label="X_train", shape="(38 400, 28)",
                    texts=texts)
    arrow(ax, (xe + 0.15, top - rows * cell / 2), (xe + 1.75, top - rows * cell / 2), color=INK,
          text="shuffle=True", text_offset=(0, 0.12), fontsize=11)
    ax.text(xe + 0.95, top - rows * cell / 2 - 0.15, "nowa kolejność\nw każdej epoce", ha="center", va="top",
            fontsize=10, color=MUTED)

    x2 = xe + 1.9
    for i, src in enumerate(order):
        color = batch_colors[i // 4]
        for j in range(cols):
            ax.add_patch(Rectangle((x2 + j * cell, top - (i + 1) * cell), cell, cell, fc=tint(color, 0.3),
                                   ec="white", lw=1.5))
        ax.text(x2 + 0.5 * cell, top - (i + 0.5) * cell, str(src + 1), ha="center", va="center", fontsize=10)
    ax.text(x2 + cols * cell / 2, top + 0.1, "po wymieszaniu", ha="center", va="bottom", fontsize=13)

    x3 = x2 + cols * cell + 1.6
    for k, color in enumerate(batch_colors):
        yb = top - (4 * k + 2) * cell
        ax.plot([x2 + cols * cell + 0.12] * 2, [top - 4 * k * cell - 0.05, top - 4 * (k + 1) * cell + 0.05],
                color=color, lw=3, solid_capstyle="round")
        arrow(ax, (x2 + cols * cell + 0.2, yb), (x3 - 0.1, yb), color=color, lw=1.8)
        box(ax, x3 + 1.25, yb, f"paczka {k + 1}", color, w=2.3, h=0.7, fontsize=13)
        ax.text(x3 + 1.25, yb - 0.5, "(256, 28)", ha="center", va="top", fontsize=10, color=MUTED)
        arrow(ax, (x3 + 2.45, yb), (x3 + 3.2, yb), color=color, lw=1.8)
        box(ax, x3 + 4.55, yb, None, color, w=2.6, h=0.7, filled=0.08)
        ax.text(x3 + 4.55, yb, f"krok {k + 1}", ha="center", va="center", fontsize=13)
    ax.text(x3 + 4.55, top + 0.1, "zero_grad, forward,\nbackward, step", ha="center", va="bottom",
            fontsize=10, color=MUTED, family=MONO)

    ax.text(x3 + 2.9, 0.3, "1 epoka = 38 400 / 256 = 150 paczek = 150 kroków optymalizatora\n"
            "(full batch GD: 1 epoka = 1 krok)", ha="center", va="center", fontsize=12, color=INK)
    save(fig, "minibatche.png")


def mlp_tabularny():
    fig, ax = canvas(15.6, 4.3)
    yc = 2.5
    steps = [
        (0.8, "X", GRAY, 0.9, "(B, 28)"),
        (3.1, "nn.Linear\n(28, 256)", BLUE, 1.75, "(B, 256)"),
        (5.7, "nn.ReLU()", AQUA, 1.45, "(B, 256)"),
        (8.3, "nn.Dropout\n(dropout)", GRAY, 1.75, "(B, 256)"),
        (11.45, "nn.Linear\n(256, 1)", BLUE, 1.75, "(B, 1)"),
        (14.35, "logit z", ORANGE, 1.25, ""),
    ]
    for i, (x, text, color, w, shape) in enumerate(steps):
        box(ax, x, yc, None, color, w=w, h=0.95, filled=0.15)
        ax.text(x, yc, text, ha="center", va="center", fontsize=11.5, family=None if i in (0, 5) else MONO)
        if i + 1 < len(steps):
            nx, nw = steps[i + 1][0], steps[i + 1][3]
            arrow(ax, (x + w / 2 + 0.02, yc), (nx - nw / 2 - 0.02, yc), lw=1.6)
            if shape:
                ax.text((x + w / 2 + nx - nw / 2) / 2, yc + 0.12, shape, ha="center", va="bottom", fontsize=9.5,
                        color=MUTED)

    # bracket around the repeated hidden layer
    x0, x1 = steps[1][0] - steps[1][3] / 2 - 0.15, steps[3][0] + steps[3][3] / 2 + 0.15
    ax.add_patch(FancyBboxPatch((x0, yc - 0.75), x1 - x0, 1.5, boxstyle="round,pad=0.02,rounding_size=0.15",
                                fc="none", ec=AQUA, lw=1.5, ls=(0, (4, 3))))
    ax.text((x0 + x1) / 2, yc + 0.95, "warstwa ukryta - powtórzona n_layers razy", ha="center", va="bottom",
            fontsize=12, color=AQUA)
    ax.text((x0 + x1) / 2, yc - 0.95, "w kolejnych warstwach: nn.Linear(256, 256)", ha="center", va="top",
            fontsize=10.5, color=MUTED, family=MONO)
    ax.text(steps[4][0], yc - 0.95, "warstwa wyjściowa", ha="center", va="top", fontsize=10.5, color=MUTED)
    ax.text(steps[5][0], yc - 0.7, "nn.BCEWithLogitsLoss\n(trening)\n\ntorch.sigmoid(z) = p̂\n(predykcja)",
            ha="center", va="top", fontsize=9.5, color=MUTED)
    ax.text(0.2, 0.25, "B - liczba przykładów w paczce, 28 - liczba cech", fontsize=10.5, color=MUTED)
    save(fig, "mlp_tabularny.png")


def _siec(ax, x0, dropped, title, sizes=(4, 6, 6, 1), dx=1.25, dy=0.62, r=0.2):
    """Mała sieć MLP; dropped - zbiór (warstwa, neuron) wyłączonych przez dropout."""
    pos = {}
    for layer, n in enumerate(sizes):
        for k in range(n):
            pos[layer, k] = (x0 + layer * dx, 2.3 + (k - (n - 1) / 2) * dy)
    for layer in range(len(sizes) - 1):
        for a in range(sizes[layer]):
            for b in range(sizes[layer + 1]):
                off = (layer, a) in dropped or (layer + 1, b) in dropped
                ax.plot(*zip(pos[layer, a], pos[layer + 1, b]), color=tint(BLUE, 0.15 if off else 0.55),
                        lw=0.6 if off else 1.0, zorder=0)
    for (layer, k), (x, y) in pos.items():
        color = GRAY if layer == 0 else (ORANGE if layer == len(sizes) - 1 else AQUA)
        if (layer, k) in dropped:
            ax.add_patch(Circle((x, y), r, fc="white", ec=tint(GRAY, 0.6), lw=1.2))
            ax.plot([x - r * 0.7, x + r * 0.7], [y - r * 0.7, y + r * 0.7], color=RED, lw=1.8)
            ax.plot([x - r * 0.7, x + r * 0.7], [y + r * 0.7, y - r * 0.7], color=RED, lw=1.8)
        else:
            ax.add_patch(Circle((x, y), r, fc=tint(color, 0.35), ec=color, lw=1.4))
    ax.text(x0 + dx * (len(sizes) - 1) / 2, 4.55, title, ha="center", va="center", fontsize=12.5)


def dropout_schemat():
    fig, ax = canvas(14.6, 5.0)
    _siec(ax, 0.6, {(1, 1), (1, 4), (2, 0), (2, 2), (2, 5)}, "trening, krok 1")
    _siec(ax, 5.2, {(1, 0), (1, 2), (1, 3), (2, 3), (2, 4)}, "trening, krok 2")
    _siec(ax, 10.3, set(), "predykcja: wszystkie neurony")
    ax.plot([9.6, 9.6], [0.2, 4.8], color=tint(GRAY, 0.5), lw=1)
    ax.text(4.78, 0.15, "model.train(): w każdym kroku losowe neurony ukryte są zerowane (tu p = 0.5),\n"
            "a pozostałe mnożone przez 1 / (1 - p)", ha="center", va="center", fontsize=10.5, color=MUTED)
    ax.text(12.18, 0.15, "model.eval(): dropout wyłączony", ha="center", va="center", fontsize=10.5,
            color=MUTED)
    save(fig, "dropout.png")


def _galaz(ax, x0, yc, skip):
    """Gałąź F(x) bloku: Linear -> ReLU -> Dropout -> Linear. Zwraca x końca ostatniej warstwy."""
    layers = [("Linear", BLUE), ("ReLU", AQUA), ("Dropout", GRAY), ("Linear", BLUE)]
    w, gap = 1.15, 0.38
    box(ax, x0, yc, "$x$", GRAY, w=0.6, h=0.6, fontsize=14)
    x = x0 + 0.3 + gap
    for k, (name, color) in enumerate(layers):
        cx = x + w / 2
        arrow(ax, (x - gap + 0.02, yc), (x - 0.02, yc), lw=1.4)
        box(ax, cx, yc, None, color, w=w, h=0.6, filled=0.15)
        ax.text(cx, yc, name, ha="center", va="center", fontsize=11, family=MONO)
        x += w + gap
    ax.add_patch(FancyBboxPatch((x0 + 0.3 + gap - 0.12, yc - 0.5), x - gap - (x0 + 0.3 + gap) + 0.24, 1.0,
                                boxstyle="round,pad=0.02,rounding_size=0.15", fc="none", ec=BLUE, lw=1.2,
                                ls=(0, (4, 3))))
    ax.text((x0 + 0.3 + x - gap) / 2 + 0.2, yc - 0.62, "$F(x)$", ha="center", va="top", fontsize=13,
            color=BLUE)
    end = x - gap
    if skip:
        px = end + 0.75
        arrow(ax, (end + 0.02, yc), (px - 0.25, yc), lw=1.4)
        ax.add_patch(Circle((px, yc), 0.23, fc="white", ec=INK, lw=1.4))
        ax.text(px, yc, "+", ha="center", va="center", fontsize=15)
        ax.plot([x0, x0, px], [yc + 0.3, yc + 1.05, yc + 1.05], color=ORANGE, lw=2)
        arrow(ax, (px, yc + 1.05), (px, yc + 0.25), color=ORANGE, lw=2)
        ax.text((x0 + px) / 2, yc + 1.15, "połączenie rezydualne: x bez zmian", ha="center", va="bottom",
                fontsize=11, color=ORANGE)
        arrow(ax, (px + 0.25, yc), (px + 0.85, yc), lw=1.4)
        box(ax, px + 1.2, yc, "$h$", INK, w=0.6, h=0.6, fontsize=14, filled=0.05)
    else:
        arrow(ax, (end + 0.02, yc), (end + 0.6, yc), lw=1.4)
        box(ax, end + 0.95, yc, "$h$", INK, w=0.6, h=0.6, fontsize=14, filled=0.05)


def blok_rezydualny():
    fig, ax = canvas(14.4, 6.8)
    ax.text(0.2, 6.5, "Zwykłe warstwy", fontsize=14, color=INK, va="center")
    _galaz(ax, 0.6, 5.55, skip=False)
    ax.text(9.6, 5.55, r"$h = F(x)$", fontsize=16, va="center")
    ax.text(12.0, 5.55, r"$\frac{\partial h}{\partial x} = \frac{\partial F}{\partial x}$", fontsize=17,
            va="center", color=ORANGE)

    ax.plot([0.2, 14.2], [4.45, 4.45], color=tint(GRAY, 0.4), lw=1)
    ax.text(0.2, 4.05, "Blok rezydualny", fontsize=14, color=INK, va="center")
    _galaz(ax, 0.6, 2.3, skip=True)
    ax.text(9.6, 2.3, r"$h = x + F(x)$", fontsize=16, va="center")
    ax.text(12.0, 2.3, r"$\frac{\partial h}{\partial x} = 1 + \frac{\partial F}{\partial x}$", fontsize=17,
            va="center", color=ORANGE)
    ax.text(7.2, 0.45, "wejście i wyjście bloku mają ten sam kształt (N, hidden_size), więc można je dodać;\n"
            "gradient wraca do x także przez ścieżkę z czynnikiem 1, więc nie zanika w głębokich sieciach",
            ha="center", va="center", fontsize=11, color=MUTED)
    save(fig, "blok_rezydualny.png")


if __name__ == "__main__":
    petla_uczenia(LOOP_GENERIC, "petla_uczenia.png")
    petla_uczenia(LOOP_TORCH, "petla_uczenia_pytorch.png", code=True,
                  loop_label="powtórz - kolejna epoka (Zadanie 7)")
    granica_decyzyjna()
    mlp_architektura()
    minibatche()
    mlp_tabularny()
    dropout_schemat()
    blok_rezydualny()
    print("Zapisano schematy w", OUT)
