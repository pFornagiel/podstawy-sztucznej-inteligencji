"""Funkcje sprawdzające do notatnika lab_3.ipynb.

Każda funkcja `sprawdz_*` przyjmuje Twoje funkcje/klasy jako argumenty i porównuje je
z niezależną implementacją referencyjną. Nie musisz czytać tego pliku.
Na końcu notatnika `podsumowanie()` wypisuje stan wszystkich zadań.
"""

import copy
import math
import numbers
import traceback

import numpy as np

__all__ = [
    "sprawdz_zadanie_1",
    "sprawdz_zadanie_2",
    "sprawdz_zadanie_3",
    "sprawdz_zadanie_4",
    "sprawdz_zadanie_5",
    "sprawdz_zadanie_6",
    "sprawdz_zadanie_7",
    "sprawdz_zadanie_8",
    "sprawdz_trening_mlp",
    "podsumowanie",
]

_ZADANIA = {
    "sprawdz_zadanie_1": "Zadanie 1 - pochodna analityczna i numeryczna",
    "sprawdz_zadanie_2": "Zadanie 2 - sigmoida i model (przejście w przód)",
    "sprawdz_zadanie_3": "Zadanie 3 - funkcja kosztu BCE",
    "sprawdz_zadanie_4": "Zadanie 4 - przejście wstecz (reguła łańcuchowa)",
    "sprawdz_zadanie_5": "Zadanie 5 - pętla ucząca w NumPy",
    "sprawdz_zadanie_6": "Zadanie 6 - regresja logistyczna w PyTorchu",
    "sprawdz_zadanie_7": "Zadanie 7 - pętla ucząca w PyTorchu",
    "sprawdz_zadanie_8": "Zadanie 8 - sieć MLP",
    "sprawdz_trening_mlp": "Trening MLP na make_moons",
}
_WYNIKI = {}


# ---- dane referencyjne (takie same jak w notatniku) -------------------------
def _f(x):
    return 3 * x**2 - 4 * x + 5


def _sigmoid(z):
    return 1 / (1 + np.exp(-z))


def _bce_per_sample(y, p, eps=1e-12):
    p = np.clip(p, eps, 1 - eps)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def _make_linear_dataset(n=200, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-3, 3, size=(3 * n, 2))
    z = X @ np.array([1.5, -1.0]) + 0.5
    keep = np.abs(z) > 0.6
    X, z = X[keep][:n], z[keep][:n]
    return X, (z > 0).astype(float)


# ---- narzędzia ---------------------------------------------------------------
def _close(a, b, tol=1e-4):
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def _is_number(x):
    return isinstance(x, numbers.Real) and not isinstance(x, bool) or (
        isinstance(x, np.ndarray) and x.ndim == 0)


def _expect(got, want, what, tol=1e-4):
    assert _is_number(got), f"{what}: oczekiwano liczby, otrzymano {got!r}"
    got = float(got)
    assert _close(got, want, tol), f"{what}: otrzymano {got:.10g}, oczekiwano {want:.10g}"


def _expect_array(arr, shape, what):
    assert isinstance(arr, np.ndarray), f"{what}: oczekiwano tablicy numpy, otrzymano {type(arr).__name__}"
    assert arr.shape == shape, f"{what}: oczekiwano kształtu {shape}, otrzymano {arr.shape}"


def _expect_allclose(got, want, what, tol=1e-6):
    err = np.max(np.abs(np.asarray(got, dtype=float) - want))
    assert err <= tol * max(1.0, np.max(np.abs(want))), \
        f"{what}: maksymalna różnica względem wartości referencyjnych wynosi {err:.3g}"


def _sprawdzian(fn):
    def wrapper(*args, **kwargs):
        try:
            fn(*args, **kwargs)
        except NotImplementedError as e:
            tb = e.__traceback__
            while tb.tb_next is not None:
                tb = tb.tb_next
            print(f"⏳ Funkcja `{tb.tb_frame.f_code.co_qualname}` czeka na uzupełnienie.")
            _WYNIKI[fn.__name__] = "⏳"
        except AssertionError as e:
            print(f"❌ {e}")
            _WYNIKI[fn.__name__] = "❌"
        except Exception as e:  # błąd w kodzie studenta, np. zły kształt w mnożeniu macierzy
            where = traceback.extract_tb(e.__traceback__)[-1]
            print(f"❌ Twój kod rzucił wyjątek {type(e).__name__}: {e}\n"
                  f"    (funkcja `{where.name}`, linia: {where.line})")
            _WYNIKI[fn.__name__] = "❌"
        else:
            print("✅ Zaliczone!")
            _WYNIKI[fn.__name__] = "✅"
    wrapper.__name__ = fn.__name__
    wrapper.__doc__ = fn.__doc__
    return wrapper


def podsumowanie():
    """Wypisuje stan wszystkich zadań sprawdzonych w tej sesji notatnika."""
    print("Stan zadań (uruchom ponownie cały notatnik, żeby odświeżyć):")
    for key, name in _ZADANIA.items():
        print(f"  {_WYNIKI.get(key, '·')}  {name}")
    done = sum(v == "✅" for v in _WYNIKI.values())
    print(f"\nZaliczone: {done} / {len(_ZADANIA)}   (· = komórka sprawdzająca nie została uruchomiona)")


# ---- Część I -------------------------------------------------------------------
@_sprawdzian
def sprawdz_zadanie_1(analytical_df, df):
    _expect(analytical_df(3.0), 14.0, "analytical_df(3)")
    _expect(analytical_df(0.0), -4.0, "analytical_df(0)")
    _expect(analytical_df(2 / 3), 0.0, "analytical_df(2/3)")
    xs = np.array([-2.0, 0.5, 4.0])
    out = analytical_df(xs)
    _expect_array(out, (3,), "analytical_df(tablica)")
    _expect_allclose(out, 6 * xs - 4, "analytical_df(tablica)")

    got = float(df(math.sin, 0.5))
    assert abs(got - math.cos(0.5)) < 1e-8, (
        f"df(sin, 0.5) = {got:.10f}, a cos(0.5) = {math.cos(0.5):.10f} - błąd {abs(got - math.cos(0.5)):.1e}"
        " jest za duży; czy używasz różnicy centralnej (f(x+h) - f(x-h)) / (2h)?")
    _expect(df(_f, 3.0), 14.0, "df(f, 3)", tol=1e-6)
    _expect(df(np.exp, 1.0, h=1e-4), math.e, "df(exp, 1, h=1e-4)", tol=1e-6)
    out = df(np.sin, np.array([0.0, 1.0, 2.0]))
    _expect_array(out, (3,), "df(np.sin, tablica)")
    _expect_allclose(out, np.cos([0.0, 1.0, 2.0]), "df(np.sin, tablica)")

    print("    x       analytical_df(x)   df(f, x)        różnica")
    for x in (-2.0, 0.0, 2 / 3, 3.0):
        a, n = float(analytical_df(x)), float(df(_f, x))
        print(f"    {x:<8.3f}{a:>13.6f}{n:>16.6f}{abs(a - n):>13.1e}")


@_sprawdzian
def sprawdz_zadanie_2(sigmoid, LogisticRegression):
    _expect(sigmoid(0.0), 0.5, "sigmoid(0)")
    _expect(sigmoid(2.0), 1 / (1 + math.exp(-2.0)), "sigmoid(2)")
    zs = np.linspace(-8, 8, 9)
    out = sigmoid(zs)
    _expect_array(out, zs.shape, "sigmoid(tablica)")
    _expect_allclose(out, _sigmoid(zs), "sigmoid(tablica)")

    m = LogisticRegression(3, seed=0)
    assert hasattr(m, "w") and hasattr(m, "b"), "model musi mieć atrybuty self.w i self.b"
    _expect_array(m.w, (3,), "LogisticRegression(3).w")
    assert np.all(np.abs(m.w) < 1), "wagi startowe powinny być małe (odchylenie standardowe 0.1)"
    assert not np.allclose(m.w, 0), "wagi startowe powinny być losowe, a nie zerowe"
    assert not np.allclose(m.w, LogisticRegression(3, seed=1).w), \
        "różne wartości seed powinny dawać różne wagi - użyj generatora rng"
    _expect(m.b, 0.0, "LogisticRegression(3).b")

    X, _ = _make_linear_dataset(n=7, seed=5)
    m = LogisticRegression(2, seed=0)
    m.w, m.b = np.array([1.0, -2.0]), 0.5
    p = m.forward(X)
    _expect_array(p, (7,), "forward(X) dla X o kształcie (7, 2)")
    _expect_allclose(p, _sigmoid(X @ m.w + m.b), "forward(X)")
    print(f"    sigmoid(0) = {float(sigmoid(0.0)):.3f},  forward(X)[:3] = {np.round(p[:3], 4)}")


@_sprawdzian
def sprawdz_zadanie_3(bce_per_sample, bce):
    y = np.array([1.0, 0.0, 1.0, 0.0])
    p = np.array([0.9, 0.2, 0.5, 0.99])
    want = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    out = bce_per_sample(y, p)
    _expect_array(out, (4,), "bce_per_sample(y, p) dla 4 punktów")
    assert np.all(out >= 0), "koszt nie może być ujemny - czy nie brakuje minusa przed sumą logarytmów?"
    _expect_allclose(out, want, "bce_per_sample")
    _expect(bce(y, p), want.mean(), "bce(y, p)")
    _expect(bce(y, y), 0.0, "bce(y, y) - koszt idealnej predykcji", tol=1e-6)
    worst = float(bce(np.array([1.0]), np.array([0.0])))
    assert math.isfinite(worst), "bce dla p = 0 i y = 1 powinno być skończone - nie usuwaj np.clip"
    print("    y    p      koszt punktu")
    for yi, pi, li in zip(y, p, out):
        print(f"    {yi:<5.0f}{pi:<7.2f}{li:>8.4f}")
    print(f"    średnia (bce) = {float(bce(y, p)):.4f}")


def _ref_grads(w, b, X, y):
    p = _sigmoid(X @ w + b)
    return X.T @ (p - y) / len(y), np.sum(p - y) / len(y)


def _full_numeric_grads(w, b, X, y, h=1e-6):
    def L(w_, b_):
        return np.mean(_bce_per_sample(y, _sigmoid(X @ w_ + b_)))

    gw = np.array([(L(w + h * e, b) - L(w - h * e, b)) / (2 * h) for e in np.eye(len(w))])
    gb = (L(w, b + h) - L(w, b - h)) / (2 * h)
    return gw, gb


@_sprawdzian
def sprawdz_zadanie_4(backward, LogisticRegression):
    X, y = _make_linear_dataset(n=50, seed=3)
    m = LogisticRegression(2, seed=0)
    m.w, m.b = np.array([0.7, -1.2]), 0.3
    out = backward(m, X, y)
    assert isinstance(out, tuple) and len(out) == 2, "backward musi zwracać krotkę (grad_w, grad_b)"
    gw, gb = out
    _expect_array(gw, (2,), "grad_w (kształt jak model.w)")
    assert _is_number(gb), f"grad_b powinno być liczbą, otrzymano {type(gb).__name__} {np.shape(gb)}"
    assert np.allclose(m.w, [0.7, -1.2]) and m.b == 0.3, "backward nie powinien zmieniać parametrów modelu"

    rw, rb = _ref_grads(m.w, m.b, X, y)
    nw, nb = _full_numeric_grads(m.w, m.b, X, y)
    print(f"    {'parametr':<10}{'Twój backward':>15}{'cała strata ±h':>17}{'wzór analityczny':>19}")
    for name, mine, num, ref in (("w1", gw[0], nw[0], rw[0]), ("w2", gw[1], nw[1], rw[1]),
                                 ("b", float(gb), nb, rb)):
        print(f"    {name:<10}{mine:>15.6f}{num:>17.6f}{ref:>19.6f}")
    for name, mine, ref in (("w1", gw[0], rw[0]), ("w2", gw[1], rw[1]), ("b", float(gb), rb)):
        if _close(mine, ref * len(y), 1e-4) and not _close(mine, ref, 1e-4):
            raise AssertionError(f"gradient {name} jest {len(y)} razy za duży - pamiętaj o 1/N ze średniej")
        if _close(mine, -ref, 1e-4) and not _close(mine, ref, 1e-4):
            raise AssertionError(f"gradient {name} ma odwrotny znak - sprawdź kolejność w różnicy w df")
        assert _close(mine, ref, 1e-5), f"gradient {name}: otrzymano {mine:.6f}, oczekiwano {ref:.6f}"


@_sprawdzian
def sprawdz_zadanie_5(train, LogisticRegression):
    X, y = _make_linear_dataset()
    m = LogisticRegression(2, seed=0)
    w0, b0 = np.array([0.05, -0.1]), 0.0
    m.w, m.b = w0.copy(), b0
    hist = train(m, X, y, lr=0.5, epochs=100)
    assert hist is not None and len(hist) == 100, \
        f"historia powinna mieć 100 wpisów (po jednym na epokę), ma {None if hist is None else len(hist)}"
    assert all(_is_number(h) for h in hist), "historia powinna zawierać liczby (wartości kosztu)"

    # referencja: ten sam start, wzór analityczny na gradient
    w, b, ref = w0.copy(), b0, []
    for _ in range(100):
        ref.append(np.mean(_bce_per_sample(y, _sigmoid(X @ w + b))))
        gw, gb = _ref_grads(w, b, X, y)
        w, b = w - 0.5 * gw, b - 0.5 * gb

    _expect(hist[0], ref[0], "koszt w epoce 0 (przed pierwszą aktualizacją)")
    for i in (0, 1, 10, 50, 99):
        print(f"    epoka {i:3d}   koszt {float(hist[i]):.5f}   (referencja {ref[i]:.5f})")
    assert all(b_ <= a_ + 1e-9 for a_, b_ in zip(hist, hist[1:])), \
        "koszt w którejś epoce wzrósł - sprawdź znak w kroku aktualizacji (w = w - lr * grad)"
    _expect(hist[-1], ref[-1], "koszt w ostatniej epoce", tol=1e-5)
    _expect_allclose(m.w, w, "wyuczone w (model powinien być zmieniony w miejscu)", tol=1e-5)
    _expect(m.b, b, "wyuczone b", tol=1e-5)
    acc = np.mean((_sigmoid(X @ m.w + m.b) >= 0.5) == y)
    print(f"    dokładność na zbiorze treningowym: {acc:.1%}")
    assert acc == 1.0, "na danych liniowo separowalnych model powinien klasyfikować wszystko poprawnie"


# ---- Część II ------------------------------------------------------------------
def _linear_layers(model):
    import torch.nn as nn
    return [mod for mod in model.modules() if isinstance(mod, nn.Linear)]


@_sprawdzian
def sprawdz_zadanie_6(LogisticRegressionTorch):
    import torch
    import torch.nn as nn

    for d in (2, 5):
        model = LogisticRegressionTorch(d)
        assert isinstance(model, nn.Module), "model musi dziedziczyć po nn.Module"
        assert isinstance(getattr(model, "linear", None), nn.Linear), \
            "model musi mieć atrybut self.linear typu nn.Linear"
        assert isinstance(getattr(model, "activation", None), nn.Sigmoid), \
            "model musi mieć atrybut self.activation typu nn.Sigmoid"
        lin = model.linear
        assert (lin.in_features, lin.out_features) == (d, 1), \
            f"dla n_features={d} warstwa liniowa powinna mieć {d} wejść i 1 wyjście, " \
            f"a ma {lin.in_features} i {lin.out_features}"
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params == d + 1, f"model powinien mieć {d + 1} parametrów, ma {n_params}"

        X = torch.randn(6, d)
        out = model(X)
        assert isinstance(out, torch.Tensor), "forward musi zwracać tensor"
        assert out.shape == (6, 1), f"wyjście dla X o kształcie (6, {d}) powinno mieć kształt (6, 1), " \
                                    f"ma {tuple(out.shape)}"
        with torch.no_grad():
            want = torch.sigmoid(X @ lin.weight.T + lin.bias)
        assert torch.allclose(out, want, atol=1e-6), "forward powinien zwracać sigmoid(linear(X))"
    print(f"    parametry: weight {tuple(model.linear.weight.shape)}, bias {tuple(model.linear.bias.shape)}")


def _ref_train_torch(model, optimizer, X, y, epochs, zero_grad=True):
    import torch.nn as nn

    loss_fn, hist = nn.BCELoss(), []
    for _ in range(epochs):
        if zero_grad:
            optimizer.zero_grad()
        loss = loss_fn(model(X), y)
        hist.append(loss.item())
        loss.backward()
        optimizer.step()
    return hist


@_sprawdzian
def sprawdz_zadanie_7(train_torch, LogisticRegressionTorch):
    import torch
    import torch.optim as optim

    X_np, y_np = _make_linear_dataset()
    X = torch.tensor(X_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.float32).unsqueeze(1)

    torch.manual_seed(0)
    model = LogisticRegressionTorch(2)
    ref_model, bad_model = copy.deepcopy(model), copy.deepcopy(model)
    hist = train_torch(model, optim.SGD(model.parameters(), lr=0.5), X, y, epochs=50)
    assert hist is not None and len(hist) == 50, \
        f"historia powinna mieć 50 wpisów, ma {None if hist is None else len(hist)}"
    assert all(isinstance(h, float) for h in hist), \
        "historia powinna zawierać liczby float - zapisuj loss.item(), a nie sam tensor loss"
    ref = _ref_train_torch(ref_model, optim.SGD(ref_model.parameters(), lr=0.5), X, y, 50)
    bad = _ref_train_torch(bad_model, optim.SGD(bad_model.parameters(), lr=0.5), X, y, 50, zero_grad=False)
    if np.allclose(hist, bad, atol=1e-5) and not np.allclose(hist, ref, atol=1e-5):
        raise AssertionError("gradienty się kumulują - wyzeruj je w każdej epoce (optimizer.zero_grad())")
    for i in (0, 1, 10, 49):
        print(f"    epoka {i:3d}   koszt {hist[i]:.5f}   (referencja {ref[i]:.5f})")
    assert np.allclose(hist, ref, atol=1e-5), \
        "historia kosztu różni się od referencyjnej - sprawdź kolejność kroków w pętli"
    for p, q in zip(model.parameters(), ref_model.parameters()):
        assert torch.allclose(p, q, atol=1e-5), "parametry po treningu różnią się od referencyjnych"
    with torch.no_grad():
        acc = ((model(X) >= 0.5).float() == y).float().mean().item()
    print(f"    dokładność na zbiorze treningowym: {acc:.1%}")


# ---- Część III -----------------------------------------------------------------
@_sprawdzian
def sprawdz_zadanie_8(MoonMLP):
    import torch
    import torch.nn as nn

    for h in (3, 7):
        model = MoonMLP(hidden_size=h)
        assert isinstance(getattr(model, "layers", None), nn.Sequential), \
            "model musi mieć atrybut self.layers typu nn.Sequential"
        kinds = [type(mod) for mod in model.layers]
        assert kinds == [nn.Linear, nn.Tanh, nn.Linear, nn.Sigmoid], (
            "self.layers powinno zawierać kolejno: Linear, Tanh, Linear, Sigmoid; "
            f"zawiera: {', '.join(k.__name__ for k in kinds)}")
        l1, l2 = model.layers[0], model.layers[2]
        assert (l1.in_features, l1.out_features) == (2, h), \
            f"pierwsza warstwa dla hidden_size={h} powinna być Linear(2, {h}), " \
            f"jest Linear({l1.in_features}, {l1.out_features})"
        assert (l2.in_features, l2.out_features) == (h, 1), \
            f"druga warstwa powinna być Linear({h}, 1), jest Linear({l2.in_features}, {l2.out_features})"
        X = torch.randn(5, 2)
        out = model(X)
        assert out.shape == (5, 1), f"wyjście powinno mieć kształt (5, 1), ma {tuple(out.shape)}"
        assert torch.allclose(out, model.layers(X)), "forward powinien po prostu wywołać self.layers"
    n_params = sum(p.numel() for p in MoonMLP(hidden_size=3).parameters())
    print(f"    MoonMLP(hidden_size=3) ma {n_params} parametrów (2·3 + 3 + 3·1 + 1)")


@_sprawdzian
def sprawdz_trening_mlp(model, X, y):
    import torch

    model.eval()
    with torch.no_grad():
        acc = ((model(X) >= 0.5).float() == y).float().mean().item()
    print(f"    dokładność na make_moons: {acc:.1%}")
    assert acc >= 0.97, "dokładność poniżej 97% - sprawdź architekturę i liczbę epok treningu"
