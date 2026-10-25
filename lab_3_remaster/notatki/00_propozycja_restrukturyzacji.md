# Lab 3 remaster - restructure proposal

Scope: mainly Part II (PyTorch) and Part III (MLP), plus the hooks the future part (real data, minibatches,
regularization, GPU) will need. Part I gets a light pass only. Cell numbers `[N]` refer to `lab_3_solution.ipynb`
as of 2026-10-25 17:15 (the newer of the two notebooks). Draft texts for students are in Polish; everything else is
for you.

## 0. Summary

1. **Keep the current skeleton and all 8 tasks.** Only two task-level changes are proposed: `lr=0.01 -> 0.03` for
   the MLP (reliability, see 4.2) and one given `model.train()` line in the `train_torch` template (see 5).
2. **Cut Part II prose about 50% (13.2k -> ~6.5k chars) and Part III about 35% (4.6k -> ~3k).** The biggest single
   cut is the optimizer cell [50] (3.3k chars). It teaches minibatch SGD, momentum, Adam, AdamW and LR scheduling
   before any of them is used. Each of these moves to the place where it is first used.
3. **Turn one long explanation into a written question** (`// skomentuj tutaj`, as in labs 1/2): the "two linear
   layers = one linear layer" derivation becomes a question under the `nn.Identity` comparison, which already
   shows it on a plot.
4. **Fix two claims that only hold for `torch.manual_seed(0)`** (MLP accuracy check, optimizer comparison text).
5. **Align the notebook frame with labs 1/2**: header block, per-lab `pyproject.toml`/`uv.lock`/`requirements.txt`,
   Colab instructions, `ex`-tagged task cells, points on every task, 10 + 2 points in total.
6. **Notes are private**: full texts are now in `notatki/`. The student notebook still links
   `notatki_gradient_backprop.md` from cell [24], so that link has to go. The repo is public (see 2.4).

## 1. What to keep

From labs 1/2:
- frame: `## Wstęp` (goal + **Uwagi**), `## Wykorzystywane biblioteki`, `## Konfiguracja środowiska` (local `uv` / Colab),
  `### Podstawowe importy`;
- `### Zadanie N (X punktu)` as a separate markdown cell tagged `ex`, followed by template, check, and optionally
  `// skomentuj tutaj`;
- short interpretation paragraphs right after plots ("what you should see and why"), not before;
- a bonus task section at the end (`## Zadanie dodatkowe`).

From the current remaster (these are its strengths, do not lose them):
- the NumPy -> PyTorch mapping table [33]: one table ties the whole part to the student's own functions;
- the "same gradients" cell [48]-[49]: NumPy `backward` vs `loss.backward()` on the same parameters, the best payoff
  in the lab;
- `train_torch` with the optimizer as an argument, reused for every later experiment;
- the visual Part III: plateau, optimizer comparison, hidden-neuron lines + 3D hidden space, architecture comparison;
- `checkers.py` with diagnostic messages (sign error, missing 1/N, gradient accumulation). For from-scratch math
  this works better than the bare `assert`s in labs 1/2, and inline checks would reveal the reference solution.
  Keep it, and add the future tasks to it so that `podsumowanie()` stays complete.

## 2. Global changes

### 2.1 Header block (as in labs 1/2)

```
# Sieci neuronowe
## Wstęp                      - 1 paragraph goal + "Uwagi" (checkers, `// skomentuj tutaj`, run top to bottom)
## Wykorzystywane biblioteki  - NumPy, PyTorch, scikit-learn (only make_moons), matplotlib - 1 line each
## Konfiguracja środowiska
### Środowisko lokalne        - same uv steps as lab 2
### Google Colab              - upload checkers.py + figures/ ; PyTorch is preinstalled
### Podstawowe importy
```

Colab detail: `<img src="figures/...">` does not render in Colab even after upload. Either accept it (images are
supporting, not required) or switch the 6 used figures to raw GitHub URLs once the lab is merged upstream.

### 2.2 Environment

Labs 1/2 each ship `pyproject.toml`, `uv.lock` and `requirements.txt`; `lab_3_remaster/` has none. The root
`.venv` currently has `torch 2.14.0+cu130`, which is a multi-GB CUDA download that students do not need here.
Proposal: give lab 3 its own `pyproject.toml` with the CPU-only PyTorch index (uv documents this setup):

```toml
[tool.uv.sources]
torch = [{ index = "pytorch-cpu" }]

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true
```

The future GPU section is marked "dla zainteresowanych" in the original lab, so students who want it can install
the CUDA build themselves. Write the device code so that it runs anywhere (see 5).

### 2.3 Task format and points

Currently, the student notebook has `### Zadanie 1: Definiowanie pochodnej`, while the solution has
`#### Definiowanie pochodnej` + `#### Zadanie 1 (0.5 punktu)`. Tasks 3 and 5-8 have no points. Proposal:
`### <Temat>` for the section, then a separate `ex` cell `### Zadanie N (X punktu)`.

Points: labs 1 and 2 are both 10 + 2 bonus. The future part also needs points, so a suggested split:

| Task | Pts | Note |
|---|---|---|
| 1 pochodna | 0.5 | |
| 2 sigmoid + model | 1 | as you set it |
| 3 BCE | 0.5 | one line of code |
| 4 backward | 1 | as you set it |
| 5 pętla NumPy | 0.5 | |
| 6 `LogisticRegressionTorch` | 0.5 | |
| 7 `train_torch` | 1 | |
| 8 `MoonMLP` + written question | 1 | question from 4.4 |
| **Parts I-III** | **6** | |
| future part | 4 | e.g. minibatch loop 1.5, regularization/early stopping 1.5, final comparison 1 |
| bonus | 2 | e.g. Optuna (original lab), GPU timing |

### 2.4 Private material and the public repo

`origin` is `github.com/pFornagiel/...`, a public fork of public `agh-cs-ai/...`, and previous labs were built
from student PRs. Both `notatki/` and `lab_3_solution.ipynb` (already committed in `4d4fd91`) will be visible
once pushed. Options: gitignore `notatki/` and keep the solution in a private repo, or keep everything local
until the term ends.

### 2.5 Sync the two notebooks

Markdown has drifted between them (cells 1, 3-4, 7, 13-14, 24-25 differ). The solution is newer
(17:15 vs 15:54), so the student notebook should be regenerated from it by stripping solutions cell by cell,
not the other way round.

## 3. Part II - PyTorch, cell by cell

| Cell | Now (chars) | Action | Target |
|---|---|---|---|
| [33] intro | 1575 | drop the PyTorch history; keep the "NumPy + GPU + autograd" sentence, module list and mapping table; drop figure `petla_uczenia_pytorch.png` (the table says the same) or keep it, your call | ~650 |
| [35] tensory | 802 | 2 short bullets (dtype, shape) | ~450 |
| [36] code | - | keep | - |
| [37] autograd | 1949 | 4 one-line steps; 2 links instead of 4 (autograd tutorial + micrograd) | ~750 |
| [38] code | - | keep | - |
| [39] akumulacja | 429 | 2 sentences | ~230 |
| [40] code | - | keep | - |
| [41] no_grad/detach | 454 | 2 sentences | ~300 |
| [42] code | - | keep | - |
| [43] `nn.Module` | 1560 | table 7 rows -> 4 rows; `train()`/`eval()` move to [54], where they are used; the `zero_grad()` row duplicates [50] | ~750 |
| [44] code | - | keep | - |
| [45] Zadanie 6 | 751 | keep the task; links in markdown only (the student template repeats them in code comments) | ~500 |
| [48]-[49] same gradients | 211 | keep as is | 211 |
| [50] optymalizatory | 3321 | keep only what Task 7 needs; move the rest (see table below) | ~550 |
| [51] Zadanie 7 | 830 | keep; one tutorial link + docs | ~550 |
| [54] torch_proba | 523 | 2 sentences + `train()`/`eval()` | ~350 |
| [56] make_moons | 238 | keep | 238 |
| [58] ograniczenia | 572 | 2 sentences; "capacity" -> Polish | ~300 |
| **Total** | **13 215** | | **~6 500** |

Where the text cut from [50] goes (the full original text is in `03_czesc_II_pytorch.md`):

| Content | New place |
|---|---|
| full-batch GD vs SGD vs minibatch, batch noise as regularization, `batch_size` as a hyperparameter | future part, right before `DataLoader` (this is where the original lab 3 had it) |
| momentum, Adam | Part III, right before the first MLP training (2 bullets, see 4.3) |
| AdamW + weight decay | future part, regularization section |
| LR scheduling, "typical lr" table | future part, hyperparameters section, or notes only |

### 3.1 Drafts (Polish, student-facing)

**[33] -> intro**

```markdown
---
## Część II - PyTorch

W praktyce pętli uczącej nie piszemy od zera, tylko korzystamy z frameworka. PyTorch to w uproszczeniu
NumPy + GPU + [automatyczne różniczkowanie (autograd)](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html):
gradienty, które w Części I liczyliśmy ręcznie, policzy za nas.

Najważniejsze moduły:
- `torch` - tensory i podstawowe funkcje,
- `torch.nn` - warstwy i funkcje kosztu, np. `nn.Linear`, `nn.Sigmoid`, `nn.BCELoss`,
- `torch.optim` - optymalizatory.

Pętla ucząca ma te same kroki co w Części I:

| Krok | Część I (NumPy) | PyTorch |
|---|---|---|
| ... (bez zmian) |

### Tensory
```

**[35] -> tensory**

```markdown
API tensorów jest bardzo podobne do NumPy. Dwie pułapki, na które trafisz w tym laboratorium:

- **typ liczb** - NumPy domyślnie tworzy `float64`, a warstwy PyTorcha pracują na `float32`. Dane z NumPy zamieniamy
  przez `.float()` albo `torch.tensor(..., dtype=torch.float32)`.
- **kształt** - model zwraca kolumnę `(N, 1)`, a etykiety mamy jako wektor `(N,)`. Funkcja kosztu wymaga tego samego
  kształtu, więc etykiety zamieniamy w kolumnę przez `.unsqueeze(1)` (w drugą stronę: `.squeeze(1)`).
```

**[37] -> autograd**

```markdown
### Autograd

Autograd robi automatycznie to samo, co nasza funkcja `backward`:

1. tensory z `requires_grad=True` są śledzone (parametry warstw mają tę flagę od razu, dane i etykiety nie),
2. każda operacja na nich dopisuje węzeł do grafu obliczeń - widać go w atrybucie `grad_fn`,
3. `.backward()` wywołane na koszcie przechodzi graf od końca i mnoży pochodne lokalne (reguła łańcuchowa);
   pochodne są liczone ze wzorów, więc są dokładne, a nie przybliżone jak w `df`,
4. wynik trafia do atrybutu `.grad` każdego śledzonego tensora.

Co dzieje się w środku: [A Gentle Introduction to torch.autograd](https://docs.pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html)
oraz [micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0) Andreja Karpathy'ego - autograd napisany od zera.

Sprawdźmy to na funkcji $f(x) = 3x^2 - 4x + 5$ z Zadania 1:
```

**[39] -> akumulacja**

```markdown
PyTorch nie nadpisuje gradientu, tylko **dodaje** nowy do tego, który już jest w `.grad`. W pętli uczącej trzeba więc
przed każdym `backward()` wyzerować gradienty wszystkich parametrów - robi to `optimizer.zero_grad()`.
```

**[41] -> no_grad / detach**

```markdown
Gdy gradienty nie są potrzebne (predykcja, wykresy), wyłączamy śledzenie blokiem `with torch.no_grad():`.
`.detach()` odcina tensor od grafu (np. przed `.numpy()`), a `.item()` zamienia jednoelementowy tensor na liczbę Pythona.
```

**[43] -> `nn.Module`**

```markdown
### Modele: `nn.Module`

Model w PyTorchu to klasa dziedzicząca po `nn.Module`, podobnie jak nasza `LogisticRegression` z Zadania 2:

| Element | Co robi |
|---|---|
| `__init__` | zaczyna od `super().__init__()`, tworzy warstwy i zapisuje je jako atrybuty - ich parametry są rejestrowane automatycznie |
| `forward(X)` | przejście w przód; jedyna metoda, którą piszemy |
| `model(X)` | wywołuje `forward` - nie wołamy `forward` wprost |
| `model.parameters()` | wszystkie parametry modelu; przekazujemy je optymalizatorowi |

Metody `backward` nie piszemy - autograd zbuduje przejście wstecz z operacji wykonanych w `forward`.
Podstawowa warstwa to [`nn.Linear(in_features, out_features)`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html),
czyli nasze `X @ w + b`. Zobaczmy, jakie parametry rejestruje:
```

**[50] -> optymalizatory**

```markdown
### Optymalizatory

Optymalizator wykonuje krok 4 pętli uczącej. Przy tworzeniu dostaje parametry modelu i stałą uczącą,
np. `optim.SGD(model.parameters(), lr=0.5)`, a potem:

- `optimizer.zero_grad()` zeruje gradienty wszystkich tych parametrów,
- `optimizer.step()` aktualizuje parametry na podstawie gradientów, które `loss.backward()` zapisał w `.grad`.

`optim.SGD` wykonuje dokładnie krok z Części I: $\theta \leftarrow \theta - \eta \, \nabla_\theta L$. Gradient liczymy
tu na całym zbiorze, więc jest to zwykły spadek wzdłuż gradientu. Trening na losowych paczkach danych i inne
optymalizatory poznamy w dalszej części laboratorium.
```

**[54] -> torch_proba**

```markdown
Do predykcji przełączamy model w tryb ewaluacji (`model.eval()`) i wyłączamy gradienty (`torch.no_grad()`).
Tryb ma znaczenie dla warstw, które działają inaczej podczas treningu (poznamy je dalej), dlatego pętla ucząca
zaczyna się od `model.train()`. Funkcja `torch_proba` robi to za nas i zwraca wynik jako tablicę NumPy dla naszych
funkcji rysujących.

Wynik jest taki sam jak dla Twojej implementacji w NumPy - krzywe różnią się tylko na starcie, bo modele zaczynają
z innych losowych parametrów.
```

**[58] -> ograniczenia**

```markdown
Koszt przestaje maleć, a granica decyzyjna regresji logistycznej to zawsze prosta - dłuższy trening ani inna stała
ucząca tego nie zmienią. Zamiast ręcznie projektować nieliniowe cechy, użyjemy sieci neuronowej, która nauczy się ich sama.
```

## 4. Part III - MLP

| Cell | Now (chars) | Action | Target |
|---|---|---|---|
| [59] wstęp MLP | 3504 | keep neuron = our logreg, hidden layer = learned features, figure, 2 formulas, `nn.Sequential` note; universal approximation and the long hidden-layer definition -> notes; linear-collapse derivation -> written question (4.4) | ~1300 |
| new, before [62] | - | momentum + Adam, 2 bullets (moved from [50]) | ~450 |
| [62] code | - | `lr=0.01 -> 0.03` (4.2) | - |
| [63] plateau | 475 | keep; adjust the epoch count after the lr change | ~400 |
| [64] code | - | 3 seeds per optimizer, thin lines (4.3) | - |
| [65] conclusion + hidden space | 505 | rewrite the conclusion so it does not depend on the seed | ~450 |
| [67] architektury | 151 | keep | 151 |
| new after [68] | - | `// skomentuj tutaj` question, part of Task 8 | ~300 |
| [69] `podsumowanie()` | - | move to the very end of the lab (after the future part) | - |

### 4.1 Draft [59]

```markdown
---
## Część III - sieć neuronowa MLP

**Neuron** to iloczyn skalarny wag z wejściem, bias i **funkcja aktywacji** - nasza regresja logistyczna to dokładnie
jeden neuron z aktywacją sigmoidalną. **MLP (multilayer perceptron)** układa neurony w kolejne warstwy. Warstwy między
wejściem a wyjściem to **warstwy ukryte**: ich wyjście $h$ to nowe cechy (**ukryta reprezentacja**), których sieć uczy
się sama. Warstwa wyjściowa to zwykła regresja logistyczna, tyle że na cechach $h$ zamiast $x$.

Nasza sieć ma warstwę ukrytą z 3 neuronami $\tanh$:

<img src="figures/mlp_architektura.png" width="820" alt="architektura MLP">

$$
h = \tanh(x W_1 + b_1) \qquad \hat p = \sigma(h W_2 + b_2)
$$

gdzie $W_1$ ma kształt $2 \times 3$, a $W_2$ - $3 \times 1$. Warstwy łączymy
[`nn.Sequential`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Sequential.html) - nie zwykłą listą
Pythona, bo jej parametry nie trafiłyby do `model.parameters()`.
```

The Task 8 cell stays as is, except: "z optymalizatorem **Adam** (opis poniżej)".

### 4.2 MLP reliability: `lr=0.01 -> 0.03`

The `sprawdz_trening_mlp` threshold is 97%. I trained `MoonMLP(3)` with Adam for 1000 epochs over several seeds:

| Setting | Seeds reaching >= 97% | Plateau escape (cost < 0.2) |
|---|---|---|
| Adam lr=0.01 (current) | 8/10 (seeds 6 and 8 give 0.90 and 0.88) | epoch 556-933 |
| Adam lr=0.03 | 20/20 | epoch 216-304 |
| Adam lr=0.01, 2000 epochs | 20/20 | - |
| 8 hidden neurons, lr=0.01 | 10/10 | - |

Seed 0 passes either way, and the template fixes the layer order, so student inits should match. But any extra
random call before construction (e.g. a student creating a test model) shifts the seed. `lr=0.03` keeps 3 neurons
(needed for the 3D hidden-space plot), keeps a clearly visible plateau (~250 epochs), and halves the time to a good
result. Use the same lr in `build_and_train` [68] and in the Adam entry of [64].

### 4.3 Optimizer comparison [64]-[65]

The current text says momentum "zaczyna z niego wychodzić dopiero pod koniec treningu" and Adam "jako jedyny osiąga
100% dokładności". That holds for seed 0 only. Over 5 seeds: plain SGD (lr 0.1) never escapes (0.87-0.88);
momentum reaches 100% on 3 of 5 (escape at epoch 455-606, otherwise 0.88-0.92); Adam reaches 100% on all of them.
Proposal: plot seeds 0-2 for each optimizer as thin lines in the same color (9 runs, a few seconds) and use a
conclusion that holds across seeds.

New cell before [62] (moved from [50]):

```markdown
Sieć uczymy optymalizatorem **Adam**. Zwykły krok SGD zależy tylko od bieżącego gradientu; dwa popularne ulepszenia:
- **momentum** (`optim.SGD(..., momentum=0.9)`) - krok uwzględnia też kierunek poprzednich kroków, jak kula tocząca
  się z górki; pomaga przejść przez płaskie fragmenty funkcji kosztu,
- **[Adam](https://arxiv.org/abs/1412.6980)** (`optim.Adam`) - momentum + osobna, automatycznie dobierana wielkość
  kroku dla każdego parametru; dobrze działa nawet przy niedokładnie dobranej stałej uczącej i jest domyślnym wyborem
  przy treningu sieci.
```

New [65] conclusion:

```markdown
Zwykłe SGD w 1000 epok nie schodzi z plateau i zostaje na poziomie regresji logistycznej. SGD z momentum zwykle z niego
wychodzi, ale później i nie przy każdym starcie. Adam wychodzi najwcześniej i za każdym razem.
```

### 4.4 Written question (replaces the derivation in [59], part of Task 8)

The derivation `z = x (W_1 W_2) + (b_1 W_2 + b_2)` is short and good, but it is lecture material, and the
`nn.Identity` panel in [68] already shows the result. Asking for it turns 900 chars of reading into one
minute of thinking, which matches labs 1/2. Put the full derivation in the solution notebook.

```markdown
Skomentuj wyniki:
- Dlaczego sieć z 8 neuronami bez aktywacji wyznacza prostą granicę, a sieć z 2 neuronami $\tanh$ już nie?
  Wskazówka: podstaw $h = x W_1 + b_1$ do wzoru na logit warstwy wyjściowej.
- Dlaczego jeden neuron ukryty nie wystarcza do rozdzielenia klas?
```

```markdown
// skomentuj tutaj
```

## 5. Hooks for the future part

Things Parts II/III should hand over, so that the later tasks extend them instead of rewriting them:

1. **`model.train()` as a given first line of `train_torch`** (template, not student code). `torch_proba` switches
   the model to `eval()` and nothing switches it back. That is harmless now, but it silently disables dropout and
   freezes BatchNorm statistics as soon as the future part uses them. The official PyTorch quickstart also calls
   `model.train()` in its train function. The checker is unaffected.
2. **Optional: `loss_fn=None` argument in `train_torch`** (default `nn.BCELoss()`). Lets the future part pass
   `nn.BCEWithLogitsLoss(pos_weight=...)` without a new function. Positional calls in the checker still work.
3. **Logits**: keep `nn.Sigmoid` + `nn.BCELoss` in Parts II/III (it mirrors NumPy 1:1 and the checkers rely on it).
   Introduce `nn.BCEWithLogitsLoss` in the future part with one paragraph: the PyTorch docs recommend it over
   Sigmoid + BCELoss for numerical stability (log-sum-exp), and it has `pos_weight` for class imbalance. Note that
   `torch_proba` assumes probabilities, so logit models need `torch.sigmoid` there (a flag or a second helper).
4. **Text already moved there** (section 3): minibatch / SGD / batch noise, AdamW + weight decay, LR scheduling.
   The full text is in `03_czesc_II_pytorch.md`, cell [50].
5. **Device code**: the original lab hard-codes `.to('cuda')`, which fails on CPU-only machines and Macs. The current
   PyTorch tutorials use
   `device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"`.
6. **Data**: the original lab uses `!wget` from the old UCI URL and `OneHotEncoder(sparse=False)`. That argument
   was removed in scikit-learn 1.4 (the venv has 1.9.1), so the original preprocessing crashes. Ship the data as
   `.csv.gz` like labs 1/2, and use `sparse_output=False`.
7. **Train-set accuracy caveat** in [63]: keep one sentence; the future part introduces validation/test splits.
8. **Checkers**: add the future tasks to `_ZADANIA` and `__all__`, keep `podsumowanie()` as the last cell.
9. **Runtime budget**: Parts I-III currently run in 15 s end to end on this machine, so the future part has plenty of time.

## 6. Part I - light pass (not the focus)

- Typos: `Regersja Logistyczna` [4], `wzdłóż` [3], `funckje` [4], `Mówiac` [4], `(batch),ale Ściśle` [4],
  `zaimplementowanego przez **algorytmu` [4], `później PyTorchu` -> `w PyTorchu` [4], `traningu` [7],
  `optymalizowanie parametrów korzystające` [28]; `Numpy` -> `NumPy` [33, 35]; `Torchu` [43].
- The update formula `θ ← θ - η∇L` is shown in [4], [24] and [50]. Keep it once in [4] and refer back.
- [7]: keep the GIF + central difference; the limit definition is lecture material (it is in the notes).
- [24]: remove the student-notebook link to `notatki_gradient_backprop.md` (private now; a copy is in `notatki/`).
- `figures/petla_uczenia_zad1..5.png` are generated by `make_figures.py` but used nowhere. Either use them
  (one at the top of Tasks 1-5, showing which loop step is being implemented; orientation without text) or remove
  them from `make_figures.py`.
- The intro [3] can lose the second "Dotychczas" sentence; the four-step list in [4] is good, keep it.

Expected effect: Part I ~12k -> ~9-10k chars without touching any task.

## 7. Suggested order of work

1. Decide 2.4 (privacy) and gitignore `notatki/` before the next commit.
2. Fix the frame (2.1, 2.2) and the task headings and points (2.3) in the solution notebook.
3. Part II rewrites (section 3), then Part III (section 4), including the lr change and the new question.
4. Add `model.train()` (and optionally `loss_fn`) to `train_torch`; update the checker message if needed.
5. Part I light pass (section 6).
6. Regenerate the student notebook from the solution, cell by cell (2.5); remove the notes link.
7. Rerun both notebooks top to bottom; the student notebook should show ⏳ for every task, the solution ✅.

## 8. Decisions for you

- Points split (2.3) and whether Task 8 includes the written question.
- Keep or drop `petla_uczenia_pytorch.png` in [33].
- Use or delete the unused `petla_uczenia_zad1..5.png`.
- `loss_fn` argument in `train_torch` now, or later in the future part.
- Where private material lives (gitignore vs separate private repo).

## Sources used

- uv, PyTorch integration (CPU index config): https://docs.astral.sh/uv/guides/integration/pytorch/
- PyTorch quickstart (`torch.accelerator` device selection, `model.train()` in the train loop):
  https://docs.pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
- `Optimizer.zero_grad` (`set_to_none=True` by default, so grads become `None`, not zeros - worth knowing when
  students print `.grad` after `zero_grad()`): https://docs.pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html
- `BCEWithLogitsLoss` (more stable than Sigmoid + BCELoss, `pos_weight`):
  https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
- Seed experiments: run locally (scripts in the session scratchpad); numbers in 4.2 and 4.3.
