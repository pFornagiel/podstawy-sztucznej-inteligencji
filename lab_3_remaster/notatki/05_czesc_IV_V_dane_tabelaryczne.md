# Notatki: Części IV i V - sieci na danych tabelarycznych, ResNet

Materiał prowadzącego: dlaczego taki zbiór i takie ustawienia, wyniki eksperymentów, które nie trafiły do notebooka,
oraz tło z literatury. Liczby AUROC dotyczą zbioru testowego, jeśli nie napisano inaczej.

## Punktacja

| Zadanie | Pkt | Temat |
|---|---|---|
| 9 | 0.5 | `train_epoch` - pętla na paczkach |
| 10 | 0.5 | `TabularMLP` |
| 11 | 0.5 | `EarlyStopping` |
| 12 | 0.5 | `MyDropout` (dropout od zera) |
| 13 | 0.5 | `ReduceLROnPlateau` + komentarz do wykresu |
| 14 | 1 | `ResidualBlock` + `TabularResNet` |
| 15 | 0.5 | `NormResidualBlock` + komentarz do porównania |
| **Części IV-V** | **4** | razem z Częściami I-III: 10 pkt |
| dodatkowe 1 | 1 | sieć vs boosting na danych hotelowych z lab. 2 |
| dodatkowe 2 | 1 | random search hiperparametrów |

## Wybór zbioru danych

Wymagania: prawdziwe dane tabelaryczne, każda technika (minibatch, early stopping, dropout, scheduler, ResNet) daje
widoczną i powtarzalną poprawę, trening na CPU w sekundach, plik danych < 5 MB (limit pre-commit).

Sprawdzone kandydaty (MLP 3x256, Adam lr=1e-3, 40 epok, QuantileTransformer, podział 64/16/20; `HGB` =
`HistGradientBoostingClassifier`; jedno ziarno):

| Zbiór | Typ | HGB | MLP bez regularyzacji, ostatnia / najlepsza epoka | Uwagi |
|---|---|---|---|---|
| California Housing | regresja, 8 cech | RMSE 0.463 | RMSE 0.516 / 0.508 | słabe przeuczenie, dropout nie pomaga |
| Adult | binarna, mieszane cechy | 0.926 | 0.875 / 0.913 | boosting wyraźnie lepszy, zyski z dropoutu małe |
| Credit default (Tajwan) | binarna | 0.780 | 0.696 / 0.775 | małe różnice, wynik zaszumiony |
| Hotel (lab 2) | binarna, mieszane cechy | 0.941 | 0.917 / 0.937 | sieć nie dogania boostingu - użyte w zadaniu dodatkowym 1 |
| **HIGGS (98k)** | binarna, 28 cech liczbowych | 0.796 | 0.735 / 0.785 | duże przeuczenie, każda technika pomaga, ResNet > HGB |

HIGGS wybrany, bo daje najczystszą historię "krok po kroku" i ma ładne tło (Baldi i in. 2014 pokazali na nim, że głębokie
sieci uczą się cech wysokiego poziomu z niskopoziomowych). Wada: cechy fizyczne są mniej intuicyjne niż w lab. 1-2,
dlatego w notebooku jest krótki opis i jeden wykres rozkładów (pik `m_bb`, płaski `jet1phi`).

### Plik danych

`higgs_data.csv.gz` (3.8 MB) to losowa próbka 60 000 z 98 049 kompletnych wierszy wersji OpenML 23512, zaokrąglona do
4 cyfr znaczących. Pełne 98k wierszy przy 4 cyfrach to 6.2 MB (powyżej limitu), przy 3 cyfrach 4.85 MB.

```python
import pandas as pd
from sklearn.datasets import fetch_openml

df = fetch_openml(data_id=23512, as_frame=True, parser="auto").frame.dropna()
df["class"] = df["class"].astype(int)
df = df[[c for c in df.columns if c != "class"] + ["class"]]
df = df.sample(n=60000, random_state=0).reset_index(drop=True)
df.to_csv("higgs_data.csv.gz", index=False, float_format="%.4g")
```

## Wyniki - 3 ziarna losowości

Ustawienia jak w notebooku (StandardScaler, Adam lr=3e-3, batch 256, early stopping patience 6,
`ReduceLROnPlateau(factor=0.5, patience=2)`); w notebooku jest ziarno 0.

| Model | AUROC (średnia ± odch. std., 3 ziarna) | ziarno 0 |
|---|---|---|
| regresja logistyczna | 0.690 | 0.690 |
| gradient boosting (HGB, domyślny) | 0.795 | 0.795 |
| MLP, ostatnia z 40 epok | 0.733 ± 0.005 | 0.731 |
| + early stopping | 0.780 ± 0.001 | 0.780 |
| + dropout 0.3 | 0.788 ± 0.001 | 0.787 |
| + ReduceLROnPlateau | 0.793 ± 0.001 | 0.792 |
| ResNet, 3 bloki x 128, bez BN | 0.795 ± 0.001 | 0.797 |
| ResNet + batch norm (pre-norm) | 0.802 ± 0.001 | 0.803 |

Progi w `check_experiment` są ustawione z zapasem ok. 1-2 punktów poniżej tych wartości.

### Dlaczego lr = 3e-3

Przy lr = 1e-3 scheduler daje tylko +0.1-0.2 pkt (w granicach szumu). Przy 3e-3 poprawa jest systematyczna: bez
schedulera 0.788 ± 0.0014, z `ReduceLROnPlateau` 0.793 ± 0.0005 - każde ziarno ze schedulerem jest lepsze od każdego
bez niego. Przy lr = 1e-2: 0.780 -> 0.788. `CosineAnnealingLR` i `OneCycleLR` dają podobne zyski; wybrany został
`ReduceLROnPlateau`, bo reaguje na koszt walidacyjny, więc łączy się z logiką early stopping (ta sama "cierpliwość").

### Dropout

| p | AUROC (lr 1e-3, 3 ziarna) | najlepsza epoka |
|---|---|---|
| 0 | 0.774 | ~5 |
| 0.2 | 0.786 | ~15 |
| 0.3 | 0.790 | ~20 |
| 0.5 | 0.793 | ~35 (wymaga >40 epok) |

### Skąd zysk ResNetu - uczciwy rozkład

Wariant z BN także w głowicy (jak u Gorishniy i in.), lr 3e-3, 3 bloki, 3 ziarna: z połączeniami rezydualnymi
0.805 ± 0.001, ten sam model bez nich (`forward` zwraca `self.block(self.norm(X))`) 0.802 ± 0.0004. Czyli przy
3 blokach prawie cały zysk względem MLP (0.793) daje **batch norm**, a połączenia rezydualne dokładają ok. 0.3 pkt.
Dlatego w notebooku:
- połączenia rezydualne są pokazane przede wszystkim jako sposób na trenowanie głębokich sieci (MLP z 20 warstwami:
  AUROC 0.50, koszt stoi na ln 2; ResNet z 10 blokami trenuje się jak płytka sieć),
- tekst mówi wprost, że ResNet jest tylko nieznacznie lepszy od MLP (zgodnie z Gorishniy i in.),
- batch norm jest osobnym krokiem (Zadanie 15).

Demonstracja głębokości jest bez BN, bo BN sam częściowo ratuje głębokie sieci: zwykłe MLP z BN po każdej warstwie
po 10 epokach (lr 1e-3, QuantileTransformer) - 4 warstwy: koszt treningowy 0.497, AUROC 0.772; 20 warstw: 0.553, 0.748; 50 warstw: 0.689, 0.537.
ResNet z BN z 2, 10 i 25 blokami trenuje się tak samo dobrze (koszt 0.517-0.526, AUROC ~0.79). To odtwarza "degradation
problem" z pracy He i in. (2016) - można o tym wspomnieć na wykładzie.

Normy gradientu przy inicjalizacji (jedna paczka, MLP 20 x 128): pierwsza warstwa 1.8e-9, ostatnia 2.2e-2 (spadek
ok. 2.3 raza na warstwę - domyślna inicjalizacja `nn.Linear` to U(-1/sqrt(fan_in), 1/sqrt(fan_in)), czyli wariancja
1/(3 fan_in), a ReLU zeruje połowę). ResNet 10 bloków: wszystkie warstwy 5e-2 - 4e-1.

### Skalowanie cech

MLP (dropout 0.3 + scheduler) na surowych cechach: 0.756 wobec 0.792 po standaryzacji. ResNet z BN na surowych cechach:
0.799 wobec 0.800 (jedno ziarno) - batch norm normalizuje wewnątrz sieci (w notebooku tylko sugestia, żeby to sprawdzić).
QuantileTransformer (domyślny w pracy Gorishniy i in.) daje na HIGGS praktycznie to samo co StandardScaler (0.793 vs
0.791), więc w notebooku jest znany z lab. 1 StandardScaler.

### Cechy nieinformatywne (nie weszło do notebooka)

Usunięcie 6 kątów `*phi` (izotropia detektora - pojedynczy kąt azymutalny nie niesie informacji): MLP 0.791 -> 0.799
(+0.8 pkt), HGB 0.795 -> 0.796 (+0.1 pkt). To dokładnie obserwacja 2 z Grinsztajn i in. (2022): nieinformatywne cechy
szkodzą MLP bardziej niż drzewom. W notebooku jest tylko wzmianka w podsumowaniu (`jet1phi` na wykresie EDA) - można
to dać jako pytanie dodatkowe albo pokazać na wykładzie.

### Czas działania

Na Apple M (15 rdzeni): Części I-III ok. 5 s, Część IV ok. 25 s, Część V ok. 30 s, zadanie dodatkowe 1 ok. 5 s,
zadanie dodatkowe 2 (random search, 15 treningów) ok. 2 minut. Pojedynczy trening to 4-8 s, boosting poniżej 3 s.
Na Colabie (2 vCPU) należy się spodziewać kilka razy dłużej.

## Tło z literatury

**Gorishniy, Rubachev, Khrulkov, Babenko - "Revisiting Deep Learning Models for Tabular Data" (NeurIPS 2021),
https://arxiv.org/abs/2106.11959**
- MLP: `MLP(x) = Linear(MLPBlock(...(MLPBlock(x))))`, `MLPBlock(x) = Dropout(ReLU(Linear(x)))` - to jest nasz `TabularMLP`.
- ResNet: `ResNetBlock(x) = x + Dropout(Linear(Dropout(ReLU(Linear(BatchNorm(x))))))`,
  `Prediction(x) = Linear(ReLU(BatchNorm(x)))`. Nasz `NormResidualBlock` to ten blok bez drugiego dropoutu i bez
  osobnej (szerszej) warstwy wewnętrznej; głowica bez BN.
- Protokół: AdamW, bez harmonogramu lr, early stopping z patience 16, batch 256-1024, QuantileTransformer.
- Wnioski: "MLP is still a good sanity check", "ResNet turns out to be an effective baseline that none of the
  competitors can consistently outperform", FT-Transformer najlepszy średnio; po strojeniu GBDT wygrywa na części
  zbiorów (California Housing, Adult, Yahoo) - "there is still no universally superior solution".
- Na Higgs small (98k), accuracy: pojedyncze sieci (tab. 2) MLP 0.723, ResNet 0.727, FT-T 0.729; ensemble boostingu
  (tab. 4) XGBoost 0.717 (domyślny) / 0.728 (strojony), CatBoost 0.728 / 0.729 - czyli na HIGGS sieci i boosting są
  na równi, co zgadza się z naszymi wynikami.

**Ye, Liu, Cai, Zhou, Zhan - "A Closer Look at Deep Learning Methods on Tabular Datasets" (TALENT, 300+ zbiorów),
https://arxiv.org/abs/2407.00956**
- Boosting (CatBoost, LightGBM) to nadal bardzo mocne punkty odniesienia; dostrojone MLP (RealMLP, MLP-PLR)
  zmniejszają różnicę, a modele pretrenowane (TabPFN v2, TabICL) dorównują lub wygrywają na wielu zadaniach.
- Kluczowa jest **niejednorodność** danych (cechy kategoryczne obok liczbowych, różne rozkłady): na zbiorach mieszanych
  zwykłe MLP wypada najgorzej, a drzewa najlepiej. Na danych czysto liczbowych sieci radzą sobie lepiej.
- Ensembling pomaga i drzewom, i sieciom. Wczesna dynamika treningu pozwala przewidzieć późniejszy wynik walidacyjny.

**Grinsztajn, Oyallon, Varoquaux - "Why do tree-based models still outperform deep learning on typical tabular data?"
(NeurIPS 2022), https://arxiv.org/abs/2207.08815** - trzy powody: sieci preferują zbyt gładkie funkcje, są wrażliwe na
nieinformatywne cechy i są niezmiennicze względem obrotów przestrzeni cech (a dane tabelaryczne nie są).

Inne, które mogą się przydać na wykładzie: Kadra i in. 2021 "Well-tuned Simple Nets Excel on Tabular Datasets"
(koktajl regularyzacji dla MLP), Holzmüller i in. 2024 "Better by Default" (RealMLP), He i in. 2016 (ResNet),
Ioffe, Szegedy 2015 (batch norm), Srivastava i in. 2014 (dropout).

## Celowo pominięte w notebooku

Żeby część była wprowadzająca, nie ma w niej: QuantileTransformer, embeddingów cech liczbowych (PLE, periodic),
FT-Transformera i TabPFN, AdamW/weight decay (L2 była w lab. 1), warmup i harmonogramów kosinusowych (tylko wzmianka),
ensemblingu sieci. Każdy z tych tematów nadaje się na rozszerzenie lub zadanie dodatkowe w kolejnych edycjach.
