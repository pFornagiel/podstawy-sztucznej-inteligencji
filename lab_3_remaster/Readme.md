# Cel

Celem laboratorium jest zrozumienie, jak uczy się sieć neuronowa: od pochodnej, przez spadek wzdłuż gradientu,
do treningu małej sieci w PyTorchu, a następnie trening większych sieci na prawdziwych danych tabelarycznych.

0. Uczenie przez spadek wzdłuż gradientu: inicjalizacja parametrów, przejście w przód, funkcja kosztu,
   gradient, aktualizacja parametrów.
1. Regresja logistyczna od zera w NumPy na danych liniowo separowalnych: pochodna numeryczna, model,
   binarna entropia krzyżowa, propagacja wsteczna (reguła łańcuchowa), pętla ucząca.
2. Wprowadzenie do PyTorcha: tensory, autograd, `nn.Module`, optymalizatory. Regresja logistyczna na zbiorze
   `make_moons` i ograniczenia modeli liniowych.
3. Sieć MLP, która rozdziela klasy w `make_moons`.
4. Sieć MLP na danych tabelarycznych (zbiór HIGGS): trening na paczkach danych (`DataLoader`), zbiór walidacyjny,
   early stopping, dropout, zmiana stałej uczącej w trakcie treningu. Porównanie z gradient boostingiem.
5. Głębsze sieci: zanikający gradient, połączenia rezydualne (ResNet) i normalizacja wsadowa.

Zawartość laboratorium dostępna jest w [notebooku](lab_3.ipynb). Pliki `checkers.py`, `higgs_data.csv.gz` oraz
katalog `figures/` muszą leżeć obok notebooka (plik `hotel_data.csv.gz` jest potrzebny tylko do zadania dodatkowego).
Schematy można wygenerować ponownie poleceniem `python figures/make_figures.py`.
