# Cel

Celem laboratorium jest zrozumienie, jak uczy się sieć neuronowa: od pochodnej, przez spadek wzdłuż gradientu,
do treningu małej sieci w PyTorchu.

0. Uczenie przez spadek wzdłuż gradientu: inicjalizacja parametrów, przejście w przód, funkcja kosztu,
   gradient, aktualizacja parametrów.
1. Regresja logistyczna od zera w NumPy na danych liniowo separowalnych: pochodna numeryczna, model,
   binarna entropia krzyżowa, propagacja wsteczna (reguła łańcuchowa), pętla ucząca.
2. Wprowadzenie do PyTorcha: tensory, autograd, `nn.Module`, optymalizatory. Regresja logistyczna na zbiorze
   `make_moons` i ograniczenia modeli liniowych.
3. Sieć MLP, która rozdziela klasy w `make_moons`.

Zawartość laboratorium dostępna jest w [notebooku](lab_3.ipynb). Pliki `checkers.py` oraz katalog `figures/`
muszą leżeć obok notebooka. Schematy można wygenerować ponownie poleceniem `python figures/make_figures.py`.
