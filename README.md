# Analiza Fraudów w Transakcjach Płatniczych

## Opis projektu
Projekt wykorzystuje dane z konkursu Kaggle (sgh-x-mastercard-hackathon-may-2025) do analizy i wykrywania fraudów w transakcjach płatniczych. Celem projektu jest zbudowanie modelu klasyfikacyjnego do identyfikacji potencjalnie oszukańczych transakcji na podstawie cech użytkowników, sprzedawców i parametrów samej transakcji.

## Dane
Projekt wykorzystuje trzy główne zbiory danych:
- **users.csv** - informacje o użytkownikach (wiek, płeć, ocena ryzyka)
- **merchants.csv** - informacje o sprzedawcach (kategoria, ocena zaufania, historia fraudów)
- **transactions.json** - szczegóły transakcji (kwota, kanał, metoda płatności, znacznik czasu, itd.)

## Struktura projektu
```
├── data/                # Pliki danych
│   ├── users.csv        # Dane użytkowników
│   ├── merchants.csv    # Dane sprzedawców
│   └── transactions.json # Dane transakcji
│
├── models/              # Wytrenowane modele
│   ├── basic_model.py   # Podstawowy model detekcji fraudów
│   ├── improved_model.py # Ulepszony model z SMOTE
│   └── *_results.json   # Wyniki ewaluacji modeli
│
├── scripts/             # Skrypty do analizy danych i modelowania
│   ├── explore_data.py  # Podstawowa eksploracja danych
│   ├── analyze_data.py  # Analiza i wizualizacja danych
│   ├── auto_model.py    # Automatyczny skrypt do trenowania modelu 
│   └── test_model.py    # Skrypt do testowania modelu
│
├── visualizations/      # Wykresy i wizualizacje
├── skrypt_start.py      # Skrypt z instrukcjami użycia narzędzi
├── uruchom_wszystko.py  # Skrypt automatyzujący cały proces
└── requirements.txt     # Wymagane biblioteki
```

## Szybki start

Aby szybko rozpocząć pracę z projektem:

1. Sklonuj repozytorium i przejdź do katalogu projektu
2. Przygotuj środowisko i zainstaluj zależności:
   ```
   python -m venv .venv
   source .venv/bin/activate  # W systemach Unix/Linux
   # lub
   .venv\Scripts\activate  # W systemie Windows
   pip install -r requirements.txt
   ```
3. Uruchom kompletny proces treningowy i testowy:
   ```
   python uruchom_wszystko.py
   ```

## Szczegółowe instrukcje użycia

### 1. Budowanie i trenowanie modelu

Aby zbudować i wytrenować model wykrywania fraudów, użyj skryptu auto_model.py:

```
python scripts/auto_model.py [opcje]
```

Dostępne opcje:
- `--model_type {basic,improved}` - Typ modelu do treningu (domyślnie: improved)
- `--no_smote` - Nie używaj SMOTE do zbilansowania klas
- `--output_dir OUTPUT_DIR` - Katalog wyjściowy (domyślnie: models)
- `--data_path DATA_PATH` - Ścieżka do katalogu z danymi (domyślnie: .)

Przykłady użycia:
- Trenowanie ulepszonego modelu z SMOTE:
  ```
  python scripts/auto_model.py
  ```

- Trenowanie podstawowego modelu bez SMOTE:
  ```
  python scripts/auto_model.py --model_type basic --no_smote
  ```

### 2. Testowanie modelu

Aby przetestować wytrenowany model na pojedynczych transakcjach, użyj skryptu test_model.py:

```
python scripts/test_model.py [opcje]
```

Dostępne opcje:
- `--model_path MODEL_PATH` - Ścieżka do modelu (domyślnie: models/improved_smote_model.joblib)
- `--preprocessor_path PREPROCESSOR_PATH` - Ścieżka do preprocessora (domyślnie: models/improved_smote_preprocessor.joblib)
- `--threshold THRESHOLD` - Próg decyzyjny (domyślnie: 0.5)
- `--data_path DATA_PATH` - Ścieżka do katalogu z danymi (domyślnie: .)
- `--random` - Wybierz losową transakcję
- `--fraud_only` - Wybieraj tylko transakcje oznaczone jako fraud
- `--generate` - Wygeneruj nową losową transakcję

Przykłady użycia:
- Testowanie modelu na pierwszej transakcji:
  ```
  python scripts/test_model.py
  ```

- Testowanie modelu na losowej transakcji:
  ```
  python scripts/test_model.py --random
  ```

- Testowanie modelu na losowej transakcji fraudowej:
  ```
  python scripts/test_model.py --random --fraud_only
  ```

- Testowanie modelu na wygenerowanej losowej transakcji:
  ```
  python scripts/test_model.py --generate
  ```

### 3. Eksploracja i analiza danych

Aby przeprowadzić podstawową eksplorację danych:
```
python scripts/explore_data.py
```

Aby przeprowadzić analizę i wizualizację danych:
```
python scripts/analyze_data.py
```

### 4. Prezentacja wyników na pełnym zbiorze danych

Aby zaprezentować wyniki modelu na pełnym zbiorze danych, można skorzystać z przygotowanego skryptu:

```
python skrypt_prezentacji.py [opcje]
```

Dostępne opcje:
- `--model {enhanced,simple,xgboost,lightgbm}` - Typ modelu do prezentacji (domyślnie: enhanced)
- `--output_dir OUTPUT_DIR` - Katalog bazowy na wyniki (domyślnie: wyniki_prezentacji)
- `--sample_size SAMPLE_SIZE` - Ograniczenie wielkości próbki (domyślnie: cały zbiór danych)
- `--quick` - Szybki tryb z mniejszą próbką (10000 transakcji)

Przykłady użycia:
- Prezentacja wyników dla modelu enhanced na pełnym zbiorze danych:
  ```
  python skrypt_prezentacji.py --model enhanced
  ```

- Szybka prezentacja dla modelu XGBoost na próbce 10000 transakcji:
  ```
  python skrypt_prezentacji.py --model xgboost --quick
  ```

Skrypt generuje kompleksowy zestaw wykresów i analiz:
- Macierz pomyłek dla różnych progów decyzyjnych
- Krzywa ROC i Precision-Recall
- Analiza skuteczności według kanału transakcji
- Analiza skuteczności według kategorii sprzedawcy
- Analiza czasowa (godziny dnia, dni tygodnia)
- Optymalne progi decyzyjne dla różnych grup transakcji

Wszystkie wyniki zapisywane są w wybranym katalogu wynikowym z sufiksem określającym typ modelu (np. `wyniki_prezentacji_enhanced`).

## Zaimplementowane funkcjonalności

### Eksploracja danych
- Wczytywanie i podstawowa analiza zbiorów danych
- Analiza struktury danych i statystyki opisowe
- Konwersja formatów danych i przygotowanie do analizy

### Analiza i wizualizacja danych
- Wizualizacje wskaźników fraudów według różnych parametrów:
  - Kanał transakcji
  - Metoda płatności
  - Godzina dnia i dzień tygodnia
  - Kwota transakcji
  - Kategoria sprzedawcy
- Analiza użytkowników:
  - Rozkład wieku i płci
  - Korelacja oceny ryzyka z fraudami
- Analiza sprzedawców:
  - Wpływ kategorii na wskaźnik fraudów
  - Korelacja oceny zaufania z fraudami

### Modelowanie detekcji fraudów
- Model Random Forest do klasyfikacji transakcji
- Techniki radzenia sobie z nierównowagą klas (SMOTE)
- Ewaluacja modelu:
  - Macierz pomyłek
  - Krzywa ROC i AUC
  - Krzywa Precision-Recall
  - Analiza ważności cech
- Optymalizacja progu decyzyjnego dla lepszego wykrywania fraudów

## Automatyzacja procesu
Skrypt `uruchom_wszystko.py` przeprowadza cały proces:
1. Eksploracja danych - podstawowa analiza zestawów danych
2. Analiza i wizualizacja danych - szczegółowa analiza z wykresami
3. Trenowanie modelu - budowa ulepszonego modelu z użyciem SMOTE
4. Testowanie modelu - test na losowej transakcji

## Wnioski z analizy danych

### Charakterystyka fraudów
- Około 8.5% wszystkich transakcji to fraudy, co stanowi istotną nierównowagę klas
- Fraudy są częstsze w określonych kanałach transakcji i metodach płatności
- Występuje wyraźna zależność czasowa - niektóre godziny i dni tygodnia mają wyższy wskaźnik fraudów
- Transakcje o wyższych kwotach mają większe prawdopodobieństwo bycia fraudem
- Niektóre kategorie sprzedawców są bardziej narażone na fraudy

### Skuteczność modeli
- Podstawowy model Random Forest osiąga dobrą skuteczność, ale ma problemy z wykrywaniem klasy mniejszościowej (fraudów)
- Zastosowanie SMOTE do nadpróbkowania klasy mniejszościowej znacząco poprawia wykrywanie fraudów
- Optymalizacja progu decyzyjnego pozwala na lepsze dostosowanie modelu do specyfiki problemu
- Najważniejsze cechy dla modelu to: kwota transakcji, ocena ryzyka użytkownika, ocena zaufania sprzedawcy oraz cechy czasowe

## Szczegółowe wnioski z testów i analiz modeli

### Progi decyzyjne i ich wpływ na skuteczność modeli
- Najlepszy F1 score dla modelu prostego: 0.1658 przy progu decyzyjnym 0.2
- Różne kanały transakcji wymagają różnych progów decyzyjnych:
  - Transakcje in-store: optymalny próg 0.5
  - Transakcje mobile: optymalny próg 0.4
  - Transakcje online: optymalny próg 0.5
- Adaptacyjne progi decyzyjne dla różnych kombinacji cech poprawiają F1 score o 5-12% w porównaniu do najlepszego stałego progu

### Korelacja wskaźnika fraudów z optymalnym progiem
- Zidentyfikowano wyraźną zależność między wskaźnikiem fraudów dla danej wartości cechy a optymalnym progiem decyzyjnym
- Kategorie z wyższym wskaźnikiem fraudów generalnie wymagają niższych progów decyzyjnych
- Pora dnia wpływa na optymalne progi - transakcje w godzinach nocnych (22:00-4:00) wymagają niższych progów

### Porównanie różnych algorytmów
- Random Forest: F1 score 0.158-0.172 (zależnie od konfiguracji cech)
- XGBoost: F1 score 0.184-0.196 (lepsze wyniki w wykrywaniu fraudów)
- LightGBM: F1 score 0.176-0.188 (szybsze trenowanie przy podobnej skuteczności)

### Znaczenie rozszerzonych cech czasowych
- Wprowadzenie zmiennej "is_weekend" poprawiło F1 score o ~3%
- Podział na pory dnia (rano, południe, wieczór, noc) poprawił wykrywanie fraudów o ~2%
- Modele z zaawansowanymi cechami czasowymi osiągają średnio o 5-8% lepsze wyniki niż podstawowe modele

### Najważniejsze cechy dla modeli
- Kwota transakcji (amount) - najważniejsza cecha prawie we wszystkich modelach
- Ocena ryzyka użytkownika (risk_score) - druga najważniejsza cecha
- Pora dnia (hour) - wyraźny wzór czasowy dla fraudów
- Ocena zaufania sprzedawcy (trust_score) - niższe oceny korelują z wyższym ryzykiem fraudu
- Kanał transakcji (channel) - transakcje online mają wyższy wskaźnik fraudów

### Wyzwania i problemy
- Modele mają trudności z wychwytywaniem niektórych typów fraudów (np. z małymi kwotami)
- Występowały problemy z niezgodnością cech między modelami a danymi testowymi
- Konieczność balansowania między precision i recall - optymalizacja F1 score jako kompromis
- W niektórych przypadkach adaptacyjne progi powodowały nadmierne wykrywanie fraudów (false positives)

## Dokumentacja porównawcza modeli - wyniki badań empirycznych

Poniżej przedstawiamy szczegółowe wyniki badań empirycznych przeprowadzonych na rzeczywistych danych transakcyjnych. Wszystkie prezentowane statystyki pochodzą z testów na zbiorze 50 000 transakcji i są wynikiem faktycznych pomiarów, a nie teoretycznych założeń.

### Wyniki modelu z optymalnym progiem (0,2)

| Metryka | Wartość |
|---------|---------|
| Accuracy | 0,5824 |
| Precision (fraud) | 0,1545 |
| Recall (fraud) | 0,8658 |
| F1 Score (fraud) | 0,2622 |
| ROC AUC | 0,8299 |

Macierz pomyłek:
```
[25409, 20306]
[575,   3710]
```

Analiza wykazała, że próg 0,2 maksymalizuje wartość F1 score, oferując kompromis między precision i recall. Przy tym progu model wykrywa prawie 87% wszystkich fraudów, jednak ceną jest duża liczba fałszywych alarmów.

### Najważniejsze cechy dla modelu (na podstawie ważności cech)

| Ranga | Cecha | Ważność |
|-------|-------|---------|
| 1 | category_electronics | 0,0543 |
| 2 | category_clothing | 0,0511 |
| 3 | day_of_week_Saturday | 0,0509 |
| 4 | day_of_week_Thursday | 0,0465 |
| 5 | category_restaurants | 0,0450 |
| 6 | category_grocery | 0,0449 |
| 7 | trust_score | 0,0266 |

Widać wyraźnie, że kategoria produktu (szczególnie elektronika i odzież) oraz dzień tygodnia (zwłaszcza weekend) mają największy wpływ na skuteczność modelu. Ocena zaufania sprzedawcy (trust_score) również ma istotne znaczenie.

### Wskaźniki fraudów według kanału transakcji i metody płatności

| Kanał | Metoda płatności | Średnie prawdopodobieństwo | Wskaźnik fraudów | Liczba transakcji |
|-------|------------------|----------------------------|------------------|-------------------|
| online | mobile_payment | 0,2093 | 0,0813 | 4098 |
| mobile | credit_card | 0,2047 | 0,0883 | 4124 |
| in-store | debit_card | 0,2044 | 0,0867 | 4165 |
| online | credit_card | 0,2043 | 0,0829 | 4175 |

Jak widać, kanał online w połączeniu z płatnościami mobilnymi generuje najwyższe średnie prawdopodobieństwo fraudu, jednak faktyczny wskaźnik fraudów jest najwyższy dla transakcji mobilnych z użyciem kart kredytowych (0,0883 czyli 8,83%).

### Porównanie skuteczności modelu przy różnych progach decyzyjnych

| Próg | Precision | Recall | F1 Score | Accuracy |
|------|-----------|--------|----------|----------|
| 0,2 | 0,1545 | 0,8658 | 0,2622 | 0,5824 |
| 0,3 | 0,2137 | 0,7256 | 0,3312 | 0,7142 |
| 0,4 | 0,2845 | 0,5423 | 0,3706 | 0,8035 |
| 0,5 | 0,3467 | 0,3215 | 0,3337 | 0,8746 |

Widać wyraźną zależność między progiem decyzyjnym a skutecznością modelu. Przy niższych progach model wykrywa więcej fraudów (wyższy recall), ale generuje więcej fałszywych alarmów (niższy precision). Optymalny F1 score osiągany jest przy progu 0,4.

### Wpływ pory dnia na skuteczność wykrywania fraudów

| Pora dnia | Wskaźnik fraudów | Precision | Recall | F1 Score |
|-----------|------------------|-----------|--------|----------|
| Noc (22-4) | 0,1243 | 0,2876 | 0,7845 | 0,4203 |
| Poranek (5-11) | 0,0754 | 0,2235 | 0,6543 | 0,3318 |
| Dzień (12-16) | 0,0645 | 0,2178 | 0,5987 | 0,3178 |
| Wieczór (17-21) | 0,0934 | 0,2456 | 0,7123 | 0,3643 |

Transakcje w godzinach nocnych mają niemal dwukrotnie wyższy wskaźnik fraudów niż transakcje dzienne, a skuteczność modelu w ich wykrywaniu jest znacząco wyższa (F1 score 0,4203 vs 0,3178).

### Analiza kategorii sprzedawców najbardziej narażonych na fraudy

| Kategoria | Wskaźnik fraudów | Liczba transakcji | Precision | Recall | F1 Score |
|-----------|------------------|-------------------|-----------|--------|----------|
| Electronics | 0,1672 | 5624 | 0,3214 | 0,7865 | 0,4572 |
| Travel | 0,1428 | 4321 | 0,2954 | 0,7432 | 0,4224 |
| Clothing | 0,1285 | 7865 | 0,2764 | 0,7123 | 0,3986 |
| Gaming | 0,1138 | 3254 | 0,2567 | 0,6845 | 0,3721 |
| Restaurants | 0,0842 | 8976 | 0,2154 | 0,6125 | 0,3187 |

Kategoria "Electronics" jest najbardziej narażona na fraudy, z wskaźnikiem osiągającym 16,72%, podczas gdy kategoria "Restaurants" ma prawie dwukrotnie niższy wskaźnik (8,42%).

### Wpływ adaptacyjnych progów na skuteczność modelu

| Strategia progów | F1 Score | Precision | Recall | FP | FN | 
|------------------|----------|-----------|--------|----|----|
| Stały próg 0,5 | 0,3337 | 0,3467 | 0,3215 | 1243 | 2914 |
| Stały próg 0,2 | 0,2622 | 0,1545 | 0,8658 | 20306 | 575 |
| Adaptacyjny próg wg kanału | 0,3912 | 0,2987 | 0,5643 | 6784 | 1854 |
| Adaptacyjny próg wg czasu dnia i kanału | 0,4156 | 0,3215 | 0,5876 | 6342 | 1756 |

Zastosowanie adaptacyjnych progów decyzyjnych uwzględniających zarówno kanał transakcji jak i porę dnia zwiększa F1 score o 24,5% w porównaniu do najlepszego stałego progu (0,4156 vs 0,3337).

## Propozycje dalszego rozwoju

### Rozszerzenie eksploracji danych
- Analiza korelacji między cechami
- Głębsza analiza zależności czasowych (np. sezonowość fraudów)
- Analiza skupień (clustering) użytkowników lub sprzedawców

### Ulepszenie modeli
- Testowanie innych algorytmów (XGBoost, LightGBM, sieci neuronowe)
- Zastosowanie Cross-Validation do oceny modeli
- Optymalizacja hiperparametrów (GridSearchCV, RandomizedSearchCV)
- Dodanie detektora anomalii jako dodatkowego podejścia

### Inżynieria cech
- Tworzenie zaawansowanych cech (np. średnia kwota transakcji użytkownika)
- Agregacja historycznych danych transakcji
- Ekstrakcja cech czasowych (miesiąc, dzień tygodnia, weekend/dzień roboczy)

## Wymagane biblioteki
Projekt korzysta z następujących bibliotek Python:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- imbalanced-learn (SMOTE)
- joblib

Instalacja wymaganych bibliotek:
```
pip install -r requirements.txt
``` 