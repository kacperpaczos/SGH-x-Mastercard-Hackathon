#!/usr/bin/env python
# -*- coding: utf-8 -*-

def main():
    print("""
╔════════════════════════════════════════════════════════════════════════════════╗
║                                                                                ║
║           NARZĘDZIA DO WYKRYWANIA FRAUDÓW W TRANSAKCJACH PŁATNICZYCH          ║
║                                                                                ║
╚════════════════════════════════════════════════════════════════════════════════╝

1. BUDOWANIE I TRENOWANIE MODELU
--------------------------------

python scripts/auto_model.py [opcje]

Opcje:
    --model_type {basic,improved}  Typ modelu (domyślnie: improved)
    --no_smote                     Bez SMOTE
    --output_dir OUTPUT_DIR        Katalog wyjściowy (domyślnie: models)
    --data_path DATA_PATH          Ścieżka do danych (domyślnie: .)

Przykłady:
    python scripts/auto_model.py
    python scripts/auto_model.py --model_type basic --no_smote
    python scripts/auto_model.py --data_path /ścieżka/do/danych


2. TESTOWANIE MODELU
-------------------

python scripts/test_model.py [opcje]

Opcje:
    --model_path MODEL_PATH               Ścieżka do modelu
    --preprocessor_path PREPROCESSOR_PATH Ścieżka do preprocessora
    --threshold THRESHOLD                 Próg decyzyjny (0.5)
    --data_path DATA_PATH                 Ścieżka do danych
    --random                              Losowa transakcja
    --fraud_only                          Tylko fraudy
    --generate                            Generuj transakcję

Przykłady:
    python scripts/test_model.py
    python scripts/test_model.py --random
    python scripts/test_model.py --random --fraud_only
    python scripts/test_model.py --generate
    python scripts/test_model.py --threshold 0.3


3. ANALIZA DANYCH
----------------

python scripts/explore_data.py
python scripts/analyze_data.py


4. BIBLIOTEKI
------------

pip install -r requirements.txt

Wymagane:
    • pandas
    • numpy
    • matplotlib
    • seaborn
    • scikit-learn
    • imbalanced-learn
    • joblib


5. STRUKTURA
-----------

data/                   # Dane
├── users.csv          # Użytkownicy
├── merchants.csv      # Sprzedawcy
└── transactions.json  # Transakcje

models/                # Modele
scripts/              # Skrypty
visualizations/       # Wizualizacje
""")

if __name__ == "__main__":
    main() 