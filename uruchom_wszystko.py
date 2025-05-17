#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import subprocess
import sys
import time

def clear_screen():
    """Czyści ekran terminala."""
    os.system('cls' if os.name == 'nt' else 'clear')

def run_command(command, description):
    """Uruchamia podaną komendę i wyświetla opis."""
    clear_screen()
    print(f"\n{'-'*80}")
    print(f"KROK: {description}")
    print(f"{'-'*80}\n")
    
    print(f"Uruchamiam: {command}\n")
    time.sleep(1)  # Krótka pauza dla czytelności
    
    # Uruchomienie komendy
    process = subprocess.Popen(command, shell=True)
    process.wait()
    
    print(f"\n{'-'*80}")
    print(f"Zakończono: {description}")
    print(f"{'-'*80}\n")
    
    input("Naciśnij ENTER, aby kontynuować...")

def main():
    # Folder ze skryptami
    scripts_dir = "scripts"
    
    clear_screen()
    print("""
╔════════════════════════════════════════════════════════════════════════════════╗
║                                                                                ║
║           AUTOMATYCZNY PROCES TRENOWANIA MODELU WYKRYWANIA FRAUDÓW            ║
║                                                                                ║
╚════════════════════════════════════════════════════════════════════════════════╝

Proces składa się z następujących kroków:
1. Eksploracja danych
2. Analiza i wizualizacja danych
3. Trenowanie modelu
4. Testowanie modelu

""")
    
    input("Naciśnij ENTER, aby rozpocząć proces...")
    
    # Krok 1: Eksploracja danych
    run_command(
        f"python {scripts_dir}/explore_data.py",
        "Eksploracja danych"
    )
    
    # Krok 2: Analiza danych
    run_command(
        f"python {scripts_dir}/analyze_data.py",
        "Analiza i wizualizacja danych"
    )
    
    # Krok 3: Trenowanie modelu
    run_command(
        f"python {scripts_dir}/auto_model.py --model_type improved",
        "Trenowanie ulepszonego modelu z SMOTE"
    )
    
    # Krok 4: Testowanie modelu
    run_command(
        f"python {scripts_dir}/test_model.py --random",
        "Testowanie modelu na losowej transakcji"
    )
    
    clear_screen()
    print("""
╔════════════════════════════════════════════════════════════════════════════════╗
║                                                                                ║
║                 PROCES ZAKOŃCZONY POMYŚLNIE                                    ║
║                                                                                ║
╚════════════════════════════════════════════════════════════════════════════════╝

Dostępne opcje:
1. Test na większej liczbie transakcji:
   python scripts/test_model.py --random --generate

2. Trenowanie z innymi parametrami:
   python scripts/auto_model.py --model_type basic --no_smote

3. Wyniki analizy: folder visualizations/
""")

if __name__ == "__main__":
    main()