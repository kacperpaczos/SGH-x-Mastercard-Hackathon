#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns

def load_data():
    print("Wczytywanie danych...")
    
    users_df = pd.read_csv('users.csv')
    print(f"Użytkownicy: {users_df.shape[0]} wierszy, {users_df.shape[1]} kolumn")
    
    merchants_df = pd.read_csv('merchants.csv')
    print(f"Sprzedawcy: {merchants_df.shape[0]} wierszy, {merchants_df.shape[1]} kolumn")
    
    transactions_data = []
    with open('transactions.json', 'r') as f:
        for line in f:
            if line.strip():
                transactions_data.append(json.loads(line))
    
    transactions_df = pd.DataFrame(transactions_data)
    print(f"Transakcje: {transactions_df.shape[0]} wierszy, {transactions_df.shape[1]} kolumn")
    
    return users_df, merchants_df, transactions_df

def explore_data(users_df, merchants_df, transactions_df):
    print("\n--- Użytkownicy ---")
    print(users_df.info())
    print("\nPrzykład:")
    print(users_df.head())
    
    print("\n--- Sprzedawcy ---")
    print(merchants_df.info())
    print("\nPrzykład:")
    print(merchants_df.head())
    
    print("\n--- Transakcje ---")
    print(transactions_df.info())
    print("\nPrzykład:")
    print(transactions_df.head())

if __name__ == "__main__":
    users_df, merchants_df, transactions_df = load_data()
    explore_data(users_df, merchants_df, transactions_df)
    
    print("\nDane gotowe do analizy.")
    print("Zmienne: users_df, merchants_df, transactions_df") 