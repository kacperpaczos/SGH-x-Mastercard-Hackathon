#!/usr/bin/env python
# -*- coding: utf-8 -*-

from explore_data import load_data
from analyze_data import parse_timestamp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Wczytanie danych
print("Wczytywanie danych...")
users_df, merchants_df, transactions_df = load_data()

# Przetwarzanie danych czasowych
print("Przetwarzanie danych czasowych...")
transactions_df = parse_timestamp(transactions_df)

# Przygotowanie danych
print("Przygotowanie danych do modelu...")

# Wyodrębnienie danych o lokalizacji
print("Wyodrębnienie danych o lokalizacji...")
transactions_df['latitude'] = transactions_df['location'].apply(lambda x: x['lat'] if isinstance(x, dict) and 'lat' in x else None)
transactions_df['longitude'] = transactions_df['location'].apply(lambda x: x['long'] if isinstance(x, dict) and 'long' in x else None)

# Łączenie danych z użytkownikami i sprzedawcami
merged_data = transactions_df.merge(users_df[['user_id', 'age', 'sex', 'risk_score']], 
                                    on='user_id', how='left')
merged_data = merged_data.merge(merchants_df[['merchant_id', 'category', 'trust_score', 'has_fraud_history']], 
                               on='merchant_id', how='left')

# Wybór cech do modelu
features = [
    # Cechy transakcji
    'amount', 'channel', 'device', 'latitude', 'longitude', 'payment_method', 
    'is_international', 'session_length_seconds', 'is_first_time_merchant',
    'hour', 'day_of_week', 
    
    # Cechy użytkownika
    'age', 'sex', 'risk_score',
    
    # Cechy sprzedawcy
    'category', 'trust_score', 'has_fraud_history'
]

# Definiowanie cech kategorialnych i numerycznych
categorical_features = [
    'channel', 'device', 'payment_method', 'day_of_week', 'sex', 'category'
]
numerical_features = [f for f in features if f not in categorical_features]

# Przygotowanie transformera cech
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Przygotowanie danych
X = merged_data[features]
y = merged_data['is_fraud']

# Podział na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"Rozmiar zbioru treningowego: {X_train.shape[0]} przykładów")
print(f"Rozmiar zbioru testowego: {X_test.shape[0]} przykładów")
print(f"Procent fraudów w zbiorze treningowym: {100 * y_train.mean():.4f}%")
print(f"Procent fraudów w zbiorze testowym: {100 * y_test.mean():.4f}%")

# Budowa modelu
print("\nTrening modelu Random Forest...")
model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
])

# Trening modelu
model.fit(X_train, y_train)

# Ewaluacja modelu
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Wyświetlenie wyników
print("\n--- Raport klasyfikacji ---")
print(classification_report(y_test, y_pred))

# Macierz pomyłek
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Przewidziana klasa')
plt.ylabel('Prawdziwa klasa')
plt.title('Macierz pomyłek')
plt.savefig('confusion_matrix.png')
print("Macierz pomyłek zapisana jako 'confusion_matrix.png'")

# Krzywa ROC
plt.figure(figsize=(8, 6))
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Krzywa ROC')
plt.legend(loc="lower right")
plt.savefig('roc_curve.png')
print("Krzywa ROC zapisana jako 'roc_curve.png'")

# Krzywa Precision-Recall
plt.figure(figsize=(8, 6))
precision, recall, _ = precision_recall_curve(y_test, y_prob)
plt.plot(recall, precision, color='blue', lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Krzywa Precision-Recall')
plt.savefig('precision_recall_curve.png')
print("Krzywa Precision-Recall zapisana jako 'precision_recall_curve.png'")

# Ważność cech
if hasattr(model['classifier'], 'feature_importances_'):
    # Odczytanie ważności cech
    feature_names = []
    
    # Zbierz nazwy cech numerycznych
    for name in numerical_features:
        feature_names.append(name)
    
    # Zbierz nazwy cech kategorialnych po transformacji OneHotEncoder
    cat_encoder = model['preprocessor'].transformers_[1][1]
    if hasattr(cat_encoder, 'get_feature_names_out'):
        cat_feature_names = cat_encoder.get_feature_names_out(categorical_features)
        feature_names.extend(cat_feature_names)
    else:
        # W przypadku gdy get_feature_names_out nie jest dostępne
        print("Uwaga: Nie można odczytać nazw cech kategorialnych po transformacji.")
        
    # Ważność cech
    importances = model['classifier'].feature_importances_
    
    if len(importances) == len(feature_names):
        # Sortowanie cech według ważności
        indices = np.argsort(importances)[::-1]
        
        # Wydrukowanie 15 najważniejszych cech
        print("\n--- Top 15 najważniejszych cech ---")
        for i, idx in enumerate(indices[:15]):
            print(f"{i+1}. {feature_names[idx]}: {importances[idx]:.4f}")
        
        # Wykres ważności cech
        plt.figure(figsize=(12, 8))
        plt.title("Ważność cech")
        plt.bar(range(min(15, len(indices))), importances[indices[:15]], align="center")
        plt.xticks(range(min(15, len(indices))), [feature_names[i] for i in indices[:15]], rotation=90)
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        print("Wykres ważności cech zapisany jako 'feature_importance.png'")
    else:
        print(f"Uwaga: Liczba cech nie zgadza się (ważności: {len(importances)}, nazwy: {len(feature_names)})")

print("\nAnaliza modelu zakończona!") 