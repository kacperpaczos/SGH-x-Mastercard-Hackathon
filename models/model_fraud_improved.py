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
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE

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

# Przetwarzanie danych
print("\nPrzetwarzanie cech kategorialnych i numerycznych...")

# Przetwarzanie cech
categorical_transformer = OneHotEncoder(handle_unknown='ignore')
numerical_transformer = StandardScaler()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Zastosowanie preprocessora
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

# Zastosowanie SMOTE do zbilansowania klas w zbiorze treningowym
print("Zastosowanie SMOTE do zbilansowania klas...")
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_preprocessed, y_train)

print(f"Rozmiar zbioru treningowego po SMOTE: {X_train_resampled.shape[0]} przykładów")
print(f"Procent fraudów w zbiorze treningowym po SMOTE: {100 * y_train_resampled.mean():.4f}%")

# Trenowanie modelu
print("\nTrening modelu Random Forest z parametrami dostosowanymi do detekcji fraudów...")
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_split=10,
    min_samples_leaf=4,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train_resampled, y_train_resampled)

# Predykcje
y_pred = rf_model.predict(X_test_preprocessed)
y_prob = rf_model.predict_proba(X_test_preprocessed)[:, 1]

# Ewaluacja modelu
print("\n--- Raport klasyfikacji ---")
print(classification_report(y_test, y_pred))

# Macierz pomyłek
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Przewidziana klasa')
plt.ylabel('Prawdziwa klasa')
plt.title('Macierz pomyłek')
plt.savefig('improved_confusion_matrix.png')
print("Macierz pomyłek zapisana jako 'improved_confusion_matrix.png'")

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
plt.savefig('improved_roc_curve.png')
print("Krzywa ROC zapisana jako 'improved_roc_curve.png'")

# Krzywa Precision-Recall
plt.figure(figsize=(8, 6))
precision, recall, _ = precision_recall_curve(y_test, y_prob)
avg_precision = average_precision_score(y_test, y_prob)
plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AP = {avg_precision:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Krzywa Precision-Recall')
plt.legend(loc="upper right")
plt.savefig('improved_pr_curve.png')
print("Krzywa Precision-Recall zapisana jako 'improved_pr_curve.png'")

# Ważność cech
if hasattr(rf_model, 'feature_importances_'):
    # Zbieramy nazwy cech po przekształceniu
    feature_names = []
    
    # Cechy numeryczne (nie zmieniają nazw)
    for name in numerical_features:
        feature_names.append(name)
    
    # Cechy kategoryczne po one-hot encoding
    if hasattr(preprocessor.transformers_[1][1], 'get_feature_names_out'):
        cat_feature_names = preprocessor.transformers_[1][1].get_feature_names_out(categorical_features)
        feature_names.extend(cat_feature_names)
    
    # Ważność cech
    importances = rf_model.feature_importances_
    
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
        plt.savefig('improved_feature_importance.png')
        print("Wykres ważności cech zapisany jako 'improved_feature_importance.png'")
    else:
        print(f"Uwaga: Liczba cech nie zgadza się (ważności: {len(importances)}, nazwy: {len(feature_names)})")

# Znalezienie optymalnego progu decyzyjnego
# Często w przypadku nierównowagi klas warto dostosować próg decyzyjny

print("\n--- Optymalizacja progu decyzyjnego ---")
thresholds = np.arange(0, 1, 0.05)
f1_scores = []

for threshold in thresholds:
    y_pred_thresh = (y_prob >= threshold).astype(int)
    report = classification_report(y_test, y_pred_thresh, output_dict=True)
    f1 = report['1']['f1-score']
    f1_scores.append(f1)
    print(f"Próg: {threshold:.2f}, F1-score dla fraudów: {f1:.4f}")

# Znalezienie optymalnego progu
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]
optimal_f1 = f1_scores[optimal_idx]

print(f"\nOptymalny próg decyzyjny: {optimal_threshold:.2f} (F1-score: {optimal_f1:.4f})")

# Zastosowanie optymalnego progu
y_pred_optimal = (y_prob >= optimal_threshold).astype(int)

print("\n--- Raport klasyfikacji z optymalnym progiem ---")
print(classification_report(y_test, y_pred_optimal))

# Macierz pomyłek z optymalnym progiem
plt.figure(figsize=(8, 6))
cm_optimal = confusion_matrix(y_test, y_pred_optimal)
sns.heatmap(cm_optimal, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Przewidziana klasa')
plt.ylabel('Prawdziwa klasa')
plt.title(f'Macierz pomyłek (próg = {optimal_threshold:.2f})')
plt.savefig('optimal_threshold_confusion_matrix.png')
print("Macierz pomyłek z optymalnym progiem zapisana jako 'optimal_threshold_confusion_matrix.png'")

print("\nAnaliza ulepszonego modelu zakończona!") 