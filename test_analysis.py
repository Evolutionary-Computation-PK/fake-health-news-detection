"""
Przykładowa analiza danych FakeHealth
Ocena rzetelności artykułów medycznych - regresja i interpretacja
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Ustawienia wizualizacji
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')

print("="*80)
print("ANALIZA DANYCH FAKEHEALTH - Ocena rzetelności artykułów")
print("="*80)

# =============================================================================
# 1. WCZYTANIE DANYCH
# =============================================================================
print("\n[1/9] Wczytywanie danych...")

# Wczytanie HealthStory
with open('HealthStory_combined.json', 'r', encoding='utf-8') as f:
    healthstory_data = json.load(f)
df_story = pd.DataFrame(healthstory_data)
df_story['dataset'] = 'HealthStory'

# Wczytanie HealthRelease
with open('HealthRelease_combined.json', 'r', encoding='utf-8') as f:
    healthrelease_data = json.load(f)
df_release = pd.DataFrame(healthrelease_data)
df_release['dataset'] = 'HealthRelease'

# Połączenie (opcjonalnie - możesz użyć tylko jednego datasetu)
df_all = pd.concat([df_story, df_release], ignore_index=True)

print(f"  HealthStory: {len(df_story)} artykułów")
print(f"  HealthRelease: {len(df_release)} artykułów")
print(f"  Łącznie: {len(df_all)} artykułów")

# =============================================================================
# 2. EKSPLORACJA DANYCH
# =============================================================================
print("\n[2/9] Eksploracja danych...")

# Rozkład ratingu
print("\nRozkład ratingów:")
print(df_all['rating'].value_counts().sort_index())

print(f"\nŚredni rating: {df_all['rating'].mean():.2f}")
print(f"Mediana ratingu: {df_all['rating'].median():.0f}")

# Długość tekstów
df_all['text_length'] = df_all['text'].str.len()
print(f"\nŚrednia długość tekstu: {df_all['text_length'].mean():.0f} znaków")

# Korelacja kryteriów z ratingiem
correlation = df_all[['rating', 'num_satisfactory', 'num_not_satisfactory']].corr()
print("\nKorelacja z ratingiem:")
print(f"  Satisfactory:     {correlation.loc['num_satisfactory', 'rating']:.3f}")
print(f"  Not Satisfactory: {correlation.loc['num_not_satisfactory', 'rating']:.3f}")

# =============================================================================
# 3. PRZYGOTOWANIE DANYCH
# =============================================================================
print("\n[3/9] Przygotowanie danych do modelowania...")

# Użyj df_story dla pojedynczego datasetu lub df_all dla obu
df = df_story.copy()  # Zmień na df_all jeśli chcesz

# Usuwanie braków
df = df.dropna(subset=['text', 'rating'])

# Połączenie tytułu i tekstu
df['full_text'] = df['title'].fillna('') + ' ' + df['text'].fillna('')

# Podział na train/test
X = df['full_text']
y = df['rating']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"  Zbiór treningowy: {len(X_train)} artykułów")
print(f"  Zbiór testowy: {len(X_test)} artykułów")

# =============================================================================
# 4. TF-IDF VECTORIZATION
# =============================================================================
print("\n[4/9] Tworzenie reprezentacji TF-IDF...")

tfidf = TfidfVectorizer(
    max_features=1000,
    min_df=2,
    max_df=0.8,
    ngram_range=(1, 2),
    stop_words='english'
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print(f"  Rozmiar macierzy: {X_train_tfidf.shape}")
print(f"  Liczba unikalnych słów/fraz: {len(tfidf.get_feature_names_out())}")

# =============================================================================
# 5. MODEL 1: REGRESJA LINIOWA
# =============================================================================
print("\n[5/9] Trenowanie modelu Regresji Liniowej...")

lr_model = LinearRegression()
lr_model.fit(X_train_tfidf, y_train)

y_pred_lr_test = lr_model.predict(X_test_tfidf)
y_pred_lr_test = np.clip(y_pred_lr_test, 1, 5)

print(f"  R² Score: {r2_score(y_test, y_pred_lr_test):.4f}")
print(f"  RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_lr_test)):.4f}")
print(f"  MAE: {mean_absolute_error(y_test, y_pred_lr_test):.4f}")

# =============================================================================
# 6. MODEL 2: RANDOM FOREST
# =============================================================================
print("\n[6/9] Trenowanie modelu Random Forest...")

rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=20,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train_tfidf, y_train)

y_pred_rf_test = rf_model.predict(X_test_tfidf)
y_pred_rf_test = np.clip(y_pred_rf_test, 1, 5)

print(f"  R² Score: {r2_score(y_test, y_pred_rf_test):.4f}")
print(f"  RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_rf_test)):.4f}")
print(f"  MAE: {mean_absolute_error(y_test, y_pred_rf_test):.4f}")

# =============================================================================
# 7. INTERPRETACJA: FEATURE IMPORTANCE (Random Forest)
# =============================================================================
print("\n[7/9] Analiza Feature Importance (Random Forest)...")

feature_names = tfidf.get_feature_names_out()
feature_importance = rf_model.feature_importances_

indices = np.argsort(feature_importance)[::-1][:20]
top_features = [(feature_names[i], feature_importance[i]) for i in indices]

print("\nTop 20 najważniejszych słów/fraz:")
for i, (feature, importance) in enumerate(top_features, 1):
    print(f"  {i:2d}. {feature:30s} - {importance:.6f}")

# =============================================================================
# 8. INTERPRETACJA: WSPÓŁCZYNNIKI REGRESJI LINIOWEJ
# =============================================================================
print("\n[8/9] Analiza współczynników Regresji Liniowej...")

coefficients = lr_model.coef_

# Słowa zwiększające rating (pozytywne)
top_positive_indices = np.argsort(coefficients)[::-1][:10]
top_positive = [(feature_names[i], coefficients[i]) for i in top_positive_indices]

# Słowa zmniejszające rating (negatywne)
top_negative_indices = np.argsort(coefficients)[:10]
top_negative = [(feature_names[i], coefficients[i]) for i in top_negative_indices]

print("\nSłowa ZWIĘKSZAJĄCE rating (wskazują na RZETELNOŚĆ):")
for word, coef in top_positive:
    print(f"  {word:30s} -> +{coef:.4f}")

print("\nSłowa ZMNIEJSZAJĄCE rating (wskazują na FAKE NEWS):")
for word, coef in top_negative:
    print(f"  {word:30s} -> {coef:.4f}")

# =============================================================================
# 9. PORÓWNANIE MODELI
# =============================================================================
print("\n[9/9] Porównanie modeli...")

results = pd.DataFrame({
    'Model': ['Regresja Liniowa', 'Random Forest'],
    'R²': [
        r2_score(y_test, y_pred_lr_test),
        r2_score(y_test, y_pred_rf_test)
    ],
    'RMSE': [
        np.sqrt(mean_squared_error(y_test, y_pred_lr_test)),
        np.sqrt(mean_squared_error(y_test, y_pred_rf_test))
    ],
    'MAE': [
        mean_absolute_error(y_test, y_pred_lr_test),
        mean_absolute_error(y_test, y_pred_rf_test)
    ]
})

print("\n" + "="*80)
print("PODSUMOWANIE - PORÓWNANIE MODELI")
print("="*80)
print(results.to_string(index=False))
print("="*80)

# =============================================================================
# 10. PRZYKŁADOWA PREDYKCJA
# =============================================================================
print("\n" + "="*80)
print("PRZYKŁADOWA PREDYKCJA")
print("="*80)

sample_idx = 0
sample_text = X_test.iloc[sample_idx]
sample_true_rating = y_test.iloc[sample_idx]

sample_tfidf = tfidf.transform([sample_text])
sample_pred_lr = lr_model.predict(sample_tfidf)[0]
sample_pred_rf = rf_model.predict(sample_tfidf)[0]

print(f"\nTekst artykułu (pierwsze 400 znaków):")
print("-" * 80)
print(sample_text[:400] + "...")
print("-" * 80)

print(f"\nRzeczywisty rating:           {sample_true_rating}")
print(f"Predykcja (Regresja Liniowa): {sample_pred_lr:.2f}")
print(f"Predykcja (Random Forest):    {sample_pred_rf:.2f}")

print("="*80)
print("ANALIZA ZAKOŃCZONA!")
print("="*80)

