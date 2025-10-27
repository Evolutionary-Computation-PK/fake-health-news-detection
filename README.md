# Ocena rzetelności artykułów (Fake News) – regresja i interpretacja

Projekt wykorzystujący dataset FakeHealth do przewidywania rzetelności artykułów zdrowotnych za pomocą regresji i interpretacji modeli ML.

## 📊 Dataset

**FakeHealth** to zbiór danych zawierający:
- **HealthStory**: ~1638 artykułów z różnych źródeł medialnych
- **HealthRelease**: ~599 komunikatów prasowych uniwersytetów

Każdy artykuł zawiera:
- Pełny tekst artykułu
- Profesjonalną recenzję z ocenami rzetelności (rating 1-5)
- 10 kryteriów oceny jakości dziennikarstwa medycznego
- Szczegółowe wyjaśnienia ocen

**Źródło**: [FakeHealth Repository](https://github.com/EnyanDai/FakeHealth)  
**Paper**: [Ginger Cannot Cure Cancer: Battling Fake Health News](https://arxiv.org/abs/2002.00837)

---

## 🚀 Szybki start

### Krok 1: Instalacja zależności

```bash
pip install -r requirements.txt
```

Minimalne wymagania:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### Krok 2: Przygotowanie danych

```bash
python prepare_data.py
```

Skrypt automatycznie:
- Wczyta wszystkie artykuły z folderów `dataset/content/HealthStory` i `dataset/content/HealthRelease`
- Połączy je z recenzjami z `dataset/reviews/`
- Utworzy 4 pliki wynikowe:
  - `HealthStory_combined.json` - pełne dane HealthStory (z kryteriami)
  - `HealthStory_combined.csv` - uproszczona wersja CSV
  - `HealthRelease_combined.json` - pełne dane HealthRelease
  - `HealthRelease_combined.csv` - uproszczona wersja CSV

### Krok 3: Uruchomienie przykładowej analizy

```bash
python test_analysis.py
```

---

## 📁 Struktura projektu

```
.
├── dataset/                    # Oryginalne dane FakeHealth
│   ├── content/
│   │   ├── HealthStory/       # ~1638 artykułów (pliki JSON)
│   │   └── HealthRelease/     # ~599 artykułów (pliki JSON)
│   └── reviews/
│       ├── HealthStory.json   # Recenzje dla HealthStory
│       └── HealthRelease.json # Recenzje dla HealthRelease
│
├── prepare_data.py            # Skrypt łączący content + reviews
├── test_analysis.py     # Przykładowa analiza i modele ML
├── requirements.txt           # Wymagane biblioteki
└── README.md                  # Ten plik
```

---

## 🔍 Struktura danych wynikowych

Po uruchomieniu `prepare_data.py` każdy rekord w plikach `*_combined.json` zawiera:

### Dane z artykułu:
- `news_id` - unikalny identyfikator
- `url` - adres URL artykułu
- `title` - tytuł artykułu
- `text` - **pełny tekst artykułu** (do analizy NLP)
- `authors` - lista autorów
- `publish_date` - data publikacji (timestamp)
- `keywords` - słowa kluczowe
- `source` - źródło artykułu

### Dane z recenzji:
- `rating` - **ocena rzetelności (1-5)** - zmienna celu do regresji! ⭐
  - 1 = najgorszy (fake news)
  - 5 = najlepszy (rzetelny)
- `review_title` - tytuł recenzji
- `description` - krótki opis problemu z artykułem
- `reviewers` - lista recenzentów
- `category` - kategoria źródła (np. "The Guardian", "University news release")
- `tags` - tagi tematyczne
- `review_summary` - podsumowanie recenzji
- `why_this_matters` - dlaczego to ma znaczenie

### Kryteria oceny (do interpretacji modelu):
- `criteria` - lista 10 kryteriów, każde zawiera:
  - `question` - pytanie (np. "Czy artykuł omawia koszty?")
  - `answer` - odpowiedź: "Satisfactory" / "Not Satisfactory" / "Not Applicable"
  - `explanation` - szczegółowe wyjaśnienie oceny
  
- `num_satisfactory` - liczba spełnionych kryteriów
- `num_not_satisfactory` - liczba niespełnionych kryteriów
- `num_not_applicable` - liczba kryteriów nieaplikowalnych

### 10 kryteriów oceny jakości dziennikarstwa medycznego:
1. Czy artykuł omawia **koszty** interwencji?
2. Czy **kwantyfikuje korzyści**?
3. Czy omawia **zagrożenia/skutki uboczne**?
4. Czy ocenia **jakość dowodów naukowych**?
5. Czy nie przesadza z chorobą (**disease-mongering**)?
6. Czy używa **niezależnych źródeł**?
7. Czy **porównuje z alternatywami**?
8. Czy ustala **dostępność** interwencji?
9. Czy wspomina o **nowości vs. faktycznej innowacji**?
10. Czy identyfikuje **konflikty interesów**?

---

## 🎯 Cel projektu

### Problem
Przewidywanie **ratingu rzetelności** artykułu (1-5) na podstawie jego tekstu:
- **Rating 1-2**: Fake news / wprowadzający w błąd
- **Rating 3**: Mieszane / częściowo rzetelny
- **Rating 4-5**: Rzetelny / wysokiej jakości

### Podejście
1. **Regresja**: Przewidywanie ciągłej wartości ratingu (1-5)
2. **Interpretacja**: Zrozumienie, jakie cechy tekstu wskazują na fake news

---

## 💻 Przykład użycia w Python

```python
import json
import pandas as pd

# Wczytanie danych
with open('HealthStory_combined.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Lub wersja CSV (bez zagnieżdżonych struktur)
df = pd.read_csv('HealthStory_combined.csv', encoding='utf-8-sig')

# Podstawowe statystyki
print(f"Liczba artykułów: {len(df)}")
print(f"\nRozkład ratingów:")
print(df['rating'].value_counts().sort_index())

# Analiza tekstu
print(f"\nŚrednia długość tekstu: {df['text'].str.len().mean():.0f} znaków")

# Korelacja między kryteriami a ratingiem
print(f"\nKorelacja z ratingiem:")
print(df[['num_satisfactory', 'num_not_satisfactory', 'rating']].corr())
```

---

## 🔬 Przykładowe modele

### Model 1: Regresja Liniowa z TF-IDF
- **Reprezentacja tekstu**: TF-IDF (1000 najważniejszych słów/bigramów)
- **Model**: Regresja liniowa
- **Interpretacja**: Współczynniki pokazują wpływ słów na rating
  - Słowa z pozytywnymi współczynnikami → wskazują na rzetelność
  - Słowa z negatywnymi współczynnikami → wskazują na fake news

### Model 2: Random Forest
- **Reprezentacja**: TF-IDF
- **Model**: Random Forest Regressor
- **Interpretacja**: Feature importance - najważniejsze słowa/frazy

---

## 📊 Możliwe analizy dla projektu

### 1. Regresja - przewidywanie ratingu
- **Zmienne objaśniające**: 
  - Tekst artykułu (główna zmienna)
  - Tytuł
  - Słowa kluczowe
  - Długość tekstu
  - Źródło/kategoria
- **Zmienna celu**: `rating` (1-5)
- **Metody**: 
  - Regresja liniowa z TF-IDF ✅
  - Random Forest Regressor ✅
  - XGBoost / LightGBM
  - Regresja z embeddings (Word2Vec, BERT)
  - Sieci neuronowe (LSTM, CNN)

### 2. Interpretacja modelu
- **SHAP values** - które słowa/frazy wpływają na ocenę konkretnego artykułu
- **LIME** - lokalne wyjaśnienia predykcji
- **Analiza współczynników** regresji liniowej
- **Feature importance** z tree-based models
- **Attention mechanisms** (dla deep learning)

### 3. Feature Engineering
- Długość tekstu, liczba zdań, złożoność składniowa
- Analiza sentymentu
- Wykrywanie clickbait w tytułach
- Źródło artykułu jako cecha kategoryczna
- **Wykorzystanie kryteriów** jako dodatkowych features

### 4. Dodatkowe zadania
- **Klasyfikacja binarna**: rating ≤ 2 (fake) vs rating ≥ 4 (rzetelny)
- **Multi-class classification**: 1, 2, 3, 4, 5 jako osobne klasy
- **Topic modeling** (LDA) - tematy fake vs rzetelnych artykułów
- **Multi-task learning**: przewidywanie ratingu + wszystkich 10 kryteriów jednocześnie

### 5. Wykorzystanie kryteriów oceny
Kryteria mogą być użyte do:
- **Walidacji modelu** - czy model wykrywa te same problemy co recenzenci?
- **Feature engineering** - dodatkowe zmienne objaśniające
- **Multi-task learning** - przewidywanie ratingu + kryteriów jednocześnie
- **Interpretacji** - które kryteria są najważniejsze dla ratingu?

---

## 📈 Metryki oceny

- **R² Score**: Jakość dopasowania modelu (0-1, wyższy = lepszy)
- **RMSE**: Root Mean Squared Error (niższy = lepszy)
- **MAE**: Mean Absolute Error - średni błąd w punktach ratingu (niższy = lepszy)

### Przykładowe wyniki dla modeli bazowych:
- **Regresja Liniowa**: R² ~0.30-0.40, MAE ~0.6-0.8
- **Random Forest**: R² ~0.35-0.50, MAE ~0.5-0.7

---

## 💡 Możliwości rozwoju projektu

### 1. Lepsze reprezentacje tekstu
- **Word2Vec / GloVe** embeddings
- **BERT / transformers** (state-of-the-art dla NLP)
- **Doc2Vec** dla całych dokumentów
- **FastText** z subword information

### 2. Zaawansowane modele
- **XGBoost / LightGBM** - gradient boosting
- **Sieci neuronowe**: LSTM, CNN dla tekstu, Transformers
- **Ensemble methods** - łączenie predykcji kilku modeli
- **Stacking** - wielopoziomowe modele

### 3. Cross-dataset learning
- Trening na HealthStory, test na HealthRelease (i odwrotnie)
- **Transfer learning** między datasetami
- Analiza, co różni te dwa datasety

### 4. Interpretacja i wyjaśnialność
- **SHAP values** dla każdej predykcji
- **LIME** dla lokalnych wyjaśnień
- **Attention visualization** (dla modeli z attention)
- Analiza, które frazy są najważniejsze

---

## ⚠️ Uwagi i ograniczenia

### Ograniczenia datasetu:
- Rating jest **subiektywny** (ocena ekspertów), ale konsystentny
- Dataset jest **niezbalansowany** - więcej artykułów z niskim ratingiem
- **Język angielski** - modele mogą nie działać dla innych języków
- **Rozmiar** - ~2200 artykułów (nie jest bardzo duży dla deep learning)

### Statystyki datasetu:

**HealthStory** (~1638 artykułów):
- Artykuły z różnych źródeł medialnych (gazety, portale)
- Recenzje profesjonalne z HealthNewsReview.org
- Większa różnorodność źródeł i stylów

**HealthRelease** (~599 artykułów):
- Głównie komunikaty prasowe uniwersytetów
- Bardziej homogenna grupa
- Często bardziej "marketingowe" podejście

---

## 📚 Literatura i źródła

- [FakeHealth Paper (arXiv)](https://arxiv.org/abs/2002.00837) - Opis datasetu i metodologii
- [HealthNewsReview.org](https://www.healthnewsreview.org/) - Źródło recenzji i kryteriów
- [Scikit-learn Documentation](https://scikit-learn.org/) - Machine learning w Python
- [SHAP Documentation](https://shap.readthedocs.io/) - Interpretacja modeli
- [LIME Documentation](https://lime-ml.readthedocs.io/) - Lokalne wyjaśnienia

---

## 🎓 Wskazówki do raportu/prezentacji

Warto uwzględnić w projekcie:

1. **Eksploracja danych** (EDA):
   - Rozkład ratingów (histogram, statystyki)
   - Długość tekstów i jej korelacja z ratingiem
   - Korelacja między kryteriami a ratingiem
   - Najczęstsze słowa w fake news vs rzetelnych artykułach

2. **Preprocessing i feature engineering**:
   - TF-IDF vectorization (parametry, uzasadnienie)
   - Normalizacja danych
   - Podział train/test/validation
   - Dodatkowe features (długość, sentiment, itp.)

3. **Modele**:
   - Co najmniej **2-3 różne podejścia**
   - Uzasadnienie wyboru modeli
   - Tuning hiperparametrów (grid search, cross-validation)

4. **Wyniki**:
   - **Tabele** z metrykami (R², RMSE, MAE) dla wszystkich modeli
   - **Wykresy**: predykcja vs rzeczywistość, feature importance, itp.
   - Porównanie modeli
   - Analiza błędów (które artykuły są źle klasyfikowane?)

5. **Interpretacja** (kluczowe!):
   - Jakie **słowa/frazy** wskazują na fake news?
   - Jakie **słowa/frazy** wskazują na rzetelność?
   - Czy model nauczył się rozpoznawać **kryteria jakości**?
   - Przykłady konkretnych predykcji z wyjaśnieniami

6. **Wnioski**:
   - Co model nauczył się rozpoznawać?
   - Czy wyniki mają sens z punktu widzenia dziennikarstwa?
   - Ograniczenia podejścia
   - Możliwe zastosowania praktyczne

7. **Ograniczenia i future work**:
   - Dataset size, język, subiektywność ocen
   - Możliwe ulepszenia (BERT, więcej danych, itp.)
   - Transfer learning na inne domeny
