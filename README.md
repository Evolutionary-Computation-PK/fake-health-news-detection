# Ocena rzetelności artykułów (Fake News) – regresja i interpretacja

Projekt wykorzystujący dataset FakeHealth do przewidywania rzetelności artykułów zdrowotnych za pomocą regresji i interpretacji modeli ML.

## Dataset

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

### Krok 3: Uruchomienie analizy

#### Opcja A: Jupyter Notebook (Zalecane)
Otwórz i uruchom `RandomForest_TF-IDF_Linguistic.ipynb` w Jupyter / VS Code / Google Colab:
- Zawiera pełną analizę z wizualizacjami
- 3 modele: Regresja, Binary Classification, Multi-Task
- Feature engineering: TF-IDF + 25 cech linguistic
- Interpretację wyników i wyjaśnialność

#### Opcja B: Skrypt Python
```bash
python test_analysis.py
```

---

## 📁 Struktura projektu

```
.
├── dataset/                                      # Oryginalne dane FakeHealth
│   ├── content/
│   │   ├── HealthStory/                         # ~1638 artykułów (pliki JSON)
│   │   └── HealthRelease/                       # ~599 artykułów (pliki JSON)
│   └── reviews/
│       ├── HealthStory.json                     # Recenzje dla HealthStory
│       └── HealthRelease.json                   # Recenzje dla HealthRelease
│
├── RandomForest_TF-IDF_Linguistic.ipynb         # ⭐ Główny notebook z analizą
├── results_RandomForest_TF-IDF_Linguistic.json  # Wyniki modeli
├── selected_linguistic_features.txt             # Lista 25 cech linguistic
│
├── prepare_data.py                              # Skrypt łączący content + reviews
├── test_analysis.py                             # Przykładowa analiza i modele ML
├── requirements.txt                             # Wymagane biblioteki
└── README.md                                    # Ten plik
```

---

## Struktura danych wynikowych

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

## Cel projektu

### Problem
Przewidywanie rzetelności artykułów zdrowotnych na podstawie tekstu. Projekt wykorzystuje trzy podejścia:

### Podejście 1: Single-Task (Regresja)
- Przewidywanie ciągłego **ratingu rzetelności** (0-5)
- Proste, ale niska wyjaśnialność

### Podejście 2: Single-Task (Klasyfikacja Binarna)
- Klasyfikacja: **Fake News** (rating < 3) vs **Reliable** (rating ≥ 3)
- Lepsza accuracy, ale wciąż brak szczegółów

### Podejście 3: Multi-Task (Przewidywanie 10 Kryteriów) **GŁÓWNE**
- Przewidywanie, które z **10 kryteriów jakości dziennikarstwa** zostały złamane
- **Naturalna wyjaśnialność**: "Artykuł jest fake, bo złamał kryteria: C1, C3, C6"
- Zgodność z intencją autorów FakeHealth dataset
- Każde kryterium = osobna klasyfikacja binarna (0=Not Satisfactory, 1=Satisfactory)

---

## 🔬 Zaimplementowane modele

### Features (1025 cech):
- **TF-IDF** (1000 features): Bag-of-Words z title + text
  - max_features=1000
  - ngram_range=(1, 2) - unigramy i bigramy
  - stop_words='english'
- **25 cech linguistic** z `selected_linguistic_features.txt`:
  - Cechy stylistyczno-językowe (20): Adjective, Adverb, Modal, Negation, Conditional, itp.
  - Cechy medyczno-językowe (5): biomedical_terms, commercial_terms, url_count

### Model 1: Random Forest Regressor (Single-Task)
- **Zadanie**: Przewidywanie ratingu (0-5)
- **Model**: Random Forest Regressor (100 drzew, max_depth=20)
- **Interpretacja**: Feature importance pokazuje najważniejsze słowa/cechy
- **Wyniki**: R²=0.23, RMSE=1.04, MAE=0.83

### Model 2: Random Forest Classifier (Single-Task)
- **Zadanie**: Klasyfikacja binarna fake/reliable
- **Model**: Random Forest Classifier (100 drzew, class_weight='balanced')
- **Interpretacja**: Feature importance
- **Wyniki**: Accuracy=0.74, F1=0.84, Precision=0.76, Recall=0.95
- ⚠️ **Problem**: Model słabo identyfikuje fake news (recall=21% dla fake), za to bardzo dobrze znajduje reliable (recall=95%). Klasyfikuje większość fake newsów (79%) jako reliable - typowy problem niezbalansowanego datasetu

### Model 3: Multi-Output Random Forest ⭐ **GŁÓWNY MODEL**
- **Zadanie**: Przewidywanie 10 kryteriów jakości (każde: 0=Not Satisfactory, 1=Satisfactory)
- **Model**: MultiOutputClassifier z Random Forest (class_weight='balanced')
- **Interpretacja**: 
  - **Naturalna wyjaśnialność** - dokładnie wiemy, które kryteria zostały złamane
  - Feature importance dla każdego kryterium osobno
- **Wyniki**: Średnia Accuracy=0.73, Średnia F1=0.70, Średni Recall=0.70
  - Najlepsze kryterium: C10 (Konflikty interesów) - F1=0.96
  - Najtrudniejsze: C2 (Kwantyfikacja korzyści) - F1=0.40

---

## 📈 Metryki oceny

### Dla Regresji (Model 1):
- **R² Score**: Jakość dopasowania modelu (0-1, wyższy = lepszy)
  - Training: 0.81, Test: **0.23** (możliwy overfitting)
- **RMSE**: Root Mean Squared Error (niższy = lepszy)
  - Test: **1.04** punktu ratingu
- **MAE**: Mean Absolute Error - średni błąd w punktach ratingu
  - Test: **0.83** punktu ratingu

### Dla Klasyfikacji Binarnej (Model 2):
- **Accuracy**: Odsetek poprawnych klasyfikacji
  - Training: 0.98, Test: **0.74**
- **Precision**: Jaki % przewidzianych "reliable" jest rzeczywiście reliable
  - Test: **0.76** (dla reliable), **0.63** (dla fake)
- **Recall**: Jaki % rzeczywistych przypadków model znalazł
  - Test: **0.95** dla reliable (✅ bardzo dobry - model znajduje 95% reliable)
  - Test: **0.21** dla fake (⚠️ słaby - model znajduje tylko 21% fake newsów)
- **F1-Score**: Średnia harmoniczna precision i recall
  - Test: **0.84** (dla reliable), **0.31** (dla fake)
- **ROC-AUC**: Zdolność modelu do rozróżniania klas
  - Test: **0.72**
- ⚠️ **Confusion Matrix**: Z 92 fake newsów tylko 19 sklasyfikowano poprawnie, a 73 błędnie jako reliable (79% błąd!)

### Dla Multi-Task (Model 3) ⭐:
- **Accuracy per kryterium**: Odsetek poprawnych predykcji dla danego kryterium
  - Średnia: **0.73** (od 0.58 dla C7 do 0.92 dla C10)
- **F1-Score per kryterium**: Średnia harmoniczna precision i recall
  - Średnia: **0.70** (od 0.40 dla C2 do 0.96 dla C10)
- **Recall per kryterium**: Jaki % rzeczywistych "Satisfactory" model znalazł
  - Średnia: **0.70**

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
