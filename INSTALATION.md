# Instalacja środowiska dla projektu FakeHealth

## Krok 0: Utworzenie wirtualnego środowiska (venv)

### Windows:
```bash
# Utwórz venv
python -m venv venv

# Aktywuj venv
venv\Scripts\activate
```

### Linux/Mac:
```bash
# Utwórz venv
python3 -m venv venv

# Aktywuj venv
source venv/bin/activate
```

**Dezaktywacja środowiska**:
```bash
deactivate
```


## Krok 1: Instalacja podstawowych bibliotek

```bash
pip install -r requirements.txt
```

To zainstaluje:
- **pandas, numpy** - analiza danych
- **scikit-learn** - machine learning
- **matplotlib, seaborn** - wizualizacja
- **spacy, textblob** - NLP
- **shap** - interpretacja modeli
- **tqdm** - progress bars
- **jupyter** - notebooki

## Krok 2: Pobranie modelu spaCy

spaCy wymaga osobnego pobrania modelu językowego:

```bash
python -m spacy download en_core_web_sm
```

Jeśli chcesz lepszy model (większy, dokładniejszy):
```bash
python -m spacy download en_core_web_md
# lub
python -m spacy download en_core_web_lg
```

## Krok 3: Pobranie danych TextBlob (opcjonalne)

TextBlob może wymagać dodatkowych danych:

```bash
python -m textblob.download_corpora
```

Lub w Pythonie:
```python
import nltk
nltk.download('brown')
nltk.download('punkt')
```

## Krok 4: Przygotowanie danych

```bash
python prepare_data.py
```

To utworzy pliki:
- `HealthStory_combined.json`
- `HealthStory_combined.csv`
- `HealthRelease_combined.json`
- `HealthRelease_combined.csv`

## Krok 5: Uruchomienie notebooka

```bash
jupyter notebook RandomForest_TF-IDF_Linguistic.ipynb
```

Lub w VS Code:
- Otwórz plik `.ipynb`
- Wybierz kernel Python
- Uruchom komórki

## Weryfikacja instalacji

Uruchom w Pythonie:

```python
import pandas as pd
import numpy as np
import sklearn
import spacy
from textblob import TextBlob
import shap
import matplotlib.pyplot as plt

print("Wszystkie biblioteki załadowane!")
print(f"pandas: {pd.__version__}")
print(f"numpy: {np.__version__}")
print(f"scikit-learn: {sklearn.__version__}")
print(f"spacy: {spacy.__version__}")

# Test spaCy
nlp = spacy.load("en_core_web_sm")
doc = nlp("This is a test.")
print(f"spaCy działa! Znaleziono {len(doc)} tokenów")

# Test TextBlob
blob = TextBlob("This is great!")
print(f"TextBlob działa! Sentiment: {blob.sentiment.polarity}")

print("\nWszystko gotowe!")
```

## Troubleshooting

### Problem: spaCy model nie znaleziony
```bash
# Upewnij się że model jest zainstalowany
python -m spacy download en_core_web_sm --user

# Lub ręcznie:
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.5.0/en_core_web_sm-3.5.0.tar.gz
```

### Problem: SHAP długo się instaluje
SHAP może wymagać kompilatora C++. Jeśli masz problem:
```bash
# Windows: Zainstaluj Visual Studio Build Tools
# Linux/Mac: Zainstaluj gcc
```

### Problem: Brak pamięci przy przetwarzaniu
Jeśli spaCy zużywa za dużo pamięci dla wszystkich ~1600 artykułów:
- Przetwarzaj w mniejszych batch'ach
- Użyj mniejszego modelu (`en_core_web_sm` zamiast `lg`)
- Ogranicz długość tekstu (już jest w kodzie: `text[:1000000]`)

### Problem: Jupyter kernel crashes
```bash
# Zwiększ limit pamięci
pip install --upgrade jupyter ipykernel
```

## Minimalna wersja (bez interpretacji)

Jeśli chcesz tylko podstawowe funkcjonalności bez SHAP:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn spacy textblob tqdm jupyter
python -m spacy download en_core_web_sm
```

Zakomentuj w notebooku linię:
```python
# import shap  # Zakomentuj jeśli nie potrzebujesz
```

---

**Po instalacji możesz przejść do uruchomienia notebooka!**

