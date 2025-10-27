"""
Skrypt do przygotowania danych FakeHealth do analizy.
Łączy dane o artykułach (content) z ich recenzjami (reviews).
Tworzy dwie tabele: HealthStory i HealthRelease.
"""

import json
import os
from pathlib import Path
from typing import Dict, List
import pandas as pd

def load_content_files(content_dir: str) -> Dict[str, dict]:
    """
    Wczytuje wszystkie pliki JSON z folderu content.
    Zwraca słownik {news_id: dane_artykułu}
    """
    content_data = {}
    content_path = Path(content_dir)
    
    if not content_path.exists():
        print(f"UWAGA: Folder {content_dir} nie istnieje!")
        return content_data
    
    json_files = list(content_path.glob("*.json"))
    print(f"Znaleziono {len(json_files)} plików w {content_dir}")
    
    for file_path in json_files:
        news_id = file_path.stem  # np. "story_reviews_00000"
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                content_data[news_id] = data
        except Exception as e:
            print(f"Błąd przy wczytywaniu {file_path}: {e}")
    
    return content_data

def load_reviews(review_file: str) -> List[dict]:
    """
    Wczytuje plik z recenzjami.
    Zwraca listę recenzji.
    """
    try:
        with open(review_file, 'r', encoding='utf-8') as f:
            reviews = json.load(f)
        print(f"Wczytano {len(reviews)} recenzji z {review_file}")
        return reviews
    except Exception as e:
        print(f"Błąd przy wczytywaniu {review_file}: {e}")
        return []

def combine_data(content_data: Dict[str, dict], reviews: List[dict]) -> List[dict]:
    """
    Łączy dane z content i reviews na podstawie news_id.
    Zwraca listę połączonych rekordów.
    """
    combined = []
    
    for review in reviews:
        news_id = review.get('news_id')
        
        if news_id and news_id in content_data:
            article = content_data[news_id]
            
            # Wyciągamy kryteria w bardziej czytelnej formie
            criteria_details = []
            for criterion in review.get('criteria', []):
                criteria_details.append({
                    'question': criterion.get('question'),
                    'answer': criterion.get('answer'),
                    'explanation': criterion.get('explanation')
                })
            
            # Łączymy dane
            record = {
                # Dane z artykułu
                'news_id': news_id,
                'url': article.get('url'),
                'title': article.get('title'),
                'text': article.get('text'),
                'authors': article.get('authors', []),
                'publish_date': article.get('publish_date'),
                'keywords': article.get('keywords', []),
                'source': article.get('source'),
                'canonical_link': article.get('canonical_link'),
                
                # Dane z recenzji
                'review_link': review.get('link'),
                'review_title': review.get('title'),
                'original_title': review.get('original_title'),
                'rating': review.get('rating'),
                'description': review.get('description'),
                'reviewers': review.get('reviewers', []),
                'category': review.get('category'),
                'tags': review.get('tags', []),
                'source_link': review.get('source_link'),
                
                # Summary z recenzji
                'review_summary': review.get('summary', {}).get('Our Review Summary', ''),
                'why_this_matters': review.get('summary', {}).get('Why This Matters', ''),
                
                # Kryteria oceny
                'criteria': criteria_details,
                
                # Liczba kryteriów satisfactory/not satisfactory
                'num_satisfactory': sum(1 for c in review.get('criteria', []) if c.get('answer') == 'Satisfactory'),
                'num_not_satisfactory': sum(1 for c in review.get('criteria', []) if c.get('answer') == 'Not Satisfactory'),
                'num_not_applicable': sum(1 for c in review.get('criteria', []) if c.get('answer') == 'Not Applicable'),
            }
            
            combined.append(record)
        else:
            if news_id:
                print(f"UWAGA: Brak artykułu dla recenzji {news_id}")
    
    return combined

def save_to_json(data: List[dict], output_file: str, indent: int = 2):
    """
    Zapisuje dane do pliku JSON.
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)
    print(f"Zapisano {len(data)} rekordów do {output_file}")

def save_to_csv(data: List[dict], output_file: str):
    """
    Zapisuje dane do pliku CSV (bez zagnieżdżonych struktur).
    """
    # Tworzymy uproszczoną wersję bez zagnieżdżonych list/dict dla CSV
    simplified = []
    for record in data:
        simple_record = {
            'news_id': record['news_id'],
            'url': record['url'],
            'title': record['title'],
            'text': record['text'],
            'authors': ', '.join(record.get('authors', [])),
            'publish_date': record['publish_date'],
            'keywords': ', '.join(record.get('keywords', [])),
            'source': record['source'],
            'rating': record['rating'],
            'review_title': record['review_title'],
            'description': record['description'],
            'reviewers': ', '.join(record.get('reviewers', [])),
            'category': record['category'],
            'tags': ', '.join(record.get('tags', [])),
            'review_summary': record['review_summary'],
            'why_this_matters': record['why_this_matters'],
            'num_satisfactory': record['num_satisfactory'],
            'num_not_satisfactory': record['num_not_satisfactory'],
            'num_not_applicable': record['num_not_applicable'],
        }
        simplified.append(simple_record)
    
    df = pd.DataFrame(simplified)
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"Zapisano {len(simplified)} rekordów do {output_file}")

def process_dataset(dataset_name: str, content_dir: str, review_file: str, output_prefix: str):
    """
    Przetwarza jeden dataset (HealthStory lub HealthRelease).
    """
    print(f"\n{'='*60}")
    print(f"Przetwarzanie: {dataset_name}")
    print(f"{'='*60}")
    
    # Wczytaj dane
    content_data = load_content_files(content_dir)
    reviews = load_reviews(review_file)
    
    # Połącz dane
    combined_data = combine_data(content_data, reviews)
    
    # Zapisz do plików
    json_output = f"{output_prefix}.json"
    csv_output = f"{output_prefix}.csv"
    
    save_to_json(combined_data, json_output, indent=2)
    save_to_csv(combined_data, csv_output)
    
    # Statystyki
    print(f"\nStatystyki dla {dataset_name}:")
    print(f"  - Liczba artykułów: {len(content_data)}")
    print(f"  - Liczba recenzji: {len(reviews)}")
    print(f"  - Połączonych rekordów: {len(combined_data)}")
    
    if combined_data:
        ratings = [r['rating'] for r in combined_data]
        print(f"  - Rozkład ratingów:")
        for rating in sorted(set(ratings)):
            count = ratings.count(rating)
            print(f"    Rating {rating}: {count} artykułów ({count/len(ratings)*100:.1f}%)")
    
    return combined_data

def main():
    """
    Główna funkcja skryptu.
    """
    print("="*60)
    print("Przygotowanie danych FakeHealth")
    print("="*60)
    
    # Sprawdź czy jesteśmy we właściwym katalogu
    if not os.path.exists('dataset'):
        print("\nERROR: Folder 'dataset' nie został znaleziony!")
        print("Upewnij się, że skrypt jest uruchomiony z głównego katalogu projektu.")
        return
    
    # Przetwórz HealthStory
    healthstory_data = process_dataset(
        dataset_name="HealthStory",
        content_dir="dataset/content/HealthStory",
        review_file="dataset/reviews/HealthStory.json",
        output_prefix="HealthStory_combined"
    )
    
    # Przetwórz HealthRelease
    healthrelease_data = process_dataset(
        dataset_name="HealthRelease",
        content_dir="dataset/content/HealthRelease",
        review_file="dataset/reviews/HealthRelease.json",
        output_prefix="HealthRelease_combined"
    )
    
    print(f"\n{'='*60}")
    print("ZAKOŃCZONO!")
    print(f"{'='*60}")
    print("\nUtworzone pliki:")
    print("  - HealthStory_combined.json - pełne dane HealthStory")
    print("  - HealthStory_combined.csv - uproszczona wersja CSV")
    print("  - HealthRelease_combined.json - pełne dane HealthRelease")
    print("  - HealthRelease_combined.csv - uproszczona wersja CSV")
    print("\nPliki JSON zawierają pełne dane włącznie z kryteriami oceny.")
    print("Pliki CSV są uproszczone (bez zagnieżdżonych struktur) - łatwiejsze do analizy w Excel/pandas.")

if __name__ == "__main__":
    main()

