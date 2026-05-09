import sys
import subprocess
from sklearn.feature_extraction.text import CountVectorizer

# Gerekli kütüphaneyi kontrol etme ve yükleme fonksiyonu
def install_package(package):
    """Belirtilen kütüphaneyi yükler."""
    try:
        __import__(package)
        print(f"'{package}' kütüphanesi yüklü.")
    except ImportError:
        print(f"'{package}' kütüphanesi yüklü değil, yükleniyor...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Kurulum işlemini başlatma
install_package("scikit-learn")

# Örnek metin veri seti (doküman korpusu)
corpus = [
    'Bu hızlı bir araba.',
    'Hızlı araba çok hızlı.',
    'Bu araba çok pahalı.'
]

#-------------------------------------------------------------------------------
# Bölüm 1: Bag of Words (BoW) Modeli
#-------------------------------------------------------------------------------
print("### 1. Bag of Words (BoW) Modeli Uygulaması ###")
print("-" * 50)

# CountVectorizer ile BoW modelini oluşturma
# Tokenization ve kelime sayımı işlemlerini yapar.
vectorizer_bow = CountVectorizer()

# Metin korpusunu BoW matrisine dönüştürme
X_bow = vectorizer_bow.fit_transform(corpus)

# Oluşturulan kelime sözlüğünü (vocabulary) görüntüleme
print("BoW Kelime Sözlüğü (Vocabulary):")
print(vectorizer_bow.vocabulary_)

# Oluşturulan BoW matrisini görüntüleme
# Her satır bir dokümanı, her sütun bir kelimeyi temsil eder.
print("\nBoW Matrisi (Seyrek Format):")
print(X_bow)
print("\nBoW Matrisi (Yoğun Format):")
print(X_bow.toarray())

# Matrisin sütunları (kelimeler) ile eşleşen özellik adlarını alma
feature_names_bow = vectorizer_bow.get_feature_names_out()
print(f"\nMatris Sütunları (Özellikler): {feature_names_bow}")

# Matrisin okunabilir formatı
import pandas as pd
df_bow = pd.DataFrame(X_bow.toarray(), columns=feature_names_bow)
print("\nBoW Matrisi (DataFrame Formatında):")
print(df_bow)

#-------------------------------------------------------------------------------
# Bölüm 2: N-gramlar ile Metin Temsili
#-------------------------------------------------------------------------------
print("\n### 2. N-gramlar ile Metin Temsili Uygulaması ###")
print("-" * 50)

# Bi-gram (2'li kelime grupları) kullanarak modeli oluşturma
# ngram_range=(1, 2) parametresi ile Unigram ve Bi-gram'ları kullanırız.
print("--- Bi-gram (Unigram + Bigram) Oluşturma ---")
vectorizer_ngram = CountVectorizer(ngram_range=(1, 2))

# Metni Bi-gram matrisine dönüştürme
X_ngram = vectorizer_ngram.fit_transform(corpus)

# Oluşturulan Bi-gram kelime sözlüğünü görüntüleme
print("Bi-gram Kelime Sözlüğü:")
print(vectorizer_ngram.vocabulary_)

# Oluşturulan Bi-gram matrisini görüntüleme
feature_names_ngram = vectorizer_ngram.get_feature_names_out()
df_ngram = pd.DataFrame(X_ngram.toarray(), columns=feature_names_ngram)
print("\nBi-gram Matrisi (DataFrame Formatında):")
print(df_ngram)

print("\n--- Tri-gram (Unigram + Bigram + Trigram) Oluşturma ---")
# ngram_range=(1, 3) parametresi ile Unigram, Bi-gram ve Tri-gram'ları kullanma
vectorizer_trigram = CountVectorizer(ngram_range=(1, 3))
X_trigram = vectorizer_trigram.fit_transform(corpus)
feature_names_trigram = vectorizer_trigram.get_feature_names_out()

df_trigram = pd.DataFrame(X_trigram.toarray(), columns=feature_names_trigram)
print("Tri-gram Matrisi (DataFrame Formatında):")
print(df_trigram)   