#!pip install zeyrek contractions --> Kısaltmaların Genişletilmesi (Contractions)
#!pip install textblob --> Stemming ve Lemmatization Karşılaştırması
#!pip install nltk --> Stopwords Kaldırma, Stemming ve Lemmatization Karşılaştırması

import sys
import subprocess
import nltk
import zeyrek
import contractions
from textblob import TextBlob
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

# Gerekli kütüphaneleri ve verileri kontrol etme ve yükleme fonksiyonu
def install_and_download():
    """Gerekli kütüphaneleri ve NLTK veri paketlerini yükler."""
    packages = ["contractions", "nltk", "zeyrek", "textblob"]
    for package in packages:
        try:
            __import__(package)
        except ImportError:
            print(f"'{package}' kütüphanesi yüklü değil, yükleniyor...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

    # NLTK veri paketlerini indirme
    print("\n### Gerekli NLTK Verileri İndiriliyor... ###")
    try:
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/punkt')
    except:
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
        
# Kurulum ve indirme işlemlerini başlatma
install_and_download()

#-------------------------------------------------------------------------------
# Bölüm 1: Kısaltmaların Genişletilmesi (Contractions)
#-------------------------------------------------------------------------------
print("### 1. Kısaltmaların Genişletilmesi (Contractions) ###")
print("-" * 50)

# Örnek İngilizce metin
text_with_contractions = "I don't wanna go to the cars, they're not mine."

# 'contractions' kütüphanesi ile genişletme
expanded_text = contractions.fix(text_with_contractions)

print(f"Orijinal Metin: '{text_with_contractions}'")
print(f"Genişletilmiş Metin: '{expanded_text}'")

# Metni kelimelere ayırma
words = nltk.word_tokenize(expanded_text)
print(f"Tokenlar: {words}\n")

#-------------------------------------------------------------------------------
# Bölüm 2: Stopwords (Durma Kelimeleri) Kaldırma
#-------------------------------------------------------------------------------
print("### 2. Stopwords Kaldırma ###")
print("-" * 50)

# İngilizce stopwords kümesini oluşturma
english_stopwords = set(stopwords.words('english'))
# Stopwords'leri metinden çıkarma
filtered_words_en = [word for word in words if word.lower() not in english_stopwords and word.isalpha()]
print(f"Orijinal İngilizce Tokenlar: {words}")
print(f"Stopwords'ler Kaldırıldı: {filtered_words_en}\n")

# Türkçe stopwords örneği
turkish_text = "Ben okula gidiyorum ve futbol oynamak istiyorum ama hava soğuk."
turkish_words = nltk.word_tokenize(turkish_text)
turkish_stopwords = set(stopwords.words('turkish'))
filtered_words_tr = [word for word in turkish_words if word.lower() not in turkish_stopwords and word.isalpha()]
print(f"Orijinal Türkçe Tokenlar: {turkish_words}")
print(f"Türkçe Stopwords'ler Kaldırıldı: {filtered_words_tr}\n")

#-------------------------------------------------------------------------------
# Bölüm 3: Stemming ve Lemmatization Karşılaştırması
#-------------------------------------------------------------------------------
print("### 3. Stemming ve Lemmatization Karşılaştırması ###")
print("-" * 50)

# İngilizce metin için Stemming
porter_stemmer = PorterStemmer()
stemmed_words = [porter_stemmer.stem(word) for word in filtered_words_en]
print(f"Stemming (İngilizce): {stemmed_words}")
# `running` kelimesi `run`'a dönüştü, doğru. Ama `cars` kelimesi `car`'a döndü, bu da mantıklı.

# TextBlob ile İngilizce Lemmatization
blob = TextBlob(expanded_text)
# TextBlob ile Lemmatization için kelime türü (verb, noun vb.) bilgisi de verilebilir.
lemmatized_words_en = [word.lemmatize() for word in blob.words if word.isalpha()]
print(f"Lemmatization (TextBlob - İngilizce): {lemmatized_words_en}\n")
# Görüldüğü gibi, her iki teknik de İngilizce için benzer sonuçlar verebilir.

# Türkçe için Zeyrek ile Lemmatization
zeyrek_analyzer = zeyrek.MorphAnalyzer()
turkish_text_lemmas = "Okula gidiyorum ve evde bilgisayarımı açtım."
turkish_words_lemmas = nltk.word_tokenize(turkish_text_lemmas)
lemmatized_turkish_words = [zeyrek_analyzer.lemmatize(word)[0][1] for word in turkish_words_lemmas if zeyrek_analyzer.lemmatize(word)]
print(f"Orijinal Türkçe Metin: '{turkish_text_lemmas}'")
print(f"Lemmatization (Zeyrek - Türkçe): {lemmatized_turkish_words}")
# Görüldüğü gibi `gidiyorum` kelimesi `gitmek` köküne döndürüldü.
# Bu, Türkçe gibi eklemeli diller için Stemming'e göre daha anlamlı sonuçlar verir.