import sys
import subprocess
import pandas as pd
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Gerekli kütüphaneleri kontrol etme ve yükleme fonksiyonu
def install_package(package):
    """Belirtilen kütüphaneyi yükler."""
    try:
        __import__(package)
    except ImportError:
        print(f"'{package}' kütüphanesi yüklü değil, yükleniyor...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Kurulum işlemini başlatma
install_package("textblob")
install_package("scikit-learn")
install_package("pandas")

#-------------------------------------------------------------------------------
# Bölüm 1: TextBlob ile Hızlı Duygu Analizi
#-------------------------------------------------------------------------------
print("### 1. TextBlob ile Hızlı Duygu Analizi ###")
print("-" * 50)

def analyze_sentiment_with_textblob(text):
    """Verilen metnin duygu skorunu hesaplar."""
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    
    # Kutupsallık skoruna göre etiketleme
    if polarity > 0.1:
        label = "POZİTİF"
    elif polarity < -0.1:
        label = "NEGATİF"
    else:
        label = "NÖTR"
        
    print(f"Metin: '{text}'")
    print(f"  -> Kutupsallık Skoru: {polarity:.2f} | Öznellik Skoru: {subjectivity:.2f} -> Etiket: {label}\n")

# Örnek metinler üzerinde uygulama
analyze_sentiment_with_textblob("Bu film harikaydı, herkese tavsiye ederim.")
analyze_sentiment_with_textblob("Ürün beklediğimden çok daha kötü çıktı, tam bir hayal kırıklığı.")
analyze_sentiment_with_textblob("Bugün hava çok rüzgarlıydı ve bulutluydu.")
analyze_sentiment_with_textblob("Yapay zeka hızlı bir şekilde gelişiyor.")

print("-" * 50)

#-------------------------------------------------------------------------------
# Bölüm 2: Makine Öğrenmesi ile Duygu Analizi Modeli
#-------------------------------------------------------------------------------
print("\n### 2. Makine Öğrenmesi Modeli ile Duygu Sınıflandırma ###")
print("-" * 50)

# Adım 1: Etiketli veri seti oluşturma
print("Adım 1: Etiketli Veri Seti Oluşturuluyor...")
data = {
    'text': [
        'Bu harika bir filmdi.', 'Çok beğendim.', 'Hizmet çok iyiydi.', 'Mükemmel bir deneyim.',
        'Korkunç bir deneyimdi.', 'Hiç memnun kalmadım.', 'Berbat bir ürün.', 'Çok kötü bir filmdi.'
    ],
    'sentiment': [
        'pozitif', 'pozitif', 'pozitif', 'pozitif',
        'negatif', 'negatif', 'negatif', 'negatif'
    ]
}
df = pd.DataFrame(data)
print(df)
print("\nVeri setimiz hazır. 'text' ve 'sentiment' sütunlarından oluşuyor.")
print("-" * 50)

# Adım 2: Metinleri Sayısal Veriye Dönüştürme (BoW)
print("Adım 2: Metinler Sayısal Verilere Dönüştürülüyor (BoW)...")
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['text'])
y = df['sentiment']

print("Oluşturulan Matrisin Boyutu:", X.shape)
print("Özellik Adları (Kelime Sözlüğü):", vectorizer.get_feature_names_out())
print("-" * 50)

# Adım 3: Sınıflandırma Modelini Eğitme
print("Adım 3: Makine Öğrenmesi Modeli Eğitiliyor (Lojistik Regresyon)...")
# Veri setini eğitim ve test olarak ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Modeli oluşturma ve eğitme
model = LogisticRegression()
model.fit(X_train, y_train)
print("Model başarıyla eğitildi.")
print("-" * 50)

# Adım 4: Modelin Performansını Değerlendirme
print("Adım 4: Modelin Performansı Değerlendiriliyor...")
y_pred = model.predict(X_test)

# Doğruluk (Accuracy) ve diğer metrikleri yazdırma
print(f"Model Doğruluk Oranı: {accuracy_score(y_test, y_pred):.2f}")
print("\nSınıflandırma Raporu:\n", classification_report(y_test, y_pred))
print("-" * 50)

# Adım 5: Yeni Bir Cümle Üzerinde Tahmin Yapma
print("Adım 5: Yeni Bir Cümle Üzerinde Tahmin Yapma...")
new_text = ["Bu ürün beklentilerimi aştı, çok memnun kaldım."]

# Yeni metni BoW vektörüne dönüştürme (modelin anladığı formata getirme)
new_text_vector = vectorizer.transform(new_text)

# Model ile tahmin yapma
prediction = model.predict(new_text_vector)

print(f"\nTahmin Edilecek Metin: '{new_text[0]}'")
print(f"Modelin Tahmini: {prediction[0].upper()}")