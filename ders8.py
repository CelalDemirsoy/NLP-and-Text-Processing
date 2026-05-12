import sys
import subprocess
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, LSTM, Dense

# Gerekli kütüphaneleri kontrol etme ve yükleme fonksiyonu
def install_package(package):
    """Belirtilen kütüphaneyi yükler."""
    try:
        __import__(package)
    except ImportError:
        print(f"'{package}' kütüphanesi yüklü değil, yükleniyor...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Kurulum işlemini başlatma
install_package("tensorflow")
install_package("numpy")

#-------------------------------------------------------------------------------
# Bölüm 1: Veri Seti Hazırlığı
#-------------------------------------------------------------------------------
print("### 1. Veri Seti Hazırlığı ve Ön İşleme ###")
print("-" * 50)

# Örnek film yorumları (0: Negatif, 1: Pozitif)
sentences = [
    'Bu film çok kötü, hiç beğenmedim.',
    'Tam bir hayal kırıklığıydı, berbat.',
    'Oyunculuklar zayıftı ve senaryo sıkıcıydı.',
    'Harika bir filmdi, tekrar izlemek isterim!',
    'Mükemmel bir deneyim, kesinlikle tavsiye ederim.',
    'Çok başarılı bir yapım olmuş, bayıldım.'
]
labels = np.array([0, 0, 0, 1, 1, 1]) # 0 = Negatif, 1 = Pozitif

# Metinleri tamsayı dizilerine dönüştürme (Tokenization)
# num_words=50: En sık geçen 50 kelimeyi al.
tokenizer = Tokenizer(num_words=50, oov_token="<oov>")
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index

# Metinleri sayısal dizilere (sequence) dönüştürme
sequences = tokenizer.texts_to_sequences(sentences)

# Dizi uzunluklarını eşitleme (Padding)
padded_sequences = pad_sequences(sequences, padding='post')

print("Orijinal Metinler:")
for i, s in enumerate(sentences):
    print(f"  - '{s}'")
print("\nSayısal Dizi Temsili (Padding Sonrası):")
print(padded_sequences)
print("-" * 50)

#-------------------------------------------------------------------------------
# Bölüm 2: Basit RNN Modeli Oluşturma ve Eğitme
#-------------------------------------------------------------------------------
print("\n### 2. Basit RNN Modeli ###")
print("-" * 50)

# RNN modeli tanımlama
model_rnn = Sequential([
    Embedding(input_dim=50, output_dim=16),
    SimpleRNN(32),
    Dense(1, activation='sigmoid') # İkili sınıflandırma için sigmoid
])

# Modeli derleme
model_rnn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Modeli eğitme
print("RNN Modeli Eğitiliyor...")
model_rnn.fit(padded_sequences, labels, epochs=5, verbose=0)
print("RNN Modeli Eğitimi Tamamlandı.")

# Modelin performansını değerlendirme
loss_rnn, accuracy_rnn = model_rnn.evaluate(padded_sequences, labels, verbose=0)
print(f"RNN Modelinin Doğruluk Oranı: {accuracy_rnn:.2f}")
print("-" * 50)

#-------------------------------------------------------------------------------
# Bölüm 3: LSTM Modeli Oluşturma ve Eğitme
#-------------------------------------------------------------------------------
print("\n### 3. LSTM Modeli ###")
print("-" * 50)

# LSTM modeli tanımlama
# Sadece SimpleRNN katmanını LSTM ile değiştiriyoruz
model_lstm = Sequential([
    Embedding(input_dim=50, output_dim=16),
    LSTM(32),
    Dense(1, activation='sigmoid')
])

# Modeli derleme
model_lstm.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Modeli eğitme
print("LSTM Modeli Eğitiliyor...")
model_lstm.fit(padded_sequences, labels, epochs=5, verbose=0)
print("LSTM Modeli Eğitimi Tamamlandı.")

# Modelin performansını değerlendirme
loss_lstm, accuracy_lstm = model_lstm.evaluate(padded_sequences, labels, verbose=0)
print(f"LSTM Modelinin Doğruluk Oranı: {accuracy_lstm:.2f}")
print("-" * 50)

#-------------------------------------------------------------------------------
# Bölüm 4: Sonuç ve Yorum
#-------------------------------------------------------------------------------
print("\n### 4. Karşılaştırma ve Sonuç ###")
print("-" * 50)

print(f"RNN Modeli Doğruluk: {accuracy_rnn:.2f}")
print(f"LSTM Modeli Doğruluk: {accuracy_lstm:.2f}")

# Yorum
print("\nYorum:")
print("• Bu basit örnekte, her iki modelin de yüksek doğruluk elde ettiğini görebilirsiniz.")
print("• Ancak, LSTM modelleri özellikle daha uzun ve karmaşık metinlerde, geçmiş bilgileri daha iyi koruduğu için basit RNN'lere göre çok daha iyi performans gösterir.")
print("• Gerçek hayattaki büyük veri setlerinde LSTM'in üstünlüğü daha belirgin hale gelir.")
print("• Transformer mimarileri ise (GPT, BERT gibi), bu sorunu 'attention' mekanizması ile çözerek bugün NLP'de en çok tercih edilen modeller haline gelmiştir.")