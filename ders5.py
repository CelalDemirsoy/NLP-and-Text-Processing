import sys
import subprocess
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from gensim.models import Word2Vec

# Gerekli kütüphaneleri ve verileri kontrol etme ve yükleme fonksiyonu
def install_and_download():
    """Gerekli kütüphaneleri ve NLTK veri paketlerini yükler."""
    packages = ["scikit-learn", "gensim", "pandas"]
    for package in packages:
        try:
            __import__(package)
        except ImportError:
            print(f"'{package}' kütüphanesi yüklü değil, yükleniyor...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    try:
        nltk.data.find('tokenizers/punkt')
    except:
        nltk.download('punkt', quiet=True)

# Kurulum ve indirme işlemlerini başlatma
install_and_download()

# Örnek metin veri seti (doküman korpusu)
corpus = [
    "Makine öğrenmesi modelleri için veri bilimi çok önemlidir.",
    "Doğal dil işleme, yapay zekanın önemli bir alt dalıdır.",
    "Yapay zeka ve makine öğrenmesi, veri bilimi alanında kullanılır.",
    "İnsanlar makine öğrenmesi modellerini geliştirir ve veri bilimine katkı sağlar."
]

# Metinleri tokenlara ayırma (Word2Vec için gerekli)
tokenized_corpus = [nltk.word_tokenize(doc.lower()) for doc in corpus]

#-------------------------------------------------------------------------------
# Bölüm 1: TF-IDF (Term Frequency-Inverse Document Frequency)
#-------------------------------------------------------------------------------
print("### 1. TF-IDF Uygulaması: Kelime Önemini Ölçme ###")
print("-" * 50)

# TfidfVectorizer ile TF-IDF matrisini oluşturma
# Stopwords'leri ve tek karakterli kelimeleri kaldırarak daha temiz bir vektör elde ederiz.
tfidf_vectorizer = TfidfVectorizer(stop_words='english', token_pattern=r'\b[^\d\W]+\b')
X_tfidf = tfidf_vectorizer.fit_transform(corpus)

# Kelime sözlüğünü ve matrisi görüntüleme
feature_names_tfidf = tfidf_vectorizer.get_feature_names_out()
df_tfidf = pd.DataFrame(X_tfidf.toarray(), columns=feature_names_tfidf)
print("TF-IDF Matrisi (DataFrame Formatında):\n")
print(df_tfidf)

print("\nTF-IDF Yorumu: 'bilim' ve 'zeka' gibi kelimelerin, 'için' veya 'bir' gibi sık kullanılan kelimelere göre daha yüksek skorlar aldığını görebilirsiniz.")
print("-" * 50)

#-------------------------------------------------------------------------------
# Bölüm 2: Word2Vec: Kelime Gömme (Embeddings)
#-------------------------------------------------------------------------------
print("\n### 2. Word2Vec: Kelimelerin Anlamsal Vektörleri ###")
print("-" * 50)

# Word2Vec modelini eğitme
# min_count=1 ile tüm kelimeleri dahil ederiz, vector_size=100 ile vektör boyutunu belirleriz.
model_w2v = Word2Vec(tokenized_corpus, min_count=1, vector_size=100)

# 'veri' kelimesinin en benzer 3 kelimesini bulma
print(" 'veri' kelimesine en benzer 3 kelime:")
similar_words = model_w2v.wv.most_similar('veri', topn=3)
print(similar_words)

# Klasik bir Word2Vec örneği: King - Man + Woman = Queen
# Bu modelimiz küçük olduğu için benzer bir örnek verelim
# Örnek: 'makine' ve 'yapay' ilişkisinden yola çıkarak 'bilimi' kelimesinin benzerini bulma
print("\n'zeka - yapay + makine' = ? (Semantik İlişki Tahmini)")
semantic_example = model_w2v.wv.most_similar(positive=['makine', 'zeka'], negative=['yapay'])
print(semantic_example)

print("\nWord2Vec Yorumu: Kelimeler arasındaki anlamsal ilişkileri, sadece birlikte geçme sıklıklarından yola çıkarak öğrenir. Bu, kelimeleri çok boyutlu bir uzayda temsil etmemizi sağlar.")
print("-" * 50)

#-------------------------------------------------------------------------------
# Bölüm 3: Kosinüs Benzerliği: İki Doküman Arasındaki Benzerlik
#-------------------------------------------------------------------------------
print("\n### 3. Kosinüs Benzerliği: Dokümanları Karşılaştırma ###")
print("-" * 50)

# Örnek olarak 0. ve 2. dokümanları alalım.
# İki doküman arasında benzerlik bekliyoruz çünkü ikisi de 'yapay zeka' ve 'makine öğrenmesi' içeriyor.
doc1_index = 0
doc2_index = 2
doc1 = corpus[doc1_index]
doc2 = corpus[doc2_index]

print(f"Doküman 1: '{doc1}'")
print(f"Doküman 2: '{doc2}'")

# Dokümanların TF-IDF vektörlerini alalım
vec1 = X_tfidf[doc1_index:doc1_index+1]
vec2 = X_tfidf[doc2_index:doc2_index+1]

# Kosinüs benzerliğini hesapla
similarity_score = cosine_similarity(vec1, vec2)
print(f"\nDoküman 1 ve 2 Arasındaki Kosinüs Benzerliği: {similarity_score[0][0]:.4f}")

# Benzer olmayan bir dokümanla karşılaştıralım (0. ve 1. doküman)
doc3_index = 1
vec3 = X_tfidf[doc3_index:doc3_index+1]
similarity_score2 = cosine_similarity(vec1, vec3)
print(f"Doküman 1 ve 3 Arasındaki Kosinüs Benzerliği: {similarity_score2[0][0]:.4f}")

print("\nKosinüs Benzerliği Yorumu: Skor 1'e ne kadar yakınsa, dokümanlar o kadar benzerdir. Gördüğünüz gibi, konu olarak daha benzer olan 1. ve 3. dokümanlar arasında daha yüksek bir benzerlik skoru elde ettik.")