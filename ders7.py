import sys
import subprocess
import nltk
import spacy

# Gerekli kütüphaneleri ve verileri kontrol etme ve yükleme fonksiyonu
def install_and_download():
    """Gerekli kütüphaneleri ve NLTK/SpaCy verilerini yükler."""
    packages = ["nltk", "spacy"]
    for package in packages:
        try:
            __import__(package)
        except ImportError:
            print(f"'{package}' kütüphanesi yüklü değil, yükleniyor...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    print("\n### Gerekli NLTK Verileri İndiriliyor... ###")
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger')
        nltk.data.find('chunkers/maxent_ne_chunker')
        nltk.data.find('corpora/words')
    except:
        nltk.download('averaged_perceptron_tagger', quiet=True)
        nltk.download('maxent_ne_chunker', quiet=True)
        nltk.download('words', quiet=True)
        
    print("\n### SpaCy İngilizce Dil Modeli Kontrol Ediliyor... ###")
    try:
        spacy.load("en_core_web_sm")
        print("SpaCy modeli 'en_core_web_sm' yüklü.")
    except OSError:
        print("SpaCy modeli 'en_core_web_sm' indirilmesi gerekiyor. İndiriliyor...")
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])

# Kurulum ve indirme işlemlerini başlatma
install_and_download()

# SpaCy modelini yükleme
nlp = spacy.load("en_core_web_sm")

# Örnek gerçek hayat metni: Bir haber başlığı
text = "Apple's CEO, Tim Cook, announced a new iPhone model in London last week."

print(f"Orijinal Metin: '{text}'")
print("=" * 70)

#-------------------------------------------------------------------------------
# Bölüm 1: NLTK ile POS ve NER Uygulaması
#-------------------------------------------------------------------------------
print("### 1. NLTK ile Uygulama ###")
print("-" * 50)

# Adım 1: Kelimeleri token'lara ayırma
tokens = nltk.word_tokenize(text)
print(f"Tokenlar: {tokens}\n")

# Adım 2: POS (Part-of-Speech) Etiketleme
# NLTK, her kelimenin sözcük türünü belirler.
# NNP: Proper Noun, NN: Noun, VBZ: Verb, vb.
pos_tags = nltk.pos_tag(tokens)
print("NLTK POS Etiketleri:")
print(pos_tags)
print("\nNLTK Yorumu: 'Apple' ve 'Tim Cook' gibi kelimeleri 'NNP' (Özel İsim) olarak etiketledi.")
print("-" * 50)

# Adım 3: NER (Named Entity Recognition)
# NLTK, POS etiketlerine dayanarak isimlendirilmiş varlıkları gruplar.
ner_tags = nltk.ne_chunk(pos_tags, binary=False)
print("NLTK NER Çıktısı (Ağaç Yapısı):\n")
print(ner_tags)

print("\nNLTK NER Yorumu: Çıktı, 'PERSON' (Kişi), 'ORGANIZATION' (Organizasyon) gibi etiketlerle bir ağaç yapısında gelir. Ancak çıktı çok detaylı değildir ve her varlığı doğru tanımayabilir.")
print("=" * 70)

#-------------------------------------------------------------------------------
# Bölüm 2: SpaCy ile POS ve NER Uygulaması
#-------------------------------------------------------------------------------
print("\n### 2. SpaCy ile Uygulama ###")
print("-" * 50)

# SpaCy, tüm işlemleri tek bir adımda yapar
doc = nlp(text)

# Adım 1: Kelimeleri, POS ve NER etiketlerini tek bir döngüde görüntüleme
print("SpaCy POS ve NER Etiketleri (Entegre Çıktı):")
for token in doc:
    print(f"  - Kelime: {token.text:12} | POS: {token.pos_:8} | Etiket: {token.ent_type_:10}")

# Adım 2: Sadece NER varlıklarını daha detaylı görüntüleme
print("\nSpaCy NER Çıktısı (Varlık Bazında):")
for ent in doc.ents:
    print(f"  - Varlık Metni: {ent.text:15} | Etiket: {ent.label_:10} | Açıklama: {spacy.explain(ent.label_)}")
    
print("\nSpaCy Yorumu: SpaCy, POS ve NER etiketlerini aynı anda ve daha anlaşılır bir formatta sunar. 'Apple' kelimesini 'ORG' (Organizasyon), 'Tim Cook'u 'PERSON' (Kişi) ve 'London'ı 'GPE' (Jeopolitik Varlık) olarak doğru ve detaylı bir şekilde etiketledi.")
print("=" * 70)

### Karşılaştırma ve Sonuç
print("\n### NLTK vs. SpaCy Karşılaştırması ###")
print("-" * 50)
print("• Kolaylık: SpaCy, önceden eğitilmiş modelleri sayesinde daha az kod satırı ve daha entegre bir çıktı sunar. NLTK, her adım için ayrı fonksiyonlar gerektirir.")
print("• Çıktı Kalitesi: SpaCy, genellikle daha doğru ve daha detaylı NER etiketleri sağlar ('GPE' gibi). NLTK'nin çıktısı ise daha çok ağaç yapısında bir gramer analizi gibidir.")
print("• Hız ve Performans: SpaCy, üretim ortamları için tasarlanmış olup çok daha hızlı çalışır.")
print("• Öğrenme: NLTK, her modülün ayrı ayrı nasıl çalıştığını anlamak için harika bir öğrenme aracıdır. SpaCy ise daha çok 'kullanıma hazır' bir çözümdür.")