import sys
import subprocess

# Gerekli kütüphaneleri kontrol etme ve yükleme fonksiyonu
def install_package(package):
    """Belirtilen kütüphaneyi yükler."""
    try:
        __import__(package)
    except ImportError:
        print(f"'{package}' kütüphanesi yüklü değil, yükleniyor...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Kurulum işlemini başlatma
install_package("transformers")
install_package("torch")  # Transformers için PyTorch gerekli olabilir.

from transformers import pipeline

#-------------------------------------------------------------------------------
# Bölüm 1: Transfer Öğrenme ile Duygu Analizi
#-------------------------------------------------------------------------------
print("### 1. Transfer Öğrenme: Önceden Eğitilmiş Model Kullanımı ###")
print("-" * 50)

# 'pipeline' ile önceden eğitilmiş bir duygu analizi modelini yüklüyoruz.
# Bu model, milyarlarca metin üzerinde eğitildiği için sıfırdan bir model
# eğitmeye gerek kalmadan yüksek performans gösterir.
classifier = pipeline("sentiment-analysis")

# Duygu analizi yapılacak metinler
texts_to_analyze = [
    "Bu ürün harika! Kesinlikle tavsiye ederim.",
    "Beni hayal kırıklığına uğrattı, hiç memnun kalmadım.",
    "Hava bugün ne çok sıcak ne de çok soğuk."
]

print("Önceden Eğitilmiş Model ile Duygu Analizi:")
for text in texts_to_analyze:
    result = classifier(text)[0]
    label = result['label']
    score = result['score']
    print(f"Metin: '{text}'")
    print(f"  -> Sonuç: {label} (Güven Skoru: {score:.2f})\n")

print("\nTransfer Öğrenme Yorumu: Gördüğünüz gibi, bir modelin tüm karmaşıklığını sıfırdan inşa etmeden, sadece birkaç satır kodla zor bir görevi başarıyla tamamladık. Bu, alanında uzmanlaşmış devasa bir modelin bilgisini, küçük bir görev için transfer etmenin gücüdür.")
print("-" * 50)

#-------------------------------------------------------------------------------
# Bölüm 2: Üretken Yapay Zeka ile Metin Üretimi
#-------------------------------------------------------------------------------
print("\n### 2. Üretken Yapay Zeka: Metin Üretimi ###")
print("-" * 50)

# Metin üretimi için önceden eğitilmiş bir modeli (gpt2) yüklüyoruz.
# Bu model, Transformers kütüphanesi ile yerel olarak çalıştırılabilir.
generator = pipeline('text-generation', model='gpt2')

# Metin üretimi için bir başlangıç metni (prompt)
prompt_text = "Yapay zeka hızla gelişiyor ve gelecekte"

print("Modelin Üreteceği Metin:")
print(f"  -> Başlangıç Metni (Prompt): '{prompt_text}'\n")

# Modeli kullanarak metin üretimi
# max_length: Üretilecek metnin maksimum uzunluğu
# num_return_sequences: Kaç farklı metin üretileceği
generated_texts = generator(
    prompt_text,
    max_length=50,
    num_return_sequences=1
)

# Üretilen metni görüntüleme
for i, generated_text in enumerate(generated_texts):
    print(f"Üretilen Metin {i+1}:")
    print(f"  -> {generated_text['generated_text']}\n")

print("Üretken Yapay Zeka Yorumu: Model, verdiğimiz başlangıç cümlesini mantıklı ve tutarlı bir şekilde tamamladı. Bu, sohbet botları, içerik üretimi ve otomatik özetleme gibi alanlarda kullanılan üretken yapay zekanın temel prensibidir.")