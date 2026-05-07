
# !pip install nltk spacy textblob --> spaCy NLTK ve TextBlob'u yükler 
# !python -m spacy download en_core_web_sm --> spaCy'ın en_core_web_sm modellerini indirir, ingilizce için temel bir modeldir.


# =========================
# Tek Hücre (KESİN ÇALIŞAN): spaCy & NLTK Tanıtım + Karşılaştırma (Kaggle)
# =========================
# Bu hücre, doğru sürümleri kurar ve NLTK'nin yeni adlandırılmış kaynaklarını (özellikle
# 'averaged_perceptron_tagger_eng' ve 'maxent_ne_chunker_tab') indirir. Ardından spaCy–NLTK
# karşılaştırma demolarını koşturur. NLTK NER kaynakları bulunamazsa zarifçe devre dışı bırakır.


# ——— İçe Aktarımlar ve Model/Kaynak Hazırlığı ———
import sys, time
import spacy
import nltk

# spaCy küçük İngilizce modeli
nlp = spacy.load("en_core_web_sm")

# Güvenli indirme yardımcıları
def safe_download(pack):
    try:
        nltk.download(pack, quiet=True)
    except Exception as e:
        print(f"[Uyarı] '{pack}' indirilemedi: {e}")

def ensure_resource(resource_hint, downloader_name):
    """nltk.data.find ile kaynağı arar; yoksa indirir."""
    try:
        nltk.data.find(resource_hint)
    except LookupError:
        safe_download(downloader_name)

# Zorunlu NLTK veri paketleri (yeni/eski adlar birlikte)
for pkg in [
    "punkt", "punkt_tab",
    "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng",
    "wordnet", "omw-1.4", "stopwords",
    "maxent_ne_chunker", "maxent_ne_chunker_tab",  # NER için hem eski hem yeni
    "words"
]:
    safe_download(pkg)

# Kaynakların varlığını doğrula (özellikle yeni adlar)
ensure_resource("tokenizers/punkt", "punkt")
ensure_resource("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng")
ensure_resource("corpora/wordnet", "wordnet")
ensure_resource("corpora/stopwords", "stopwords")
# NER chunker için yeni klasör yapısı
ensure_resource("chunkers/maxent_ne_chunker_tab/english_ace_multiclass", "maxent_ne_chunker_tab")
ensure_resource("corpora/words", "words")

from pprint import pprint
from nltk import word_tokenize, sent_tokenize, pos_tag, ne_chunk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords, wordnet

print("=== SÜRÜMLER ===")
print("spaCy:", spacy.__version__)
print("NLTK :", nltk.__version__)
print()

# ——— Örnek Metin ———
text = (
    "OpenAI released a new model in 2024. "
    "Dr. Ada Lovelace presented the results at the AI Research Summit in London. "
    "Many developers integrated the model into their products within weeks."
)

print("=== AMAÇ & GENEL BAKIŞ ===")
print(
    "- Bu hücre spaCy ve NLTK'yı giriş düzeyinde tanıtır ve karşılaştırır.\n"
    "- Görevler: Tokenization, Sentence Segmentation, POS, Lemmatization, Stemming, NER, Stop-words, Mikro Hız Testi.\n"
    "- Özet: NLTK = eğitsel/klasik NLP yapıtaşları; spaCy = üretim odaklı, hızlı pipeline.\n"
)

# ——— 1) Cümle/Kelime Bölütleme ———
print("\n=== 1) CÜMLE & KELİME BÖLÜTLEME ===")
doc = nlp(text)
print("spaCy cümleleri:")
for i, s in enumerate(doc.sents, 1):
    print(f"  {i}. {s.text}")
first_sent_spacy = list(doc.sents)[0]
print("spaCy (1. cümledeki tokenlar):")
print(" ", [t.text for t in first_sent_spacy])

print("\nNLTK cümleleri:")
nltk_sents = sent_tokenize(text)
for i, s in enumerate(nltk_sents, 1):
    print(f"  {i}. {s}")
first_sent_nltk = nltk_sents[0]
print("NLTK (1. cümledeki tokenlar):")
print(" ", word_tokenize(first_sent_nltk))

# ——— 2) POS (Sözcük Türü Etiketleme) ———
print("\n=== 2) POS (Part-of-Speech) ===")
print("spaCy (2. cümle):")
second_sent_spacy = list(doc.sents)[1]
for tok in second_sent_spacy:
    print(f"  {tok.text:<15} pos={tok.pos_:<6} tag={tok.tag_:<5} dep={tok.dep_:<10} head={tok.head.text}")

print("\nNLTK (2. cümle):")
second_sent_nltk = nltk_sents[1]
tokens_nltk = word_tokenize(second_sent_nltk)
# pos_tag: yeni tagger ismini gerektirebilir; eksikse indirmeyi dener.
try:
    pprint(pos_tag(tokens_nltk))
except LookupError:
    print("[Bilgi] 'averaged_perceptron_tagger_eng' indiriliyor ve tekrar deneniyor...")
    safe_download("averaged_perceptron_tagger_eng")
    pprint(pos_tag(tokens_nltk))

# ——— 3) Lemmatization ve Stemming ———
print("\n=== 3) LEMMATIZATION & STEMMING ===")
words_list = ["presented", "results", "integrated", "products", "better", "mice", "running"]

print("spaCy lemmatization:")
doc_words = nlp(" ".join(words_list))
for w, t in zip(words_list, doc_words):
    print(f"  {w:<12} -> {t.lemma_}")

def nltk_pos_to_wn(tag):
    if tag.startswith("J"): return wordnet.ADJ
    if tag.startswith("V"): return wordnet.VERB
    if tag.startswith("N"): return wordnet.NOUN
    if tag.startswith("R"): return wordnet.ADV
    return wordnet.NOUN

print("\nNLTK lemmatization (WordNet, POS ipucu ile):")
wnl = WordNetLemmatizer()
nltk_pos_list = pos_tag(words_list)
for w, p in nltk_pos_list:
    lemma = wnl.lemmatize(w, pos=nltk_pos_to_wn(p))
    print(f"  {w:<12} ({p}) -> {lemma}")

print("\nNLTK stemming (Porter):")
stemmer = PorterStemmer()
for w in words_list:
    print(f"  {w:<12} -> {stemmer.stem(w)}")

# ——— 4) Varlık Tanıma (NER) ———
print("\n=== 4) NER (Named Entity Recognition) ===")
print("spaCy NER:")
for ent in doc.ents:
    print(f"  {ent.text:<30} -> {ent.label_}")

print("\nNLTK ne_chunk:")
nltk_tokens_all = word_tokenize(text)
nltk_pos_all = pos_tag(nltk_tokens_all)
nltk_ner_tree = None
try:
    nltk_ner_tree = ne_chunk(nltk_pos_all, binary=False)
except LookupError as e:
    # Yeni paket adı gerekiyorsa indir ve tekrar dene
    msg = str(e)
    if "maxent_ne_chunker_tab" in msg:
        print("[Bilgi] 'maxent_ne_chunker_tab' indiriliyor...")
        safe_download("maxent_ne_chunker_tab")
    if "maxent_ne_chunker" in msg:
        print("[Bilgi] 'maxent_ne_chunker' indiriliyor...")
        safe_download("maxent_ne_chunker")
    if "words" in msg:
        print("[Bilgi] 'words' indiriliyor...")
        safe_download("words")
    # ikinci deneme
    try:
        nltk_ner_tree = ne_chunk(nltk_pos_all, binary=False)
    except LookupError:
        print("[Uyarı] NLTK NER kaynakları indirilemedi; NLTK NER adımı atlanıyor.")

if nltk_ner_tree is not None:
    print("NLTK ne_chunk (ilk ~30 yaprak):")
    print(" ", nltk_ner_tree[:30])

# ——— 5) Stop-words ———
print("\n=== 5) STOP-WORDS ===")
from spacy.lang.en.stop_words import STOP_WORDS as SPACY_STOP
nltk_stop = set(stopwords.words("english"))
print("Örnek 10 (spaCy):", sorted(list(SPACY_STOP))[:10])
print("Örnek 10 (NLTK) :", sorted(list(nltk_stop))[:10])

tokens_alpha = [t.text for t in doc if t.is_alpha]
spacy_filtered = [t for t in tokens_alpha if t.lower() not in SPACY_STOP]
nltk_filtered  = [t for t in tokens_alpha if t.lower() not in nltk_stop]
print("Filtrelenmiş (spaCy):", spacy_filtered)
print("Filtrelenmiş (NLTK) :", nltk_filtered)

# ——— 6) Mikro Hız Karşılaştırması ———
'''
print("\n=== 6) MİKRO HIZ KARŞILAŞTIRMASI (Oyuncak Ölçek) ===")
corpus = [text] * 1000  # Kaggle CPU için makul

t0 = time.time()
_ = [nlp(t) for t in corpus]  # tokenizer + tagger + lemmatizer + NER
spacy_time = time.time() - t0

def nltk_pipeline(t):
    sents = sent_tokenize(t)
    out = []
    for s in sents:
        toks = word_tokenize(s)
        pos  = pos_tag(toks)
        # NLTK NER kaynakları yoksa bu adımı es geç
        try:
            ne   = ne_chunk(pos, binary=False)
        except LookupError:
            ne   = pos  # sadece POS çıktısını döndür
        out.append(ne)
    return out

t0 = time.time()
_ = [nltk_pipeline(t) for t in corpus]
nltk_time = time.time() - t0

print(f"spaCy toplam (n={len(corpus)}): {spacy_time:.3f}s")
print(f"NLTK  toplam (n={len(corpus)}): {nltk_time:.3f}s")
print("Not: Gerçek iş yüklerinde spaCy, toplu işlemde (nlp.pipe) daha da avantajlı olabilir.")
'''

# ——— Uygulamalı Karşılaştırma Özeti ———
print("\n=== UYGULAMALI KARŞILAŞTIRMA (Kısa Özet) ===")
comparison = [
    ("Odak", "Üretim/pipeline, hız", "Eğitsel/klasik NLP yapıtaşları"),
    ("Kurulum", "spacy + model (en_core_web_sm)", "nltk + corpora/models indirme"),
    ("Tokenizasyon/Cümle", "Hızlı tokenizer + sentencizer/parser", "Punkt + klasik tokenizers"),
    ("POS/Morfoloji", "pos_/tag_/lemma_ (modern modeller)", "pos_tag + WordNet lemmatizer"),
    ("NER", "Güçlü OOTB neural NER", "ne_chunk (chunker); daha zayıf OOTB"),
    ("Hız/Bellek", "Cython optimize, verimli Doc", "Genelde yavaş (pipeline kurulumlu)"),
]
for k, s, n in comparison:
    print(f"- {k:<18} | spaCy: {s} | NLTK: {n}")

# ——— Bonus: spaCy toplu işleme paterni ———
print("\n=== BONUS: spaCy toplu işleme (nlp.pipe) örneği ===")
large_corpus = [text] * 2000
ents_preview = None
cnt = 0
for d in nlp.pipe(large_corpus, batch_size=256, n_process=1):
    if ents_preview is None:
        ents_preview = [(e.text, e.label_) for e in d.ents][:5]
    cnt += 1
print(f"nlp.pipe ile {cnt} doküman işlendi. Önizleme (ilk dokümanın ilk 5 varlığı): {ents_preview}")

print("\n=== NOTLAR ===")
print(
    "- Örnekler İngilizce metin üzerindedir. Türkçe için spaCy 'tr_core_news_sm' kurulabilir; "
    "NLTK tarafında Türkçe için ek kural/kaynak gerekir.\n"
    "- NLTK 3.9+ sürümünde POS/NER kaynak adları güncellenmiştir; bu hücre yeni adlarla uyumludur."
)
