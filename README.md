# NLP-and-Text-Processing



1.Ders : Regex ile Basit Kelime Arama ve Nokta (.) Joker Karakteri

Bu kod, her bir Regex kavramını ayrı bir örnekle göstererek adım adım öğrenmenizi sağlar:
Kısım 1: re.findall() fonksiyonu ile basit kelime arama ve nokta (.) joker karakterinin nasıl çalıştığını görürsünüz.
Kısım 2: \d ve [\d/] gibi meta karakterleri kullanarak daha karmaşık kalıpları (telefon numarası, tarih) nasıl bulacağınızı deneyimlersiniz.
Kısım 3: Gerçek bir örnek olan e-posta adresi bulma ile Regex'in pratik faydasını görürsünüz. Bu örnek, \w (kelime karakteri) ve . (nokta karakteri) gibi meta karakterleri birleştirmenin gücünü gösterir.
Kısım 4: re.sub() fonksiyonunu kullanarak, bulunan bir kalıbın (telefon numarası) nasıl başka bir metinle (gizlenmiş numara) değiştirilebileceğini öğrenirsiniz. Bu, metin temizleme ve veri maskeleme gibi görevler için çok önemlidir.


*********************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************


2.Ders : NLTK ve Spacy ile Karşılaştırmalı Tokenization

Bu kod, metin parçalama işlemlerinin kütüphaneye göre nasıl farklılık gösterebileceğini net bir şekilde ortaya koyar:

NLTK Bölümü: word_tokenize ve sent_tokenize fonksiyonlarını kullanarak metni kelime ve cümle bazında ayırır. NLTK, basit ve kural tabanlı bir tokenizer olduğu için, Dr. kısaltmasını veya 25.08.2025 tarihini tam olarak tanıyamaz ve bu kısımları ayırarak token'lara böler.

Spacy Bölümü: Spacy, daha gelişmiş bir yaklaşım sunar. nlp nesnesi, önceden eğitilmiş Türkçe dil modelini kullanarak metni işler. Bu model, Dr. gibi yaygın kısaltmaları veya 25.08.2025 gibi tarih formatlarını tek bir token olarak algılayabilir. Bu, Spacy'nin daha bağlamsal ve akıllı bir tokenization yaptığının kanıtıdır. Ayrıca token nesnesinin is_punct, is_space gibi özelliklerini inceleyerek her bir token hakkında detaylı bilgi alabilirsiniz.

Karşılaştırma Bölümü: Kodun son kısmı, her iki kütüphanenin çıktılarındaki temel farkları vurgulayarak, hangi kütüphanenin ne zaman daha uygun olabileceğine dair bir fikir verir. Bu karşılaştırmayı doğrudan çıktıları okuyarak yapabilir ve her iki yaklaşımın güçlü ve zayıf yönlerini anlayabilirsiniz.

*********************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************


3.Ders : Metin Ön İşleme Teknikleri ve Örnekler

Contractions kütüphanesi, metin içindeki kısaltmaların genişletilmesini sağlar. 
Stopwords, metin içindeki durma kelimelerini kaldırmak için kullanılan bir arayüz sağlar. 
Stemming, metin içindeki kelimelerinin kısaltılmasını (ek olarak) sağlar.
Zeyrek, metin içindeki kelimelerinin kısaltılmasını ve küçük harf yapmasını sağlar. 


*********************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************

4.Ders : BoW ve N-gramlar ile Metin Temsili

Bu kod, metin içindeki kelimelerinin birbirlerine bağlı olarak bir araya getirilmesi için kullanılan metin ön İşleme tekniklerini gösterir. Bu, BoW (Bag of Words) ve N-gramlar ile metin temsili olarak çalışır.

BoW (Bag of Words) ile metin temsili yapmak için, metin içindeki tüm kelimeleri bir araya getirir. Bu, bir metin içindeki tüm kelimelerin birbirlerine bağlı olarak bir araya getirilmesi için kullanılır. BoW, metin içindeki tüm kelimelerin bir araya getirilmesini sağlar.   


N-gramlar, metin içindeki tüm kelimelerin birbirlerine bağlı olarak bir araya getirilmesi için kullanılan bir metin ön İşleme teknikidir.  
                                                                                        
                                                                                        
*********************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************

5.Ders : TF-IDF ve Word2Vec ile Kelime Önemini Ölçme

Bu kod, TF-IDF ve Word2Vec ile kelime önemini ölçme işlemlerini gösterir. Bu, TF-IDF ve Word2Vec'in kullanımını ve önemini ölçme işlemlerini gösterir.

TF-IDF, metin içindeki tüm kelimelerin birbirlerine bağlı olarak bir araya getirilmesi için kullanılan bir metin ön İşleme teknikidir. TF-IDF, bir metin içindeki tüm kelimelerin birbirlerine bağlı olarak bir araya getirilmesi için kullanılır. TF-IDF, kelimelerin önemini ölçmek için kullanılır.

Word2Vec, metin içindeki tüm kelimelerin birbirlerine bağlı olarak bir araya getirilmesi için kullanılan bir metin ön İşleme teknikidir. Word2Vec, bir metin içindeki tüm kelimelerin birbirlerine bağlı olarak bir araya getirilmesi için kullanılır. Word2Vec, kelimelerin anlamsal vektörlerini üretir.

*********************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************

6.Ders :  Makine Öğrenmesi ile Duygu Analizi Modeli

Bu kod, metin içindeki duygu analizi işlemlerini gösterir. Bu, TextBlob ile duygu analizi, Makine Öğrenmesi ile duygu analizi modelini kullanarak duygu analizi işlemlerini gösterir.

TextBlob, metin içindeki duygu analizi işlemlerini gösterir. 


Makine Öğrenmesi ile duygu analizi modelini kullanarak duygu analizi işlemlerini gösterir. 


*********************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************
 7.Ders : NLTK ve Spacy ile Karşılaştırmalı POS ve NER Uygulaması

Bu kod, metin içindeki POS ve NER işlemlerini gösterir. Bu, NLTK ve Spacy ile POS ve NER uygulamasını gösterir.

NLTK, her modülün ayrı ayrı nasıl çalıştığını anlamak için harika bir öğrenme aracıdır. SpaCy ise daha çok 'kullanıma hazır' bir çözümdür.


*********************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************

8.Ders : RNN ile Makine Öğrenmesi

Bu kod, metin içindeki duygu analizi işlemlerini gösterir. Bu, TextBlob ile duygu analizi, Makine Öğrenmesi ile duygu analizi modelini kullanarak duygu analizi işlemlerini gösterir.

TextBlob, metin içindeki duygu analizi işlemlerini gösterir. 


Makine Öğrenmesi ile duygu analizi modelini kullanarak duygu analizi işlemlerini gösterir. 


*********************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************  

9.Ders : Transfer Öğrenme ile Duygu Analizi

Bu kod, metin içindeki duygu analizi işlemlerini gösterir. Bu, TextBlob ile duygu analizi, Makine Öğrenmesi ile duygu analizi modelini kullanarak duygu analizi işlemlerini gösterir.

TextBlob, metin içindeki duygu analizi işlemlerini gösterir. 


Makine Öğrenmesi ile duygu analizi modelini kullanarak duygu analizi işlemlerini gösterir.