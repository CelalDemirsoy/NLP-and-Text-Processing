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