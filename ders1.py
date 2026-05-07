import re

#--------------------------------------------------------------------------------
# Kısım 1: Basit Regex Kalıpları
# Nokta(.), yıldız(*) ve artı(+) gibi temel joker karakterleri deneyelim.
#--------------------------------------------------------------------------------
print("### Kısım 1: Basit Regex Kalıpları ###")
print("---------------------------------------")

text_1 = "Bugün hava güneşli. yarın da güneşli olacak."

# Örnek 1: 'güneşli' kelimesini arama
pattern_1 = r"güneşli"
result_1 = re.findall(pattern_1, text_1)
print(f"Metin: '{text_1}'")
print(f"Aranan Kalıp: '{pattern_1}'")
print(f"Sonuç: {result_1}")
print("-" * 50)

# Örnek 2: Nokta (.) joker karakteri
# 'g.neşli' kalıbı, 'g' ve 'n' arasında herhangi bir karakter olan kelimeleri bulur.
pattern_2 = r"g.neşli"
result_2 = re.findall(pattern_2, text_1)
print(f"Metin: '{text_1}'")
print(f"Aranan Kalıp: '{pattern_2}'")
print(f"Sonuç: {result_2}")
print("-" * 50)

#--------------------------------------------------------------------------------
# Kısım 2: Meta Karakterler ve Karakter Sınıfları
# \d (rakam), \s (boşluk), [] (karakter seti) gibi meta karakterleri deneyelim.
#--------------------------------------------------------------------------------
print("### Kısım 2: Meta Karakterler ve Karakter Sınıfları ###")
print("---------------------------------------------------------")

text_2 = "Telefon numaram 555-123-4567. Bugünün tarihi 25/08/2025."

# Örnek 3: Bir telefon numarasını bulma
# \d{3} -> tam 3 rakam
# \d{4} -> tam 4 rakam
pattern_3 = r"\d{3}-\d{3}-\d{4}"
result_3 = re.findall(pattern_3, text_2)
print(f"Metin: '{text_2}'")
print(f"Aranan Kalıp: '{pattern_3}'")
print(f"Sonuç: {result_3}")
print("-" * 50)

# Örnek 4: Bir tarihi bulma
# [\d/] -> rakam veya / işareti
# + -> bir veya daha fazla tekrar
pattern_4 = r"[\d/]+"
result_4 = re.findall(pattern_4, text_2)
print(f"Metin: '{text_2}'")
print(f"Aranan Kalıp: '{pattern_4}'")
print(f"Sonuç: {result_4}")
print("-" * 50)

#--------------------------------------------------------------------------------
# Kısım 3: Gerçek Dünya Uygulaması - E-posta Adresi Bulma
# Gruplama ve @ sembolünü kullanarak e-posta adreslerini çıkaralım.
#--------------------------------------------------------------------------------
print("### Kısım 3: Gerçek Dünya Uygulaması - E-posta Adresi Bulma ###")
print("----------------------------------------------------------------")

text_3 = "İletişim için bana ahmet@mail.com veya mehmet.kara@yandex.net adresinden ulaşabilirsiniz."

# Örnek 5: E-posta adresi bulma kalıbı
# \w+ -> bir veya daha fazla kelime karakteri
# @ -> @ sembolü
# \. -> nokta karakteri (özel bir anlamı olduğu için kaçış karakteri \ kullanılır)
pattern_5 = r"\w+@\w+\.\w+"
result_5 = re.findall(pattern_5, text_3)
print(f"Metin: '{text_3}'")
print(f"Aranan Kalıp: '{pattern_5}'")
print(f"Sonuç: {result_5}")
print("-" * 50)

#--------------------------------------------------------------------------------
# Kısım 4: Değiştirme (Substitution) İşlemi
# Bulunan bir kalıbı başka bir ifadeyle değiştirelim.
#--------------------------------------------------------------------------------
print("### Kısım 4: Değiştirme (Substitution) İşlemi ###")
print("---------------------------------------------------")

text_4 = "Gizli numaram: 555-123-4567."
# Telefon numarasını gizlemek için re.sub() kullanma
pattern_6 = r"\d{3}-\d{3}-\d{4}"
replacement = "***-***-****"
# re.sub(kalıp, yeni metin, orijinal metin)
new_text = re.sub(pattern_6, replacement, text_4)
print(f"Orijinal Metin: '{text_4}'")
print(f"Değiştirme Kalıbı: '{pattern_6}' -> '{replacement}'")
print(f"Sonuç: '{new_text}'")