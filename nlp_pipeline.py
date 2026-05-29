import re
import nltk
from google.cloud import bigquery
import pandas as pd
from snowballstemmer import stemmer

# NLTK Stopwords listesini indirelim
nltk.download('stopwords')
from nltk.corpus import stopwords

# 1. BigQuery'den Ham Veriyi Çekelim
client = bigquery.Client(project=PROJECT_ID)
query_raw = f"SELECT * FROM `{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}`"
df_raw = client.query(query_raw).to_dataframe()

# 2. Türkçe NLP Ön İşleme Fonksiyonu
turkce_stemmer = stemmer('turkish')
stop_words = set(stopwords.words('turkish'))

# Telekom özelinde ekstra etkisiz kelimeler ekleyebiliriz
extra_stopwords = {'bir', 'bu', 've', 'için', 'de', 'da', 'mi', 'mu', 'ya', 'veya'}
stop_words.update(extra_stopwords)

def clean_turkish_text(text):
    if not isinstance(text, str):
        return ""

    # Küçük harfe çevirme (Türkçe karakter duyarlı)
    text = text.replace('İ', 'i').replace('I', 'ı').replace('Ş', 'ş').replace('Ç', 'ç').replace('Ğ', 'ğ').replace('Ü', 'ü').replace('Ö', 'ö')
    text = text.lower()

    # Noktalama işaretlerini ve sayıları kaldırma (Sadece harfler kalsın)
    text = re.sub(r'[^a-zıişğüçö\s]', '', text)

    # Kelimelere ayırma (Tokenization)
    words = text.split()

    # Stop-words temizliği ve Kök indirgeme (Stemming)
    cleaned_words = []
    for word in words:
        if word not in stop_words:
            # Kelimenin kökünü bul
            stemmed_word = turkce_stemmer.stemWord(word)
            cleaned_words.append(stemmed_word)

    return " ".join(cleaned_words)

# 3. Pipeline'ı Çalıştıralım
print("Metin ön işleme adımı başladı, lütfen bekleyin...")
df_raw['cleaned_text'] = df_raw['raw_text'].apply(clean_turkish_text)
print("Metin ön işleme başarıyla tamamlandı!")

# Sonucu görelim
print("\n--- Orijinal vs. Temizlenmiş Metin Örnekleri ---");
print(df_raw[['raw_text', 'cleaned_text']].head(5))
