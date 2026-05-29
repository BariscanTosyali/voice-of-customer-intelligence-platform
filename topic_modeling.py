import pandas as pd
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel

def prepare_lda_corpus(cleaned_texts):
    """
    Temizlenmiş metin listesini Gensim LDA modeli için 
    Dictionary (Sözlük) ve Corpus (Bag-of-Words) yapılarına dönüştürür.
    """
    # Metinleri kelimelerine ayırarak liste listesi (tokenized) haline getiriyoruz
    tokenized_texts = [text.split() for text in cleaned_texts]
    
    # Gensim sözlüğü oluşturma
    dictionary = Dictionary(tokenized_texts)
    
    # Kelimeleri frekans sınırlarına göre filtreleme (opsiyonel ama başarıyı artırır)
    dictionary.filter_extremes(no_below=2, no_above=0.9)
    
    # Bag-of-Words (BoW) matrisi oluşturma
    corpus = [dictionary.doc2bow(text) for text in tokenized_texts]
    
    return corpus, dictionary, tokenized_texts

def train_lda_model(corpus, dictionary, num_topics=3):
    """
    Gensim LDA modelini eğitir.
    """
    print(f"[INFO] {num_topics} konu kırılımı için LDA Konu Modellemesi eğitiliyor...")
    lda_model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=42,
        passes=10,
        alpha='auto'
    )
    return lda_model

def get_dominant_topics(lda_model, corpus, tokenized_texts):
    """
    Her bir döküman için en yüksek olasılığa sahip (baskın) konuyu bulur
     ve anlaşılır iş etiketlerine dönüştürür.
    """
    topic_mapping = {
        0: "Şebeke, Altyapı & Paket Sorunları",
        1: "Müşteri İlişkileri & Yüksek Fatura",
        2: "Dijital Kanallar & Uygulama Hataları"
    }
    
    dominant_topics = []
    
    for i, row in enumerate(lda_model[corpus]):
        # row içeriği: [(topic_id, probability), (topic_id, probability)...]
        # Olasılığa göre sıralayıp en yüksek olanı alıyoruz
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        dominant_topic_id = row[0][0]
        
        # İş biriminin anlayacağı etikete çevirme
        topic_name = topic_mapping.get(dominant_topic_id, "Diğer Operasyonel Konular")
        dominant_topics.append(topic_name)
        
    return dominant_topics

def apply_topic_modeling(df, num_topics=3):
    """
    Tüm konu modelleme sürecini orkestre eden ana fonksiyon.
    """
    corpus, dictionary, tokenized_texts = prepare_lda_corpus(df['cleaned_text'].values)
    lda_model = train_lda_model(corpus, dictionary, num_topics=num_topics)
    
    # Baskın konuları tespit et ve DataFrame'e ekle
    df['dominant_topic_name'] = get_dominant_topics(lda_model, corpus, tokenized_texts)
    print("[SUCCESS] LDA Konu Modellemesi tamamlandı ve etiketler atandı.")
    
    return df

if __name__ == "__main__":
    print("[WARN] Bu modül entegrasyon hattı üzerinden çağrılmalıdır.")
