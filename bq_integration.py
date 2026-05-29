import os
import pandas as pd
from google.cloud import bigquery

# Diğer yazdığımız modüllerden fonksiyonları içeri aktarıyoruz
from generate_data import generate_synthetic_data  # İlk adımda yazdığımız veri üretici
from nlp_pipeline import clean_feedback_data        # Metin temizleme fonksiyonun
from sentiment_analysis import train_sentiment_model # Eğittiğimiz Derin Öğrenme fonksiyonu
from topic_modeling import apply_topic_modeling     # Yukarıdaki LDA fonksiyonu

# GCP ve BigQuery Konfigürasyonları
PROJECT_ID = "voc-intelligence-platform"
DATASET_ID = "voc-dataset"
TABLE_ID = "processed_customer_feedback"
TABLE_REF = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}"

def upload_dataframe_to_bigquery(df):
    """
    Elde edilen nihai analitik DataFrame'i Google BigQuery'ye yazar/günceller.
    """
    print(f"[INFO] BigQuery bağlantısı kuruluyor: {TABLE_REF}")
    client = bigquery.Client(project=PROJECT_ID)
    
    # Tablo yazma konfigürasyonu (Eğer tablo varsa üzerine yazar: WRITE_TRUNCATE)
    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_TRUNCATE",
        autodetect=True
    )
    
    print("[INFO] Veriler BigQuery'ye yükleniyor...")
    job = client.load_table_from_dataframe(df, TABLE_REF, job_config=job_config)
    job.result()  # Yüklemenin tamamlanmasını bekler
    
    print(f"[SUCCESS] Veriler başarıyla BigQuery'ye aktarıldı! Toplam Satır: {len(df)}")

def run_end_to_end_pipeline():
    """
    Uçtan uca (End-to-End) tüm yapay zeka ve veri hattını tetikler.
    """
    print("=== VOICE OF CUSTOMER INTELLIGENCE PLATFORM PIPELINE BAŞLIYOR ===\n")
    
    # 1. Adım: Ham Veri Simülasyonu
    raw_df = generate_synthetic_data(num_records=10000)
    
    # 2. Adım: Gelişmiş Türkçe NLP Ön İşleme
    processed_df = clean_feedback_data(raw_df)
    
    # 3. Adım: Derin Öğrenme (Sequential Embedding) Duygu Analizi Modeli
    model, tokenizer = train_sentiment_model(processed_df)
    
    # Model tahminlerini DataFrame'e ekleme aşaması
    # (Eğitim fonksiyonu içinde etiketlendiği için burada tahmin eşleştirmesini yapıyoruz)
    sentiment_mapping = {0: "Negatif", 1: "Pozitif", 2: "Nötr"}
    processed_df['predicted_sentiment'] = processed_df['label'].map(sentiment_mapping)
    
    # 4. Adım: Gözetimsiz Öğrenme (Gensim LDA) Konu Modellemesi
    final_df = apply_topic_modeling(processed_df)
    
    # BigQuery için gereksiz ara kolonları (örneğin listeleri veya numerik label'ı) temizleme
    if 'label' in final_df.columns:
        final_df = final_df.drop(columns=['label'])
        
    # 5. Adım: BigQuery Veri Ambarına Yükleme
    upload_dataframe_to_bigquery(final_df)
    
    print("\n=== PIPELINE BAŞARIYLA TAMAMLANDI: DASHBOARD GÜNCELLENMEYE HAZIR ===")

if __name__ == "__main__":
    # Pipeline'ı lokalde veya sunucuda tetiklemek için ana giriş noktası
    run_end_to_end_pipeline()
