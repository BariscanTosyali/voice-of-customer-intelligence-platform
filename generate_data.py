import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# 1. Şablon Müşteri Yorumları (Telekom Odaklı)
positive_reviews = [
    "İnternet hızı gerçekten çok başarılı, teşekkürler.",
    "Müşteri hizmetlerindeki arkadaş çok yardımcı oldu, sorunum hemen çözüldü.",
    "Yeni paketime geçiş yaptım, çekim gücü her yerde çok iyi.",
    "Mobil uygulama arayüzü çok kullanışlı olmuş, fatura ödemek çok pratik.",
    "Yıllardır bu hattı kullanıyorum, çekim kalitesinden çok memnunum."
]

negative_reviews = [
    "Faturam bu ay inanılmaz yüksek geldi, hiçbir bilgilendirme yapılmadı!",
    "Evde internet sürekli kopuyor, uzaktan çalışırken rezil oldum.",
    "Müşteri temsilcisine bağlanmak için yarım saat telefonda bekledim, kimse ilgilenmedi.",
    "Metrodaki çekim gücü sıfır, acil arama bile yapamıyorum.",
    "Yurt dışı paketi aldım ama veri akışı sağlanamadı, boşuna para ödedim.",
    "Taahhüdüm bitmek üzere diye sürekli arayıp rahatsız ediyorlar, bıktım.",
    "Hattım durup dururken servis dışı kaldı, sim kart hatası veriyor."
]

neutral_reviews = [
    "Fatura detaylarını inceledim, standart prosedür uygulanmış.",
    "Yeni kampanya hakkında bilgi almak için uygulamaya giriş yaptım.",
    "Kullanım detaylarımı SMS ile talep ettim, biraz geç de olsa geldi.",
    "Tarifemi değiştirmeyi düşünüyorum ama henüz karar veremedim.",
    "Paket fiyatları piyasa ortalamasında görünüyor."
]

def generate_voc_data(num_records=10000):
    channels = ['Twitter', 'Call_Center', 'SMS', 'Mobile_App']
    segments = ['Gold', 'Silver', 'Standard']

    data = []
    start_date = datetime(2026, 1, 1)

    for i in range(num_records):
        # Rastgele tarih üretimi
        random_days = random.randint(0, 120)
        random_hours = random.randint(0, 23)
        random_minutes = random.randint(0, 59)
        record_date = start_date + timedelta(days=random_days, hours=random_hours, minutes=random_minutes)

        # Yorum türü seçimi (Gerçekçi olması için şikayet oranını biraz yüksek tutalım)
        rand_val = random.random()
        if rand_val < 0.50:  # %50 Negatif
            text = random.choice(negative_reviews)
        elif rand_val < 0.80:  # %30 Pozitif
            text = random.choice(positive_reviews)
        else:  # %20 Nötr
            text = random.choice(neutral_reviews)

        data.append({
            'interaction_id': f"VOC_{1000000 + i}",
            'timestamp': record_date,
            'channel': random.choice(channels),
            'customer_segment': random.choice(segments),
            'raw_text': text
        })

    df = pd.DataFrame(data)
    # Tarihe göre sıralayalım
    df = df.sort_values(by='timestamp').reset_index(drop=True)
    return df

# Test için 10,000 satır üretelim
df_voc = generate_voc_data(10000)
df_voc.to_csv('voc_raw_data.csv', index=False, encoding='utf-8')
print(f"Başarıyla {len(df_voc)} satır veri üretildi ve 'voc_raw_data.csv' olarak kaydedildi.")
