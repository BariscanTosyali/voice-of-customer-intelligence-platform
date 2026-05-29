import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Konfigürasyon ve Hiperparametreler
MAX_WORDS = 5000  # En sık geçen kelime sınırı
MAX_LEN = 20      # Sabit metin boyutu
EMBEDDING_DIM = 32

def assign_label(text):
    """
    Metin içerisindeki anahtar kelimelere göre kural tabanlı 
    ground-truth etiketlerini simüle eder.
    """
    pos_keywords = ["başarılı", "teşekkürler", "yardımcı", "çözüldü", "iyi", "pratik", "memnunum"]
    neg_keywords = ["yüksek", "kopuyor", "rezil", "bekledim", "ilginmedi", "sıfır", "boşuna", "bıktım", "hata"]
    
    if any(word in text for word in neg_keywords):
        return 0  # Negatif
    elif any(word in text for word in pos_keywords):
        return 1  # Pozitif
    else:
        return 2  # Nötr

def prepare_data_for_training(df):
    """
    Veriyi etiketler, Train-Test olarak böler ve Tokenization işlemlerini tamamlar.
    """
    print("[INFO] Veri seti etiketleniyor ve eğitim için hazırlanıyor...")
    df['label'] = df['raw_text'].apply(assign_label)
    
    X = df['cleaned_text'].values
    y = to_categorical(df['label'].values, num_classes=3)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Tokenization & Padding
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)
    
    X_train_padded = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=MAX_LEN, padding='post', truncating='post')
    X_test_padded = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=MAX_LEN, padding='post', truncating='post')
    
    return X_train_padded, X_test_padded, y_train, y_test, tokenizer

def build_embedding_model(vocab_size):
    """
    Keras Embedding tabanlı Derin Öğrenme mimarisini kurar.
    """
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM),
        GlobalAveragePooling1D(),
        Dense(24, activation='relu'),
        Dropout(0.2),
        Dense(3, activation='softmax')  # 3 Sınıf: Negatif, Pozitif, Nötr
    ])
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def train_sentiment_model(df):
    """
    Modeli eğitir, test seti üzerinde değerlendirir ve eğitilmiş modeli döndürür.
    """
    X_train_padded, X_test_padded, y_train, y_test, tokenizer = prepare_data_for_training(df)
    vocab_size = len(tokenizer.word_index) + 1
    
    model = build_embedding_model(vocab_size)
    model.summary()
    
    print("\n[INFO] Derin Öğrenme Modeli eğitimi başlıyor...")
    model.fit(
        X_train_padded, y_train, 
        epochs=5, 
        validation_data=(X_test_padded, y_test), 
        batch_size=32,
        verbose=1
    )
    
    loss, accuracy = model.evaluate(X_test_padded, y_test, verbose=0)
    print(f"\n[SUCCESS] Test Seti Doğruluk Oranı (Accuracy): {accuracy:.4f}")
    
    return model, tokenizer

if __name__ == "__main__":
    print("[WARN] Bu modül tek başına çalıştırılmak yerine entegrasyon hattı üzerinden çağrılmalıdır.")
