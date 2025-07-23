# BİSMİLLAHİRRAHMANİRRAHİM
"""
RNN ile bir sınıflandırma problemi çözme
Duygu analizi --> Bir cümlenin olumlu mu olumsuz mu olduğunu belirleme
Restorant yorumları değerlendirmesi
"""

# Konsolda uyarı ve bilgilendirme mesajlarını kapat
#import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  ## TF Log: 0=Tümü, 1=Uyarı+Hata, 2=Hata, 3=Hiçbirini gösterir

# Kütüphanleri içeri aktar
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, SimpleRNN
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import load_model


# Veriseti oluştur/çek
veri = pd.read_json('RNN/veriseti.json', encoding='utf-8')

# text ve label sayısını yazdır
print(f"\nVeri setinde {len(veri['text'])} metin ve {len(veri['label'])} etiket var.\n")

# Verisetinden rasgele 7 örnek al
print("Rasgele 7 örnek:")
print(veri.sample(7, random_state=42).to_string(index=False))


# Metin temizleme ve ön işleme
# jetonlayiciyı oluştur ve metinleri sayısal verilere dönüştür
jetonlayici = Tokenizer()
jetonlayici.fit_on_texts(veri['text'])
metinler = jetonlayici.texts_to_sequences(veri['text'])

# Metinleri sabit uzunlukta olacak şekilde doldur
maks_uzunluk = max(len(metin) for metin in metinler)
print(f"\nMaksimum metin uzunluğu: {maks_uzunluk}")
X = pad_sequences(metinler, maxlen=maks_uzunluk)
print(X.shape)

# Etiketleri sayısal verilere dönüştür
etiketleyici = LabelEncoder()
y = etiketleyici.fit_transform(veri['label'])
# Hangi etiket hangi encode değerine denk geliyor
etiketler_dict = dict(zip(etiketleyici.classes_, range(len(y))))
print(f"\nEtiketler ve encode değerleri: {etiketler_dict}")

# Veriyi eğitim ve test setlerine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=19)


# Metin temisili: word embedding: word2vec
# Word2Vec modelini oluştur
kelimeler = veri['text'].apply(lambda x: x.split())
gomme_byt = 70
word2vec_model = Word2Vec(sentences=kelimeler, vector_size=gomme_byt, window=5, min_count=1, workers=4)

# Kelimelerin vektörlerini al
kelime_vektorleri = word2vec_model.wv
print(f"\nKelime vektörlerinin boyutu: {gomme_byt}")
print(kelime_vektorleri.most_similar('güzel', topn=5))  # 'yemekler' kelimesine en yakın 5 kelimeyi yazdır

# Kelime vektörlerini embedding matrisi olarak hazırlama
gomme_matrisi = np.zeros((len(jetonlayici.word_index) + 1, gomme_byt))
for kelime, indeks in jetonlayici.word_index.items():
    if kelime in kelime_vektorleri:
        gomme_matrisi[indeks] = kelime_vektorleri[kelime]


# RNN modelini oluştur
model = Sequential()
model.add(Embedding(input_dim=len(jetonlayici.word_index) + 1, output_dim=gomme_byt, weights=[gomme_matrisi], input_length=maks_uzunluk, trainable=False))
model.add(SimpleRNN(33, return_sequences=False))    # RNN katmanı
model.add(Dense(19, activation='relu'))   # Gizli katman
model.add(Dense(len(etiketler_dict), activation='softmax'))   # Çıkış katmanı

# Modeli derle
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()  # Model özetini yazdır

# Modeli eğit
model.fit(X_train, y_train, epochs=19, batch_size=3, validation_data=(X_test, y_test))

# Modeli değerlendir
kayip, dogruluk = model.evaluate(X_test, y_test)
print(f"\nTest kaybı: {kayip:.4f}, Test doğruluğu: {dogruluk:.4f}")

# Modeli kaydet
model.save('RNN/rnn_sentiment_analysis_model.h5')

# Modeli yükle ve test et
yuklenen_model = load_model('RNN/rnn_sentiment_analysis_model.h5')


# Cümle sınıflandırma fonksiyonu
def cumle_siniflandir(cumle):
    cumle = [cumle]  # Cümleyi listeye al
    cumle = jetonlayici.texts_to_sequences(cumle)  # Metni sayısal verilere dönüştür
    cumle = pad_sequences(cumle, maxlen=maks_uzunluk)  # Sabit uzunlukta doldur
    tahmin = yuklenen_model.predict(cumle)  # Tahmin yap
    etiket_indeksi = np.argmax(tahmin, axis=1)[0]  # En yüksek olasılıklı etiket indeksini al
    etiket = etiketleyici.inverse_transform([etiket_indeksi])[0]  # Etiketi geri dönüştür
    return etiket


# Örnek cümleleri sınıflandır
ornek_cumleler = [
    "Yemekler çok lezzetliydi, tekrar geleceğim.",
    "Servis çok yavaştı, memnun kalmadım.",
    "Mekan çok güzel ve temizdi.",
    "Tatlılar bayattı, hiç beğenmedim.",
    "Garsonlar çok ilgisizdi, bir daha gelmem.",
    "Fiyatlar çok yüksekti ama yemekler güzeldi.",
    "Yemekler fena değildi, ortalama bir deneyimdi.",
]

for cumle in ornek_cumleler:
    sonuc = cumle_siniflandir(cumle)
    print(cumle, " => ", sonuc)