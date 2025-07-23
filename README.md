# 🤖 RNN ile Türkçe Duygu Analizi

Bu proje, restoran yorumları üzerinden bir duygu analizi (sentiment analysis) sistemi geliştirmeyi amaçlamaktadır. Yorumlar `olumlu`, `olumsuz` ve `normal` olarak sınıflandırılmıştır. RNN (Recurrent Neural Network) mimarisi ile eğitilen model, Türkçe metinleri başarılı şekilde analiz edebilmektedir.

## 🧠 Kullanılan Teknolojiler

- Python
- TensorFlow & Keras
- Gensim (Word2Vec)
- Scikit-learn
- Pandas, NumPy

## 🗃️ Veri Kümesi

Veri kümesi 200 adet Türkçe restoran yorumu içermektedir. Her yorum, aşağıdaki etiketlerden biriyle sınıflandırılmıştır:

- `olumlu`
- `olumsuz`
- `normal`

Eğitim amaçlı 200 adet veri kullanılmıştır. İsteğe göre veri sayısı `veriseti.json` dosyası içinde çoğaltılabilir.

## 🧱 Model Mimarisi

Modelin mimarisi aşağıdaki gibidir:

```text
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 embedding (Embedding)       (None, 11, 70)            39130      
 simple_rnn (SimpleRNN)      (None, 33)                3432      
 dense (Dense)               (None, 19)                646       
 dense_1 (Dense)             (None, 3)                 60        
=================================================================
Toplam parametre sayısı: 43,268 (169.02 KB)  
Eğitilebilir parametre sayısı: 4,138 (16.16 KB)
Dondurulmuş parametre sayısı: 39,130 (152.85 KB)
```

## 📉 Test Sonuçları

Kayıp: 0.0797  
Doğruluk: 0.9756

## 🧪 Örnek Tahminler

Yemekler çok lezzetliydi, tekrar geleceğim.  ==>  olumsuz  
Servis çok yavaştı, memnun kalmadım.  ==>  olumsuz  
Mekan çok güzel ve temizdi.  ==>  olumlu  
Tatlılar bayattı, hiç beğenmedim.  ==>  olumsuz  
Garsonlar çok ilgisizdi, bir daha gelmem.  ==>  olumlu  
Fiyatlar çok yüksekti ama yemekler güzeldi.  ==>  olumlu  
Yemekler fena değildi, ortalama bir deneyimdi.  ==>  normal  

## 📌 Notlar

- Bu proje eğitim amacıyla hazırlanmıştır.
- Veri seti manuel olarak oluşturulmuş olup sınırlı sayıda örnek içermektedir.
- Geliştirmek için daha büyük bir veri kümesi ve farklı modeller denenebilir (LSTM, GRU, Transformer tabanlı modeller gibi).

