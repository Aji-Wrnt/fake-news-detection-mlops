import pandas as pd
import numpy as np
import nltk
import re
import string
import pickle
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Download data bahasa
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load Data dari folder rawData
print("Load data")
df_true = pd.read_csv('rawData/True.csv')
df_fake = pd.read_csv('rawData/Fake.csv')

df_true['label'] = 0
df_fake['label'] = 1
df = pd.concat([df_true, df_fake], axis=0).sample(frac=1).reset_index(drop=True)

# Pembersihan Teks
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    words = text.split()
    words = [w for w in words if not w in stop_words]
    return " ".join(words)

print("Cleaning")
df['text'] = df['text'].apply(clean_text)

# Split & Vectorize
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)

# Training Model
print("Training")
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)


# Evaluasi
print("evaluasi Model")
# Transformasikan data uji (X_test) menggunakan vectorizer yang sudah di-fit
X_test_tfidf = vectorizer.transform(X_test)

# Prediksi label data uji menggunakan model
y_pred = model.predict(X_test_tfidf)

# Cetak laporan performa ke terminal Docker
print("\n======================= REPORT ========================")
print(classification_report(y_test, y_pred, target_names=['True News (0)', 'Fake News (1)']))
print(f"Total Accuracy Score: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("========================================================\n")


# Save Model
print("Saving artifacts")
with open('model_fakeTrue_news.pkl', 'wb') as m_file:
    pickle.dump(model, m_file)
with open('vectorizer.pkl', 'wb') as v_file:
    pickle.dump(vectorizer, v_file)

print("Selesai!.")