from flask import Flask, request, jsonify, render_template
import pickle
import re
import string
import nltk
from nltk.corpus import stopwords

app = Flask(__name__)

# Unduh dan inisialisasi stopwords NLTK
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

# Load model dan vectorizer
model = None
vectorizer = None
try:
    with open('model_fakeTrue_news.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
except FileNotFoundError:
    print("Peringatan: model atau vectorizer belum ditemukan. Jalankan train.py terlebih dahulu.")

# Pembersihan Teks (Disamakan persis dengan train.py)
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

# Endpoint untuk HTML
@app.route('/')
def home():
    return render_template('simple.html')

# Endpoint untuk Postman / Frontend
@app.route('/predict', methods=['POST'])
def predict():
    if model is None or vectorizer is None:
        return jsonify({'error': 'Model atau vectorizer belum tersedia di server. Jalankan train.py terlebih dahulu.'}), 500

    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'Masukkan teks berita dengan key "text"'}), 400
    
    raw_text = data['text']
    if not isinstance(raw_text, str) or not raw_text.strip():
        return jsonify({'error': 'Teks berita tidak boleh kosong.'}), 400
    
    cleaned_text = clean_text(raw_text)
    if not cleaned_text.strip():
        return jsonify({'error': 'Teks tidak memiliki kata yang valid setelah dibersihkan.'}), 400

    text_vector = vectorizer.transform([cleaned_text])
    prediction = model.predict(text_vector)
    
    result = "Fake News (Berita hoax)" if prediction[0] == 1 else "True News (Berita asli)"
    
    return jsonify({
        'status': 'success',
        'prediction': result
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000) 