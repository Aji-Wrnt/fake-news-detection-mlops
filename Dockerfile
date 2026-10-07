# 1. Gunakan sistem dasar Python versi ringan
FROM python:3.9-slim

# 2. Buat folder kerja di dalam Docker
WORKDIR /app

# 3. Copy file requirements lalu install library-nya
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Download data stopwords untuk pembersihan teks di dalam Docker
RUN python -m nltk.downloader stopwords

# 5. Copy seluruh isi folder laptopmu ke dalam Docker
COPY . .

# 6. Jalankan otomatis train.py untuk bikin model, lalu jalankan app.py (Flask)
CMD python train.py && python app.py