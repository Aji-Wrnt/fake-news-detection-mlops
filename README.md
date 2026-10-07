# Fake News Detection System

An end-to-end Machine Learning and Natural Language Processing (NLP) system designed to detect and classify news articles into real or fake news. The project includes an automated data preprocessing and training pipeline, a Flask REST API with a simple web interface, and containerized deployment using Docker.

---

## Overview

The proliferation of misinformation and fake news poses significant challenges to information integrity. This project implements a binary text classifier using TF-IDF feature extraction and Logistic Regression to categorize English news articles:

- **True News (0)**: Factual and verified news articles.
- **Fake News (1)**: Fabricated or misleading news stories.

---

## Tech Stack

- **Programming Language**: Python 3.9+
- **Machine Learning & NLP**: scikit-learn, NLTK, pandas, numpy
- **Web Framework**: Flask
- **Containerization**: Docker, Docker Compose
- **Frontend**: HTML5, CSS3, JavaScript (Fetch API)

---

## Repository Structure

```text
.
├── rawData/
│   ├── .gitkeep                 # Preserves directory structure in version control
│   ├── Fake.csv                 # Fake news dataset (excluded from git)
│   └── True.csv                 # True news dataset (excluded from git)
├── templates/
│   └── simple.html              # Web user interface template
├── app.py                       # Flask application serving predictions and web UI
├── train.py                     # Pipeline for data preprocessing, vectorization, and training
├── Dockerfile                   # Docker container specification
├── docker-compose.yml           # Docker Compose configuration
├── requirements.txt             # Python dependencies
├── model_fakeTrue_news.pkl      # Serialized trained Logistic Regression model
├── vectorizer.pkl               # Serialized fitted TF-IDF vectorizer
├── .gitignore                   # Git ignore configuration
└── README.md                    # Project documentation
```

---

## Dataset

The model is trained on the ISOT Fake News Dataset, containing two separate subsets:

- **True.csv**: Articles sourced from reputable agencies (predominantly Reuters).
- **Fake.csv**: Articles collected from untrustworthy websites and flagged fact-checking sources.

### Schema
Each dataset contains the following attributes:
- `title`: Headline of the news article.
- `text`: Full body text of the article.
- `subject`: Category/topic (e.g., politics, world news).
- `date`: Publication date.

> **Note**: Due to GitHub file size limitations (both files combined exceed 115 MB), raw CSV files are tracked by `.gitignore`. To train the model locally, place `True.csv` and `Fake.csv` inside the `rawData/` directory.

---

## Pipeline Architecture

1. **Text Preprocessing**:
   - Lowercase normalization.
   - Removal of bracketed text, URLs, HTML tags, punctuation, newline characters, and alphanumeric strings with digits.
   - English stopword removal via NLTK corpus.
2. **Feature Extraction**:
   - Term Frequency-Inverse Document Frequency (`TfidfVectorizer`) restricted to the top 5,000 most informative features.
3. **Model Training & Evaluation**:
   - 80/20 train-test split (`random_state=42`).
   - Logistic Regression classifier trained on vectorized representations.
   - Evaluation metrics: Accuracy, Precision, Recall, and F1-score.
4. **Artifact Serialization**:
   - Persistence of the trained classifier (`model_fakeTrue_news.pkl`) and vocabulary vectorizer (`vectorizer.pkl`) via Python `pickle`.

---

## Getting Started

### Option 1: Local Environment

#### Prerequisites
- Python 3.9 or higher
- pip package manager

#### Setup Steps

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Create and activate a virtual environment:
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # Linux / macOS
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Train the model (optional if `.pkl` files are already present):
   ```bash
   python train.py
   ```

5. Run the web server:
   ```bash
   python app.py
   ```
   Open your browser and navigate to `http://localhost:5000`.

---

### Option 2: Docker & Docker Compose

#### Prerequisites
- Docker Engine and Docker Compose installed and running.

#### Run with Docker Compose
```bash
docker compose up --build
```

Access the application at `http://localhost:5000`.

To stop the container:
```bash
docker compose down
```

---

## API Documentation

### Web Interface
- **Endpoint**: `/`
- **Method**: `GET`
- **Description**: Serves the browser interface for interactive text analysis.

---

### Prediction Endpoint
- **Endpoint**: `/predict`
- **Method**: `POST`
- **Headers**: `Content-Type: application/json`

#### Request Body
```json
{
  "text": "The government passed a comprehensive budget reform bill earlier today in parliament."
}
```

#### Successful Response (200 OK)
```json
{
  "status": "success",
  "prediction": "True News (Berita asli)"
}
```
*or for flagged content:*
```json
{
  "status": "success",
  "prediction": "Fake News (Berita hoax)"
}
```

#### Error Response (400 Bad Request)
```json
{
  "error": "Teks tidak memiliki kata yang valid setelah dibersihkan."
}
```
