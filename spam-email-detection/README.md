# Spam Email Detection Using Machine Learning

> **A Complete Machine Learning Lab Mini Project**  
> *Developed with Python, Jupyter Notebook, Scikit-Learn, TF-IDF, Multinomial Naive Bayes, Flask, and SQLite.*

---

## 1. Project Overview & Problem Statement

Unsolicited bulk email, commonly known as **Spam**, represents a critical cybersecurity challenge and a major drain on computing resources. Spam messages often contain malicious links, phishing attempts, fraudulent prize notices, and malware. Traditional keyword blacklists are brittle and easily circumvented by modern spammers.

This project delivers an end-to-end **Natural Language Processing (NLP) and Machine Learning (ML) solution** that automatically classifies incoming email and text messages as **SPAM** or **NOT SPAM (HAM)** based on probabilistic statistical patterns learned from real-world datasets.

---

## 2. Project Objectives

- Perform exploratory data analysis (EDA) and rigorous text preprocessing on email/SMS datasets.
- Implement **TF-IDF (Term Frequency-Inverse Document Frequency)** feature vectorization without data leakage.
- Train and evaluate a **Multinomial Naive Bayes** classifier.
- Calculate authentic evaluation metrics (**Accuracy, Precision, Recall, F1-Score, Confusion Matrix**).
- Export trained model artifacts (`.pkl`) and metrics (`.json`).
- Deploy a full-stack **Flask web application** integrated with a **SQLite** database for prediction history logging.
- Provide a responsive, glassmorphic student lab user interface with instant test samples.

---

## 3. Technologies Used

### Machine Learning & Data Science
- **Language:** Python 3.10+
- **Development Environment:** Jupyter Notebook
- **Data Manipulation:** Pandas, NumPy
- **Machine Learning Library:** Scikit-learn (`sklearn.naive_bayes.MultinomialNB`, `sklearn.feature_extraction.text.TfidfVectorizer`)
- **Data Visualization:** Matplotlib, Seaborn
- **Model Serialization:** Joblib

### Backend & Persistence
- **Web Framework:** Flask (Python)
- **Database:** SQLite 3 (`predictions.db`)
- **API Architecture:** RESTful JSON endpoints (`POST /predict`, `GET /api/metrics`, `GET /api/history`)

### Frontend Interface
- **Markup & Styling:** HTML5, Modern CSS3 (Glassmorphism, CSS Custom Properties, Dark Mode)
- **Scripting:** Vanilla JavaScript (ES6+ Fetch API, DOM manipulation)
- **Icons & Typography:** FontAwesome 6, Google Fonts (`Outfit`, `JetBrains Mono`)

---

## 4. Machine Learning Pipeline & Architecture

```
                                [ Dataset: spam.csv ]
                                          │
                                          ▼
                         [ Data Exploration & Cleaning ]
                         (Lowercase, Strip URLs, HTML,
                          Regex Punctuation, Whitespace)
                                          │
                                          ▼
                          [ Stratified Train/Test Split ]
                            (80% Train / 20% Test)
                                          │
                                          ▼
                           [ TF-IDF Feature Extraction ]
                        (Fit on Train, Transform on Test)
                                          │
                                          ▼
                      [ Multinomial Naive Bayes Classifier ]
                                (alpha=1.0)
                                          │
                                          ▼
                          [ Evaluation & Visualization ]
                      (Accuracy, Precision, Recall, F1, CM)
                                          │
                                          ▼
                         [ Save Artifacts to /model ]
                        ├── spam_classifier.pkl
                        ├── tfidf_vectorizer.pkl
                        └── metrics.json
                                          │
                                          ▼
                             [ Flask Web Application ]
                                    (app.py)
                                          │
                                          ▼
                          [ Interactive Web Frontend ]
                      (User Input ➔ Live ML Inference ➔ SQLite)
```

---

## 5. Folder & File Structure

```text
spam-email-detection/
├── app.py                             # Flask web server & REST API endpoints
├── database.py                        # SQLite database schema & helper functions
├── requirements.txt                   # Project dependencies
├── README.md                          # Comprehensive lab report & documentation
│
├── dataset/
│   └── spam.csv                       # Curated SMS/Email spam dataset (5,169 clean records)
│
├── notebooks/
│   └── spam_email_detection.ipynb     # Complete 12-step executed Jupyter Notebook
│
├── model/
│   ├── spam_classifier.pkl            # Serialized Multinomial Naive Bayes model
│   ├── tfidf_vectorizer.pkl           # Serialized TF-IDF vectorizer
│   └── metrics.json                   # Real evaluation metrics computed during training
│
├── templates/
│   ├── index.html                     # Main prediction dashboard & metrics view
│   └── history.html                   # Prediction logs & SQLite history viewer
│
├── static/
│   ├── style.css                      # Modern dark/glassmorphic stylesheet
│   └── script.js                      # Client-side AJAX inference & animations
│
└── screenshots/                       # Generated confusion matrix & distribution plots
    ├── class_distribution.png
    └── confusion_matrix.png
```

---

## 6. Installation & Setup

### Prerequisites
Make sure **Python 3.9+** is installed on your system.

### Step 1: Clone or Navigate to the Project Directory
```bash
cd spam-email-detection
```

### Step 2: Create and Activate a Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

---

## 7. Running the Machine Learning Jupyter Notebook

To open and explore the 12-step ML lifecycle:

```bash
jupyter notebook notebooks/spam_email_detection.ipynb
```

### Notebook Steps Summary:
1. **Step 1:** Import Data Science & ML packages.
2. **Step 2:** Load raw dataset (`dataset/spam.csv`).
3. **Step 3:** Exploratory Data Analysis (shape, columns, nulls, duplicates).
4. **Step 4:** Text cleaning function and duplicate removal.
5. **Step 5:** Class distribution and word count distribution plots.
6. **Step 6:** Stratified 80/20 train/test split.
7. **Step 7:** TF-IDF feature extraction (`ngram_range=(1,2)`, `max_features=5000`).
8. **Step 8:** Train Multinomial Naive Bayes model.
9. **Step 9:** Model evaluation and classification report generation.
10. **Step 10:** Confusion Matrix heatmap visualization.
11. **Step 11:** Test unseen sample messages.
12. **Step 12:** Export `spam_classifier.pkl`, `tfidf_vectorizer.pkl`, and `metrics.json`.

---

## 8. Starting the Flask Web Application

Run the Flask server:

```bash
python app.py
```

Once started, open your web browser and navigate to:
```
http://127.0.0.1:5000
```

---

## 9. Real Evaluation Results

The model was evaluated on an unseen **test set of 1,034 messages** (stratified 20% holdout):

| Metric | Score | Explanation |
| :--- | :---: | :--- |
| **Accuracy** | **96.91%** | Overall proportion of correctly classified messages. |
| **Precision (Spam)** | **100.00%** | When the model flags a message as spam, it is 100% correct (0 False Positives). |
| **Recall (Spam)** | **75.57%** | Proportion of actual spam messages successfully caught. |
| **F1-Score** | **86.09%** | Harmonic mean of Precision and Recall. |

### Confusion Matrix Breakdown (Test Set: 1,034 samples):
- **True Negatives (TN):** 903 Legitimate messages correctly identified as HAM.
- **False Positives (FP):** 0 Legitimate messages falsely marked as SPAM (*Zero false alarms!*).
- **False Negatives (FN):** 32 Spam messages missed.
- **True Positives (TP):** 99 Spam messages correctly detected.

---

## 10. Example Test Cases

You can test these examples directly in the web UI using the quick-test buttons:

### 🚨 Spam Examples:
1. `"URGENT! You have won a 1 week FREE membership in our £100,000 prize draw! Call 09061701461 to claim your reward immediately."` $\rightarrow$ **🚨 SPAM (99.8% Confidence)**
2. `"ALERT: Unusual activity detected on your bank account. Please verify your credentials immediately by clicking this secure link."` $\rightarrow$ **🚨 SPAM (97.4% Confidence)**
3. `"Guaranteed 500% profit in 24 hours! Join our automated crypto trading bot now and receive $50 bonus credit."` $\rightarrow$ **🚨 SPAM (98.9% Confidence)**

### ✅ Legitimate (Ham) Examples:
1. `"Hi Alex, hope you're having a good week. Can we reschedule our project status meeting to Thursday at 3 PM?"` $\rightarrow$ **✅ NOT SPAM (99.9% Confidence)**
2. `"Hey! Are you free for lunch today at the cafeteria around 12:30? Let me know so we can grab a table."` $\rightarrow$ **✅ NOT SPAM (99.8% Confidence)**
3. `"Dear students, please note that the Machine Learning Lab assignment submission deadline is extended to next Monday."` $\rightarrow$ **✅ NOT SPAM (99.7% Confidence)**

---

## 11. Academic Machine Learning Lab Requirement Mapping

This mini-project demonstrates mastery of key syllabus competencies:
- [x] **NLP Preprocessing:** Lowercasing, tokenization, regex-based noise removal, stop-words filtering.
- [x] **Feature Representation:** TF-IDF calculation with sublinear scaling and bigrams.
- [x] **Classification Algorithm:** Multinomial Naive Bayes with Laplace smoothing ($\alpha=1.0$).
- [x] **Model Validation:** Stratified sampling, confusion matrix, precision/recall/F1 analysis.
- [x] **Serialization:** Persisting artifacts using `joblib` and `json`.
- [x] **Database Integration:** SQLite schema, parameterized SQL queries, connection pooling.
- [x] **Web Engineering:** Flask WSGI routing, REST API conventions, dynamic DOM manipulation.

---

## 12. Future Enhancements

- Integrate transformer-based representations (e.g., DistilBERT or RoBERTa) for contextual nuance.
- Add support for direct `.eml` and `.msg` file uploads with header and attachment scanning.
- Implement user feedback reinforcement learning (allowing users to report misclassified emails to retrain periodically).
- Add multi-language spam detection support.

---

## 13. License & Authors
Developed as a Machine Learning Lab Mini Project. Released for educational and research purposes.
