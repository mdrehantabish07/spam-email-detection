# Spam Email Detection Using Machine Learning

> **A Complete Machine Learning Lab Mini Project**  
> *Developed with Python, Jupyter Notebook, Scikit-Learn, TF-IDF, Multinomial Naive Bayes, Flask, SQLite, and Light/Dark Theme support.*

---

## 1. Project Overview & Problem Statement

Unsolicited bulk email, commonly known as **Spam**, represents a critical cybersecurity challenge and a major drain on computing resources. Spam messages often contain malicious links, phishing attempts, fraudulent prize notices, and malware. Traditional keyword blacklists are brittle and easily circumvented by modern spammers.

This project delivers an end-to-end **Natural Language Processing (NLP) and Machine Learning (ML) solution** that automatically classifies incoming email and text messages as **SPAM** or **NOT SPAM (HAM)** based on probabilistic statistical patterns learned from real-world datasets.

---

## 2. Project Objectives

- Perform exploratory data analysis (EDA) and robust text preprocessing on email/SMS datasets.
- Normalize critical spam signals including currency symbols (`$`, `£`, `€`, `₹`), URLs, email addresses, and exclamations.
- Implement **TF-IDF (Term Frequency-Inverse Document Frequency)** feature vectorization with sublinear scaling and n-grams without data leakage.
- Benchmark and compare multiple algorithms: **Multinomial Naive Bayes**, **Complement Naive Bayes**, **Logistic Regression**, and **Linear Support Vector Classifier (SVC)**.
- Evaluate the best model (**Multinomial Naive Bayes with tuned $\alpha=0.1$**) using authentic metrics (**Accuracy, Precision, Recall, F1-Score, Confusion Matrix**).
- Export trained model artifacts (`.pkl`) and metrics (`.json`).
- Deploy a full-stack **Flask web application** integrated with a **SQLite** database for prediction history logging.
- Provide a responsive, glassmorphic student lab user interface with **Light & Dark Mode** and one-click quick test samples.

---

## 3. Technologies Used

### Machine Learning & Data Science
- **Language:** Python 3.10+
- **Development Environment:** Jupyter Notebook (`notebooks/spam_email_detection.ipynb`)
- **Data Manipulation:** Pandas, NumPy
- **Machine Learning Library:** Scikit-learn (`sklearn.naive_bayes.MultinomialNB`, `sklearn.feature_extraction.text.TfidfVectorizer`, `sklearn.linear_model.LogisticRegression`, `sklearn.svm.LinearSVC`)
- **Data Visualization:** Matplotlib, Seaborn
- **Model Serialization:** Joblib

### Backend & Persistence
- **Web Framework:** Flask (Python)
- **Database:** SQLite 3 (`predictions.db`)
- **API Architecture:** RESTful JSON endpoints (`POST /predict`, `GET /api/metrics`, `GET /api/history`)

### Frontend Interface
- **Markup & Styling:** HTML5, Modern CSS3 (Glassmorphism, CSS Custom Properties, Light/Dark Modes)
- **Scripting:** Vanilla JavaScript (ES6+ Fetch API, DOM manipulation, persistent `localStorage`)
- **Icons & Typography:** FontAwesome 6, Google Fonts (`Outfit`, `JetBrains Mono`)

---

## 4. Machine Learning Pipeline & Architecture

```
                                [ Dataset: spam.csv ]
                                          │
                                          ▼
                         [ Data Exploration & Cleaning ]
                         (Lowercase, Tokenize URLs, Emails,
                          Currency Symbols [$£€₹], Numbers)
                                          │
                                          ▼
                          [ Stratified Train/Test Split ]
                            (80% Train / 20% Test)
                                          │
                                          ▼
                           [ TF-IDF Feature Extraction ]
                        (Unigrams + Bigrams, sublinear_tf)
                                          │
                                          ▼
                      [ Multinomial Naive Bayes Classifier ]
                                  (alpha=0.1)
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
│   ├── style.css                      # Modern dark/light glassmorphic stylesheet
│   └── script.js                      # Client-side AJAX inference & theme manager
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
4. **Step 4:** Text cleaning function (normalizing currency symbols, URLs, email addresses, and exclamations).
5. **Step 5:** Class distribution and word count distribution plots.
6. **Step 6:** Stratified 80/20 train/test split.
7. **Step 7:** TF-IDF feature extraction (`ngram_range=(1,2)`, `max_features=7500`, `sublinear_tf=True`).
8. **Step 8:** Multi-model comparison (MultinomialNB, ComplementNB, Logistic Regression, Linear SVC).
9. **Step 9:** Best model evaluation and classification report generation.
10. **Step 10:** Confusion Matrix heatmap visualization.
11. **Step 11:** Test unseen real-world sample messages.
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
| **Accuracy** | **98.26%** | Overall proportion of correctly classified messages. |
| **Precision (Spam)** | **97.48%** | Low false positive rate (900 True Negatives vs 3 False Positives). |
| **Recall (Spam)** | **88.55%** | High spam catch rate (116 True Positives caught out of 131). |
| **F1-Score** | **92.80%** | Harmonic mean of Precision and Recall. |

### Algorithm Benchmark Comparison:
| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **MultinomialNB ($\alpha=0.1$)** | **98.26%** | **97.48%** | **88.55%** | **92.80%** |
| ComplementNB | 97.29% | 85.03% | 95.42% | 89.93% |
| Linear SVC (Balanced) | 97.87% | 92.91% | 90.08% | 91.47% |
| Logistic Regression (Balanced) | 97.49% | 90.08% | 90.08% | 90.08% |

---

## 10. Example Test Cases Verified

### 🚨 Spam Examples:
1. `"Congratulations! You have won ₹50,000 in our lucky draw. Claim your prize now by clicking this link. Offer expires today!"` $\rightarrow$ **🚨 SPAM (100.0% Confidence)**
2. `"URGENT! Your bank account will be blocked today. Verify your account immediately by clicking this link."` $\rightarrow$ **🚨 SPAM (83.0% Confidence)**
3. `"You have been selected to receive a FREE iPhone. Click here now to claim your prize before the offer expires."` $\rightarrow$ **🚨 SPAM (99.9% Confidence)**
4. `"Earn ₹10,000 every day from home with zero investment. Register now and start earning immediately."` $\rightarrow$ **🚨 SPAM (95.8% Confidence)**
5. `"You are the lucky winner of a ₹1,00,000 lottery prize. Send your details now to receive your money."` $\rightarrow$ **🚨 SPAM (99.8% Confidence)**

### ✅ Legitimate (Ham) Examples:
1. `"Hi team, our project meeting is scheduled for tomorrow at 10 AM. Please bring your progress updates."` $\rightarrow$ **✅ NOT SPAM (99.5% Confidence)**
2. `"Dear Professor, I have completed my Machine Learning assignment and will submit it before the deadline. Thank you."` $\rightarrow$ **✅ NOT SPAM (98.8% Confidence)**
3. `"Hey, are we still meeting at 5 PM today? Let me know if you need to change the time."` $\rightarrow$ **✅ NOT SPAM (99.9% Confidence)**
4. `"Hi Rahul, I have attached the updated report. Please review it and let me know if any changes are required."` $\rightarrow$ **✅ NOT SPAM (99.3% Confidence)**
5. `"Hi Mom, I will reach home around 8 PM today. Please don't wait for me for dinner."` $\rightarrow$ **✅ NOT SPAM (99.9% Confidence)**

---

## 11. Academic Machine Learning Lab Requirement Mapping

This mini-project demonstrates mastery of key syllabus competencies:
- [x] **NLP Preprocessing:** Lowercasing, tokenization, currency token preservation, URL/email tokenization, stop-words filtering.
- [x] **Feature Representation:** TF-IDF calculation with sublinear scaling, unigrams, and bigrams.
- [x] **Classification Algorithms & Comparison:** Multinomial Naive Bayes, ComplementNB, Logistic Regression, Linear SVC.
- [x] **Model Validation:** Stratified sampling, confusion matrix, precision/recall/F1 analysis.
- [x] **Serialization:** Persisting artifacts using `joblib` and `json`.
- [x] **Database Integration:** SQLite schema, parameterized SQL queries, connection pooling.
- [x] **Web Engineering:** Flask WSGI routing, REST API conventions, dynamic DOM manipulation, Light/Dark theme switching.
