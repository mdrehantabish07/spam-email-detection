import os
import re
import json
import joblib
from flask import Flask, render_template, request, jsonify
import database

# Initialize Flask application
app = Flask(__name__)

# Base directory paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'spam_classifier.pkl')
VECTORIZER_PATH = os.path.join(BASE_DIR, 'model', 'tfidf_vectorizer.pkl')
METRICS_PATH = os.path.join(BASE_DIR, 'model', 'metrics.json')

# Initialize SQLite Database on startup
database.init_db()

# Load Trained Model and Vectorizer
print("Loading trained Machine Learning model and TF-IDF vectorizer...")
try:
    classifier = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    print("Model and vectorizer loaded successfully!")
except Exception as e:
    print(f"Error loading model artifacts: {e}")
    classifier = None
    vectorizer = None

# Load Actual Performance Metrics from JSON
try:
    with open(METRICS_PATH, 'r') as f:
        MODEL_METRICS = json.load(f)
    print("Model metrics loaded successfully!")
except Exception as e:
    print(f"Error loading metrics: {e}")
    MODEL_METRICS = {}


def preprocess_text(text):
    """
    Applies the exact text preprocessing used during Jupyter Notebook training:
    - Lowercase conversion
    - Removes URL links
    - Removes HTML tags
    - Cleans punctuation and non-alphanumeric symbols
    - Removes multiple extra whitespaces
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


@app.route('/')
def home():
    """Renders the main dashboard page."""
    return render_template('index.html', metrics=MODEL_METRICS)


@app.route('/history')
def history_page():
    """Renders the prediction history page with SQLite logs."""
    predictions = database.get_all_predictions()
    stats = database.get_prediction_stats()
    return render_template('history.html', predictions=predictions, stats=stats)


@app.route('/predict', methods=['POST'])
def predict():
    """
    API Endpoint: Receives email/message text, preprocesses it,
    runs inference via the trained TF-IDF + Naive Bayes pipeline,
    records the result to SQLite, and returns JSON response.
    """
    if classifier is None or vectorizer is None:
        return jsonify({
            'success': False,
            'error': 'Machine learning model is not loaded. Please ensure model files are present.'
        }), 500

    data = request.get_json(silent=True) or request.form
    email_text = data.get('message', '').strip()

    if not email_text:
        return jsonify({
            'success': False,
            'error': 'Please enter some email or message text to analyze.'
        }), 400

    try:
        # Preprocessing text
        cleaned_text = preprocess_text(email_text)

        if not cleaned_text:
            cleaned_text = email_text.lower()

        # Vectorization
        transformed_vector = vectorizer.transform([cleaned_text])

        # Inference via Multinomial Naive Bayes
        prediction_index = int(classifier.predict(transformed_vector)[0])
        prediction_probabilities = classifier.predict_proba(transformed_vector)[0]

        # 1 = SPAM, 0 = NOT SPAM (HAM)
        prediction_label = "SPAM" if prediction_index == 1 else "NOT SPAM"
        confidence = float(prediction_probabilities[prediction_index] * 100)

        # Store to SQLite Database
        database.insert_prediction(
            message=email_text,
            prediction=prediction_label,
            confidence=confidence
        )

        return jsonify({
            'success': True,
            'prediction': prediction_label,
            'confidence': round(confidence, 2),
            'probabilities': {
                'ham': round(float(prediction_probabilities[0]) * 100, 2),
                'spam': round(float(prediction_probabilities[1]) * 100, 2)
            },
            'cleaned_text': cleaned_text
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Prediction error: {str(e)}'
        }), 500


@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    """API Endpoint: Returns evaluation metrics computed in Jupyter Notebook."""
    return jsonify({
        'success': True,
        'metrics': MODEL_METRICS
    })


@app.route('/api/history', methods=['GET'])
def get_history_api():
    """API Endpoint: Returns recent predictions list from SQLite."""
    try:
        limit = int(request.args.get('limit', 50))
        records = database.get_all_predictions(limit=limit)
        stats = database.get_prediction_stats()
        return jsonify({
            'success': True,
            'predictions': records,
            'stats': stats
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/clear-history', methods=['POST'])
def clear_history_api():
    """API Endpoint: Clears all prediction history from SQLite."""
    try:
        database.clear_all_predictions()
        return jsonify({
            'success': True,
            'message': 'Prediction history cleared successfully.'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


if __name__ == '__main__':
    # Run development server
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting Spam Email Detection Flask App at http://127.0.0.1:{port}")
    app.run(host='127.0.0.1', port=port, debug=True)
