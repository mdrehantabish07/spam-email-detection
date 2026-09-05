import sqlite3
import os
from datetime import datetime

# Default database file path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB_PATH = os.path.join(BASE_DIR, 'predictions.db')


def get_connection(db_path=None):
    """Establishes and returns a connection to the SQLite database."""
    if db_path is None:
        db_path = DEFAULT_DB_PATH
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path=None):
    """Initializes the database schema if the tables do not already exist."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            message TEXT NOT NULL,
            prediction TEXT NOT NULL,
            confidence REAL NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()


def insert_prediction(message, prediction, confidence, db_path=None):
    """Inserts a new prediction record into the SQLite database."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO predictions (message, prediction, confidence, created_at)
        VALUES (?, ?, ?, datetime('now', 'localtime'))
    ''', (message, prediction, float(confidence)))
    conn.commit()
    record_id = cursor.lastrowid
    conn.close()
    return record_id


def get_all_predictions(limit=100, db_path=None):
    """Retrieves all past prediction records ordered by most recent first."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT id, message, prediction, confidence, created_at
        FROM predictions
        ORDER BY id DESC
        LIMIT ?
    ''', (limit,))
    rows = cursor.fetchall()
    predictions = [
        {
            'id': row['id'],
            'message': row['message'],
            'prediction': row['prediction'],
            'confidence': round(row['confidence'], 2),
            'created_at': row['created_at']
        }
        for row in rows
    ]
    conn.close()
    return predictions


def clear_all_predictions(db_path=None):
    """Clears all prediction history from the database."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute('DELETE FROM predictions')
    conn.commit()
    conn.close()


def get_prediction_stats(db_path=None):
    """Returns aggregated stats from prediction history."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) as total FROM predictions')
    total = cursor.fetchone()['total']
    
    cursor.execute("SELECT COUNT(*) as spam_count FROM predictions WHERE prediction = 'SPAM'")
    spam_count = cursor.fetchone()['spam_count']
    
    cursor.execute("SELECT COUNT(*) as ham_count FROM predictions WHERE prediction != 'SPAM'")
    ham_count = cursor.fetchone()['ham_count']
    
    conn.close()
    return {
        'total': total,
        'spam_count': spam_count,
        'ham_count': ham_count
    }


if __name__ == '__main__':
    # Initialize database when run directly
    init_db()
    print("SQLite database initialized successfully at:", DEFAULT_DB_PATH)
