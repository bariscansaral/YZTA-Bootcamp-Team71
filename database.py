import sqlite3
import json

class MedicalDB:
    def __init__(self, db_name="neuroscan_system.db"):
        self.conn = sqlite3.connect(db_name)
        self.create_tables()

    def create_tables(self):
        cursor=self.conn.cursor()
        cursor.execute("""CREATE TABLE IF NOT EXISTS users(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        full_name TEXT NOT NULL,
        age INTEGER NOT NULL,
        is_doctor BOOLEAN DEFAULT 0
        )
        """)

        cursor.execute("""CREATE TABLE IF NOT EXISTS medical_records(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        doctor_id INTEGER,
        results TEXT,
        data TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(user_id) REFERENCES users(id),
        FOREIGN KEY(doctor_id) REFERENCES users(id)
        )
        """)
        self.conn.commit()

    def add_user(self, full_name, age, is_doctor=False):
        cursor = self.conn.cursor()
        cursor.execute('INSERT INTO users (full_name, age, is_doctor) VALUES (?, ?, ?)',
                       (full_name, age, is_doctor))
        self.conn.commit()
        return cursor.lastrowid

    def add_medical_record(self, user_id, doctor_id, results, data):
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO medical_records (user_id, doctor_id, results, data)
            VALUES (?, ?, ?, ?)
        ''', (user_id, doctor_id, json.dumps(results), json.dumps(data)))
        self.conn.commit()

    def get_user_id(self, full_name):
        cursor = self.conn.cursor()
        cursor.execute("SELECT id FROM users WHERE full_name = ?", (full_name,))
        result=cursor.fetchone()
        return result[0] if result else None