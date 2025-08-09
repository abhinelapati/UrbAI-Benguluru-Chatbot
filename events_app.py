import sqlite3
from datetime import datetime

def get_db_connection():
    conn = sqlite3.connect('events.db')
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    conn.execute('''CREATE TABLE IF NOT EXISTS events
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  name TEXT NOT NULL,
                  date TEXT NOT NULL,
                  location TEXT NOT NULL,
                  description TEXT NOT NULL)''')
    conn.commit()
    conn.close()

def add_event(name, date, location, description):
    conn = get_db_connection()
    conn.execute('INSERT INTO events (name, date, location, description) VALUES (?, ?, ?, ?)',
                 (name, date, location, description))
    conn.commit()
    conn.close()

def get_upcoming_events():
    conn = get_db_connection()
    cursor = conn.execute('SELECT * FROM events WHERE date >= ? ORDER BY date ASC',
                          (datetime.now().date().isoformat(),))
    events = cursor.fetchall()
    conn.close()
    return [dict(event) for event in events]