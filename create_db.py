import sqlite3

conn = sqlite3.connect("waste_predictions.db")
c = conn.cursor()

c.execute('''
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_name TEXT,
    predicted_class TEXT
)
''')

conn.commit()
conn.close()
print("Database and table created!")
