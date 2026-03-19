import sqlite3

conn = sqlite3.connect("sample_data.db")
cursor = conn.cursor()

cursor.execute("""
INSERT INTO data_table (name, value, timestamp)
VALUES ('Test', 5.5, '2026-03-18 19:30:00')
""")

conn.commit()
conn.close()