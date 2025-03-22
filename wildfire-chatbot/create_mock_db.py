import sqlite3

# Connect or create the database file
conn = sqlite3.connect('wildfire_data.db')
cursor = conn.cursor()

# Create the zone_risk table
cursor.execute('''
CREATE TABLE IF NOT EXISTS zone_risk (
    zone_name TEXT PRIMARY KEY,
    ndvi REAL,
    nbr REAL,
    ndwi REAL,
    summary TEXT
)
''')

# Insert mock data
mock_data = [
    ("Zone A", 0.12, 0.25, 0.20, "Zone A has dry vegetation and moderate burn severity."),
    ("Zone B", 0.35, 0.15, 0.40, "Zone B shows healthy vegetation and low fire risk."),
    ("Zone C", 0.10, 0.30, 0.18, "Zone C is at high risk due to low NDVI and high burn index."),
]

cursor.executemany('INSERT OR REPLACE INTO zone_risk VALUES (?, ?, ?, ?, ?)', mock_data)

conn.commit()
conn.close()

print("✅ Mock wildfire_data.db created.")
