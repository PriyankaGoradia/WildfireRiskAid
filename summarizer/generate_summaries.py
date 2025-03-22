from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from summarizer_engine import summarize_zone
from db_models import SpectralIndex, Base #wherever the db_models is located. SpectralIndex is the 1 table we know so far.
import json


from collections import defaultdict

# connection to database
DB_URI = "postgresql://username:password@localhost:5432/wildfire_db"  # update this with the proper link
engine = create_engine(DB_URI)
Session = sessionmaker(bind=engine)
session = Session()

# Statistics for each zone
zones = defaultdict(dict)
all_entries = session.query(SpectralIndex).all()

for entry in all_entries:
    #This is heavily change when we get the actual table columns
    # Working off the idea that it's based off of "zones"
    zone_id = entry.index_id  # assuming this maps to a zone id
    zones[zone_id] = {
        "name" : entry.index_name,
        "min": entry.min_value,
        "max": entry.max_value,
        "mean": entry.mean_value
    }
    # zones[zone_id]["file_path"] = entry.file_path

# creating the summaries
summaries = {}

for zone_id, stats in zones.items():
    summary = summarize_zone(zone_id, stats)
    summaries[zone_id] = summary

#put all this info into a file
with open("zone_summaries.json", "w") as f:
    json.dump(summaries, f, indent=2)

