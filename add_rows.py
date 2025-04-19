import csv
from datetime import datetime

from tracker import Base, CatHealthLog, SessionLocal  # assuming tracker.py has your models

# Path to your CSV
CSV_FILE = "cat_logs.csv"

# Create DB session
db = SessionLocal()

with open(CSV_FILE, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        # Clean and convert values
        date_value = datetime.strptime(row['date'], "%Y-%m-%d").date()
        visible_issues = row['visible_issues'].strip()
        
        # Remove brackets from visible_issues string
        if visible_issues.startswith("[") and visible_issues.endswith("]"):
            visible_issues = visible_issues[1:-1].strip()
        
        # Create log entry
        log_entry = CatHealthLog(
            date=date_value,
            sleep_estimate=row['sleep_estimate'],
            food_range=row['food_range'],
            mood=row['mood'],
            activity_level=row['activity_level'],
            vocalization_level=row['vocalization_level'],
            affection_level=row['affection_level'],
            visible_issues=visible_issues,
            notes=row['notes']
        )
        db.add(log_entry)

# Commit once after all inserts
db.commit()
db.close()

print("✅ Database populated from CSV.")
