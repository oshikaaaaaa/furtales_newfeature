from tracker import SessionLocal, CatHealthLog

db = SessionLocal()
print("✅ Connected to DB")
print("Total rows in table:", db.query(CatHealthLog).count())
db.close()