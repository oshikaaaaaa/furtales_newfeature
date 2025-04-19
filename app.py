from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy import extract

from sqlalchemy.orm import sessionmaker, Session

from datetime import date
from typing import List, Optional
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from tracker import CatHealthLog, CatHealthLogCreate,CatHealthLogResponse

from datetime import datetime


from tracker import Base, CatHealthLog, SessionLocal 

import pandas as pd
from anamoly_detection import analyze_new_cat_log, detect_anomalies_detailed
from load_model import load_model

# Load the model and encoders when the app starts
model, model_features, label_encoders, i = load_model()# assuming tracker.py has your models

# Path to your CSV


# Create DB session
# FastAPI app
app = FastAPI()

# Serve static files (HTML, CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Route to create/update a log entry
@app.post("/log_cat_health/", response_model=dict)
def log_cat_health(log: CatHealthLogCreate, db: Session = Depends(get_db)):
    # Check if entry for this date already exists
    existing_log = db.query(CatHealthLog).filter(CatHealthLog.date == log.date).first()
    
    if existing_log:
        # Update existing entry
        for key, value in log.dict().items():
            setattr(existing_log, key, value)
        db.commit()
        log_entry = existing_log
    else:
        # Create new entry
        db_log = CatHealthLog(**log.model_dump())
        db.add(db_log)
        db.commit()
        db.refresh(db_log)
        log_entry = db_log
    
    # Convert ORM object to dict for anomaly detection
    log_dict = {
        'date': log_entry.date.isoformat(),
        'sleep_estimate': log_entry.sleep_estimate,
        'food_range': log_entry.food_range,
        'mood': log_entry.mood,
        'activity_level': log_entry.activity_level,
        'vocalization_level': log_entry.vocalization_level,
        'affection_level': log_entry.affection_level,
        'visible_issues': log_entry.visible_issues,
        'notes': log_entry.notes
    }
    
    # Run anomaly detection
    analysis = analyze_new_cat_log(log_dict, model, model_features, label_encoders)
    print("analysis done------------------------------------------------")
    print(analysis)
    
    return {
        "message": "Log entry saved successfully", 
        "log_id": log_entry.id,
        "analysis": {
            'date': log_entry.date.isoformat(),
            "is_anomaly": bool(analysis['is_anomaly']),
            "anomaly_score": float(analysis['anomaly_score']),
            "alert_level": analysis['alert_level'],
            "insights": analysis['insights'],
            "recommendation": analysis['recommendation']
        }
    }

# Route to get log by date
@app.get("/cat_health_log/{date_str}", response_model=Optional[CatHealthLogResponse])
def get_cat_health_log(date_str: str, db: Session = Depends(get_db)):
    try:
        log_date = date.fromisoformat(date_str)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
    
    log = db.query(CatHealthLog).filter(CatHealthLog.date == log_date).first()
    if not log:
        return None
    return log

# Route to get logs for a specific month
@app.get("/cat_health_logs/{year}/{month}", response_model=List[CatHealthLogResponse])
def get_month_logs(year: int, month: int, db: Session = Depends(get_db)):
    logs = db.query(CatHealthLog).filter(
        extract('year', CatHealthLog.date) == year,
        extract('month', CatHealthLog.date) == month
    ).all()
    return logs


from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import Dict, Any, List

@app.get("/cat_health_daily_insights/{date}")
def get_daily_insights(date: str, db: Session = Depends(get_db)):
    """
    Get daily insights for a specific date.
    
    This endpoint retrieves the health log for the specified date,
    analyzes it, and returns insights about the cat's health for that day.
    """
    # Get the entry for the specific date
    log_entry = db.query(CatHealthLog).filter(CatHealthLog.date == date).first()

    
    if not log_entry:
        raise HTTPException(status_code=404, detail="No data found for this date")
    
    # Convert ORM object to dict for analysis
    log_dict = {
        'date': log_entry.date.isoformat(),
        'sleep_estimate': log_entry.sleep_estimate,
        'food_range': log_entry.food_range,
        'mood': log_entry.mood,
        'activity_level': log_entry.activity_level,
        'vocalization_level': log_entry.vocalization_level,
        'affection_level': log_entry.affection_level,
        'visible_issues': log_entry.visible_issues,
        'notes': log_entry.notes
    }
    print(log_dict)
    
    # Run anomaly detection on this single log entry
    analysis = analyze_new_cat_log(log_dict, model, model_features, label_encoders)
    
    # Format the response
    # return {
    #     "trends": analysis["insights"],
    #     "summary": generate_summary(log_entry),
    #     "recommendations": [analysis["recommendation"]],
    #     "is_anomaly": bool(analysis["is_anomaly"]),
    #     "anomaly_score": float(analysis["anomaly_score"]),
    #     "alert_level": analysis["alert_level"]
    # }

    return {
        "message": f'Daily Insights for {date}',
        "log_id": log_entry.id,
        "analysis": {
            "is_anomaly": bool(analysis['is_anomaly']),
            "anomaly_score": float(analysis['anomaly_score']),
            "alert_level": analysis['alert_level'],
            "insights": analysis['insights'],
            "recommendation": analysis['recommendation'],
            "date": log_dict["date"]
        }
    }

def generate_summary(log_entry: CatHealthLog) -> str:
    """Generate a summary of the cat's health for a single day."""
    summary = f"On {log_entry.date.strftime('%Y-%m-%d')}, your cat was "
    
    # Add mood to summary
    if log_entry.mood:
        summary += f"{log_entry.mood.lower()}"
    else:
        summary += "in an unrecorded mood"
    
    # Add activity level to summary if present
    if log_entry.activity_level:
        summary += f" with {log_entry.activity_level.lower()} activity"
    
    # Add sleep information if present
    if log_entry.sleep_estimate:
        summary += f" and {log_entry.sleep_estimate.lower()} sleep"
    
    # Add food information if present
    if log_entry.food_range:
        summary += f". Food intake was {log_entry.food_range}"
    
    # Add visible issues if any
    if log_entry.visible_issues and log_entry.visible_issues.strip():
        summary += f". You noted these issues: {log_entry.visible_issues}"
    
    summary += "."
    return summary



# --- Helper Functions ---
def count_weekly_anomalies(logs_df, model, model_features, label_encoders):
    count = 0
    daily_analysis = []
    for _, row in logs_df.iterrows():
        single_df = pd.DataFrame([row])
        anomaly_date_map = {} 
        anomaly_categories = {}
        category_date_map = {}
        result = detect_anomalies_detailed(single_df, model, model_features, label_encoders)
        if result["is_anomaly"]:
            count += 1

            date_str = row["date"]
            anomaly_date_map[date_str] = result["insights"]

            for insight in result["insights"]:
                key = insight.lower()
                anomaly_categories[key] = anomaly_categories.get(key, 0) + 1

                if key not in category_date_map:
                    category_date_map[key] = set()
                category_date_map[key].add(date_str)
        daily_analysis.append({
            "date": row["date"],
            "is_anomaly": bool(result["is_anomaly"]),
            "score": result["anomaly_score"],
            "insights": result["insights"]
        })
    return count, daily_analysis

def detect_behavior_shift(logs_df, feature_name):
    mid = len(logs_df) // 2
    first_half = logs_df.iloc[:mid][feature_name]
    second_half = logs_df.iloc[mid:][feature_name]
    
    if logs_df[feature_name].dtype == int or logs_df[feature_name].dtype == object:
        shift = first_half.mode()[0] != second_half.mode()[0]
    else:
        shift = abs(first_half.mean() - second_half.mean()) > 10

    return shift, first_half.mode()[0], second_half.mode()[0]

def generate_weekly_summary(logs_df, model, model_features, label_encoders):
    summary = {}

    anomaly_count, daily_results = count_weekly_anomalies(logs_df, model, model_features, label_encoders)
    summary["anomalies"] = f"{anomaly_count} out of {len(logs_df)} days had unusual patterns."

    low_food_days = logs_df[logs_df['food_midpoint'] < 100]
    summary["food"] = f"{len(low_food_days)} days had low food intake."

    mood_shift, before_mood, after_mood = detect_behavior_shift(logs_df, "mood_encoded")
    if mood_shift:
        old = label_encoders['mood'].inverse_transform([before_mood])[0]
        new = label_encoders['mood'].inverse_transform([after_mood])[0]
        summary["mood"] = f"Mood shifted from {old} to {new}"
    else:
        summary["mood"] = "No major mood shift detected"

    activity_shift, before_act, after_act = detect_behavior_shift(logs_df, "activity_level_encoded")
    if activity_shift:
        old = label_encoders['activity_level'].inverse_transform([before_act])[0]
        new = label_encoders['activity_level'].inverse_transform([after_act])[0]
        summary["activity"] = f"Activity level changed from {old} to {new}"
    else:
        summary["activity"] = "Activity levels were stable"

    return summary, daily_results
from fastapi import HTTPException
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from typing import List, Dict
from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy.orm import Session

from anamoly_detection import analyze_new_cat_log
from load_model import load_model


@app.get("/weekly_insights/{date_str}")
def get_weekly_insights(date_str: str, db: Session = Depends(get_db)):
    """
    Get a weekly summary of cat health including trends and anomaly summaries.
    """
    try:
        end_date = datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")

    start_date = end_date - timedelta(days=6)

    # Get all logs from the past 7 days
    logs = db.query(CatHealthLog).filter(
        CatHealthLog.date >= start_date,
        CatHealthLog.date <= end_date
    ).order_by(CatHealthLog.date).all()

    if len(logs) < 3:
        return {
            "message": f"Only {len(logs)} log(s) found between {start_date} and {end_date} — not enough data for weekly insights.",
            "recommendation": "Log at least 3 days per week for accurate weekly trend analysis.",
            "weekly_insights": [],
            "has_message": False
        }

    # Analyzing trends and anomalies
    daily_insights = []
    anomaly_count = 0
    food_low_days = 0
    issues_reported = []
    anomaly_categories = {}
    category_date_map = {}  # Map categories to dates
    sleep_levels = {}
    activity_levels = {}
    affection_levels = {}
    mood_levels = {}

    for log in logs:
        log_dict = {
            'date': log.date.isoformat(),
            'sleep_estimate': log.sleep_estimate,
            'food_range': log.food_range,
            'mood': log.mood,
            'activity_level': log.activity_level,
            'vocalization_level': log.vocalization_level,
            'affection_level': log.affection_level,
            'visible_issues': log.visible_issues,
            'notes': log.notes
        }

        result = analyze_new_cat_log(log_dict, model, model_features, label_encoders)
        
        current_date = log.date.isoformat()
        daily_insights.append({
            "date": current_date,
            "is_anomaly": bool(result["is_anomaly"]),
            "anomaly_score": float(result["anomaly_score"]),
            "insights": result["insights"],
            "alert_level": result["alert_level"]
        })

        if result["is_anomaly"]:
            anomaly_count += 1
            for insight in result["insights"]:
                key = insight.lower()
                anomaly_categories[key] = anomaly_categories.get(key, 0) + 1
                
                # Store which dates this anomaly category appeared on
                if key not in category_date_map:
                    category_date_map[key] = set()
                category_date_map[key].add(current_date)  # Use current log date

        # Feature tracking
        def count_level(level_dict, value):
            if value:
                key = value.lower()
                level_dict[key] = level_dict.get(key, 0) + 1

        count_level(sleep_levels, log.sleep_estimate)
        count_level(activity_levels, log.activity_level)
        count_level(affection_levels, log.affection_level)
        count_level(mood_levels, log.mood)

        # Food logic
        if log.food_range:
            level = log.food_range.lower()
            if level in ["very low", "low"]:
                food_low_days += 1

        if log.visible_issues:
            issues_reported.append(log.visible_issues)

    summary = []

    if anomaly_count > 0:
        summary.append(f"{anomaly_count} anomaly log(s) detected this week.")
        if anomaly_categories:
            sorted_anomalies = sorted(anomaly_categories.items(), key=lambda x: x[1], reverse=True)
            for category, count in sorted_anomalies:
                if category == "unusual pattern detected but no single clear cause":
                    continue  # Skip vague categories

                summary.append(f"'{category}' anomaly appeared {count} time(s).")

                # Add which dates this category occurred
                dates = sorted(list(category_date_map.get(category, [])))
                if dates:
                    summary.append(f"   ↳ Occurred on: {', '.join(dates)}")

    if food_low_days >= 2:
        summary.append(f"{food_low_days} day(s) had low food intake.")

    if issues_reported:
        summary.append(f"Visible issues were reported on {len(issues_reported)} day(s): {', '.join(set(issues_reported))}.")

    def summarize_feature_trend(name, level_dict):
        sorted_levels = sorted(level_dict.items(), key=lambda x: x[1], reverse=True)
        if sorted_levels:
            top_level, count = sorted_levels[0]
            if count >= 3:
                summary.append(f"{name} was mostly '{top_level}' ({count} day(s)).")

    summarize_feature_trend("Sleep", sleep_levels)
    summarize_feature_trend("Activity level", activity_levels)
    summarize_feature_trend("Affection level", affection_levels)
    summarize_feature_trend("Mood", mood_levels)

    if not summary:
        summary.append("No major patterns or concerns found this week.")

    return {
        "message": f"Weekly insights from {start_date} to {end_date}",
        "summary": summary,
        "daily_insights": daily_insights,
        "has_message": True
    }
# Serve the main HTML page
@app.get("/")
def read_root():
    return FileResponse("static/index.html")

@app.get("/heatmap.html")
def get_heatmap():
    return FileResponse("static/heatmap.html")


@app.get("/stepgraph.png")
def get_step():
    return FileResponse("static/stepgraph.png")


from mangum import Mangum

# Create the handler
handler = Mangum(app)


