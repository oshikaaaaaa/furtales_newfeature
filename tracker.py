from fastapi import Depends
from sqlalchemy import create_engine, Column, Integer, String, Date, Text,Float, Boolean
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker
from pydantic import BaseModel
from datetime import date
from typing import List


# MySQL connection string
SQLALCHEMY_DATABASE_URL = "mysql+pymysql://root:Cooperation322060#@localhost:3306/pet_tracker"
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# SQLAlchemy models
class CatHealthLog(Base):
    __tablename__ = "cat_health_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    date = Column(Date, nullable=False)
    sleep_estimate = Column(String(50))
    food_range = Column(String(50))
    mood = Column(String(50))
    activity_level = Column(String(50))
    vocalization_level = Column(String(50))
    affection_level = Column(String(50))
    visible_issues = Column(Text)
    notes = Column(Text)
   
    
    # Analysis fields
    is_anomaly = Column(Boolean, default=False)
    anomaly_score = Column(Float, default=0.0)
    alert_level = Column(String(20), default="Normal")
    insights = Column(Text, default="")
    recommendation = Column(Text, default="")

# Create the table
Base.metadata.create_all(bind=engine)

# Pydantic schemas
class CatHealthLogCreate(BaseModel):
    date: date
    sleep_estimate: str
    food_range: str
    mood: str
    activity_level: str
    vocalization_level: str
    affection_level: str
    visible_issues: str
    notes: str

class CatHealthLogResponse(BaseModel):
    id: int
    date: date
    sleep_estimate: str
    food_range: str
    mood: str
    activity_level: str
    vocalization_level: str
    affection_level: str
    visible_issues: str
    notes: str
    is_anomaly: bool = False
    anomaly_score: float = 0.0
    alert_level: str = "Normal"
    insights: str = ""
    recommendation: str = ""
    
    class Config:
        orm_mode = True

class AnalysisResponse(BaseModel):
    is_anomaly: bool
    anomaly_score: float
    alert_level: str
    insights: List[str]
    recommendation: str
