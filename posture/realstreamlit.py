import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pymongo import MongoClient
from datetime import datetime, timedelta
client = MongoClient("mongodb://localhost:27017/")
db = client["posture_analysis"]
collection = db["best_posture_scores"]
st.title("Posture Progress Dashboard")
st.markdown("This dashboard shows the historical posture scores and progress over time.")

data_from_mongo = list(collection.find({}, {"_id": 0, "rula_score": 1, "reba_score": 1, "timestamp": 1}))

if not data_from_mongo:
    st.warning("No data found in MongoDB.")
else:
    df = pd.DataFrame(data_from_mongo)
    df['timestamp'] = pd.to_datetime(df['timestamp'])  
    # Session
    if len(df) == 1:
        session_range = st.slider("Select Session Range", 1, 2, (1, 2)) 
    else:
        session_range = st.slider("Select Session Range", 1, len(df), (1, min(10, len(df))))
    filtered_data = df.iloc[session_range[0]-1:session_range[1]]

    fig, ax = plt.subplots()
    ax.plot(filtered_data['timestamp'], filtered_data['rula_score'], marker='o', color='blue', label='RULA Score')
    ax.plot(filtered_data['timestamp'], filtered_data['reba_score'], marker='o', color='green', label='REBA Score')
    ax.set_title("Posture Assessment Scores Over Time")
    ax.set_xlabel("Date/Time")
    ax.set_ylabel("Score")
    ax.legend()

    st.pyplot(fig)

#Improvement
    rula_improvement = filtered_data['rula_score'].iloc[0] - filtered_data['rula_score'].iloc[-1]
    reba_improvement = filtered_data['reba_score'].iloc[0] - filtered_data['reba_score'].iloc[-1]

    st.markdown(f"### Progress Summary")
    st.markdown(f"RULA improved from {filtered_data['rula_score'].iloc[0]} to {filtered_data['rula_score'].iloc[-1]}.")
    st.markdown(f"REBA improved from {filtered_data['reba_score'].iloc[0]} to {filtered_data['reba_score'].iloc[-1]}.")

client.close()