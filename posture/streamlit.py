import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

np.random.seed(42)  
dates = [datetime(2025, 1, 1) + timedelta(days=i) for i in range(90)]
rula_scores = np.random.randint(1, 11, 90)
reba_scores = np.maximum(rula_scores - np.random.randint(0, 2, 90), 1) 

data = pd.DataFrame({'Date': dates, 'RULA Score': rula_scores, 'REBA Score': reba_scores})

st.title("Posture Progress Dashboard")
st.markdown("This dashboard shows the historical posture scores and progress over time.")

session_range = st.slider("Select Session Range", 1, 90, (1, 10))
filtered_data = data.iloc[session_range[0]-1:session_range[1]]
fig, ax = plt.subplots()
ax.plot(filtered_data['Date'], filtered_data['RULA Score'], marker='o', color='blue', label='RULA Score')
ax.plot(filtered_data['Date'], filtered_data['REBA Score'], marker='o', color='green', label='REBA Score')
ax.set_title("Posture Assessment Scores Over Time")
ax.set_xlabel("Date")
ax.set_ylabel("Score")
ax.legend()

st.pyplot(fig)
rula_improvement = rula_scores[0] - rula_scores[-1]
reba_improvement = reba_scores[0] - reba_scores[-1]
st.markdown(f"### Progress Summary")
st.markdown(f"RULA improved from {rula_scores[0]} to {rula_scores[-1]}.")
st.markdown(f"REBA improved from {reba_scores[0]} to {reba_scores[-1]}.")