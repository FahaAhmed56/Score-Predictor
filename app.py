import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title="Student Score Predictor", page_icon="🤖")
st.title("🤖 Student Exam Score Predictor")
st.write("A small Machine Learning project using Linear Regression.")

model_path = Path("student_score_model.pkl")
if not model_path.exists():
    st.error("Model not found. Run: python train_model.py")
    st.stop()

model = joblib.load(model_path)

study_hours = st.slider("Study hours per day", 0.0, 12.0, 5.0, 0.5)
attendance = st.slider("Attendance (%)", 0, 100, 75)
previous_score = st.slider("Previous exam score", 0, 100, 65)

if st.button("Predict Score"):
    input_data = pd.DataFrame([{
        "study_hours": study_hours,
        "attendance": attendance,
        "previous_score": previous_score
    }])
    prediction = float(model.predict(input_data)[0])
    prediction = max(0, min(100, prediction))
    st.success(f"Predicted Exam Score: {prediction:.1f} / 100")
    st.progress(int(prediction))
    st.caption("This is a demo model trained on a small sample dataset.")
