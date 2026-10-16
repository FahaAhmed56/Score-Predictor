# Student Exam Score Predictor

A beginner-friendly Machine Learning project.

## What it does
The model learns from sample student data and predicts an exam score using:
- Study hours per day
- Attendance percentage
- Previous exam score

It uses **Linear Regression** from scikit-learn and a **Streamlit** web interface.

## How to run on Windows

1. Extract the ZIP.
2. Open the extracted `student_score_ml` folder in VS Code.
3. Open **Terminal > New Terminal**.
4. Create a virtual environment (recommended):

   `python -m venv venv`

5. Activate it:

   `venv\Scripts\activate`

6. Install packages:

   `pip install -r requirements.txt`

7. Train the ML model:

   `python train_model.py`

   This creates `student_score_model.pkl`.

8. Start the web app:

   `streamlit run app.py`

9. A browser page should open automatically. If it does not, copy the Local URL shown in the terminal into your browser.

## How the ML works
`data.csv` is the training data.

Inputs (features):
- `study_hours`
- `attendance`
- `previous_score`

Output (target):
- `exam_score`

`train_model.py` splits the data into training/testing sets, trains Linear Regression, prints MAE and R2 evaluation values, and saves the trained model.

`app.py` loads the saved model and lets a user make predictions through a simple UI.

## Demo
Try:
- Study hours: 7
- Attendance: 85%
- Previous score: 75

Then click **Predict Score**.

## Files
- `data.csv` - sample ML dataset
- `train_model.py` - model training/evaluation
- `app.py` - Streamlit UI
- `requirements.txt` - Python dependencies
- `README.md` - setup instructions
