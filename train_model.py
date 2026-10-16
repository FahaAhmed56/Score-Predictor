import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import joblib

df = pd.read_csv("data.csv")
X = df[["study_hours", "attendance", "previous_score"]]
y = df["exam_score"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print("Model trained successfully!")
print(f"MAE: {mean_absolute_error(y_test, predictions):.2f}")
print(f"R2 score: {r2_score(y_test, predictions):.2f}")

joblib.dump(model, "student_score_model.pkl")
print("Saved: student_score_model.pkl")
