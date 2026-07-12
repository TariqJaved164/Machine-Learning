# Importing required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, AdaBoostRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
import joblib


# 1. LOAD AND SPLIT DATA
df = pd.read_csv("student_dataset_10000_rows.csv")

# Drop the target and unnecessary columns.
X = df.drop(columns=['exam_score', 'placement_status'])
y = df['exam_score']

# Train test split
X_train , X_test , y_train , y_test = train_test_split(X , y , random_state=42 , test_size=0.20)

# 2. DEFINE PREPROCESSING & MODEL PIPELINE
# List features to ensure deterministic mapping
numerical_features = ['study_hours', 'attendance', 'sleep_hours', 'internet_usage', 'assignments_completed', 'previous_score']

# Define preprocessor
preprocessor = ColumnTransformer(
                transformers=[('num' , StandardScaler() , numerical_features)])

# Combine preprocessor and model in a single pipeline.
pipeline = Pipeline(steps=[
                    ('preprocessor' , preprocessor),
                    ('regressor'  , RandomForestRegressor(random_state=42))])

# 3. TRAIN THE PIPELINE
print("Training the machine learning Pipeline....")
pipeline.fit(X_train , y_train)
print("Pipeline trained successfully....")

# 4. EVALUATE PERFORMANCE
y_pred = pipeline.predict(X_test)

print("--- Pipeline Performance Metrics ---")
print(f"Mean Absolute Error (MAE): {mean_absolute_error(y_test, y_pred):.2f} marks")
print(f"R-squared (R²) Score: {r2_score(y_test, y_pred):.2f}")

# 5. SAVE THE PIPELINE
joblib.dump(pipeline , 'student_score_pipeline.pkl')

# 6. INFERENCE / PRODUCTION TEST
new_student_data = pd.DataFrame([{
    'study_hours': 7,
    'attendance': 92,
    'sleep_hours': 8,
    'internet_usage': 3,
    'assignments_completed': 18,
    'previous_score': 85
}])
# Load the saved pipeline back up
deployed_pipeline = joblib.load('student_score_pipeline.pkl')
# Generate prediction seamlessly
predicted_grade = deployed_pipeline.predict(new_student_data)[0]
print(f"Predicted Final Exam Score: {predicted_grade:.2f}/100")
