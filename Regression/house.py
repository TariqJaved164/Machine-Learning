# California House Price Prediction WITHOUT using Pipeline module

# Import necessary libraries
import pandas as pd  # For loading and analyzing data
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For plotting graphs
import seaborn as sns  # For visualizations

# Machine learning libraries
from sklearn.model_selection import train_test_split  # To split data into train and test sets
from sklearn.linear_model import LinearRegression  # Linear Regression model
from sklearn.preprocessing import StandardScaler  # For feature scaling
from sklearn.impute import SimpleImputer  # To handle missing values
from sklearn.metrics import mean_squared_error, r2_score  # For model evaluation

# Load the dataset (CSV downloaded from Kaggle)
data = pd.read_csv("california_housing.csv")  # Replace with actual file name if needed

# Display the first 5 rows
data.head()

# Check missing values
print("Missing values:\n", data.isnull().sum())

# Drop rows with missing target value if any
data = data.dropna(subset=["median_house_value"])

# Separate features and target variable
X = data.drop("median_house_value", axis=1)  # Features
y = data["median_house_value"]  # Target

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle missing values in numerical features
imputer = SimpleImputer(strategy="mean")  # Replace missing values with mean
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

# Train a Linear Regression model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Make predictions on test data
y_pred = model.predict(X_test_scaled)

# Evaluate the model
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("Root Mean Squared Error (RMSE):", rmse)
print("R² Score:", r2)

# Plot predicted vs actual values
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual House Prices")
plt.ylabel("Predicted House Prices")
plt.title("Actual vs Predicted House Prices (No Pipeline)")
plt.show()
