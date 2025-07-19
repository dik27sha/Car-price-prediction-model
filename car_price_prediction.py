# car_price_prediction.py

# STEP 1: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# STEP 2: Load the dataset
df = pd.read_csv("car_data.csv")  # Make sure this file is in your project folder

print("First 5 rows:\n", df.head())
print("\nDataset Info:\n")
print(df.info())

# STEP 3: Check missing values
print("\nMissing values:\n", df.isnull().sum())

# STEP 4: Feature engineering
df['Car_Age'] = 2025 - df['year']
df.drop(['year'], axis=1, inplace=True)

# STEP 5: Drop unnecessary column (optional)
df.drop(['name'], axis=1, inplace=True)

# STEP 6: Encode categorical variables using get_dummies
df = pd.get_dummies(df, drop_first=True)

# STEP 7: Scale numerical columns
scaler = StandardScaler()
numerical_cols = ['km_driven', 'Car_Age']  # ✅ Corrected case
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# STEP 8: Split into X and y
X = df.drop('selling_price', axis=1)
y = df['selling_price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# STEP 9: Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# STEP 10: Make predictions
y_pred = model.predict(X_test)

# STEP 11: Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"\nModel Evaluation:")
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

# STEP 12: Feature importance
coefficients = pd.Series(model.coef_, index=X.columns)
print("\nFeature Importance:\n")
print(coefficients.sort_values(ascending=False))

# STEP 13: Plot feature importance
plt.figure(figsize=(10,6))
sns.barplot(x=coefficients.values, y=coefficients.index)
plt.title("Feature Importance")
plt.xlabel("Coefficient Value")
plt.tight_layout()
plt.show()
#  we run this code in terminal 
#  .\venv\Scripts\activate
#   python car_price_prediction.py
