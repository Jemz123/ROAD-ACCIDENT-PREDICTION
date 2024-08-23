import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import xgboost as xgb

# Load the dataset
df = pd.read_csv(r"C:\Users\Administrator\Desktop\pythonprojects\RTA Dataset.csv")

# Display the first few rows and column names
print("Initial Data Sample:")
print(df.head())
print("\nColumns in the Dataset:")
print(df.columns)

# Handle missing values
print("\nMissing Values Count:")
print(df.isnull().sum())

# Drop rows with missing target values if any
df = df.dropna(subset=['Accident_severity'])

# Drop rows with missing values in feature columns if necessary
df = df.dropna()

# Convert categorical variables to dummy variables
categorical_columns = ['Day_of_week', 'Age_band_of_driver', 'Sex_of_driver', 
                        'Educational_level', 'Vehicle_driver_relation', 
                        'Driving_experience', 'Type_of_vehicle', 'Owner_of_vehicle', 
                        'Service_year_of_vehicle', 'Defect_of_vehicle', 'Area_accident_occured', 
                        'Lanes_or_Medians', 'Road_allignment', 'Types_of_Junction', 
                        'Road_surface_type', 'Road_surface_conditions', 'Light_conditions', 
                        'Weather_conditions', 'Type_of_collision', 'Vehicle_movement', 
                        'Casualty_class', 'Sex_of_casualty', 'Age_band_of_casualty', 
                        'Work_of_casuality', 'Fitness_of_casuality', 'Pedestrian_movement', 
                        'Cause_of_accident']

df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

# Feature engineering: Extract hour from 'Time' and create 'hour_of_day'
df['hour_of_day'] = pd.to_datetime(df['Time'], format='%H:%M:%S').dt.hour

# Drop the 'Time' column as it's no longer needed
df = df.drop(columns=['Time'])

# Check and handle missing values in 'Accident_severity'
df['Accident_severity'] = df['Accident_severity'].map({
    'Slight Injury': 1,
    'Serious Injury': 2,
    'Fatal Injury': 3
})

# Binary Classification Example: Convert to binary classification
# For example, classify as 'serious or fatal' vs 'slight'
df['Accident_severity'] = df['Accident_severity'].apply(lambda x: 1 if x > 1 else 0)

# Define features and target variable
X = df.drop(columns=['Accident_severity'])
y = df['Accident_severity']

# Convert all columns in X to numeric (if not already)
X = X.apply(pd.to_numeric, errors='coerce')

# Drop any rows where conversion to numeric resulted in NaN values
X = X.dropna()
y = y[X.index]

# Ensure there are no NaN values in X or y
print("\nMissing Values After Processing:")
print(X.isnull().sum())
print(y.isnull().sum())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Initialize and train the model
model = XGBClassifier(use_label_encoder=False, objective='binary:logistic')
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print("\nModel Accuracy:")
print(accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Plot feature importance
xgb.plot_importance(model)
plt.title('Feature Importance')
plt.show()
