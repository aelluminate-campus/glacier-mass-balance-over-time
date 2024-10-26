# **IMPORT LIBRARIES**
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import warnings



# **PREFERENCES**
# Suppress warnings
warnings.filterwarnings('ignore')

# Set the default Seaborn style
sns.set_theme(style='whitegrid', font='serif')

# DATASET URL - https://datahub.io/core/glacier-mass-balance



# **EXPLORATORY DATA ANALYSIS (EDA)**

# Step 1: Load Dataset
df = pd.read_csv("glaciers.csv")

# Step 2: Initial Dataset Inspection
print("=== Dataset Information ===")
print(df.info())  # Display data types and non-null counts for each column
print("\n=== First 5 Rows of Data ===")
print(df.head())  # Display first 5 rows



# **DATA CLEANING AND PREPROCESSING**

# Step 3: Check for Missing Values
print("\n=== Missing Values in Each Column ===")
missing_values = df.isnull().sum()
print(missing_values)

# Remove rows with missing values
df.dropna(inplace=True)

# Check data types
print("\n=== Data Types ===")
print(df.dtypes)

# Step 4: Statistical Summary of Dataset
print("\n=== Statistical Summary ===")
print(df.describe())



# **DATA VISUALIZATIONS**

# Step 5: Glacier Mass Balance Over Time
plt.figure(figsize=(12, 6))
sns.lineplot(x='Year', y='Mean cumulative mass balance', data=df, marker='o', linewidth=2)
plt.title('Glacier Mass Balance Over Time')
plt.xlabel('Year')
plt.ylabel('Mean Cumulative Mass Balance (in mm)')
plt.grid(True, linestyle='--', alpha=0.7)  # Enhanced gridlines
plt.xticks(rotation=45)
plt.show()

# Step 6: Yearly Change Calculation
df['Annual Change in Mass Balance'] = df['Mean cumulative mass balance'].diff()

# Step 7: Year-over-Year Change Visualization
plt.figure(figsize=(12, 6))
sns.barplot(x='Year', y='Annual Change in Mass Balance', data=df, palette="coolwarm")
plt.title('Year-over-Year Change in Glacier Mass Balance')
plt.xlabel('Year')
plt.ylabel('Annual Change in Mass Balance (in mm)')
plt.axhline(0, color='red', linestyle='--')  # Horizontal line for reference
plt.xticks(rotation=45, fontsize=8)
plt.grid()
plt.show()

# Step 8: Correlation Heatmap
plt.figure(figsize=(10, 8))  # Adjusted figure size for better visibility
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f", cbar_kws={'shrink': .8}, linewidths=.5)
plt.xticks(rotation=0, ha='center')  # Keep x-axis labels horizontal and centered
plt.yticks(rotation=0)  # Keep y-axis labels horizontal
plt.title('Correlation Matrix of Features')
plt.tight_layout()  # Ensure everything fits without cutting off
plt.show()



# **MODEL EVALUATION**

# Step 9: Data Preparation for Modeling
X = df[['Year']]  # Features
y = df['Mean cumulative mass balance']  # Target variable

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 10: Initialize Models for Regression
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(),
    "Support Vector Machine": SVR()
}

# Step 11: Train Models and Evaluate Performance
results = {}
print("\n=== Model Performance Results ===")
for model_name, model in models.items():
    model.fit(X_train, y_train)  # Train the model
    y_pred = model.predict(X_test)  # Make predictions
    mse = mean_squared_error(y_test, y_pred)  # Calculate Mean Squared Error
    r2 = r2_score(y_test, y_pred)  # Calculate R-squared
    results[model_name] = {"MSE": mse, "R^2": r2}  # Store metrics
    
    # Convert R² to percentage
    r2_percentage = r2 * 100
    print(f"{model_name} - MSE: {mse:.2f}, R^2: {r2_percentage:.2f}%")

# Step 12: Model Evaluation Summary
print("\n=== Model Evaluation Results ===")
for model_name, metrics in results.items():
    r2_percentage = metrics["R^2"] * 100  # Convert R² to percentage
    if metrics["R^2"] >= 0.85:
        print(f"{model_name} meets the performance requirement with R^2: {r2_percentage:.2f}%")
    else:
        print(f"{model_name} does not meet the performance requirement with R^2: {r2_percentage:.2f}%")
