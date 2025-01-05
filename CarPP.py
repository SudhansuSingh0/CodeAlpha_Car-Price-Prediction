# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import seaborn as sns

# Load the dataset
file_path = 'cardata.csv'  # Replace with the correct path if needed
car_data = pd.read_csv(file_path)

# Data preprocessing
# Drop irrelevant column
data = car_data.drop(['Car_Name'], axis=1)

# Create a new column for car age
data['Car_Age'] = 2024 - data['Year']
data.drop(['Year'], axis=1, inplace=True)

# One-hot encoding for categorical variables
categorical_features = ['Fuel_Type', 'Selling_type', 'Transmission']
encoder = OneHotEncoder(sparse_output=False, drop='first')  # Updated parameter name
encoded_data = encoder.fit_transform(data[categorical_features])
encoded_columns = encoder.get_feature_names_out(categorical_features)
encoded_df = pd.DataFrame(encoded_data, columns=encoded_columns, index=data.index)


# Combine numerical and encoded data
numerical_data = data.drop(categorical_features, axis=1)
processed_data = pd.concat([numerical_data, encoded_df], axis=1)

# Splitting the data
X = processed_data.drop(['Selling_Price'], axis=1)
y = processed_data['Selling_Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling the numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = RandomForestRegressor(random_state=42)
model.fit(X_train_scaled, y_train)

# Predictions
y_pred = model.predict(X_test_scaled)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display the results
print("Mean Absolute Error (MAE):", mae)
print("R² Score:", r2)
# 1. Distribution of Selling Price

plt.figure(figsize=(8, 5))
sns.histplot(car_data['Selling_Price'], kde=True, color='blue', bins=30)
plt.title("Distribution of Selling Price")
plt.xlabel("Selling Price (in lakhs)")
plt.ylabel("Frequency")
plt.show()

# 2. Correlation Heatmap
numeric_data = car_data.select_dtypes(include=['float64', 'int64'])

plt.figure(figsize=(10, 8))
sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Heatmap")
plt.show()


# 3. Scatter Plot: Selling Price vs. Present Price
plt.figure(figsize=(8, 5))
sns.scatterplot(x='Present_Price', y='Selling_Price', data=car_data, hue='Fuel_Type')
plt.title("Selling Price vs. Present Price")
plt.xlabel("Present Price (in lakhs)")
plt.ylabel("Selling Price (in lakhs)")
plt.legend(title="Fuel Type")
plt.show()

# 4. Bar Chart: Count of Cars by Fuel Type
plt.figure(figsize=(6, 4))
sns.countplot(x='Fuel_Type', data=car_data, hue='Fuel_Type', dodge=False, palette='Set2', legend=False)
plt.title("Count of Cars by Fuel Type")
plt.xlabel("Fuel Type")
plt.ylabel("Number of Cars")
plt.show()


# 5. Box Plot: Selling Price by Transmission Type
plt.figure(figsize=(8, 5))
sns.boxplot(x='Transmission', y='Selling_Price', data=car_data, hue='Transmission', dodge=False, palette='Set3', legend=False)
plt.title("Selling Price by Transmission Type")
plt.xlabel("Transmission Type")
plt.ylabel("Selling Price (in lakhs)")
plt.show()
