import pandas as pd
import numpy as np
import matplotlib as mplt

df=pd.read_csv("update_temperature.csv",encoding="latin1")


print(df)

print(df.info())
#to understand max and min loss
print(df.describe())

#Finding missing values

print(df.isnull().sum())

if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    # Extract numerical features from the date (e.g., day of year)
    df['DayOfYear'] = df['Date'].dt.dayofyear
    df = df.drop('Date', axis=1)

# Assume 'Temperature' is a numerical feature and 'DeviceType' is a categorical one.
# You can adjust these column names to match your dataset.
numerical_features = ['Temperature']
categorical_features = [] # Add a categorical column name here if applicable, e.g., ['DeviceType']

# If a categorical feature was created from date, add it here
if 'DayOfYear' in df.columns:
    numerical_features.append('DayOfYear')

# Define target and features
target = 'Loss'
features = numerical_features + categorical_features

# Check if target and features exist in the DataFrame
if target not in df.columns or not all(f in df.columns for f in features):
    print("Error: Required columns are missing from the dataset.")
    print(f"Expected columns: {target} and {features}")
    exit()

X = df[features]
y = df[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")

# Create preprocessing pipelines for numerical and categorical data
# A 'preprocessor' is a powerful tool to apply different transformations to different columns.
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'
)

# ==============================================================================
# 3. Model Training
# ==============================================================================

# Create the full pipeline with preprocessing and the model
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

print("\nStarting model training...")

# Train the model
model_pipeline.fit(X_train, y_train)

print("Model training complete.")

# ==============================================================================
# 4. Model Evaluation
# ==============================================================================

print("\n--- Model Evaluation ---")

# Make predictions on the test set
y_pred = model_pipeline.predict(X_test)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Print the results
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R²): {r2:.2f}")

# You can also inspect the model's coefficients to understand feature importance
# Note: This is only meaningful if using a simple model like LinearRegression
if 'regressor' in model_pipeline.named_steps:
    regressor = model_pipeline.named_steps['regressor']
    if hasattr(regressor, 'coef_'):
        # To get feature names after one-hot encoding, you'd need a more complex
        # setup. Here, we'll just show the coefficients.
        print("\nModel Coefficients:")
        # The number of coefficients should match the number of processed features.
        print(regressor.coef_)

print("\n--- Model evaluation complete. ---")
