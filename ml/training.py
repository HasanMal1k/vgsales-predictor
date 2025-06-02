# Video Game Sales Prediction Model Training with MLflow
# This notebook trains a model to predict video game sales

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import joblib
import mlflow
import mlflow.sklearn
from datetime import datetime
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("video_game_sales_prediction")

# Load the dataset
print("Loading Video Game Sales dataset...")
df = pd.read_csv('../vgsales.csv')

print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Display basic info
print("\nDataset Info:")
print(df.info())
print("\nFirst 5 rows:")
print(df.head())

# Data preprocessing
print("\nData Preprocessing...")

# Remove rows with missing values
df_clean = df.dropna()
print(f"Shape after removing NaN: {df_clean.shape}")

# Define features and target
# We'll predict Global_Sales based on Platform, Genre, Publisher, Year
features = ['Platform', 'Genre', 'Publisher', 'Year']
target = 'Global_Sales'

X = df_clean[features].copy()
y = df_clean[target].copy()

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Analyze categorical features
print(f"\nCategorical features analysis:")
print(f"Unique Platforms: {X['Platform'].nunique()}")
print(f"Unique Genres: {X['Genre'].nunique()}")
print(f"Unique Publishers: {X['Publisher'].nunique()}")

# For publishers, we'll keep only top publishers to avoid overfitting
top_publishers = X['Publisher'].value_counts().head(20).index
X['Publisher'] = X['Publisher'].apply(lambda x: x if x in top_publishers else 'Other')

print(f"Publishers after grouping: {X['Publisher'].nunique()}")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Create preprocessing pipeline
categorical_features = ['Platform', 'Genre', 'Publisher']
numerical_features = ['Year']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
    ]
)

# Start MLflow run
with mlflow.start_run() as run:
    print(f"\nMLflow Run ID: {run.info.run_id}")
    
    # Model parameters
    n_estimators = 100
    max_depth = 10
    min_samples_split = 5
    min_samples_leaf = 2
    random_state = 42
    
    # Log parameters
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("min_samples_split", min_samples_split)
    mlflow.log_param("min_samples_leaf", min_samples_leaf)
    mlflow.log_param("random_state", random_state)
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("dataset_size", len(X))
    mlflow.log_param("top_publishers_count", 20)
    
    # Preprocess the data
    print("Preprocessing data...")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Train the model
    print("Training Random Forest model...")
    start_time = datetime.now()
    
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state
    )
    
    model.fit(X_train_processed, y_train)
    
    training_time = (datetime.now() - start_time).total_seconds()
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Make predictions
    y_pred_train = model.predict(X_train_processed)
    y_pred_test = model.predict(X_test_processed)
    
    # Calculate metrics
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    train_rmse = np.sqrt(train_mse)
    test_rmse = np.sqrt(test_mse)
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    # Log metrics
    mlflow.log_metric("train_mse", train_mse)
    mlflow.log_metric("test_mse", test_mse)
    mlflow.log_metric("train_rmse", train_rmse)
    mlflow.log_metric("test_rmse", test_rmse)
    mlflow.log_metric("train_mae", train_mae)
    mlflow.log_metric("test_mae", test_mae)
    mlflow.log_metric("train_r2", train_r2)
    mlflow.log_metric("test_r2", test_r2)
    mlflow.log_metric("training_time_seconds", training_time)
    
    # Log model
    mlflow.sklearn.log_model(model, "random_forest_model")
    
    # Print results
    print(f"\nModel Performance:")
    print(f"Training RMSE: {train_rmse:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Training MAE: {train_mae:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    print(f"Training R²: {train_r2:.4f}")
    print(f"Test R²: {test_r2:.4f}")
    
    # Feature importance analysis
    feature_names = (numerical_features + 
                    list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)))
    
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop 15 Most Important Features:")
    print(feature_importance.head(15))
    
    # Save model and preprocessor
    os.makedirs('../ml', exist_ok=True)
    
    # Save the full pipeline (preprocessor + model)
    joblib.dump({
        'preprocessor': preprocessor,
        'model': model,
        'feature_names': feature_names,
        'top_publishers': list(top_publishers)
    }, '../ml/model_pipeline.pkl')
    
    # Also save individual components for API
    joblib.dump(model, '../ml/model.pkl')
    joblib.dump(preprocessor, '../ml/preprocessor.pkl')
    joblib.dump(feature_names, '../ml/feature_names.pkl')
    joblib.dump(list(top_publishers), '../ml/top_publishers.pkl')
    
    print(f"\nModel pipeline saved to: ../ml/model_pipeline.pkl")
    
    # Save sample predictions for testing
    sample_data = X_test.head(5).copy()
    sample_predictions = y_pred_test[:5]
    sample_actual = y_test.head(5).values
    
    sample_results = pd.DataFrame({
        'Platform': sample_data['Platform'],
        'Genre': sample_data['Genre'],
        'Publisher': sample_data['Publisher'],
        'Year': sample_data['Year'],
        'Actual_Sales': sample_actual,
        'Predicted_Sales': sample_predictions
    })
    
    print(f"\nSample Predictions:")
    print(sample_results)
    
    sample_results.to_csv('../ml/sample_predictions.csv', index=False)
    
    print(f"\nMLflow tracking URI: {mlflow.get_tracking_uri()}")
    print(f"Experiment ID: {mlflow.active_run().info.experiment_id}")
    print(f"Run ID: {run.info.run_id}")

print("\nTraining completed successfully!")
print("Next steps:")
print("1. Check MLflow UI at http://localhost:5000")
print("2. Run the FastAPI backend")
print("3. Start the frontend application")

# Basic analysis
print(f"\nDataset Summary:")
print(f"- Total games: {len(df_clean)}")
print(f"- Date range: {df_clean['Year'].min():.0f} - {df_clean['Year'].max():.0f}")
print(f"- Average global sales: {df_clean['Global_Sales'].mean():.2f} million")
print(f"- Top selling game: {df_clean.loc[df_clean['Global_Sales'].idxmax(), 'Name']} ({df_clean['Global_Sales'].max():.2f}M)")
print(f"- Most common platform: {df_clean['Platform'].mode()[0]}")
print(f"- Most common genre: {df_clean['Genre'].mode()[0]}")