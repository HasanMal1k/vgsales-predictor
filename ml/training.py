# Simplified Video Game Sales Prediction Model Training (No MLflow)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import joblib
from datetime import datetime
import os

def main():
    print("Loading Video Game Sales dataset...")
    df = pd.read_csv('../vgsales.csv')

    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    # Data preprocessing
    print("\nData Preprocessing...")
    df_clean = df.dropna()
    print(f"Shape after removing NaN: {df_clean.shape}")

    # Define features and target
    features = ['Platform', 'Genre', 'Publisher', 'Year']
    target = 'Global_Sales'

    X = df_clean[features].copy()
    y = df_clean[target].copy()

    # For publishers, keep only top publishers to avoid overfitting
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

    # Model parameters
    n_estimators = 100
    max_depth = 10
    min_samples_split = 5
    min_samples_leaf = 2
    random_state = 42
    
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
    print("\nSaving model files...")
    
    # Save the full pipeline (preprocessor + model)
    joblib.dump({
        'preprocessor': preprocessor,
        'model': model,
        'feature_names': feature_names,
        'top_publishers': list(top_publishers)
    }, 'model_pipeline.pkl')
    
    # Also save individual components for API
    joblib.dump(model, 'model.pkl')
    joblib.dump(preprocessor, 'preprocessor.pkl')
    joblib.dump(feature_names, 'feature_names.pkl')
    joblib.dump(list(top_publishers), 'top_publishers.pkl')
    
    print(f"Model pipeline saved to: model_pipeline.pkl")
    
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
    
    sample_results.to_csv('sample_predictions.csv', index=False)
    
    print("\nTraining completed successfully!")
    print("Model files saved. You can now start the API server.")

if __name__ == "__main__":
    main()  