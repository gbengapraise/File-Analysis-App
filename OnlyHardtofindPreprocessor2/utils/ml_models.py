import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.impute import SimpleImputer
import plotly.express as px
import plotly.graph_objects as go

def prepare_data_for_ml(df, target_column, feature_columns=None, categorical_columns=None, test_size=0.2, random_state=42):
    """
    Prepare data for machine learning by handling missing values, encoding categorical features,
    and splitting into training and testing sets.
    
    Parameters:
    - df: pandas DataFrame, the dataset
    - target_column: str, the column to predict
    - feature_columns: list, columns to use as features (None uses all except target)
    - categorical_columns: list, categorical columns to encode (None auto-detects)
    - test_size: float, proportion of data to use for testing
    - random_state: int, random seed for reproducibility
    
    Returns:
    - X_train, X_test, y_train, y_test, feature_names, preprocessor
    """
    if df is None or target_column not in df.columns:
        st.error(f"Target column {target_column} not found in the dataframe")
        return None, None, None, None, None, None
    
    # Make a copy to avoid modifying the original
    df_ml = df.copy()
    
    # Select features
    if feature_columns is None:
        feature_columns = [col for col in df_ml.columns if col != target_column]
    else:
        # Ensure all feature columns exist in the dataframe
        feature_columns = [col for col in feature_columns if col in df_ml.columns]
    
    if not feature_columns:
        st.error("No feature columns available for modeling")
        return None, None, None, None, None, None
    
    # Create feature matrix and target vector
    X = df_ml[feature_columns]
    y = df_ml[target_column]
    
    # Auto-detect categorical columns if not provided
    if categorical_columns is None:
        categorical_columns = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Handle categorical features
    preprocessor = {}
    X_processed = X.copy()
    
    if categorical_columns:
        # Encode categorical features
        for col in categorical_columns:
            if col in X_processed.columns:
                le = LabelEncoder()
                # Fill missing values with a placeholder
                X_processed[col] = X_processed[col].fillna('missing')
                X_processed[col] = le.fit_transform(X_processed[col])
                preprocessor[col] = le
    
    # Handle missing values in numerical columns
    numerical_columns = X_processed.select_dtypes(include=np.number).columns.tolist()
    if numerical_columns:
        imputer = SimpleImputer(strategy='mean')
        X_processed[numerical_columns] = imputer.fit_transform(X_processed[numerical_columns])
        preprocessor['imputer'] = imputer
    
    # Handle missing values in target variable for regression tasks
    if y.dtype in [np.float64, np.int64]:
        y = y.fillna(y.mean())
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=test_size, random_state=random_state
    )
    
    # Scale numerical features
    scaler = StandardScaler()
    
    if numerical_columns:
        X_train[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
        X_test[numerical_columns] = scaler.transform(X_test[numerical_columns])
        preprocessor['scaler'] = scaler
    
    return X_train, X_test, y_train, y_test, feature_columns, preprocessor

def train_linear_regression(X_train, y_train):
    """
    Train a linear regression model
    
    Parameters:
    - X_train: DataFrame, training features
    - y_train: Series, training target
    
    Returns:
    - Trained model
    """
    if X_train is None or y_train is None:
        return None
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def train_random_forest_regression(X_train, y_train, n_estimators=100, max_depth=None):
    """
    Train a random forest regression model
    
    Parameters:
    - X_train: DataFrame, training features
    - y_train: Series, training target
    - n_estimators: int, number of trees
    - max_depth: int, maximum depth of trees
    
    Returns:
    - Trained model
    """
    if X_train is None or y_train is None:
        return None
    
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_logistic_regression(X_train, y_train, max_iter=1000):
    """
    Train a logistic regression model
    
    Parameters:
    - X_train: DataFrame, training features
    - y_train: Series, training target
    - max_iter: int, maximum number of iterations
    
    Returns:
    - Trained model
    """
    if X_train is None or y_train is None:
        return None
    
    model = LogisticRegression(max_iter=max_iter, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_random_forest_classifier(X_train, y_train, n_estimators=100, max_depth=None):
    """
    Train a random forest classifier model
    
    Parameters:
    - X_train: DataFrame, training features
    - y_train: Series, training target
    - n_estimators: int, number of trees
    - max_depth: int, maximum depth of trees
    
    Returns:
    - Trained model
    """
    if X_train is None or y_train is None:
        return None
    
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_regression_model(model, X_test, y_test, feature_names):
    """
    Evaluate a regression model and return metrics and visualizations
    
    Parameters:
    - model: trained regression model
    - X_test: DataFrame, testing features
    - y_test: Series, testing target
    - feature_names: list, names of features
    
    Returns:
    - metrics: dict, performance metrics
    - predictions: Series, model predictions
    - feature_importance: DataFrame (if available)
    """
    if model is None or X_test is None or y_test is None:
        return None, None, None
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    metrics = {
        'Mean Squared Error': mse,
        'Root Mean Squared Error': rmse,
        'R² Score': r2
    }
    
    # Get feature importance if available
    feature_importance = None
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
    elif hasattr(model, 'coef_'):
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': model.coef_
        }).sort_values('Coefficient', key=lambda x: abs(x), ascending=False)
    
    return metrics, pd.Series(y_pred, index=y_test.index), feature_importance

def evaluate_classification_model(model, X_test, y_test, feature_names):
    """
    Evaluate a classification model and return metrics and visualizations
    
    Parameters:
    - model: trained classification model
    - X_test: DataFrame, testing features
    - y_test: Series, testing target
    - feature_names: list, names of features
    
    Returns:
    - metrics: dict, performance metrics
    - predictions: Series, model predictions
    - feature_importance: DataFrame (if available)
    """
    if model is None or X_test is None or y_test is None:
        return None, None, None
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    metrics = {
        'Accuracy': accuracy,
        'Classification Report': report
    }
    
    # Get feature importance if available
    feature_importance = None
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
    elif hasattr(model, 'coef_'):
        if len(model.classes_) == 2:  # Binary classification
            feature_importance = pd.DataFrame({
                'Feature': feature_names,
                'Coefficient': model.coef_[0]
            }).sort_values('Coefficient', key=lambda x: abs(x), ascending=False)
        else:  # Multi-class
            feature_importance = pd.DataFrame({
                'Feature': feature_names * len(model.classes_),
                'Class': np.repeat(model.classes_, len(feature_names)),
                'Coefficient': model.coef_.flatten()
            })
    
    return metrics, pd.Series(y_pred, index=y_test.index), feature_importance

def plot_regression_results(y_test, y_pred, title="Actual vs Predicted Values"):
    """
    Create a scatter plot of actual vs predicted values for regression models
    
    Parameters:
    - y_test: Series, actual target values
    - y_pred: Series, predicted target values
    - title: str, plot title
    
    Returns:
    - plotly figure
    """
    if y_test is None or y_pred is None:
        return None
    
    # Combine actual and predicted into a DataFrame
    results_df = pd.DataFrame({
        'Actual': y_test,
        'Predicted': y_pred
    })
    
    # Create scatter plot
    fig = px.scatter(
        results_df,
        x='Actual',
        y='Predicted',
        title=title
    )
    
    # Add perfect prediction line
    min_val = min(results_df['Actual'].min(), results_df['Predicted'].min())
    max_val = max(results_df['Actual'].max(), results_df['Predicted'].max())
    
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', dash='dash')
        )
    )
    
    fig.update_layout(
        xaxis_title="Actual Values",
        yaxis_title="Predicted Values"
    )
    
    return fig

def plot_feature_importance(feature_importance, title="Feature Importance"):
    """
    Create a bar chart of feature importance
    
    Parameters:
    - feature_importance: DataFrame with 'Feature' and 'Importance'/'Coefficient' columns
    - title: str, plot title
    
    Returns:
    - plotly figure
    """
    if feature_importance is None or feature_importance.empty:
        return None
    
    # Determine importance column name
    if 'Importance' in feature_importance.columns:
        importance_col = 'Importance'
    elif 'Coefficient' in feature_importance.columns:
        importance_col = 'Coefficient'
        # Take absolute value for coefficients
        feature_importance[importance_col] = feature_importance[importance_col].abs()
    else:
        return None
    
    # Sort by importance
    sorted_df = feature_importance.sort_values(importance_col, ascending=True)
    
    # Limit to top 15 features if there are too many
    if len(sorted_df) > 15:
        sorted_df = sorted_df.tail(15)
    
    fig = px.bar(
        sorted_df,
        y='Feature',
        x=importance_col,
        orientation='h',
        title=title
    )
    
    fig.update_layout(
        xaxis_title=importance_col,
        yaxis_title="Feature"
    )
    
    return fig

def get_model_prediction(model, X_new, preprocessor, target_type="regression"):
    """
    Use a trained model to make predictions on new data
    
    Parameters:
    - model: trained model
    - X_new: DataFrame, new features
    - preprocessor: dict, preprocessing objects
    - target_type: str, 'regression' or 'classification'
    
    Returns:
    - predictions
    """
    if model is None or X_new is None or preprocessor is None:
        return None
    
    # Make a copy to avoid modifying the original
    X_processed = X_new.copy()
    
    # Preprocess categorical features
    for col, le in preprocessor.items():
        if col in X_processed.columns and isinstance(le, LabelEncoder):
            # Fill missing values with placeholder
            X_processed[col] = X_processed[col].fillna('missing')
            
            # Transform known categories
            # Handle unknown categories by assigning a default value
            X_processed[col] = X_processed[col].apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )
    
    # Impute missing values in numerical columns
    if 'imputer' in preprocessor:
        numerical_columns = X_processed.select_dtypes(include=np.number).columns.tolist()
        if numerical_columns:
            X_processed[numerical_columns] = preprocessor['imputer'].transform(X_processed[numerical_columns])
    
    # Scale numerical features
    if 'scaler' in preprocessor:
        numerical_columns = X_processed.select_dtypes(include=np.number).columns.tolist()
        if numerical_columns:
            X_processed[numerical_columns] = preprocessor['scaler'].transform(X_processed[numerical_columns])
    
    # Make predictions
    predictions = model.predict(X_processed)
    
    return predictions
