import streamlit as st
import pandas as pd
import numpy as np
from utils.ml_models import (
    prepare_data_for_ml, train_linear_regression, train_random_forest_regression,
    train_logistic_regression, train_random_forest_classifier,
    evaluate_regression_model, evaluate_classification_model,
    plot_regression_results, plot_feature_importance, get_model_prediction
)

# Set page configuration
st.set_page_config(
    page_title="Predictive Analysis",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables if they don't exist
if 'data' not in st.session_state:
    st.session_state.data = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'model_type' not in st.session_state:
    st.session_state.model_type = None
if 'preprocessor' not in st.session_state:
    st.session_state.preprocessor = None
if 'feature_names' not in st.session_state:
    st.session_state.feature_names = None
if 'target_column' not in st.session_state:
    st.session_state.target_column = None
if 'model_metrics' not in st.session_state:
    st.session_state.model_metrics = None
if 'test_predictions' not in st.session_state:
    st.session_state.test_predictions = None
if 'feature_importance' not in st.session_state:
    st.session_state.feature_importance = None

# Function to reset model state
def reset_model_state():
    st.session_state.model = None
    st.session_state.model_type = None
    st.session_state.preprocessor = None
    st.session_state.feature_names = None
    st.session_state.target_column = None
    st.session_state.model_metrics = None
    st.session_state.test_predictions = None
    st.session_state.feature_importance = None

# Function to determine if a column is categorical
def is_categorical(df, column):
    # Check if column is object or category type
    if pd.api.types.is_object_dtype(df[column]) or pd.api.types.is_categorical_dtype(df[column]):
        return True
    
    # Check if numeric but with few unique values (likely categorical)
    if pd.api.types.is_numeric_dtype(df[column]):
        unique_values = df[column].nunique()
        if unique_values <= 10:  # Threshold for considering a numeric column as categorical
            return True
    
    return False

# Page title and description
st.title("🔮 Predictive Analysis")
st.write("Build and evaluate machine learning models to make predictions")

# Check if data is loaded
if st.session_state.data is not None:
    df = st.session_state.data
    
    # Create tabs for different stages of predictive analysis
    pred_tabs = st.tabs([
        "Model Setup", "Training & Evaluation", "Prediction", "Model Insights"
    ])
    
    # Tab 1: Model Setup
    with pred_tabs[0]:
        st.header("Model Setup")
        
        # Display warning if dataset is too small
        if len(df) < 50:
            st.warning("⚠️ Your dataset is quite small for machine learning. Results may not be reliable.")
        
        # Determine task type (regression or classification)
        col1, col2 = st.columns(2)
        
        with col1:
            # Select target variable
            target_column = st.selectbox(
                "Select target variable (what you want to predict):",
                options=df.columns.tolist()
            )
            
            if target_column:
                # Determine if classification or regression based on target column
                is_classification = is_categorical(df, target_column)
                
                task_type = st.radio(
                    "Task type:",
                    options=["Classification", "Regression"],
                    index=0 if is_classification else 1,
                    help="Classification predicts categories, Regression predicts numeric values"
                )
                
                # Show unique values for classification tasks
                if task_type == "Classification":
                    unique_values = df[target_column].unique()
                    st.write(f"Unique values in target: {len(unique_values)}")
                    if len(unique_values) <= 10:
                        st.write(f"Classes: {', '.join(str(v) for v in unique_values)}")
                    
                    # Check if binary or multi-class
                    if len(unique_values) == 2:
                        st.info("Binary classification task detected")
                    elif len(unique_values) > 2:
                        st.info("Multi-class classification task detected")
                    else:
                        st.error("Need at least two classes for classification")
                else:
                    # For regression, show basic stats
                    if pd.api.types.is_numeric_dtype(df[target_column]):
                        stats = df[target_column].describe()
                        st.write(f"Target range: {stats['min']:.2f} to {stats['max']:.2f}")
                        st.write(f"Mean: {stats['mean']:.2f}, Std: {stats['std']:.2f}")
                    else:
                        st.error("Target variable should be numeric for regression")
        
        with col2:
            # Feature selection
            st.subheader("Feature Selection")
            
            # Automatically exclude the target column from feature options
            feature_options = [col for col in df.columns if col != target_column]
            
            selected_features = st.multiselect(
                "Select features to include in the model:",
                options=feature_options,
                default=feature_options[:min(5, len(feature_options))]
            )
            
            if selected_features:
                # Data splitting parameters
                test_size = st.slider(
                    "Test set size (%):",
                    min_value=10,
                    max_value=50,
                    value=20,
                    help="Percentage of data to use for testing the model"
                )
                
                # Random seed for reproducibility
                random_state = st.number_input(
                    "Random seed:",
                    value=42,
                    min_value=1,
                    help="Seed for random number generator to ensure reproducible results"
                )
        
        # Model selection based on task type
        st.subheader("Model Selection")
        
        if target_column and selected_features:
            if task_type == "Regression":
                model_type = st.selectbox(
                    "Select regression model:",
                    options=["Linear Regression", "Random Forest Regression"],
                    help="Choose the algorithm for your regression task"
                )
                
                # Model-specific parameters
                if model_type == "Random Forest Regression":
                    col1, col2 = st.columns(2)
                    with col1:
                        n_estimators = st.slider(
                            "Number of trees:",
                            min_value=10,
                            max_value=500,
                            value=100,
                            step=10
                        )
                    with col2:
                        max_depth = st.slider(
                            "Maximum tree depth:",
                            min_value=2,
                            max_value=30,
                            value=10
                        )
                    
                    model_params = {
                        "n_estimators": n_estimators,
                        "max_depth": max_depth
                    }
                else:
                    model_params = {}
            
            else:  # Classification
                model_type = st.selectbox(
                    "Select classification model:",
                    options=["Logistic Regression", "Random Forest Classifier"],
                    help="Choose the algorithm for your classification task"
                )
                
                # Model-specific parameters
                if model_type == "Logistic Regression":
                    col1, col2 = st.columns(2)
                    with col1:
                        max_iter = st.slider(
                            "Maximum iterations:",
                            min_value=100,
                            max_value=2000,
                            value=1000,
                            step=100
                        )
                    
                    model_params = {
                        "max_iter": max_iter
                    }
                    
                elif model_type == "Random Forest Classifier":
                    col1, col2 = st.columns(2)
                    with col1:
                        n_estimators = st.slider(
                            "Number of trees:",
                            min_value=10,
                            max_value=500,
                            value=100,
                            step=10
                        )
                    with col2:
                        max_depth = st.slider(
                            "Maximum tree depth:",
                            min_value=2,
                            max_value=30,
                            value=10
                        )
                    
                    model_params = {
                        "n_estimators": n_estimators,
                        "max_depth": max_depth
                    }
            
            # Prepare data button
            if st.button("Prepare Data for Modeling", key="prepare_data"):
                with st.spinner("Preparing data..."):
                    # Identify categorical features
                    categorical_columns = [col for col in selected_features if is_categorical(df, col)]
                    
                    # Prepare data for machine learning
                    test_size_frac = test_size / 100.0  # Convert percentage to fraction
                    
                    X_train, X_test, y_train, y_test, feature_names, preprocessor = prepare_data_for_ml(
                        df, 
                        target_column=target_column,
                        feature_columns=selected_features,
                        categorical_columns=categorical_columns,
                        test_size=test_size_frac,
                        random_state=random_state
                    )
                    
                    if X_train is not None:
                        # Store in session state for use in other tabs
                        st.session_state.X_train = X_train
                        st.session_state.X_test = X_test
                        st.session_state.y_train = y_train
                        st.session_state.y_test = y_test
                        st.session_state.feature_names = feature_names
                        st.session_state.preprocessor = preprocessor
                        st.session_state.target_column = target_column
                        st.session_state.model_type = model_type
                        st.session_state.model_params = model_params
                        st.session_state.task_type = task_type
                        
                        st.success(f"✅ Data prepared successfully! Training set: {len(X_train)} samples, Test set: {len(X_test)} samples")
                        
                        # Display summary of data preparation
                        st.subheader("Data Preparation Summary")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write(f"**Task:** {task_type}")
                            st.write(f"**Target:** {target_column}")
                            st.write(f"**Features:** {len(selected_features)}")
                            st.write(f"**Training samples:** {len(X_train)}")
                            st.write(f"**Test samples:** {len(X_test)}")
                        
                        with col2:
                            st.write(f"**Model:** {model_type}")
                            st.write(f"**Categorical features:** {len(categorical_columns)}")
                            st.write(f"**Numeric features:** {len(selected_features) - len(categorical_columns)}")
                            
                            # Display model parameters
                            if model_params:
                                st.write("**Model parameters:**")
                                for param, value in model_params.items():
                                    st.write(f"- {param}: {value}")
                        
                        # Display sample of preprocessed data
                        st.subheader("Sample of Preprocessed Training Data")
                        st.dataframe(X_train.head(5), use_container_width=True)
                    else:
                        st.error("Failed to prepare data. Please check your selections.")
        else:
            st.warning("Please select both target variable and features to continue")
    
    # Tab 2: Training & Evaluation
    with pred_tabs[1]:
        st.header("Model Training & Evaluation")
        
        # Check if data has been prepared
        if 'X_train' in st.session_state and st.session_state.X_train is not None:
            # Get data from session state
            X_train = st.session_state.X_train
            X_test = st.session_state.X_test
            y_train = st.session_state.y_train
            y_test = st.session_state.y_test
            feature_names = st.session_state.feature_names
            model_type = st.session_state.model_type
            model_params = st.session_state.model_params
            task_type = st.session_state.task_type
            
            # Train model button
            if st.button("Train Model", key="train_model"):
                with st.spinner(f"Training {model_type}..."):
                    try:
                        # Train appropriate model based on type
                        if model_type == "Linear Regression":
                            model = train_linear_regression(X_train, y_train)
                        elif model_type == "Random Forest Regression":
                            model = train_random_forest_regression(
                                X_train, y_train, 
                                n_estimators=model_params.get("n_estimators", 100),
                                max_depth=model_params.get("max_depth", None)
                            )
                        elif model_type == "Logistic Regression":
                            model = train_logistic_regression(
                                X_train, y_train,
                                max_iter=model_params.get("max_iter", 1000)
                            )
                        elif model_type == "Random Forest Classifier":
                            model = train_random_forest_classifier(
                                X_train, y_train,
                                n_estimators=model_params.get("n_estimators", 100),
                                max_depth=model_params.get("max_depth", None)
                            )
                        
                        # Store model in session state
                        st.session_state.model = model
                        
                        st.success(f"✅ {model_type} trained successfully!")
                        
                        # Evaluate the model
                        with st.spinner("Evaluating model performance..."):
                            if task_type == "Regression":
                                metrics, predictions, feature_importance = evaluate_regression_model(
                                    model, X_test, y_test, feature_names
                                )
                            else:  # Classification
                                metrics, predictions, feature_importance = evaluate_classification_model(
                                    model, X_test, y_test, feature_names
                                )
                            
                            # Store evaluation results
                            st.session_state.model_metrics = metrics
                            st.session_state.test_predictions = predictions
                            st.session_state.feature_importance = feature_importance
                            
                            # Display evaluation metrics
                            st.subheader("Model Performance")
                            
                            if task_type == "Regression":
                                col1, col2, col3 = st.columns(3)
                                
                                col1.metric("Mean Squared Error", f"{metrics['Mean Squared Error']:.4f}")
                                col2.metric("Root MSE", f"{metrics['Root Mean Squared Error']:.4f}")
                                col3.metric("R² Score", f"{metrics['R² Score']:.4f}")
                                
                                # Scatter plot of actual vs predicted
                                st.subheader("Actual vs Predicted Values")
                                fig = plot_regression_results(y_test, predictions)
                                st.plotly_chart(fig, use_container_width=True)
                            
                            else:  # Classification
                                col1, col2 = st.columns(2)
                                
                                col1.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
                                
                                # Classification report
                                st.subheader("Classification Report")
                                
                                # Convert classification report dict to dataframe for display
                                report = metrics['Classification Report']
                                
                                # Skip the 'support' metric and filter out the 'accuracy' row
                                class_metrics = {}
                                for class_name, values in report.items():
                                    if isinstance(values, dict) and class_name != 'accuracy':
                                        class_metrics[class_name] = {
                                            'precision': values['precision'],
                                            'recall': values['recall'],
                                            'f1-score': values['f1-score']
                                        }
                                
                                report_df = pd.DataFrame(class_metrics).transpose()
                                report_df.index.name = 'Class'
                                report_df = report_df.reset_index()
                                
                                st.dataframe(report_df, use_container_width=True)
                                
                                # Add accuracy as a note
                                st.info(f"Overall accuracy: {report['accuracy']:.4f}")
                            
                            # Feature Importance
                            if feature_importance is not None:
                                st.subheader("Feature Importance")
                                fig = plot_feature_importance(feature_importance)
                                st.plotly_chart(fig, use_container_width=True)
                    
                    except Exception as e:
                        st.error(f"Error during model training or evaluation: {str(e)}")
            
            # If model has been trained and evaluated, show results
            elif 'model' in st.session_state and st.session_state.model is not None:
                model = st.session_state.model
                metrics = st.session_state.model_metrics
                predictions = st.session_state.test_predictions
                feature_importance = st.session_state.feature_importance
                
                # Display evaluation metrics
                st.subheader("Model Performance")
                
                if task_type == "Regression":
                    col1, col2, col3 = st.columns(3)
                    
                    col1.metric("Mean Squared Error", f"{metrics['Mean Squared Error']:.4f}")
                    col2.metric("Root MSE", f"{metrics['Root Mean Squared Error']:.4f}")
                    col3.metric("R² Score", f"{metrics['R² Score']:.4f}")
                    
                    # Scatter plot of actual vs predicted
                    st.subheader("Actual vs Predicted Values")
                    fig = plot_regression_results(y_test, predictions)
                    st.plotly_chart(fig, use_container_width=True)
                
                else:  # Classification
                    col1, col2 = st.columns(2)
                    
                    col1.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
                    
                    # Classification report
                    st.subheader("Classification Report")
                    
                    # Convert classification report dict to dataframe for display
                    report = metrics['Classification Report']
                    
                    # Skip the 'support' metric and filter out the 'accuracy' row
                    class_metrics = {}
                    for class_name, values in report.items():
                        if isinstance(values, dict) and class_name != 'accuracy':
                            class_metrics[class_name] = {
                                'precision': values['precision'],
                                'recall': values['recall'],
                                'f1-score': values['f1-score']
                            }
                    
                    report_df = pd.DataFrame(class_metrics).transpose()
                    report_df.index.name = 'Class'
                    report_df = report_df.reset_index()
                    
                    st.dataframe(report_df, use_container_width=True)
                    
                    # Add accuracy as a note
                    st.info(f"Overall accuracy: {report['accuracy']:.4f}")
                
                # Feature Importance
                if feature_importance is not None:
                    st.subheader("Feature Importance")
                    fig = plot_feature_importance(feature_importance)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Option to retrain with different parameters
                if st.button("Retrain with Different Parameters", key="retrain"):
                    reset_model_state()
                    st.info("Please go back to Model Setup tab to configure new parameters")
                    st.rerun()
        else:
            st.warning("⚠️ Please prepare your data first in the Model Setup tab")
            if st.button("Go to Model Setup"):
                # Switch to the first tab
                pass
    
    # Tab 3: Prediction
    with pred_tabs[2]:
        st.header("Make Predictions")
        
        # Check if model has been trained
        if 'model' in st.session_state and st.session_state.model is not None:
            # Get model from session state
            model = st.session_state.model
            feature_names = st.session_state.feature_names
            preprocessor = st.session_state.preprocessor
            target_column = st.session_state.target_column
            task_type = st.session_state.task_type
            
            st.subheader("Prediction Options")
            
            # Create tabs for prediction methods
            pred_method_tabs = st.tabs(["Use Test Data", "Manual Input", "Upload New Data"])
            
            # Tab 1: Use Test Data for Predictions
            with pred_method_tabs[0]:
                if 'X_test' in st.session_state and st.session_state.X_test is not None:
                    X_test = st.session_state.X_test
                    y_test = st.session_state.y_test
                    
                    st.write(f"Test data contains {len(X_test)} samples")
                    
                    # Options for showing predictions
                    num_samples = st.slider(
                        "Number of samples to view:",
                        min_value=1,
                        max_value=min(100, len(X_test)),
                        value=min(10, len(X_test))
                    )
                    
                    # Button to show predictions
                    if st.button("Show Predictions on Test Data", key="show_test_pred"):
                        with st.spinner("Generating predictions..."):
                            # Get predictions from session state if available, otherwise generate new ones
                            if 'test_predictions' in st.session_state and st.session_state.test_predictions is not None:
                                predictions = st.session_state.test_predictions
                            else:
                                if task_type == "Regression":
                                    task = "regression"
                                else:
                                    task = "classification"
                                
                                predictions = get_model_prediction(model, X_test, preprocessor, target_type=task)
                            
                            # Create a DataFrame with actual and predicted values
                            results_df = pd.DataFrame({
                                'Actual': y_test.values,
                                'Predicted': predictions
                            })
                            
                            # Display results
                            st.subheader("Prediction Results")
                            st.dataframe(results_df.head(num_samples), use_container_width=True)
                            
                            # For classification, show a confusion matrix
                            if task_type == "Classification":
                                from sklearn.metrics import confusion_matrix
                                import plotly.figure_factory as ff
                                
                                # Compute confusion matrix
                                cm = confusion_matrix(y_test, predictions)
                                
                                # Get unique classes
                                classes = np.unique(np.concatenate([y_test, predictions]))
                                
                                # Create heatmap
                                fig = ff.create_annotated_heatmap(
                                    z=cm,
                                    x=classes,
                                    y=classes,
                                    annotation_text=cm,
                                    colorscale='Viridis'
                                )
                                
                                fig.update_layout(
                                    title='Confusion Matrix',
                                    xaxis=dict(title='Predicted'),
                                    yaxis=dict(title='Actual')
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                            
                            # For regression, show residual plot
                            else:
                                import plotly.express as px
                                
                                # Calculate residuals
                                results_df['Residual'] = results_df['Actual'] - results_df['Predicted']
                                
                                # Create residual plot
                                fig = px.scatter(
                                    results_df,
                                    x='Predicted',
                                    y='Residual',
                                    title='Residual Plot'
                                )
                                
                                fig.add_hline(y=0, line_dash="dash", line_color="red")
                                
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Additional metrics
                                residuals = results_df['Residual']
                                mse = (residuals ** 2).mean()
                                mae = abs(residuals).mean()
                                
                                col1, col2 = st.columns(2)
                                col1.metric("Mean Absolute Error", f"{mae:.4f}")
                                col2.metric("Mean Squared Error", f"{mse:.4f}")
                
                else:
                    st.warning("Test data not available. Please prepare your data in the Model Setup tab.")
            
            # Tab 2: Manual Input for Predictions
            with pred_method_tabs[1]:
                st.write("Enter values for each feature to get a prediction")
                
                # Create input fields for each feature
                input_values = {}
                
                # Create columns for better layout
                cols = st.columns(2)
                
                for i, feature in enumerate(feature_names):
                    col_idx = i % 2  # Alternate between columns
                    
                    # Check if categorical or numerical input should be used
                    if preprocessor and feature in preprocessor and isinstance(preprocessor[feature], object):
                        # Categorical feature with label encoder
                        options = [str(c) for c in preprocessor[feature].classes_]
                        value = cols[col_idx].selectbox(f"{feature}:", options=options)
                        input_values[feature] = value
                    else:
                        # Numerical feature
                        # Try to get the range from the original data
                        try:
                            min_val = float(df[feature].min())
                            max_val = float(df[feature].max())
                            mean_val = float(df[feature].mean())
                            
                            value = cols[col_idx].slider(
                                f"{feature}:",
                                min_value=min_val,
                                max_value=max_val,
                                value=mean_val
                            )
                        except:
                            # Fallback for errors
                            value = cols[col_idx].number_input(f"{feature}:", value=0.0)
                        
                        input_values[feature] = value
                
                # Predict button
                if st.button("Make Prediction", key="manual_predict"):
                    with st.spinner("Generating prediction..."):
                        try:
                            # Create a DataFrame with the input values
                            input_df = pd.DataFrame([input_values])
                            
                            # Determine prediction type
                            if task_type == "Regression":
                                pred_type = "regression"
                            else:
                                pred_type = "classification"
                            
                            # Make prediction
                            prediction = get_model_prediction(model, input_df, preprocessor, target_type=pred_type)
                            
                            # Display result
                            st.subheader("Prediction Result")
                            
                            if task_type == "Regression":
                                st.success(f"Predicted {target_column}: **{prediction[0]:.4f}**")
                            else:
                                st.success(f"Predicted {target_column}: **{prediction[0]}**")
                            
                            # Show input values
                            st.subheader("Input Values")
                            st.dataframe(input_df, use_container_width=True)
                        
                        except Exception as e:
                            st.error(f"Error making prediction: {str(e)}")
            
            # Tab 3: Upload New Data for Predictions
            with pred_method_tabs[2]:
                st.write("Upload a new CSV or Excel file with the same features to make predictions")
                
                uploaded_file = st.file_uploader(
                    "Choose a file",
                    type=["csv", "xlsx", "xls"],
                    key="prediction_file_upload"
                )
                
                if uploaded_file:
                    try:
                        # Read file
                        file_extension = uploaded_file.name.split('.')[-1].lower()
                        
                        if file_extension == 'csv':
                            new_data = pd.read_csv(uploaded_file)
                        elif file_extension in ['xls', 'xlsx']:
                            new_data = pd.read_excel(uploaded_file)
                        else:
                            st.error("Unsupported file format")
                            new_data = None
                        
                        if new_data is not None:
                            # Check if all required features are present
                            missing_features = [f for f in feature_names if f not in new_data.columns]
                            
                            if missing_features:
                                st.error(f"Missing required features: {', '.join(missing_features)}")
                            else:
                                st.write(f"Loaded data with {len(new_data)} rows and {len(new_data.columns)} columns")
                                
                                # Preview the data
                                st.subheader("Data Preview")
                                st.dataframe(new_data.head(), use_container_width=True)
                                
                                # Make predictions button
                                if st.button("Make Predictions", key="batch_predict"):
                                    with st.spinner("Generating predictions..."):
                                        # Extract features
                                        X_new = new_data[feature_names].copy()
                                        
                                        # Determine prediction type
                                        if task_type == "Regression":
                                            pred_type = "regression"
                                        else:
                                            pred_type = "classification"
                                        
                                        # Make predictions
                                        predictions = get_model_prediction(model, X_new, preprocessor, target_type=pred_type)
                                        
                                        # Add predictions to the data
                                        result_df = new_data.copy()
                                        result_df[f'Predicted_{target_column}'] = predictions
                                        
                                        # Display results
                                        st.subheader("Prediction Results")
                                        st.dataframe(result_df.head(20), use_container_width=True)
                                        
                                        # Option to download results
                                        csv = result_df.to_csv(index=False).encode('utf-8')
                                        st.download_button(
                                            label="Download Results as CSV",
                                            data=csv,
                                            file_name="prediction_results.csv",
                                            mime="text/csv"
                                        )
                    
                    except Exception as e:
                        st.error(f"Error processing file or making predictions: {str(e)}")
        else:
            st.warning("⚠️ Please train a model first in the Training & Evaluation tab")
            if st.button("Go to Training & Evaluation"):
                # Switch to the training tab
                pass
    
    # Tab 4: Model Insights
    with pred_tabs[3]:
        st.header("Model Insights and Interpretation")
        
        # Check if model has been trained
        if 'model' in st.session_state and st.session_state.model is not None:
            model = st.session_state.model
            model_type = st.session_state.model_type
            task_type = st.session_state.task_type
            feature_names = st.session_state.feature_names
            
            # Create tabs for different insights
            insight_tabs = st.tabs([
                "Feature Importance", "Model Details", "What-If Analysis"
            ])
            
            # Tab 1: Feature Importance
            with insight_tabs[0]:
                if 'feature_importance' in st.session_state and st.session_state.feature_importance is not None:
                    feature_importance = st.session_state.feature_importance
                    
                    st.subheader("Feature Importance")
                    
                    # Display feature importance plot
                    fig = plot_feature_importance(feature_importance)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display feature importance as a table
                    st.subheader("Feature Importance (Table)")
                    st.dataframe(feature_importance, use_container_width=True)
                    
                    # For tree-based models, add option to visualize a tree
                    if model_type in ["Random Forest Regression", "Random Forest Classifier"]:
                        st.subheader("Tree Visualization")
                        st.info("Tree-based models consist of multiple decision trees. Below is a visualization of the first tree.")
                        
                        # Import necessary libraries
                        from sklearn.tree import export_graphviz
                        import graphviz
                        
                        # Get the first tree from the forest
                        estimator = model.estimators_[0]
                        
                        # Create DOT data
                        dot_data = export_graphviz(
                            estimator,
                            feature_names=feature_names,
                            filled=True,
                            max_depth=3,  # Limit depth for visibility
                            impurity=False,
                            proportion=True
                        )
                        
                        # Convert to graphviz object
                        graph = graphviz.Source(dot_data)
                        
                        # Display the tree
                        st.graphviz_chart(dot_data)
                        
                        # Add note about depth limitation
                        st.info("Note: The tree visualization is limited to a depth of 3 for clarity. The actual model may use deeper trees.")
                else:
                    st.warning("Feature importance data not available.")
            
            # Tab 2: Model Details
            with insight_tabs[1]:
                st.subheader("Model Information")
                
                # Basic model details
                st.write(f"**Model Type:** {model_type}")
                st.write(f"**Task Type:** {task_type}")
                st.write(f"**Target Variable:** {st.session_state.target_column}")
                st.write(f"**Number of Features:** {len(feature_names)}")
                
                # Display model parameters
                st.subheader("Model Parameters")
                
                # Get model parameters
                params = model.get_params()
                
                # Filter out unnecessary parameters
                important_params = {}
                for param, value in params.items():
                    # Skip complex nested parameters
                    if not isinstance(value, (list, dict, object)) or isinstance(value, (str, bool, int, float)):
                        important_params[param] = value
                
                # Display as a table
                params_df = pd.DataFrame({
                    'Parameter': list(important_params.keys()),
                    'Value': list(important_params.values())
                })
                
                st.dataframe(params_df, use_container_width=True)
                
                # Model performance metrics
                if 'model_metrics' in st.session_state and st.session_state.model_metrics is not None:
                    metrics = st.session_state.model_metrics
                    
                    st.subheader("Performance Metrics")
                    
                    if task_type == "Regression":
                        metrics_df = pd.DataFrame({
                            'Metric': ['Mean Squared Error', 'Root Mean Squared Error', 'R² Score'],
                            'Value': [
                                metrics['Mean Squared Error'],
                                metrics['Root Mean Squared Error'],
                                metrics['R² Score']
                            ]
                        })
                        
                        st.dataframe(metrics_df, use_container_width=True)
                    else:
                        st.write(f"**Accuracy:** {metrics['Accuracy']:.4f}")
                        
                        # Display classification report
                        report = metrics['Classification Report']
                        
                        # Skip the 'support' metric and filter out the 'accuracy' row
                        class_metrics = {}
                        for class_name, values in report.items():
                            if isinstance(values, dict) and class_name != 'accuracy':
                                class_metrics[class_name] = {
                                    'precision': values['precision'],
                                    'recall': values['recall'],
                                    'f1-score': values['f1-score']
                                }
                        
                        report_df = pd.DataFrame(class_metrics).transpose()
                        report_df.index.name = 'Class'
                        report_df = report_df.reset_index()
                        
                        st.dataframe(report_df, use_container_width=True)
            
            # Tab 3: What-If Analysis
            with insight_tabs[2]:
                st.subheader("What-If Analysis")
                st.write("Explore how changing feature values affects the prediction")
                
                # Get a baseline sample for analysis
                if 'X_test' in st.session_state and st.session_state.X_test is not None:
                    X_test = st.session_state.X_test
                    
                    # Select a random sample from the test set
                    sample_index = st.selectbox(
                        "Select a sample from the test set:",
                        options=list(range(len(X_test))),
                        index=0
                    )
                    
                    sample = X_test.iloc[sample_index].copy()
                    
                    # Display the sample
                    st.subheader("Baseline Sample")
                    sample_df = pd.DataFrame(sample).T
                    st.dataframe(sample_df, use_container_width=True)
                    
                    # Make baseline prediction
                    if task_type == "Regression":
                        pred_type = "regression"
                    else:
                        pred_type = "classification"
                    
                    baseline_pred = get_model_prediction(
                        model, 
                        pd.DataFrame([sample]), 
                        st.session_state.preprocessor, 
                        target_type=pred_type
                    )[0]
                    
                    # Display baseline prediction
                    st.write(f"**Baseline Prediction:** {baseline_pred}")
                    
                    # Feature to modify
                    feature_to_modify = st.selectbox(
                        "Select feature to modify:",
                        options=feature_names
                    )
                    
                    # Determine the range for modification
                    is_categorical = feature_to_modify in st.session_state.preprocessor and isinstance(
                        st.session_state.preprocessor[feature_to_modify], object
                    )
                    
                    if is_categorical:
                        # Categorical feature
                        options = [str(c) for c in st.session_state.preprocessor[feature_to_modify].classes_]
                        modified_value = st.selectbox(
                            f"Modify {feature_to_modify} to:",
                            options=options
                        )
                    else:
                        # Numerical feature
                        try:
                            # Get the range from the original data
                            original_value = sample[feature_to_modify]
                            min_val = float(X_test[feature_to_modify].min())
                            max_val = float(X_test[feature_to_modify].max())
                            
                            modified_value = st.slider(
                                f"Modify {feature_to_modify} from {original_value:.4f} to:",
                                min_value=min_val,
                                max_value=max_val,
                                value=float(original_value)
                            )
                        except:
                            # Fallback
                            original_value = sample[feature_to_modify]
                            modified_value = st.number_input(
                                f"Modify {feature_to_modify} from {original_value} to:",
                                value=float(original_value)
                            )
                    
                    # Create modified sample
                    modified_sample = sample.copy()
                    modified_sample[feature_to_modify] = modified_value
                    
                    # Make prediction with modified sample
                    modified_pred = get_model_prediction(
                        model, 
                        pd.DataFrame([modified_sample]), 
                        st.session_state.preprocessor, 
                        target_type=pred_type
                    )[0]
                    
                    # Display modified prediction
                    st.subheader("Modified Prediction")
                    st.write(f"**Modified {feature_to_modify} to:** {modified_value}")
                    st.write(f"**New Prediction:** {modified_pred}")
                    
                    # Show change
                    change = modified_pred - baseline_pred
                    change_pct = (change / baseline_pred) * 100 if baseline_pred != 0 else float('inf')
                    
                    if change > 0:
                        st.success(f"Prediction increased by {change:.4f} ({change_pct:.2f}%)")
                    elif change < 0:
                        st.error(f"Prediction decreased by {abs(change):.4f} ({abs(change_pct):.2f}%)")
                    else:
                        st.info("No change in prediction")
                    
                    # Option for sensitivity analysis
                    if st.checkbox("Perform sensitivity analysis on this feature"):
                        st.subheader("Sensitivity Analysis")
                        
                        if not is_categorical:
                            # For numerical features, create a range of values
                            original_value = float(sample[feature_to_modify])
                            feature_min = float(X_test[feature_to_modify].min())
                            feature_max = float(X_test[feature_to_modify].max())
                            
                            # Create a range of values between min and max
                            num_points = st.slider("Number of points:", 5, 50, 20)
                            feature_range = np.linspace(feature_min, feature_max, num_points)
                            
                            # Make predictions for each value
                            predictions = []
                            for value in feature_range:
                                test_sample = sample.copy()
                                test_sample[feature_to_modify] = value
                                pred = get_model_prediction(
                                    model, 
                                    pd.DataFrame([test_sample]), 
                                    st.session_state.preprocessor, 
                                    target_type=pred_type
                                )[0]
                                predictions.append(pred)
                            
                            # Create sensitivity plot
                            import plotly.express as px
                            
                            sensitivity_df = pd.DataFrame({
                                feature_to_modify: feature_range,
                                'Prediction': predictions
                            })
                            
                            fig = px.line(
                                sensitivity_df,
                                x=feature_to_modify,
                                y='Prediction',
                                title=f"Sensitivity of Prediction to {feature_to_modify}"
                            )
                            
                            # Add marker for original value
                            fig.add_vline(
                                x=original_value,
                                line_dash="dash",
                                line_color="red",
                                annotation_text="Original value"
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            # For categorical features, show prediction for each category
                            categories = [str(c) for c in st.session_state.preprocessor[feature_to_modify].classes_]
                            predictions = []
                            
                            for category in categories:
                                test_sample = sample.copy()
                                test_sample[feature_to_modify] = category
                                pred = get_model_prediction(
                                    model, 
                                    pd.DataFrame([test_sample]), 
                                    st.session_state.preprocessor, 
                                    target_type=pred_type
                                )[0]
                                predictions.append(pred)
                            
                            # Create bar chart
                            import plotly.express as px
                            
                            sensitivity_df = pd.DataFrame({
                                feature_to_modify: categories,
                                'Prediction': predictions
                            })
                            
                            fig = px.bar(
                                sensitivity_df,
                                x=feature_to_modify,
                                y='Prediction',
                                title=f"Prediction by {feature_to_modify} Categories"
                            )
                            
                            # Highlight original category
                            original_category = str(sample[feature_to_modify])
                            for i, cat in enumerate(categories):
                                if cat == original_category:
                                    fig.data[0].marker.color = ['blue' if c != cat else 'red' for c in categories]
                                    break
                            
                            st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Test data not available. Please prepare your data first.")
        else:
            st.warning("⚠️ Please train a model first in the Training & Evaluation tab")
else:
    st.warning("⚠️ Please upload a data file first on the Home page")
    if st.button("Go to Home"):
        st.switch_page("app.py")
