import streamlit as st
import pandas as pd
import numpy as np
from utils.data_processor import handle_missing_values, handle_duplicates, convert_data_types

# Set page configuration
st.set_page_config(
    page_title="Data Cleaning",
    page_icon="🧹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables if they don't exist
if 'data' not in st.session_state:
    st.session_state.data = None
if 'data_cleaned' not in st.session_state:
    st.session_state.data_cleaned = False

# Function to check if a column has missing values
def has_missing_values(df, column):
    return df[column].isnull().sum() > 0

# Page title and description
st.title("🧹 Data Cleaning")
st.write("Clean and preprocess your data before analysis")

# Check if data is loaded
if st.session_state.data is not None:
    df = st.session_state.data
    
    # Create tabs for different cleaning operations
    cleaning_tabs = st.tabs(["Missing Values", "Duplicates", "Data Types", "Filtering", "Preview"])
    
    # Tab 1: Missing Values
    with cleaning_tabs[0]:
        st.header("Handle Missing Values")
        
        # Display statistics about missing values
        missing_vals = df.isnull().sum()
        missing_percent = (df.isnull().sum() / len(df) * 100).round(2)
        
        missing_df = pd.DataFrame({
            'Column': missing_vals.index,
            'Missing Values': missing_vals.values,
            'Percentage': missing_percent.values
        })
        
        # Sort by missing values count (descending)
        missing_df = missing_df.sort_values('Missing Values', ascending=False)
        
        # Only show columns with missing values
        missing_cols = missing_df[missing_df['Missing Values'] > 0]
        
        if len(missing_cols) > 0:
            st.write("Columns with missing values:")
            st.dataframe(missing_cols, use_container_width=True)
            
            # Select columns for missing value treatment
            selected_columns = st.multiselect(
                "Select columns to handle missing values:",
                options=missing_cols['Column'].tolist(),
                default=missing_cols['Column'].tolist()
            )
            
            if selected_columns:
                # Choose strategy
                strategy = st.selectbox(
                    "Select strategy to handle missing values:",
                    options=[
                        'drop_rows', 'drop_columns', 'fill_mean', 'fill_median',
                        'fill_mode', 'fill_custom', 'fill_ffill', 'fill_bfill'
                    ],
                    format_func=lambda x: {
                        'drop_rows': 'Drop rows with missing values',
                        'drop_columns': 'Drop columns with missing values',
                        'fill_mean': 'Fill with mean (numeric only)',
                        'fill_median': 'Fill with median (numeric only)',
                        'fill_mode': 'Fill with mode (most frequent value)',
                        'fill_custom': 'Fill with custom value',
                        'fill_ffill': 'Fill with previous value (forward fill)',
                        'fill_bfill': 'Fill with next value (backward fill)'
                    }.get(x)
                )
                
                # Custom value input if selected
                custom_value = None
                if strategy == 'fill_custom':
                    custom_value = st.text_input("Enter custom value to fill missing values")
                
                # Preview the result of the operation
                if st.button("Preview Changes", key="preview_missing"):
                    with st.spinner("Processing..."):
                        df_preview = handle_missing_values(df, strategy, selected_columns, custom_value)
                        
                        # Show comparison before/after
                        col1, col2 = st.columns(2)
                        with col1:
                            st.subheader("Before")
                            st.dataframe(df[selected_columns].head(10), use_container_width=True)
                        
                        with col2:
                            st.subheader("After")
                            st.dataframe(df_preview[selected_columns].head(10), use_container_width=True)
                        
                        # Show statistics after cleaning
                        missing_after = df_preview[selected_columns].isnull().sum()
                        missing_percent_after = (missing_after / len(df_preview) * 100).round(2)
                        
                        st.subheader("Missing values after cleaning")
                        comparison_df = pd.DataFrame({
                            'Column': selected_columns,
                            'Before (count)': [df[col].isnull().sum() for col in selected_columns],
                            'Before (%)': [missing_percent[col] for col in selected_columns],
                            'After (count)': [missing_after[col] for col in selected_columns],
                            'After (%)': missing_percent_after
                        })
                        
                        st.dataframe(comparison_df, use_container_width=True)
                
                # Apply changes
                if st.button("Apply Changes", key="apply_missing"):
                    with st.spinner("Applying changes..."):
                        st.session_state.data = handle_missing_values(df, strategy, selected_columns, custom_value)
                        st.success("✅ Missing values handled successfully!")
                        st.session_state.data_cleaned = True
                        st.experimental_rerun()
        else:
            st.success("✅ No missing values found in the dataset")
    
    # Tab 2: Duplicates
    with cleaning_tabs[1]:
        st.header("Handle Duplicate Rows")
        
        # Check for duplicates
        duplicate_count = df.duplicated().sum()
        
        if duplicate_count > 0:
            st.warning(f"Found {duplicate_count} duplicate rows in the dataset ({duplicate_count/len(df)*100:.2f}%)")
            
            # Choose strategy for handling duplicates
            duplicate_strategy = st.selectbox(
                "Select strategy to handle duplicates:",
                options=['remove_first', 'remove_last', 'keep_all'],
                format_func=lambda x: {
                    'remove_first': 'Remove duplicates (keep first occurrence)',
                    'remove_last': 'Remove duplicates (keep last occurrence)',
                    'keep_all': 'Keep all duplicates'
                }.get(x)
            )
            
            # Preview duplicate rows
            if st.button("Show Duplicate Rows", key="show_duplicates"):
                st.dataframe(df[df.duplicated(keep=False)].sort_values(by=df.columns[0]), use_container_width=True)
            
            # Apply changes
            if st.button("Apply Changes", key="apply_duplicates"):
                with st.spinner("Removing duplicates..."):
                    st.session_state.data = handle_duplicates(df, duplicate_strategy)
                    st.success(f"✅ Duplicates handled successfully! {len(df) - len(st.session_state.data)} rows removed.")
                    st.session_state.data_cleaned = True
                    st.experimental_rerun()
        else:
            st.success("✅ No duplicate rows found in the dataset")
    
    # Tab 3: Data Types
    with cleaning_tabs[2]:
        st.header("Data Type Conversion")
        
        # Display current data types
        dtypes_df = pd.DataFrame({
            'Column': df.dtypes.index,
            'Current Type': df.dtypes.values.astype(str)
        })
        
        st.dataframe(dtypes_df, use_container_width=True)
        
        # Data type conversion
        col1, col2 = st.columns(2)
        
        with col1:
            selected_column = st.selectbox("Select column to convert:", options=df.columns)
        
        with col2:
            target_type = st.selectbox(
                "Convert to:",
                options=['int', 'float', 'str', 'bool', 'datetime']
            )
        
        if st.button("Preview Conversion", key="preview_convert"):
            try:
                df_preview = convert_data_types(df, selected_column, target_type)
                
                # Show comparison before/after
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Before")
                    st.write(f"Data type: {df[selected_column].dtype}")
                    st.dataframe(df[[selected_column]].head(10), use_container_width=True)
                
                with col2:
                    st.subheader("After")
                    st.write(f"Data type: {df_preview[selected_column].dtype}")
                    st.dataframe(df_preview[[selected_column]].head(10), use_container_width=True)
                
            except Exception as e:
                st.error(f"Error converting data type: {str(e)}")
        
        if st.button("Apply Conversion", key="apply_convert"):
            try:
                with st.spinner("Converting data type..."):
                    st.session_state.data = convert_data_types(df, selected_column, target_type)
                    st.success(f"✅ Column '{selected_column}' converted to {target_type} successfully!")
                    st.session_state.data_cleaned = True
                    st.experimental_rerun()
            except Exception as e:
                st.error(f"Error converting data type: {str(e)}")
    
    # Tab 4: Filtering
    with cleaning_tabs[3]:
        st.header("Filter Data")
        
        # Simple filtering interface
        col1, col2, col3 = st.columns(3)
        
        with col1:
            filter_column = st.selectbox("Select column to filter:", options=df.columns)
        
        with col2:
            # Different filter options based on column data type
            if pd.api.types.is_numeric_dtype(df[filter_column]):
                min_val = float(df[filter_column].min())
                max_val = float(df[filter_column].max())
                
                filter_type = st.selectbox(
                    "Filter type:",
                    options=["range", "equals", "greater_than", "less_than"],
                    key="numeric_filter_type"
                )
                
                if filter_type == "range":
                    filter_value = st.slider(
                        "Select range:",
                        min_value=min_val,
                        max_value=max_val,
                        value=(min_val, max_val)
                    )
                else:
                    filter_value = st.number_input(
                        "Enter value:",
                        value=min_val,
                        min_value=min_val,
                        max_value=max_val
                    )
            else:
                # For categorical/text columns
                unique_values = df[filter_column].dropna().unique()
                
                filter_type = st.selectbox(
                    "Filter type:",
                    options=["equals", "contains", "starts_with", "ends_with"],
                    key="categorical_filter_type"
                )
                
                if filter_type == "equals":
                    filter_value = st.selectbox("Select value:", options=unique_values)
                else:
                    filter_value = st.text_input("Enter text to filter:")
        
        # Apply filter button
        if st.button("Apply Filter", key="apply_filter"):
            with st.spinner("Filtering data..."):
                try:
                    # Filtering logic based on filter type and column data type
                    if pd.api.types.is_numeric_dtype(df[filter_column]):
                        if filter_type == "range":
                            filtered_df = df[(df[filter_column] >= filter_value[0]) & 
                                            (df[filter_column] <= filter_value[1])]
                        elif filter_type == "equals":
                            filtered_df = df[df[filter_column] == filter_value]
                        elif filter_type == "greater_than":
                            filtered_df = df[df[filter_column] > filter_value]
                        elif filter_type == "less_than":
                            filtered_df = df[df[filter_column] < filter_value]
                    else:
                        if filter_type == "equals":
                            filtered_df = df[df[filter_column] == filter_value]
                        elif filter_type == "contains":
                            filtered_df = df[df[filter_column].astype(str).str.contains(filter_value, na=False)]
                        elif filter_type == "starts_with":
                            filtered_df = df[df[filter_column].astype(str).str.startswith(filter_value, na=False)]
                        elif filter_type == "ends_with":
                            filtered_df = df[df[filter_column].astype(str).str.endswith(filter_value, na=False)]
                    
                    # Display filtered data
                    st.subheader("Filtered Data")
                    st.write(f"Showing {len(filtered_df)} rows out of {len(df)} total rows")
                    st.dataframe(filtered_df, use_container_width=True)
                    
                    # Option to save the filtered dataset
                    if st.button("Save Filtered Data", key="save_filtered"):
                        st.session_state.data = filtered_df
                        st.success("✅ Filtered data saved as the current dataset!")
                        st.session_state.data_cleaned = True
                        st.experimental_rerun()
                    
                except Exception as e:
                    st.error(f"Error applying filter: {str(e)}")
    
    # Tab 5: Preview
    with cleaning_tabs[4]:
        st.header("Data Preview")
        
        # Show current data
        st.subheader("Current Dataset")
        st.write(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
        st.dataframe(df.head(100), use_container_width=True)
        
        # Download cleaned data
        if st.session_state.data_cleaned:
            @st.cache_data
            def convert_df_to_csv(df):
                return df.to_csv(index=False).encode('utf-8')
            
            csv = convert_df_to_csv(df)
            st.download_button(
                label="Download Cleaned Data as CSV",
                data=csv,
                file_name="cleaned_data.csv",
                mime="text/csv",
                key="download_cleaned"
            )
            
            # Reset option
            if st.button("Reset to Original Data", key="reset_data"):
                if st.session_state.uploaded_file_name is not None:
                    st.warning("This will reset all cleaning operations! Proceed?")
                    if st.button("Confirm Reset", key="confirm_reset"):
                        from utils.data_processor import process_uploaded_file
                        uploaded_file = st.session_state.uploaded_file_name
                        # We need to re-upload the file here
                        st.error("Since we can't access the original file, please re-upload your file from the main page.")
                        st.session_state.data_cleaned = False
else:
    st.warning("⚠️ Please upload a data file first on the Home page")
    if st.button("Go to Home"):
        st.switch_page("app.py")
