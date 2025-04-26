import pandas as pd
import numpy as np
import streamlit as st

def get_initial_dataframe_info(df):
    """
    Calculate and store initial dataframe information in session state
    """
    if df is not None:
        # Store information about the dataframe
        st.session_state.columns = df.columns.tolist()
        st.session_state.numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
        st.session_state.categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        st.session_state.datetime_columns = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        # Calculate missing values percentage
        missing_percentages = (df.isnull().sum() / len(df) * 100).to_dict()
        st.session_state.missing_percentages = missing_percentages
        
        # Check for duplicates
        st.session_state.duplicate_count = df.duplicated().sum()
        
        # Sample uniqueness of categorical columns
        unique_values = {}
        for col in st.session_state.categorical_columns:
            if len(df[col].unique()) < 20:  # Only for columns with reasonable number of unique values
                unique_values[col] = df[col].unique().tolist()
        st.session_state.unique_categorical_values = unique_values

def handle_missing_values(df, strategy, columns=None, custom_value=None):
    """
    Handle missing values in the dataframe using various strategies
    
    Parameters:
    - df: pandas DataFrame
    - strategy: str, one of 'drop_rows', 'drop_columns', 'fill_mean', 'fill_median', 
                'fill_mode', 'fill_custom', 'fill_ffill', 'fill_bfill'
    - columns: list, columns to apply the strategy to (None means all columns)
    - custom_value: value to use for 'fill_custom' strategy
    
    Returns:
    - Processed pandas DataFrame
    """
    if df is None:
        return None
    
    # Make a copy to avoid modifying the original
    df_processed = df.copy()
    
    # If no columns specified, use all columns
    if columns is None:
        columns = df.columns
    
    # Apply the selected strategy
    if strategy == 'drop_rows':
        df_processed = df_processed.dropna(subset=columns)
        
    elif strategy == 'drop_columns':
        df_processed = df_processed.drop(columns=columns)
        
    elif strategy == 'fill_mean':
        for col in columns:
            if pd.api.types.is_numeric_dtype(df_processed[col]):
                df_processed[col] = df_processed[col].fillna(df_processed[col].mean())
            
    elif strategy == 'fill_median':
        for col in columns:
            if pd.api.types.is_numeric_dtype(df_processed[col]):
                df_processed[col] = df_processed[col].fillna(df_processed[col].median())
            
    elif strategy == 'fill_mode':
        for col in columns:
            df_processed[col] = df_processed[col].fillna(df_processed[col].mode()[0])
            
    elif strategy == 'fill_custom':
        for col in columns:
            df_processed[col] = df_processed[col].fillna(custom_value)
            
    elif strategy == 'fill_ffill':
        df_processed[columns] = df_processed[columns].fillna(method='ffill')
            
    elif strategy == 'fill_bfill':
        df_processed[columns] = df_processed[columns].fillna(method='bfill')
    
    return df_processed

def handle_duplicates(df, strategy):
    """
    Handle duplicate rows in the dataframe
    
    Parameters:
    - df: pandas DataFrame
    - strategy: str, one of 'remove_first', 'remove_last', 'keep_all'
    
    Returns:
    - Processed pandas DataFrame
    """
    if df is None:
        return None
    
    # Make a copy to avoid modifying the original
    df_processed = df.copy()
    
    if strategy == 'remove_first':
        df_processed = df_processed.drop_duplicates(keep='first')
    elif strategy == 'remove_last':
        df_processed = df_processed.drop_duplicates(keep='last')
    elif strategy == 'keep_all':
        # No action needed, keep all rows including duplicates
        pass
    
    return df_processed

def convert_data_types(df, column, new_type):
    """
    Convert a column to a different data type
    
    Parameters:
    - df: pandas DataFrame
    - column: str, the column to convert
    - new_type: str, the new data type ('int', 'float', 'str', 'bool', 'datetime')
    
    Returns:
    - Processed pandas DataFrame with the column converted
    """
    if df is None or column not in df.columns:
        return df
    
    # Make a copy to avoid modifying the original
    df_processed = df.copy()
    
    try:
        if new_type == 'int':
            df_processed[column] = pd.to_numeric(df_processed[column], errors='coerce').astype('Int64')
        elif new_type == 'float':
            df_processed[column] = pd.to_numeric(df_processed[column], errors='coerce')
        elif new_type == 'str':
            df_processed[column] = df_processed[column].astype(str)
        elif new_type == 'bool':
            df_processed[column] = df_processed[column].astype(bool)
        elif new_type == 'datetime':
            df_processed[column] = pd.to_datetime(df_processed[column], errors='coerce')
    except Exception as e:
        st.error(f"Error converting {column} to {new_type}: {str(e)}")
    
    return df_processed

def filter_dataframe(df, filters):
    """
    Apply filters to the dataframe
    
    Parameters:
    - df: pandas DataFrame
    - filters: list of dicts with keys 'column', 'operator', 'value'
    
    Returns:
    - Filtered pandas DataFrame
    """
    if df is None or not filters:
        return df
    
    # Make a copy to avoid modifying the original
    df_filtered = df.copy()
    
    for filter_item in filters:
        column = filter_item.get('column')
        operator = filter_item.get('operator')
        value = filter_item.get('value')
        
        if column not in df_filtered.columns:
            continue
        
        if operator == 'equals':
            df_filtered = df_filtered[df_filtered[column] == value]
        elif operator == 'not_equals':
            df_filtered = df_filtered[df_filtered[column] != value]
        elif operator == 'greater_than':
            df_filtered = df_filtered[df_filtered[column] > value]
        elif operator == 'less_than':
            df_filtered = df_filtered[df_filtered[column] < value]
        elif operator == 'contains':
            df_filtered = df_filtered[df_filtered[column].astype(str).str.contains(str(value), na=False)]
        elif operator == 'starts_with':
            df_filtered = df_filtered[df_filtered[column].astype(str).str.startswith(str(value), na=False)]
        elif operator == 'ends_with':
            df_filtered = df_filtered[df_filtered[column].astype(str).str.endswith(str(value), na=False)]
    
    return df_filtered

def get_summary_statistics(df, columns=None):
    """
    Generate summary statistics for selected columns
    
    Parameters:
    - df: pandas DataFrame
    - columns: list of column names to include (None for all numeric columns)
    
    Returns:
    - DataFrame with summary statistics
    """
    if df is None:
        return None
    
    if columns is None:
        # Use all numeric columns
        columns = df.select_dtypes(include=np.number).columns.tolist()
    
    # Filter to only include numeric columns from the selection
    numeric_columns = [col for col in columns if col in df.select_dtypes(include=np.number).columns]
    
    if not numeric_columns:
        return pd.DataFrame()
    
    # Calculate statistics
    stats_df = df[numeric_columns].describe().T
    
    # Add additional statistics
    stats_df['skew'] = df[numeric_columns].skew()
    stats_df['kurtosis'] = df[numeric_columns].kurtosis()
    stats_df['missing_count'] = df[numeric_columns].isnull().sum()
    stats_df['missing_pct'] = df[numeric_columns].isnull().mean() * 100
    
    return stats_df

def get_categorical_summary(df, columns=None):
    """
    Generate summary for categorical columns
    
    Parameters:
    - df: pandas DataFrame
    - columns: list of categorical column names (None for all object columns)
    
    Returns:
    - Dictionary with categorical summaries
    """
    if df is None:
        return {}
    
    if columns is None:
        # Use all object and category columns
        columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Filter to only include categorical columns from the selection
    cat_columns = [col for col in columns if col in df.select_dtypes(include=['object', 'category']).columns]
    
    if not cat_columns:
        return {}
    
    summaries = {}
    for col in cat_columns:
        value_counts = df[col].value_counts().reset_index()
        value_counts.columns = ['value', 'count']
        value_counts['percentage'] = value_counts['count'] / len(df) * 100
        
        summaries[col] = {
            'unique_count': df[col].nunique(),
            'missing_count': df[col].isnull().sum(),
            'missing_pct': df[col].isnull().mean() * 100,
            'value_counts': value_counts
        }
    
    return summaries
