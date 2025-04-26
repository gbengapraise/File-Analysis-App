import streamlit as st
import pandas as pd
import numpy as np
from utils.data_processor import get_summary_statistics, get_categorical_summary

# Set page configuration
st.set_page_config(
    page_title="Data Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables if they don't exist
if 'data' not in st.session_state:
    st.session_state.data = None

# Page title and description
st.title("📊 Data Analysis")
st.write("Analyze your data with descriptive statistics and insights")

# Check if data is loaded
if st.session_state.data is not None:
    df = st.session_state.data
    
    # Create tabs for different analyses
    analysis_tabs = st.tabs([
        "Summary Statistics", 
        "Categorical Analysis",
        "Group By Analysis", 
        "Correlation"
    ])
    
    # Tab 1: Summary Statistics
    with analysis_tabs[0]:
        st.header("Summary Statistics")
        
        # Select columns for analysis
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_columns:
            selected_num_columns = st.multiselect(
                "Select numeric columns for summary statistics:",
                options=numeric_columns,
                default=numeric_columns[:min(5, len(numeric_columns))]
            )
            
            if selected_num_columns:
                # Calculate statistics
                stats_df = get_summary_statistics(df, selected_num_columns)
                
                # Display statistics
                st.subheader("Descriptive Statistics")
                st.dataframe(stats_df, use_container_width=True)
                
                # Additional statistics
                st.subheader("Distribution Information")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Skewness**")
                    st.info(
                        "Skewness measures the asymmetry of a distribution:\n"
                        "- Values close to 0 indicate a symmetric distribution\n"
                        "- Positive values indicate right skew (tail extends right)\n"
                        "- Negative values indicate left skew (tail extends left)"
                    )
                    skew_df = pd.DataFrame({
                        'Column': selected_num_columns,
                        'Skewness': [df[col].skew() for col in selected_num_columns]
                    })
                    st.dataframe(skew_df, use_container_width=True)
                
                with col2:
                    st.write("**Kurtosis**")
                    st.info(
                        "Kurtosis measures the 'tailedness' of a distribution:\n"
                        "- Values close to 0 indicate a normal distribution\n"
                        "- Positive values indicate heavy tails (outlier-prone)\n"
                        "- Negative values indicate light tails (few outliers)"
                    )
                    kurtosis_df = pd.DataFrame({
                        'Column': selected_num_columns,
                        'Kurtosis': [df[col].kurtosis() for col in selected_num_columns]
                    })
                    st.dataframe(kurtosis_df, use_container_width=True)
        else:
            st.warning("No numeric columns found in the dataset for summary statistics")
    
    # Tab 2: Categorical Analysis
    with analysis_tabs[1]:
        st.header("Categorical Analysis")
        
        # Select categorical columns
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if categorical_columns:
            selected_cat_column = st.selectbox(
                "Select a categorical column to analyze:",
                options=categorical_columns
            )
            
            if selected_cat_column:
                # Get frequency distribution
                cat_summary = get_categorical_summary(df, [selected_cat_column])
                
                if selected_cat_column in cat_summary:
                    value_counts = cat_summary[selected_cat_column]['value_counts']
                    unique_count = cat_summary[selected_cat_column]['unique_count']
                    missing_count = cat_summary[selected_cat_column]['missing_count']
                    
                    # Display summary
                    st.subheader(f"Analysis of '{selected_cat_column}'")
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Unique Values", unique_count)
                    col2.metric("Missing Values", missing_count)
                    col3.metric("Fill Rate", f"{(1 - missing_count/len(df))*100:.2f}%")
                    
                    # Display frequency table
                    st.subheader("Frequency Distribution")
                    
                    # Limit to top 20 categories if there are too many
                    if len(value_counts) > 20:
                        st.info(f"Showing top 20 categories out of {len(value_counts)}")
                        value_counts = value_counts.head(20)
                    
                    st.dataframe(value_counts, use_container_width=True)
                    
                    # Cross tabulation
                    st.subheader("Cross Tabulation")
                    st.write("Analyze relationship with another categorical variable")
                    
                    other_cats = [col for col in categorical_columns if col != selected_cat_column]
                    if other_cats:
                        cross_tab_col = st.selectbox(
                            "Select column to cross-tabulate with:",
                            options=other_cats
                        )
                        
                        # Calculate and display crosstab
                        try:
                            crosstab = pd.crosstab(
                                df[selected_cat_column],
                                df[cross_tab_col],
                                margins=True,
                                normalize=st.checkbox("Show percentages", key="crosstab_pct")
                            )
                            
                            st.dataframe(crosstab, use_container_width=True)
                            
                            st.write("**Chi-Square Test for Independence**")
                            st.info(
                                "The chi-square test evaluates if there is a significant relationship "
                                "between the two categorical variables."
                            )
                            
                            # Import chi2_contingency only when needed
                            from scipy.stats import chi2_contingency
                            
                            # Perform chi-square test
                            contingency_table = pd.crosstab(
                                df[selected_cat_column], 
                                df[cross_tab_col]
                            )
                            
                            chi2, p, dof, expected = chi2_contingency(contingency_table)
                            
                            # Display results
                            chi2_results = pd.DataFrame({
                                'Statistic': ['Chi-square value', 'p-value', 'Degrees of freedom'],
                                'Value': [chi2, p, dof]
                            })
                            
                            st.dataframe(chi2_results, use_container_width=True)
                            
                            if p < 0.05:
                                st.success(
                                    f"The p-value ({p:.4f}) is less than 0.05, suggesting a significant "
                                    f"relationship between '{selected_cat_column}' and '{cross_tab_col}'."
                                )
                            else:
                                st.info(
                                    f"The p-value ({p:.4f}) is greater than 0.05, suggesting no significant "
                                    f"relationship between '{selected_cat_column}' and '{cross_tab_col}'."
                                )
                        
                        except Exception as e:
                            st.error(f"Error performing cross tabulation: {str(e)}")
                    else:
                        st.warning("No other categorical columns available for cross tabulation")
        else:
            st.warning("No categorical columns found in the dataset")
    
    # Tab 3: Group By Analysis
    with analysis_tabs[2]:
        st.header("Group By Analysis")
        
        # Select columns for grouping and aggregation
        all_columns = df.columns.tolist()
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if categorical_columns and numeric_columns:
            # Group by column selection
            group_by_cols = st.multiselect(
                "Select column(s) to group by:",
                options=categorical_columns,
                default=categorical_columns[:1] if categorical_columns else None
            )
            
            if group_by_cols:
                # Aggregation column selection
                agg_cols = st.multiselect(
                    "Select numeric column(s) to aggregate:",
                    options=numeric_columns,
                    default=numeric_columns[:2] if len(numeric_columns) >= 2 else numeric_columns
                )
                
                if agg_cols:
                    # Select aggregation functions
                    agg_functions = st.multiselect(
                        "Select aggregation functions:",
                        options=["mean", "median", "sum", "min", "max", "count", "std"],
                        default=["mean", "sum", "count"]
                    )
                    
                    if agg_functions:
                        # Create a dictionary for aggregation
                        agg_dict = {col: agg_functions for col in agg_cols}
                        
                        try:
                            # Perform groupby
                            grouped_df = df.groupby(group_by_cols).agg(agg_dict)
                            
                            # Display results
                            st.subheader("Group By Results")
                            st.dataframe(grouped_df, use_container_width=True)
                            
                            # Download option for grouped data
                            @st.cache_data
                            def convert_df_to_csv(df):
                                return df.to_csv().encode('utf-8')
                            
                            csv = convert_df_to_csv(grouped_df)
                            st.download_button(
                                label="Download Grouped Data as CSV",
                                data=csv,
                                file_name="grouped_data.csv",
                                mime="text/csv"
                            )
                            
                            # Option to pivot the results
                            if len(group_by_cols) >= 2 and st.checkbox("Create pivot table"):
                                st.subheader("Pivot Table")
                                
                                # Select columns for pivot
                                pivot_index = st.selectbox(
                                    "Select column for pivot index:", 
                                    options=group_by_cols
                                )
                                
                                pivot_columns = st.selectbox(
                                    "Select column for pivot columns:", 
                                    options=[col for col in group_by_cols if col != pivot_index]
                                )
                                
                                pivot_values = st.selectbox(
                                    "Select column for pivot values:", 
                                    options=agg_cols
                                )
                                
                                pivot_aggfunc = st.selectbox(
                                    "Select aggregation function:", 
                                    options=agg_functions
                                )
                                
                                # Create pivot table
                                pivot_table = pd.pivot_table(
                                    df,
                                    values=pivot_values,
                                    index=pivot_index,
                                    columns=pivot_columns,
                                    aggfunc=pivot_aggfunc
                                )
                                
                                st.dataframe(pivot_table, use_container_width=True)
                                
                                # Download option for pivot table
                                csv_pivot = convert_df_to_csv(pivot_table)
                                st.download_button(
                                    label="Download Pivot Table as CSV",
                                    data=csv_pivot,
                                    file_name="pivot_table.csv",
                                    mime="text/csv",
                                    key="download_pivot"
                                )
                        
                        except Exception as e:
                            st.error(f"Error performing group by analysis: {str(e)}")
                    
                    else:
                        st.warning("Please select at least one aggregation function")
                else:
                    st.warning("Please select at least one numeric column to aggregate")
            else:
                st.warning("Please select at least one column to group by")
        else:
            if not categorical_columns:
                st.warning("No categorical columns found for grouping")
            if not numeric_columns:
                st.warning("No numeric columns found for aggregation")
    
    # Tab 4: Correlation Analysis
    with analysis_tabs[3]:
        st.header("Correlation Analysis")
        
        # Select columns for correlation
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_columns) >= 2:
            selected_corr_columns = st.multiselect(
                "Select columns for correlation analysis:",
                options=numeric_columns,
                default=numeric_columns[:min(5, len(numeric_columns))]
            )
            
            if len(selected_corr_columns) >= 2:
                # Select correlation method
                corr_method = st.radio(
                    "Select correlation method:",
                    options=["pearson", "spearman", "kendall"],
                    horizontal=True,
                    help=(
                        "Pearson: Linear relationship, "
                        "Spearman: Monotonic relationship, "
                        "Kendall: Ordinal association"
                    )
                )
                
                # Calculate and display correlation matrix
                corr_matrix = df[selected_corr_columns].corr(method=corr_method)
                
                st.subheader(f"Correlation Matrix ({corr_method.capitalize()})")
                st.dataframe(corr_matrix.style.background_gradient(cmap='coolwarm'), use_container_width=True)
                
                # Interpretation guide
                st.subheader("Interpretation")
                st.info(
                    "**Correlation coefficient values range from -1 to 1:**\n\n"
                    "- **1.0**: Perfect positive correlation\n"
                    "- **0.7 to 0.9**: Strong positive correlation\n"
                    "- **0.4 to 0.6**: Moderate positive correlation\n"
                    "- **0.1 to 0.3**: Weak positive correlation\n"
                    "- **0**: No correlation\n"
                    "- **-0.1 to -0.3**: Weak negative correlation\n"
                    "- **-0.4 to -0.6**: Moderate negative correlation\n"
                    "- **-0.7 to -0.9**: Strong negative correlation\n"
                    "- **-1.0**: Perfect negative correlation"
                )
                
                # Show strongest correlations
                st.subheader("Top Correlations")
                
                # Get upper triangle of correlation matrix
                corr_upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                
                # Stack and sort to find strongest correlations
                corr_pairs = corr_upper.stack().sort_values(ascending=False)
                
                if not corr_pairs.empty:
                    # Convert to DataFrame for display
                    top_corr_df = pd.DataFrame({
                        'Feature 1': [idx[0] for idx in corr_pairs.index],
                        'Feature 2': [idx[1] for idx in corr_pairs.index],
                        'Correlation': corr_pairs.values
                    })
                    
                    st.dataframe(top_corr_df, use_container_width=True)
                    
                    # Select a pair for scatter plot
                    if st.checkbox("Show scatter plot for a correlation pair"):
                        # Allow user to select a pair from top correlations
                        scatter_pair = st.selectbox(
                            "Select a pair of variables:",
                            options=[(row['Feature 1'], row['Feature 2']) for _, row in top_corr_df.iterrows()],
                            format_func=lambda x: f"{x[0]} vs {x[1]} (r = {corr_matrix.loc[x[0], x[1]]:.2f})"
                        )
                        
                        if scatter_pair:
                            # Import visualization utilities
                            from utils.visualizer import create_scatter_plot
                            
                            fig = create_scatter_plot(
                                df, 
                                scatter_pair[0], 
                                scatter_pair[1],
                                title=f"Scatter Plot: {scatter_pair[0]} vs {scatter_pair[1]} (r = {corr_matrix.loc[scatter_pair[0], scatter_pair[1]]:.2f})"
                            )
                            
                            st.pyplot(fig)
                            
                            # Add regression line option
                            if st.checkbox("Add regression line"):
                                from utils.visualizer import create_plotly_scatter
                                import plotly.express as px
                                
                                # Create scatter plot with regression line
                                fig = px.scatter(
                                    df,
                                    x=scatter_pair[0],
                                    y=scatter_pair[1],
                                    trendline="ols",
                                    title=f"Scatter Plot with Regression Line: {scatter_pair[0]} vs {scatter_pair[1]}"
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Display regression stats
                                import statsmodels.api as sm
                                
                                X = sm.add_constant(df[scatter_pair[0]])
                                y = df[scatter_pair[1]]
                                
                                model = sm.OLS(y, X).fit()
                                st.text(model.summary().tables[1].as_text())
                else:
                    st.warning("No correlation pairs found")
            else:
                st.warning("Please select at least two columns for correlation analysis")
        else:
            st.warning("Need at least two numeric columns to calculate correlation")
else:
    st.warning("⚠️ Please upload a data file first on the Home page")
    if st.button("Go to Home"):
        st.switch_page("app.py")
