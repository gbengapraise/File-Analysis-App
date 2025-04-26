import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.visualizer import (
    set_plot_style, create_histogram, create_boxplot, create_scatter_plot,
    create_bar_chart, create_pie_chart, create_correlation_heatmap, create_line_chart,
    create_plotly_histogram, create_plotly_scatter, create_plotly_bar,
    create_plotly_pie, create_plotly_line, create_plotly_heatmap, create_plotly_box
)

# Set page configuration
st.set_page_config(
    page_title="Data Visualization",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables if they don't exist
if 'data' not in st.session_state:
    st.session_state.data = None
if 'theme' not in st.session_state:
    st.session_state.theme = "light"

# Page title and description
st.title("📈 Data Visualization")
st.write("Create various visualizations to explore and understand your data")

# Check if data is loaded
if st.session_state.data is not None:
    df = st.session_state.data
    
    # Set plot style based on theme
    plot_style = "darkgrid" if st.session_state.theme == "dark" else "whitegrid"
    set_plot_style(plot_style)
    
    # Visualization options
    viz_types = [
        "Histogram", "Box Plot", "Scatter Plot", "Bar Chart",
        "Pie Chart", "Line Chart", "Correlation Heatmap", "Custom Plot"
    ]
    
    # Create two tabs - one for simple plots and one for advanced/custom plots
    viz_tabs = st.tabs(["Basic Visualization", "Advanced Visualization", "Interactive Visualization"])
    
    # Tab 1: Basic Visualization
    with viz_tabs[0]:
        st.header("Basic Visualization")
        
        # Select visualization type
        basic_viz_type = st.selectbox(
            "Select visualization type:",
            options=viz_types[:5],  # First 5 options
            key="basic_viz_type"
        )
        
        # Column selection and parameters based on visualization type
        if basic_viz_type == "Histogram":
            # Numeric columns for histogram
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if numeric_columns:
                col1, col2 = st.columns(2)
                
                with col1:
                    hist_column = st.selectbox("Select column for histogram:", options=numeric_columns)
                    bins = st.slider("Number of bins:", min_value=5, max_value=100, value=30)
                
                with col2:
                    kde = st.checkbox("Show density curve (KDE)", value=True)
                    hist_color = st.color_picker("Histogram color:", "#4CAF50")
                
                # Create and display the histogram
                st.subheader(f"Histogram of {hist_column}")
                
                if hist_column:
                    fig = create_histogram(
                        df, hist_column, bins=bins, kde=kde, 
                        title=f"Histogram of {hist_column}", color=hist_color
                    )
                    st.pyplot(fig)
                    
                    # Display basic statistics
                    st.write("**Basic Statistics:**")
                    stats = df[hist_column].describe()
                    st.write(f"Mean: {stats['mean']:.2f}, Median: {stats['50%']:.2f}, Std Dev: {stats['std']:.2f}")
                    st.write(f"Min: {stats['min']:.2f}, Max: {stats['max']:.2f}")
                    
                    # Skewness and kurtosis
                    skew = df[hist_column].skew()
                    kurtosis = df[hist_column].kurtosis()
                    st.write(f"Skewness: {skew:.2f}, Kurtosis: {kurtosis:.2f}")
            else:
                st.warning("No numeric columns found for histogram")
        
        elif basic_viz_type == "Box Plot":
            # Numeric columns for box plot
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if numeric_columns:
                col1, col2 = st.columns(2)
                
                with col1:
                    box_column = st.selectbox("Select column for box plot:", options=numeric_columns)
                
                with col2:
                    box_color = st.color_picker("Box plot color:", "#4CAF50")
                
                # Create and display the box plot
                st.subheader(f"Box Plot of {box_column}")
                
                if box_column:
                    fig = create_boxplot(
                        df, box_column, 
                        title=f"Box Plot of {box_column}", color=box_color
                    )
                    st.pyplot(fig)
                    
                    # Display outlier information
                    st.write("**Outlier Information:**")
                    
                    # Calculate IQR and outlier boundaries
                    q1 = df[box_column].quantile(0.25)
                    q3 = df[box_column].quantile(0.75)
                    iqr = q3 - q1
                    
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    
                    outliers = df[(df[box_column] < lower_bound) | (df[box_column] > upper_bound)][box_column]
                    
                    st.write(f"Q1 (25th percentile): {q1:.2f}")
                    st.write(f"Median (50th percentile): {df[box_column].median():.2f}")
                    st.write(f"Q3 (75th percentile): {q3:.2f}")
                    st.write(f"IQR (Q3-Q1): {iqr:.2f}")
                    st.write(f"Lower outlier boundary: {lower_bound:.2f}")
                    st.write(f"Upper outlier boundary: {upper_bound:.2f}")
                    st.write(f"Number of outliers: {len(outliers)}")
                    
                    if len(outliers) > 0 and len(outliers) <= 10:
                        st.write("Outlier values:", outliers.values)
            else:
                st.warning("No numeric columns found for box plot")
        
        elif basic_viz_type == "Scatter Plot":
            # Numeric columns for scatter plot
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_columns) >= 2:
                col1, col2 = st.columns(2)
                
                with col1:
                    x_column = st.selectbox("Select X-axis column:", options=numeric_columns, index=0)
                
                with col2:
                    y_column = st.selectbox("Select Y-axis column:", options=numeric_columns, index=min(1, len(numeric_columns)-1))
                
                # Optional color grouping
                categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
                hue_column = None
                
                if categorical_columns:
                    hue_column = st.selectbox(
                        "Color points by category (optional):",
                        options=["None"] + categorical_columns
                    )
                    
                    if hue_column == "None":
                        hue_column = None
                
                # Create and display the scatter plot
                st.subheader(f"Scatter Plot: {x_column} vs {y_column}")
                
                if x_column and y_column:
                    fig = create_scatter_plot(
                        df, x_column, y_column, hue=hue_column,
                        title=f"Scatter Plot: {x_column} vs {y_column}"
                    )
                    st.pyplot(fig)
                    
                    # Display correlation
                    correlation = df[[x_column, y_column]].corr().iloc[0, 1]
                    st.write(f"**Correlation between {x_column} and {y_column}:** {correlation:.4f}")
                    
                    # Add regression line option
                    if st.checkbox("Show regression line", key="scatter_regression"):
                        import plotly.express as px
                        
                        fig = px.scatter(
                            df, x=x_column, y=y_column, 
                            color=hue_column, trendline="ols",
                            title=f"Scatter Plot with Regression Line: {x_column} vs {y_column}"
                        )
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Need at least two numeric columns for scatter plot")
        
        elif basic_viz_type == "Bar Chart":
            # Get columns appropriate for bar charts
            categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if categorical_columns:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    x_column = st.selectbox("Select category column (X-axis):", options=categorical_columns)
                
                with col2:
                    # Option to use count or a numeric value
                    use_count = st.checkbox("Use count of categories", value=True)
                    
                    if use_count:
                        y_column = None
                    else:
                        y_column = st.selectbox("Select value column (Y-axis):", options=numeric_columns)
                
                with col3:
                    bar_color = st.color_picker("Bar color:", "#4CAF50")
                    orientation = st.radio(
                        "Orientation:",
                        options=["vertical", "horizontal"],
                        horizontal=True
                    )
                
                # Create and display the bar chart
                if x_column:
                    title = f"Count of {x_column}" if use_count else f"Bar Chart: {x_column} vs {y_column}"
                    st.subheader(title)
                    
                    # Check if too many categories
                    unique_cats = df[x_column].nunique()
                    if unique_cats > 15:
                        st.warning(f"Column {x_column} has {unique_cats} unique values. The chart may be crowded.")
                        
                        # Option to limit categories
                        limit_cats = st.checkbox("Limit to top categories", value=True)
                        if limit_cats:
                            top_n = st.slider("Number of top categories to show:", 5, 20, 10)
                            
                            # Get top categories
                            top_cats = df[x_column].value_counts().nlargest(top_n).index
                            chart_df = df[df[x_column].isin(top_cats)].copy()
                            chart_df[x_column] = chart_df[x_column].astype('category')
                        else:
                            chart_df = df
                    else:
                        chart_df = df
                    
                    fig = create_bar_chart(
                        chart_df, x_column, y_column, 
                        title=title, color=bar_color,
                        orientation=orientation
                    )
                    st.pyplot(fig)
            else:
                st.warning("No categorical columns found for bar chart")
        
        elif basic_viz_type == "Pie Chart":
            # Categorical columns for pie chart
            categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            if categorical_columns:
                col1, col2 = st.columns(2)
                
                with col1:
                    pie_column = st.selectbox("Select category column for pie chart:", options=categorical_columns)
                
                with col2:
                    custom_colors = st.checkbox("Use custom color palette", value=False)
                    if custom_colors:
                        palette = st.selectbox(
                            "Select color palette:",
                            options=["viridis", "plasma", "inferno", "magma", "cividis", "Set1", "Set2", "Set3", "tab10"]
                        )
                    else:
                        palette = None
                
                # Create and display the pie chart
                if pie_column:
                    # Check if too many categories
                    unique_cats = df[pie_column].nunique()
                    
                    if unique_cats > 10:
                        st.warning(f"Column {pie_column} has {unique_cats} unique values. The chart may be cluttered.")
                        st.info("The pie chart will show the top 10 categories and group the rest as 'Others'.")
                    
                    fig = create_pie_chart(
                        df, pie_column, 
                        title=f"Distribution of {pie_column}",
                        colors=palette
                    )
                    st.pyplot(fig)
                    
                    # Display frequency table
                    st.subheader("Category Distribution")
                    value_counts = df[pie_column].value_counts()
                    percent = (df[pie_column].value_counts(normalize=True) * 100).round(2)
                    
                    dist_df = pd.DataFrame({
                        'Count': value_counts,
                        'Percentage': percent
                    })
                    
                    st.dataframe(dist_df, use_container_width=True)
            else:
                st.warning("No categorical columns found for pie chart")
    
    # Tab 2: Advanced Visualization
    with viz_tabs[1]:
        st.header("Advanced Visualization")
        
        # Select advanced visualization type
        advanced_viz_type = st.selectbox(
            "Select visualization type:",
            options=viz_types[5:] + ["Grouped Bar Chart", "Faceted Charts"],
            key="advanced_viz_type"
        )
        
        if advanced_viz_type == "Line Chart":
            # Get columns for line chart
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            all_columns = df.columns.tolist()
            
            if numeric_columns:
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    # Select x-axis column (can be any type)
                    x_column = st.selectbox("Select X-axis column:", options=all_columns)
                    
                    # Multiple y-axis columns (must be numeric)
                    y_columns = st.multiselect(
                        "Select Y-axis column(s):",
                        options=numeric_columns,
                        default=[numeric_columns[0]] if numeric_columns else None
                    )
                
                # Create and display the line chart
                if x_column and y_columns:
                    st.subheader(f"Line Chart with {x_column} on X-axis")
                    
                    # Sort data by x-column if it's a datetime or numeric
                    try:
                        if pd.api.types.is_datetime64_any_dtype(df[x_column]) or pd.api.types.is_numeric_dtype(df[x_column]):
                            line_df = df.sort_values(by=x_column)
                        else:
                            line_df = df
                    except:
                        line_df = df
                    
                    fig = create_line_chart(
                        line_df, x_column, y_columns,
                        title=f"Line Chart: {', '.join(y_columns)} by {x_column}"
                    )
                    st.pyplot(fig)
                    
                    # Option for interactive plotly version
                    if st.checkbox("Show interactive version", key="line_interactive"):
                        plotly_fig = create_plotly_line(
                            line_df, x_column, y_columns,
                            title=f"Line Chart: {', '.join(y_columns)} by {x_column}"
                        )
                        st.plotly_chart(plotly_fig, use_container_width=True)
            else:
                st.warning("No numeric columns found for line chart Y-axis")
        
        elif advanced_viz_type == "Correlation Heatmap":
            # Numeric columns for correlation heatmap
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_columns) >= 2:
                col1, col2 = st.columns(2)
                
                with col1:
                    selected_columns = st.multiselect(
                        "Select columns for correlation analysis:",
                        options=numeric_columns,
                        default=numeric_columns[:min(8, len(numeric_columns))]
                    )
                
                with col2:
                    corr_method = st.selectbox(
                        "Correlation method:",
                        options=["pearson", "spearman", "kendall"],
                        help="Pearson: linear correlation, Spearman: rank correlation, Kendall: ordinal association"
                    )
                    
                    cmap = st.selectbox(
                        "Color map:",
                        options=["viridis", "plasma", "inferno", "magma", "cividis", "coolwarm", "RdBu", "RdYlGn"]
                    )
                
                # Create and display the heatmap
                if selected_columns and len(selected_columns) >= 2:
                    st.subheader(f"{corr_method.capitalize()} Correlation Heatmap")
                    
                    fig = create_correlation_heatmap(
                        df, selected_columns, method=corr_method,
                        title=f"{corr_method.capitalize()} Correlation Heatmap",
                        cmap=cmap
                    )
                    st.pyplot(fig)
                    
                    # Display correlation matrix as a table
                    corr_matrix = df[selected_columns].corr(method=corr_method).round(2)
                    st.subheader("Correlation Matrix")
                    st.dataframe(corr_matrix.style.background_gradient(cmap=cmap), use_container_width=True)
                    
                    # Option for interactive plotly version
                    if st.checkbox("Show interactive version", key="heatmap_interactive"):
                        plotly_fig = create_plotly_heatmap(
                            df, selected_columns, method=corr_method,
                            title=f"{corr_method.capitalize()} Correlation Heatmap"
                        )
                        st.plotly_chart(plotly_fig, use_container_width=True)
            else:
                st.warning("Need at least two numeric columns for correlation heatmap")
        
        elif advanced_viz_type == "Grouped Bar Chart":
            # Get columns for grouped bar chart
            categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(categorical_columns) >= 2 and numeric_columns:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    x_column = st.selectbox("Select primary category (X-axis):", options=categorical_columns, index=0)
                
                with col2:
                    group_column = st.selectbox(
                        "Select grouping category:", 
                        options=[col for col in categorical_columns if col != x_column],
                        index=0 if len(categorical_columns) > 1 else None
                    )
                
                with col3:
                    use_count = st.checkbox("Use count of categories", value=True)
                    
                    if not use_count:
                        y_column = st.selectbox("Select value column (Y-axis):", options=numeric_columns)
                    else:
                        y_column = None
                
                # Create and display the grouped bar chart
                if x_column and group_column:
                    import seaborn as sns
                    
                    st.subheader(f"Grouped Bar Chart: {x_column} by {group_column}")
                    
                    # Check if too many categories
                    x_unique = df[x_column].nunique()
                    group_unique = df[group_column].nunique()
                    
                    if x_unique > 10 or group_unique > 5:
                        st.warning(
                            f"Many unique values detected: {x_column} has {x_unique} values and "
                            f"{group_column} has {group_unique} values. Chart may be crowded."
                        )
                        
                        # Option to limit categories
                        limit_cats = st.checkbox("Limit to top categories", value=True)
                        if limit_cats:
                            x_top_n = st.slider(f"Top values of {x_column}:", 3, 15, min(5, x_unique))
                            group_top_n = st.slider(f"Top values of {group_column}:", 2, 10, min(3, group_unique))
                            
                            # Get top categories for each column
                            x_top_cats = df[x_column].value_counts().nlargest(x_top_n).index
                            group_top_cats = df[group_column].value_counts().nlargest(group_top_n).index
                            
                            chart_df = df[
                                df[x_column].isin(x_top_cats) & 
                                df[group_column].isin(group_top_cats)
                            ].copy()
                        else:
                            chart_df = df
                    else:
                        chart_df = df
                    
                    # Create the figure
                    fig, ax = plt.subplots(figsize=(12, 8))
                    
                    if use_count:
                        # Count plot with grouping
                        grouped_data = pd.crosstab(chart_df[x_column], chart_df[group_column])
                        grouped_data.plot(kind='bar', ax=ax)
                    else:
                        # Value plot with grouping
                        sns.barplot(
                            data=chart_df,
                            x=x_column,
                            y=y_column,
                            hue=group_column,
                            ax=ax
                        )
                    
                    # Customize plot
                    ax.set_title(f"Grouped Bar Chart: {x_column} by {group_column}")
                    ax.set_xlabel(x_column)
                    
                    if use_count:
                        ax.set_ylabel("Count")
                    else:
                        ax.set_ylabel(y_column)
                    
                    # Rotate x-axis labels if there are many categories
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    
                    st.pyplot(fig)
                    
                    # Option for interactive plotly version
                    if st.checkbox("Show interactive version", key="grouped_bar_interactive"):
                        import plotly.express as px
                        
                        if use_count:
                            # For count-based plot
                            plotly_fig = px.histogram(
                                chart_df,
                                x=x_column,
                                color=group_column,
                                barmode="group",
                                title=f"Grouped Bar Chart: {x_column} by {group_column}"
                            )
                        else:
                            # For value-based plot
                            plotly_fig = px.bar(
                                chart_df,
                                x=x_column,
                                y=y_column,
                                color=group_column,
                                barmode="group",
                                title=f"Grouped Bar Chart: {x_column} by {group_column}"
                            )
                        
                        st.plotly_chart(plotly_fig, use_container_width=True)
            else:
                st.warning("Need at least two categorical columns and one numeric column for grouped bar chart")
        
        elif advanced_viz_type == "Faceted Charts":
            # Get columns for faceted charts
            categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if categorical_columns and len(numeric_columns) >= 1:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    facet_column = st.selectbox(
                        "Select column for faceting:",
                        options=categorical_columns
                    )
                    
                    chart_type = st.selectbox(
                        "Select chart type:",
                        options=["histogram", "kde", "scatter"]
                    )
                
                with col2:
                    if chart_type in ["histogram", "kde"]:
                        x_column = st.selectbox("Select data column:", options=numeric_columns)
                        y_column = None
                    else:  # scatter
                        x_column = st.selectbox("Select X-axis column:", options=numeric_columns, index=0)
                        y_column = st.selectbox(
                            "Select Y-axis column:", 
                            options=[col for col in numeric_columns if col != x_column],
                            index=0 if len(numeric_columns) > 1 else None
                        )
                
                with col3:
                    # Layout settings
                    max_cats = df[facet_column].nunique()
                    n_cols = st.slider("Number of columns in layout:", 1, 4, min(2, max_cats))
                    
                    # Color settings
                    color = st.color_picker("Chart color:", "#4CAF50")
                
                # Create and display the faceted chart
                if facet_column and x_column:
                    import seaborn as sns
                    
                    # Check if there are too many categories
                    if max_cats > 12:
                        st.warning(f"Column {facet_column} has {max_cats} unique values. The chart may be cluttered.")
                        st.info("Limiting to top 12 categories by frequency.")
                        
                        # Limit to top categories
                        top_cats = df[facet_column].value_counts().nlargest(12).index
                        chart_df = df[df[facet_column].isin(top_cats)].copy()
                    else:
                        chart_df = df
                    
                    # Calculate grid dimensions
                    n_cats = min(max_cats, 12)
                    n_rows = (n_cats + n_cols - 1) // n_cols  # Ceiling division
                    
                    # Create FacetGrid
                    g = sns.FacetGrid(
                        chart_df,
                        col=facet_column,
                        col_wrap=n_cols,
                        height=3,
                        sharey=True
                    )
                    
                    # Apply the chart type
                    if chart_type == "histogram":
                        g.map_dataframe(sns.histplot, x=x_column, color=color)
                        title = f"Histogram of {x_column} by {facet_column}"
                    elif chart_type == "kde":
                        g.map_dataframe(sns.kdeplot, x=x_column, color=color, fill=True)
                        title = f"Density Plot of {x_column} by {facet_column}"
                    else:  # scatter
                        g.map_dataframe(sns.scatterplot, x=x_column, y=y_column, color=color)
                        title = f"Scatter Plot of {x_column} vs {y_column} by {facet_column}"
                    
                    # Add title
                    g.fig.suptitle(title, y=1.02)
                    plt.tight_layout()
                    
                    st.pyplot(g.fig)
                    
                    # Option for interactive plotly version
                    if st.checkbox("Show interactive version", key="facet_interactive"):
                        import plotly.express as px
                        
                        if chart_type == "histogram":
                            plotly_fig = px.histogram(
                                chart_df,
                                x=x_column,
                                facet_col=facet_column,
                                facet_col_wrap=n_cols,
                                color_discrete_sequence=[color],
                                title=title
                            )
                        elif chart_type == "kde":
                            # Plotly doesn't have direct KDE, use histogram with density
                            plotly_fig = px.histogram(
                                chart_df,
                                x=x_column,
                                facet_col=facet_column,
                                facet_col_wrap=n_cols,
                                color_discrete_sequence=[color],
                                histnorm="probability density",
                                title=title
                            )
                        else:  # scatter
                            plotly_fig = px.scatter(
                                chart_df,
                                x=x_column,
                                y=y_column,
                                facet_col=facet_column,
                                facet_col_wrap=n_cols,
                                color_discrete_sequence=[color],
                                title=title
                            )
                        
                        st.plotly_chart(plotly_fig, use_container_width=True)
            else:
                st.warning("Need at least one categorical column and one numeric column for faceted charts")
        
        elif advanced_viz_type == "Custom Plot":
            st.subheader("Create a Custom Visualization")
            
            # Get available columns by type
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
            datetime_columns = df.select_dtypes(include=['datetime64']).columns.tolist()
            
            # User selects what type of plot to create
            plot_lib = st.radio(
                "Select plotting library:",
                options=["Matplotlib/Seaborn", "Plotly"],
                horizontal=True
            )
            
            custom_plot_type = st.selectbox(
                "Select plot type:",
                options=[
                    "Scatter", "Line", "Bar", "Histogram", "Box", "Violin",
                    "Heatmap", "Pair Plot", "Joint Plot"
                ]
            )
            
            # Configure the plot based on the type
            if custom_plot_type in ["Scatter", "Line", "Joint Plot"]:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    x_column = st.selectbox("X-axis:", options=numeric_columns + datetime_columns, key="custom_x")
                
                with col2:
                    y_column = st.selectbox("Y-axis:", options=numeric_columns, key="custom_y")
                
                with col3:
                    if categorical_columns:
                        color_by = st.selectbox(
                            "Color by (optional):",
                            options=["None"] + categorical_columns,
                            key="custom_color"
                        )
                        if color_by == "None":
                            color_by = None
                    else:
                        color_by = None
                
                # Create the plot
                if plot_lib == "Matplotlib/Seaborn":
                    import seaborn as sns
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    if custom_plot_type == "Scatter":
                        sns.scatterplot(data=df, x=x_column, y=y_column, hue=color_by, ax=ax)
                        title = f"Scatter Plot: {x_column} vs {y_column}"
                    elif custom_plot_type == "Line":
                        sns.lineplot(data=df, x=x_column, y=y_column, hue=color_by, ax=ax)
                        title = f"Line Plot: {x_column} vs {y_column}"
                    elif custom_plot_type == "Joint Plot":
                        # Joint plot is different, needs its own figure
                        st.write("**Joint Plot**")
                        fig = plt.figure(figsize=(10, 10))
                        g = sns.jointplot(
                            data=df, x=x_column, y=y_column, hue=color_by,
                            height=8, ratio=3, kind='scatter'
                        )
                        title = f"Joint Plot: {x_column} vs {y_column}"
                        st.pyplot(g.fig)
                        # Skip the regular plotting code
                        plt.close(fig)
                        fig = None
                    
                    if fig:
                        ax.set_title(title)
                        plt.tight_layout()
                        st.pyplot(fig)
                
                else:  # Plotly
                    import plotly.express as px
                    
                    if custom_plot_type == "Scatter":
                        fig = px.scatter(df, x=x_column, y=y_column, color=color_by)
                    elif custom_plot_type == "Line":
                        fig = px.line(df, x=x_column, y=y_column, color=color_by)
                    elif custom_plot_type == "Joint Plot":
                        # Joint plot in Plotly is similar to a scatter with marginals
                        fig = px.scatter(
                            df, x=x_column, y=y_column, color=color_by,
                            marginal_x="histogram", marginal_y="histogram"
                        )
                    
                    fig.update_layout(title=f"{custom_plot_type}: {x_column} vs {y_column}")
                    st.plotly_chart(fig, use_container_width=True)
            
            elif custom_plot_type in ["Bar", "Box", "Violin"]:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if categorical_columns:
                        x_column = st.selectbox("X-axis (categorical):", options=categorical_columns, key="custom_cat_x")
                    else:
                        st.warning("No categorical columns available")
                        x_column = None
                
                with col2:
                    y_column = st.selectbox("Y-axis (numeric):", options=numeric_columns, key="custom_num_y")
                
                with col3:
                    if categorical_columns and len(categorical_columns) > 1:
                        group_by = st.selectbox(
                            "Group by (optional):",
                            options=["None"] + [col for col in categorical_columns if col != x_column],
                            key="custom_group"
                        )
                        if group_by == "None":
                            group_by = None
                    else:
                        group_by = None
                
                # Create the plot
                if x_column and y_column:
                    if plot_lib == "Matplotlib/Seaborn":
                        import seaborn as sns
                        
                        fig, ax = plt.subplots(figsize=(12, 6))
                        
                        if custom_plot_type == "Bar":
                            sns.barplot(data=df, x=x_column, y=y_column, hue=group_by, ax=ax)
                            title = f"Bar Plot: {y_column} by {x_column}"
                        elif custom_plot_type == "Box":
                            sns.boxplot(data=df, x=x_column, y=y_column, hue=group_by, ax=ax)
                            title = f"Box Plot: {y_column} by {x_column}"
                        elif custom_plot_type == "Violin":
                            sns.violinplot(data=df, x=x_column, y=y_column, hue=group_by, ax=ax)
                            title = f"Violin Plot: {y_column} by {x_column}"
                        
                        ax.set_title(title)
                        plt.xticks(rotation=45, ha='right')
                        plt.tight_layout()
                        st.pyplot(fig)
                    
                    else:  # Plotly
                        import plotly.express as px
                        
                        if custom_plot_type == "Bar":
                            fig = px.bar(df, x=x_column, y=y_column, color=group_by)
                        elif custom_plot_type == "Box":
                            fig = px.box(df, x=x_column, y=y_column, color=group_by)
                        elif custom_plot_type == "Violin":
                            fig = px.violin(df, x=x_column, y=y_column, color=group_by)
                        
                        fig.update_layout(title=f"{custom_plot_type}: {y_column} by {x_column}")
                        st.plotly_chart(fig, use_container_width=True)
            
            elif custom_plot_type == "Histogram":
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    hist_column = st.selectbox("Select column:", options=numeric_columns, key="custom_hist")
                
                with col2:
                    bins = st.slider("Number of bins:", 5, 100, 30, key="custom_bins")
                
                with col3:
                    if categorical_columns:
                        color_by = st.selectbox(
                            "Color by (optional):",
                            options=["None"] + categorical_columns,
                            key="custom_hist_color"
                        )
                        if color_by == "None":
                            color_by = None
                    else:
                        color_by = None
                
                # Create the plot
                if hist_column:
                    if plot_lib == "Matplotlib/Seaborn":
                        import seaborn as sns
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        
                        if color_by:
                            for category in df[color_by].unique():
                                subset = df[df[color_by] == category]
                                sns.histplot(data=subset, x=hist_column, bins=bins, alpha=0.5, label=category, ax=ax)
                            plt.legend()
                        else:
                            sns.histplot(data=df, x=hist_column, bins=bins, ax=ax)
                        
                        ax.set_title(f"Histogram of {hist_column}")
                        plt.tight_layout()
                        st.pyplot(fig)
                    
                    else:  # Plotly
                        import plotly.express as px
                        
                        fig = px.histogram(
                            df, x=hist_column, color=color_by,
                            nbins=bins, marginal="box"
                        )
                        
                        fig.update_layout(title=f"Histogram of {hist_column}")
                        st.plotly_chart(fig, use_container_width=True)
            
            elif custom_plot_type == "Heatmap":
                numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
                
                if len(numeric_columns) >= 2:
                    st.write("**Correlation Heatmap Settings**")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        corr_columns = st.multiselect(
                            "Select columns for correlation:",
                            options=numeric_columns,
                            default=numeric_columns[:min(8, len(numeric_columns))],
                            key="custom_heatmap_cols"
                        )
                    
                    with col2:
                        corr_method = st.selectbox(
                            "Correlation method:",
                            options=["pearson", "spearman", "kendall"],
                            key="custom_heatmap_method"
                        )
                        
                        if plot_lib == "Matplotlib/Seaborn":
                            cmap = st.selectbox(
                                "Color map:",
                                options=["viridis", "plasma", "inferno", "magma", "cividis", "coolwarm", "RdBu", "RdYlGn"],
                                key="custom_heatmap_cmap"
                            )
                    
                    # Create the plot
                    if corr_columns and len(corr_columns) >= 2:
                        # Calculate correlation matrix
                        corr_matrix = df[corr_columns].corr(method=corr_method)
                        
                        if plot_lib == "Matplotlib/Seaborn":
                            import seaborn as sns
                            
                            fig, ax = plt.subplots(figsize=(10, 8))
                            
                            sns.heatmap(
                                corr_matrix,
                                annot=True,
                                cmap=cmap,
                                fmt=".2f",
                                linewidths=0.5,
                                square=True,
                                ax=ax
                            )
                            
                            ax.set_title(f"{corr_method.capitalize()} Correlation Heatmap")
                            plt.tight_layout()
                            st.pyplot(fig)
                        
                        else:  # Plotly
                            import plotly.express as px
                            
                            fig = px.imshow(
                                corr_matrix,
                                text_auto=True,
                                aspect="auto",
                                color_continuous_scale="RdBu_r"
                            )
                            
                            fig.update_layout(title=f"{corr_method.capitalize()} Correlation Heatmap")
                            st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Need at least two numeric columns for a heatmap")
            
            elif custom_plot_type == "Pair Plot":
                numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
                
                if len(numeric_columns) >= 2:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        pair_columns = st.multiselect(
                            "Select columns for pair plot:",
                            options=numeric_columns,
                            default=numeric_columns[:min(4, len(numeric_columns))],
                            key="custom_pair_cols"
                        )
                    
                    with col2:
                        if categorical_columns:
                            hue_column = st.selectbox(
                                "Color by (optional):",
                                options=["None"] + categorical_columns,
                                key="custom_pair_hue"
                            )
                            if hue_column == "None":
                                hue_column = None
                        else:
                            hue_column = None
                    
                    # Create the plot
                    if pair_columns and len(pair_columns) >= 2:
                        if plot_lib == "Matplotlib/Seaborn":
                            import seaborn as sns
                            
                            st.write("**Pair Plot**")
                            
                            # Warning for large datasets
                            if len(df) > 1000:
                                st.warning(f"Dataset has {len(df)} rows. Pair plot may be slow. Showing a sample.")
                                plot_df = df.sample(1000, random_state=42)
                            else:
                                plot_df = df
                            
                            # Create the pair plot
                            g = sns.pairplot(
                                plot_df[pair_columns + ([hue_column] if hue_column else [])],
                                hue=hue_column,
                                height=2.5
                            )
                            
                            g.fig.suptitle("Pair Plot", y=1.02)
                            plt.tight_layout()
                            st.pyplot(g.fig)
                        
                        else:  # Plotly
                            import plotly.express as px
                            
                            # Warning for large datasets
                            if len(df) > 1000:
                                st.warning(f"Dataset has {len(df)} rows. Pair plot may be slow. Showing a sample.")
                                plot_df = df.sample(1000, random_state=42)
                            else:
                                plot_df = df
                            
                            fig = px.scatter_matrix(
                                plot_df,
                                dimensions=pair_columns,
                                color=hue_column
                            )
                            
                            fig.update_layout(title="Pair Plot")
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Select at least two columns for pair plot")
                else:
                    st.warning("Need at least two numeric columns for a pair plot")
    
    # Tab 3: Interactive Visualization
    with viz_tabs[2]:
        st.header("Interactive Visualization")
        
        # Select interactive visualization type
        interactive_viz_type = st.selectbox(
            "Select visualization type:",
            options=[
                "Interactive Scatter", "Interactive Line", "Interactive Bar",
                "Interactive Histogram", "Interactive Box Plot", "Interactive Heatmap"
            ],
            key="interactive_viz_type"
        )
        
        if interactive_viz_type == "Interactive Scatter":
            # Get columns for scatter plot
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            if len(numeric_columns) >= 2:
                col1, col2 = st.columns(2)
                
                with col1:
                    x_column = st.selectbox("X-axis:", options=numeric_columns, key="i_scatter_x")
                    y_column = st.selectbox("Y-axis:", options=numeric_columns, key="i_scatter_y")
                
                with col2:
                    color_by = None
                    size_by = None
                    
                    if categorical_columns:
                        color_by = st.selectbox(
                            "Color by (optional):",
                            options=["None"] + categorical_columns,
                            key="i_scatter_color"
                        )
                        if color_by == "None":
                            color_by = None
                    
                    if numeric_columns:
                        size_by = st.selectbox(
                            "Size by (optional):",
                            options=["None"] + numeric_columns,
                            key="i_scatter_size"
                        )
                        if size_by == "None":
                            size_by = None
                
                # Create the plot
                if x_column and y_column:
                    fig = create_plotly_scatter(
                        df, x_column, y_column, color=color_by, size=size_by,
                        title=f"Interactive Scatter Plot: {x_column} vs {y_column}"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show trend line option
                    if st.checkbox("Add trend line", key="i_scatter_trend"):
                        import plotly.express as px
                        
                        trend_fig = px.scatter(
                            df, x=x_column, y=y_column, color=color_by, size=size_by,
                            trendline="ols",
                            title=f"Scatter Plot with Trend Line: {x_column} vs {y_column}"
                        )
                        
                        st.plotly_chart(trend_fig, use_container_width=True)
                        
                        # Show regression details
                        if st.checkbox("Show regression details", key="i_scatter_reg_details"):
                            import statsmodels.api as sm
                            
                            X = sm.add_constant(df[x_column])
                            y = df[y_column]
                            model = sm.OLS(y, X).fit()
                            
                            st.text(model.summary().tables[1].as_text())
            else:
                st.warning("Need at least two numeric columns for a scatter plot")
        
        elif interactive_viz_type == "Interactive Line":
            # Get columns for line chart
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            all_columns = df.columns.tolist()
            categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            if numeric_columns:
                col1, col2 = st.columns(2)
                
                with col1:
                    # X-axis can be any type, but commonly datetime or numeric
                    x_column = st.selectbox("X-axis:", options=all_columns, key="i_line_x")
                    
                    # Multiple y-axis columns (must be numeric)
                    y_columns = st.multiselect(
                        "Y-axis column(s):",
                        options=numeric_columns,
                        default=[numeric_columns[0]] if numeric_columns else None,
                        key="i_line_y"
                    )
                
                with col2:
                    if categorical_columns:
                        color_by = st.selectbox(
                            "Group by (optional):",
                            options=["None"] + categorical_columns,
                            key="i_line_color"
                        )
                        if color_by == "None":
                            color_by = None
                    else:
                        color_by = None
                
                # Create the plot
                if x_column and y_columns:
                    # Sort by x column if it's a datetime or numeric for proper line connection
                    try:
                        if pd.api.types.is_datetime64_any_dtype(df[x_column]) or pd.api.types.is_numeric_dtype(df[x_column]):
                            line_df = df.sort_values(by=x_column).copy()
                        else:
                            line_df = df.copy()
                    except:
                        line_df = df.copy()
                    
                    # Create plot
                    if color_by:
                        # If using color grouping, can only use one y column at a time
                        y_col = y_columns[0] if y_columns else None
                        
                        if y_col:
                            import plotly.express as px
                            
                            fig = px.line(
                                line_df,
                                x=x_column,
                                y=y_col,
                                color=color_by,
                                title=f"Line Chart: {y_col} by {x_column}, grouped by {color_by}"
                            )
                            
                            # Add markers for better visibility
                            fig.update_traces(mode="lines+markers")
                            
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        # Multiple lines without grouping
                        fig = create_plotly_line(
                            line_df, x_column, y_columns,
                            title=f"Line Chart: {', '.join(y_columns)} by {x_column}"
                        )
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Need at least one numeric column for the y-axis")
        
        elif interactive_viz_type == "Interactive Bar":
            # Get columns for bar chart
            categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if categorical_columns:
                col1, col2 = st.columns(2)
                
                with col1:
                    x_column = st.selectbox("X-axis (category):", options=categorical_columns, key="i_bar_x")
                    
                    use_count = st.checkbox("Use count of categories", value=True, key="i_bar_count")
                    if not use_count and numeric_columns:
                        y_column = st.selectbox("Y-axis (value):", options=numeric_columns, key="i_bar_y")
                    else:
                        y_column = None
                
                with col2:
                    if categorical_columns and len(categorical_columns) > 1:
                        color_by = st.selectbox(
                            "Group by (optional):",
                            options=["None"] + [col for col in categorical_columns if col != x_column],
                            key="i_bar_color"
                        )
                        if color_by == "None":
                            color_by = None
                    else:
                        color_by = None
                    
                    if color_by:
                        bar_mode = st.selectbox(
                            "Bar mode:",
                            options=["group", "stack", "overlay"],
                            key="i_bar_mode"
                        )
                    else:
                        bar_mode = "group"
                
                # Create the plot
                if x_column:
                    # Check if too many categories
                    n_categories = df[x_column].nunique()
                    if n_categories > 15:
                        st.warning(f"Column {x_column} has {n_categories} unique values. Chart may be crowded.")
                        limit_cats = st.checkbox("Limit to top categories", value=True, key="i_bar_limit")
                        if limit_cats:
                            top_n = st.slider("Number of top categories:", 5, 20, 10, key="i_bar_top_n")
                            top_cats = df[x_column].value_counts().nlargest(top_n).index
                            bar_df = df[df[x_column].isin(top_cats)].copy()
                        else:
                            bar_df = df
                    else:
                        bar_df = df
                    
                    # Create the plot
                    if use_count:
                        # Count-based bar chart
                        import plotly.express as px
                        
                        fig = px.histogram(
                            bar_df,
                            x=x_column,
                            color=color_by,
                            title=f"Count of {x_column}" + (f" by {color_by}" if color_by else ""),
                            barmode=bar_mode
                        )
                    else:
                        # Value-based bar chart
                        fig = create_plotly_bar(
                            bar_df, x_column, y_column, color=color_by,
                            title=f"Bar Chart of {y_column} by {x_column}" + (f", grouped by {color_by}" if color_by else "")
                        )
                        
                        # Update barmode
                        fig.update_layout(barmode=bar_mode)
                    
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Need at least one categorical column for x-axis")
        
        elif interactive_viz_type == "Interactive Histogram":
            # Get columns for histogram
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            if numeric_columns:
                col1, col2 = st.columns(2)
                
                with col1:
                    hist_column = st.selectbox("Select column:", options=numeric_columns, key="i_hist_col")
                    bins = st.slider("Number of bins:", 5, 100, 30, key="i_hist_bins")
                
                with col2:
                    if categorical_columns:
                        color_by = st.selectbox(
                            "Group by (optional):",
                            options=["None"] + categorical_columns,
                            key="i_hist_color"
                        )
                        if color_by == "None":
                            color_by = None
                    else:
                        color_by = None
                    
                    if color_by:
                        hist_mode = st.selectbox(
                            "Histogram mode:",
                            options=["group", "overlay", "stack"],
                            key="i_hist_mode"
                        )
                    else:
                        hist_mode = "overlay"
                
                # Create the plot
                if hist_column:
                    import plotly.express as px
                    
                    # Create histogram with optional grouping
                    fig = px.histogram(
                        df,
                        x=hist_column,
                        color=color_by,
                        nbins=bins,
                        barmode=hist_mode,
                        marginal="box",
                        title=f"Histogram of {hist_column}" + (f" by {color_by}" if color_by else "")
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show distribution statistics
                    if st.checkbox("Show distribution statistics", key="i_hist_stats"):
                        stats = df[hist_column].describe()
                        
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Mean", f"{stats['mean']:.2f}")
                        col2.metric("Median", f"{stats['50%']:.2f}")
                        col3.metric("Std Dev", f"{stats['std']:.2f}")
                        col4.metric("IQR", f"{stats['75%'] - stats['25%']:.2f}")
                        
                        # Additional stats
                        skew = df[hist_column].skew()
                        kurtosis = df[hist_column].kurtosis()
                        
                        col1, col2 = st.columns(2)
                        col1.metric("Skewness", f"{skew:.2f}")
                        col2.metric("Kurtosis", f"{kurtosis:.2f}")
            else:
                st.warning("Need at least one numeric column for histogram")
        
        elif interactive_viz_type == "Interactive Box Plot":
            # Get columns for box plot
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            col1, col2 = st.columns(2)
            
            with col1:
                y_column = st.selectbox("Y-axis (values):", options=numeric_columns, key="i_box_y")
                
                if categorical_columns:
                    x_column = st.selectbox(
                        "X-axis (grouping, optional):",
                        options=["None"] + categorical_columns,
                        key="i_box_x"
                    )
                    if x_column == "None":
                        x_column = None
                else:
                    x_column = None
            
            with col2:
                if categorical_columns and x_column and len(categorical_columns) > 1:
                    color_by = st.selectbox(
                        "Color by (optional):",
                        options=["None"] + [col for col in categorical_columns if col != x_column],
                        key="i_box_color"
                    )
                    if color_by == "None":
                        color_by = None
                else:
                    color_by = None
                
                # Option to show points or not
                show_points = st.checkbox("Show all data points", value=True, key="i_box_points")
            
            # Create the plot
            if y_column:
                fig = create_plotly_box(
                    df, x_column, y_column, color=color_by,
                    title=f"Box Plot of {y_column}" + (f" by {x_column}" if x_column else "")
                )
                
                # Update to show points or not
                if not show_points:
                    fig.update_traces(boxpoints=False)
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Need at least one numeric column for y-axis")
        
        elif interactive_viz_type == "Interactive Heatmap":
            # Get columns for heatmap
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_columns) >= 2:
                col1, col2 = st.columns(2)
                
                with col1:
                    selected_columns = st.multiselect(
                        "Select columns for correlation:",
                        options=numeric_columns,
                        default=numeric_columns[:min(8, len(numeric_columns))],
                        key="i_heatmap_cols"
                    )
                
                with col2:
                    corr_method = st.selectbox(
                        "Correlation method:",
                        options=["pearson", "spearman", "kendall"],
                        key="i_heatmap_method"
                    )
                    
                    color_scale = st.selectbox(
                        "Color scale:",
                        options=["RdBu_r", "Viridis", "Plasma", "Inferno", "Magma", "Cividis"],
                        key="i_heatmap_colors"
                    )
                
                # Create the plot
                if selected_columns and len(selected_columns) >= 2:
                    fig = create_plotly_heatmap(
                        df, selected_columns, method=corr_method,
                        title=f"{corr_method.capitalize()} Correlation Heatmap"
                    )
                    
                    # Update color scale
                    fig.update_traces(colorscale=color_scale)
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show correlation matrix as a table
                    if st.checkbox("Show correlation matrix as table", key="i_heatmap_table"):
                        corr_matrix = df[selected_columns].corr(method=corr_method).round(2)
                        st.dataframe(corr_matrix.style.background_gradient(cmap='RdBu_r'), use_container_width=True)
            else:
                st.warning("Need at least two numeric columns for a heatmap")
    
    # Download visualization
    st.divider()
    st.header("Export Visualizations")
    
    # Option to download visualizations
    download_format = st.selectbox(
        "Select download format:",
        options=["PNG", "PDF", "HTML (Interactive)"],
        index=0
    )
    
    # Note: The actual download functionality would require capturing the figures
    # into files, which is typically handled with matplotlib.savefig() for static
    # plots and plotly's write_html for interactive plots.
    st.info(
        "To download visualizations, capture screenshots or save individual plots "
        "from the interactive visualizations using the camera icon in the top-right "
        "corner of Plotly charts."
    )
else:
    st.warning("⚠️ Please upload a data file first on the Home page")
    if st.button("Go to Home"):
        st.switch_page("app.py")
