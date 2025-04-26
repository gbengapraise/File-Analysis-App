import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def set_plot_style(plot_style="darkgrid"):
    """Set the style for matplotlib/seaborn plots"""
    sns.set_style(plot_style)
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["font.size"] = 12

def create_histogram(df, column, bins=30, kde=True, title=None, color="#4CAF50"):
    """
    Create a histogram for a numerical column
    
    Parameters:
    - df: pandas DataFrame
    - column: str, the column to visualize
    - bins: int, number of bins
    - kde: bool, whether to show density curve
    - title: str, plot title
    - color: str, color for the histogram
    
    Returns:
    - matplotlib figure
    """
    if df is None or column not in df.columns:
        st.error(f"Column {column} not found in the dataframe")
        return None
    
    fig, ax = plt.subplots()
    sns.histplot(data=df, x=column, bins=bins, kde=kde, color=color, ax=ax)
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"Histogram of {column}")
    
    ax.set_xlabel(column)
    ax.set_ylabel("Frequency")
    
    plt.tight_layout()
    return fig

def create_boxplot(df, column, title=None, color="#4CAF50"):
    """
    Create a boxplot for a numerical column
    
    Parameters:
    - df: pandas DataFrame
    - column: str, the column to visualize
    - title: str, plot title
    - color: str, color for the boxplot
    
    Returns:
    - matplotlib figure
    """
    if df is None or column not in df.columns:
        st.error(f"Column {column} not found in the dataframe")
        return None
    
    fig, ax = plt.subplots()
    sns.boxplot(data=df, x=column, color=color, ax=ax)
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"Boxplot of {column}")
    
    ax.set_xlabel(column)
    
    plt.tight_layout()
    return fig

def create_scatter_plot(df, x_column, y_column, hue=None, title=None):
    """
    Create a scatter plot between two numerical columns
    
    Parameters:
    - df: pandas DataFrame
    - x_column: str, column for x-axis
    - y_column: str, column for y-axis
    - hue: str, column for color encoding
    - title: str, plot title
    
    Returns:
    - matplotlib figure
    """
    if df is None or x_column not in df.columns or y_column not in df.columns:
        st.error(f"Columns {x_column} or {y_column} not found in the dataframe")
        return None
    
    if hue and hue not in df.columns:
        st.error(f"Hue column {hue} not found in the dataframe")
        hue = None
    
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x=x_column, y=y_column, hue=hue, ax=ax)
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"Scatter Plot: {x_column} vs {y_column}")
    
    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)
    
    plt.tight_layout()
    return fig

def create_bar_chart(df, x_column, y_column=None, title=None, color="#4CAF50", orientation="vertical"):
    """
    Create a bar chart for categorical data
    
    Parameters:
    - df: pandas DataFrame
    - x_column: str, column for categories
    - y_column: str, column for values (uses count if None)
    - title: str, plot title
    - color: str, bar color
    - orientation: str, "vertical" or "horizontal"
    
    Returns:
    - matplotlib figure
    """
    if df is None or x_column not in df.columns:
        st.error(f"Column {x_column} not found in the dataframe")
        return None
    
    if y_column and y_column not in df.columns:
        st.error(f"Column {y_column} not found in the dataframe")
        return None
    
    fig, ax = plt.subplots()
    
    if y_column:
        if orientation == "horizontal":
            sns.barplot(data=df, y=x_column, x=y_column, color=color, ax=ax)
        else:
            sns.barplot(data=df, x=x_column, y=y_column, color=color, ax=ax)
    else:
        # Count plot if no y_column specified
        if orientation == "horizontal":
            sns.countplot(data=df, y=x_column, color=color, ax=ax)
        else:
            sns.countplot(data=df, x=x_column, color=color, ax=ax)
    
    if title:
        ax.set_title(title)
    else:
        if y_column:
            ax.set_title(f"Bar Chart: {x_column} vs {y_column}")
        else:
            ax.set_title(f"Count of {x_column}")
    
    # Rotate x-axis labels if there are many categories
    if orientation == "vertical" and len(df[x_column].unique()) > 5:
        plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    return fig

def create_pie_chart(df, column, title=None, colors=None):
    """
    Create a pie chart for a categorical column
    
    Parameters:
    - df: pandas DataFrame
    - column: str, column to visualize
    - title: str, plot title
    - colors: list, custom colors for pie segments
    
    Returns:
    - matplotlib figure
    """
    if df is None or column not in df.columns:
        st.error(f"Column {column} not found in the dataframe")
        return None
    
    # Get value counts and prepare data
    value_counts = df[column].value_counts()
    
    # Limit to top 10 categories if there are too many
    if len(value_counts) > 10:
        st.warning(f"Column {column} has more than 10 unique values. Showing top 10.")
        others_sum = value_counts[10:].sum()
        value_counts = value_counts[:10]
        if others_sum > 0:
            value_counts['Others'] = others_sum
    
    fig, ax = plt.subplots()
    wedges, texts, autotexts = ax.pie(
        value_counts,
        labels=value_counts.index,
        autopct='%1.1f%%',
        startangle=90,
        colors=colors
    )
    
    # Make the percentage labels readable
    plt.setp(autotexts, size=10, weight="bold")
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"Distribution of {column}")
    
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    plt.tight_layout()
    return fig

def create_correlation_heatmap(df, columns=None, method='pearson', title=None, cmap='viridis'):
    """
    Create a correlation heatmap for numerical columns
    
    Parameters:
    - df: pandas DataFrame
    - columns: list, columns to include (None for all numeric columns)
    - method: str, correlation method ('pearson', 'spearman', or 'kendall')
    - title: str, plot title
    - cmap: str, colormap name
    
    Returns:
    - matplotlib figure
    """
    if df is None:
        st.error("No dataframe provided")
        return None
    
    # Select numeric columns
    if columns is None:
        numeric_df = df.select_dtypes(include=np.number)
    else:
        available_columns = [col for col in columns if col in df.columns]
        numeric_df = df[available_columns].select_dtypes(include=np.number)
    
    if numeric_df.empty:
        st.error("No numeric columns available for correlation analysis")
        return None
    
    # Calculate correlation matrix
    corr_matrix = numeric_df.corr(method=method)
    
    # Create the heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    heatmap = sns.heatmap(
        corr_matrix,
        annot=True,
        cmap=cmap,
        linewidths=0.5,
        fmt=".2f",
        square=True,
        ax=ax
    )
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"Correlation Heatmap ({method})")
    
    plt.tight_layout()
    return fig

def create_line_chart(df, x_column, y_columns, title=None):
    """
    Create a line chart for time series or sequential data
    
    Parameters:
    - df: pandas DataFrame
    - x_column: str, column for x-axis
    - y_columns: list, columns for y-axis
    - title: str, plot title
    
    Returns:
    - matplotlib figure
    """
    if df is None or x_column not in df.columns:
        st.error(f"Column {x_column} not found in the dataframe")
        return None
    
    # Ensure y_columns is a list
    if isinstance(y_columns, str):
        y_columns = [y_columns]
    
    # Check if y_columns exist in the dataframe
    available_y_columns = [col for col in y_columns if col in df.columns]
    
    if not available_y_columns:
        st.error(f"None of the y-columns {y_columns} found in the dataframe")
        return None
    
    fig, ax = plt.subplots()
    
    for col in available_y_columns:
        ax.plot(df[x_column], df[col], label=col)
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"Line Chart")
    
    ax.set_xlabel(x_column)
    ax.set_ylabel("Value")
    
    if len(available_y_columns) > 1:
        ax.legend()
    
    # Rotate x-axis labels if there are many values
    if len(df[x_column].unique()) > 10:
        plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    return fig

def create_plotly_histogram(df, column, bins=30, title=None):
    """
    Create an interactive Plotly histogram
    
    Parameters:
    - df: pandas DataFrame
    - column: str, column to visualize
    - bins: int, number of bins
    - title: str, plot title
    
    Returns:
    - plotly figure
    """
    if df is None or column not in df.columns:
        st.error(f"Column {column} not found in the dataframe")
        return None
    
    if title is None:
        title = f"Histogram of {column}"
    
    fig = px.histogram(
        df,
        x=column,
        nbins=bins,
        marginal="box",
        title=title,
        color_discrete_sequence=['#4CAF50']
    )
    
    fig.update_layout(
        xaxis_title=column,
        yaxis_title="Count",
        bargap=0.1
    )
    
    return fig

def create_plotly_scatter(df, x_column, y_column, color=None, size=None, title=None):
    """
    Create an interactive Plotly scatter plot
    
    Parameters:
    - df: pandas DataFrame
    - x_column: str, column for x-axis
    - y_column: str, column for y-axis
    - color: str, column name for color encoding
    - size: str, column name for size encoding
    - title: str, plot title
    
    Returns:
    - plotly figure
    """
    if df is None or x_column not in df.columns or y_column not in df.columns:
        st.error(f"Columns {x_column} or {y_column} not found in the dataframe")
        return None
    
    if title is None:
        title = f"Scatter Plot: {x_column} vs {y_column}"
    
    if color and color not in df.columns:
        st.warning(f"Color column {color} not found in the dataframe. Ignoring.")
        color = None
    
    if size and size not in df.columns:
        st.warning(f"Size column {size} not found in the dataframe. Ignoring.")
        size = None
    
    fig = px.scatter(
        df,
        x=x_column,
        y=y_column,
        color=color,
        size=size,
        title=title,
        opacity=0.7
    )
    
    fig.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')))
    
    fig.update_layout(
        xaxis_title=x_column,
        yaxis_title=y_column
    )
    
    return fig

def create_plotly_bar(df, x_column, y_column=None, color=None, title=None):
    """
    Create an interactive Plotly bar chart
    
    Parameters:
    - df: pandas DataFrame
    - x_column: str, column for categories
    - y_column: str, column for values (uses count if None)
    - color: str, column for color encoding
    - title: str, plot title
    
    Returns:
    - plotly figure
    """
    if df is None or x_column not in df.columns:
        st.error(f"Column {x_column} not found in the dataframe")
        return None
    
    if y_column and y_column not in df.columns:
        st.error(f"Column {y_column} not found in the dataframe")
        return None
    
    if color and color not in df.columns:
        st.warning(f"Color column {color} not found in the dataframe. Ignoring.")
        color = None
    
    if y_column:
        # Bar chart with values
        if title is None:
            title = f"Bar Chart: {x_column} vs {y_column}"
        
        fig = px.bar(
            df,
            x=x_column,
            y=y_column,
            color=color,
            title=title
        )
        
        fig.update_layout(
            xaxis_title=x_column,
            yaxis_title=y_column
        )
    else:
        # Count plot
        if title is None:
            title = f"Count of {x_column}"
        
        value_counts = df[x_column].value_counts().reset_index()
        value_counts.columns = ['category', 'count']
        
        fig = px.bar(
            value_counts,
            x='category',
            y='count',
            color=color,
            title=title
        )
        
        fig.update_layout(
            xaxis_title=x_column,
            yaxis_title="Count"
        )
    
    return fig

def create_plotly_pie(df, column, title=None):
    """
    Create an interactive Plotly pie chart
    
    Parameters:
    - df: pandas DataFrame
    - column: str, column to visualize
    - title: str, plot title
    
    Returns:
    - plotly figure
    """
    if df is None or column not in df.columns:
        st.error(f"Column {column} not found in the dataframe")
        return None
    
    if title is None:
        title = f"Distribution of {column}"
    
    # Get value counts
    value_counts = df[column].value_counts().reset_index()
    value_counts.columns = ['category', 'count']
    
    # Limit to top 10 categories if there are too many
    if len(value_counts) > 10:
        st.warning(f"Column {column} has more than 10 unique values. Showing top 10.")
        others_count = value_counts['count'][10:].sum()
        value_counts = value_counts.iloc[:10]
        if others_count > 0:
            value_counts = pd.concat([
                value_counts,
                pd.DataFrame({'category': ['Others'], 'count': [others_count]})
            ], ignore_index=True)
    
    fig = px.pie(
        value_counts,
        values='count',
        names='category',
        title=title
    )
    
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label'
    )
    
    return fig

def create_plotly_line(df, x_column, y_columns, title=None):
    """
    Create an interactive Plotly line chart
    
    Parameters:
    - df: pandas DataFrame
    - x_column: str, column for x-axis
    - y_columns: list or str, column(s) for y-axis
    - title: str, plot title
    
    Returns:
    - plotly figure
    """
    if df is None or x_column not in df.columns:
        st.error(f"Column {x_column} not found in the dataframe")
        return None
    
    # Ensure y_columns is a list
    if isinstance(y_columns, str):
        y_columns = [y_columns]
    
    # Check if y_columns exist in the dataframe
    available_y_columns = [col for col in y_columns if col in df.columns]
    
    if not available_y_columns:
        st.error(f"None of the y-columns {y_columns} found in the dataframe")
        return None
    
    if title is None:
        title = f"Line Chart"
    
    fig = go.Figure()
    
    for col in available_y_columns:
        fig.add_trace(
            go.Scatter(
                x=df[x_column],
                y=df[col],
                mode='lines+markers',
                name=col
            )
        )
    
    fig.update_layout(
        title=title,
        xaxis_title=x_column,
        yaxis_title="Value",
        hovermode="x unified"
    )
    
    return fig

def create_plotly_heatmap(df, columns=None, method='pearson', title=None):
    """
    Create an interactive Plotly correlation heatmap
    
    Parameters:
    - df: pandas DataFrame
    - columns: list, columns to include (None for all numeric columns)
    - method: str, correlation method ('pearson', 'spearman', or 'kendall')
    - title: str, plot title
    
    Returns:
    - plotly figure
    """
    if df is None:
        st.error("No dataframe provided")
        return None
    
    # Select numeric columns
    if columns is None:
        numeric_df = df.select_dtypes(include=np.number)
    else:
        available_columns = [col for col in columns if col in df.columns]
        numeric_df = df[available_columns].select_dtypes(include=np.number)
    
    if numeric_df.empty:
        st.error("No numeric columns available for correlation analysis")
        return None
    
    # Calculate correlation matrix
    corr_matrix = numeric_df.corr(method=method)
    
    if title is None:
        title = f"Correlation Heatmap ({method})"
    
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale='Viridis',
        title=title
    )
    
    fig.update_layout(
        xaxis_title="Features",
        yaxis_title="Features"
    )
    
    return fig

def create_plotly_box(df, x_column=None, y_column=None, color=None, title=None):
    """
    Create an interactive Plotly box plot
    
    Parameters:
    - df: pandas DataFrame
    - x_column: str, categorical column for x-axis grouping (optional)
    - y_column: str, numerical column for y-axis
    - color: str, column for color encoding
    - title: str, plot title
    
    Returns:
    - plotly figure
    """
    if df is None:
        st.error("No dataframe provided")
        return None
    
    if y_column and y_column not in df.columns:
        st.error(f"Column {y_column} not found in the dataframe")
        return None
    
    if x_column and x_column not in df.columns:
        st.warning(f"Column {x_column} not found in the dataframe. Ignoring.")
        x_column = None
    
    if color and color not in df.columns:
        st.warning(f"Color column {color} not found in the dataframe. Ignoring.")
        color = None
    
    if title is None:
        if x_column and y_column:
            title = f"Box Plot: {y_column} by {x_column}"
        elif y_column:
            title = f"Box Plot of {y_column}"
        else:
            title = "Box Plot"
    
    fig = px.box(
        df,
        x=x_column,
        y=y_column,
        color=color,
        title=title,
        points="all"
    )
    
    fig.update_layout(
        xaxis_title=x_column if x_column else "",
        yaxis_title=y_column if y_column else ""
    )
    
    return fig
