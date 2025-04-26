import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Page config
st.set_page_config(
    page_title="Math Graph Analysis",
    page_icon="🔢",
    layout="wide"
)

# Page title
st.title("🔢 Math Graph Analysis and Visualizations")
st.write("Explore different types of visualizations")

# Upload data section
st.header("📊 Upload Data for Charting")
data_file = st.file_uploader("Upload CSV file", type=['csv'])

if data_file:
    df = pd.read_csv(data_file)
    st.write("Data Preview:")
    st.dataframe(df.head())

    # Allow users to select a chart type
    chart_type = st.selectbox("Select a chart type", ["Histogram", "Bar Chart", "Pie Chart", "Scatter Plot", "Pictogram"])

    if chart_type == "Histogram":
        st.subheader("📊 Histogram")
        column = st.selectbox("Select a column for histogram", df.columns)
        bins = st.slider("Number of bins", 5, 100, 20)

        fig = px.histogram(df, x=column, nbins=bins, title=f"Histogram of {column}")
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Bar Chart":
        st.subheader("📊 Bar Chart")
        x_column = st.selectbox("Select x-axis column for bar chart", df.columns)
        y_column = st.selectbox("Select y-axis column for bar chart", df.columns)

        fig = px.bar(df, x=x_column, y=y_column, title=f"Bar Chart of {y_column} vs {x_column}")
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Pie Chart":
        st.subheader("🍰 Pie Chart")
        pie_column = st.selectbox("Select a column for pie chart", df.columns)
        fig = px.pie(df, names=pie_column, title=f"Pie Chart of {pie_column}")
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Scatter Plot":
        st.subheader("📍 Scatter Plot")
        x_col = st.selectbox("Select x-axis column for scatter plot", df.columns)
        y_col = st.selectbox("Select y-axis column for scatter plot", df.columns)

        fig = px.scatter(df, x=x_col, y=y_col, title=f"Scatter Plot of {y_col} vs {x_col}")
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Pictogram":
        st.subheader("📊 Pictogram")
        pictogram_column = st.selectbox("Select a column for pictogram", df.columns)
        icon = st.text_input("Enter icon for pictogram (e.g., '🍏', '📦')", "📦")

        # Aggregate the data for pictogram (e.g., sum the values in the column)
        pictogram_data = df[pictogram_column].value_counts()

        fig = go.Figure(go.Bar(
            x=pictogram_data.index,
            y=pictogram_data.values,
            text=pictogram_data.values,
            hoverinfo="text",
            marker=dict(
                color='rgb(255, 165, 0)',
                line=dict(color='rgb(0, 0, 0)', width=2)
            )
        ))

        fig.update_layout(
            title=f"Pictogram Chart of {pictogram_column}",
            xaxis_title=pictogram_column,
            yaxis_title="Count",
            showlegend=False
        )

        for i, value in enumerate(pictogram_data.values):
            fig.add_trace(go.Scatter(
                x=[pictogram_data.index[i]] * value,
                y=np.arange(1, value + 1),
                mode='text',
                text=[icon] * value,
                textposition='bottom center'
            ))

        st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Please upload a CSV file to begin chart analysis.")

