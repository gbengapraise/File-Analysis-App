# 📚 Import Libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 🧠 Initialize Session State
if 'page' not in st.session_state:
    st.session_state.page = "Home"

# 🎨 App Title and Description
st.set_page_config(page_title="Math Graph and Data Analysis App", page_icon="📊", layout="wide")
st.title("📊 Math Graph and Data Analysis App")
st.write("#### Welcome! This app helps you upload data, clean it, visualize it, and predict future values easily — all without coding!")

st.markdown("---")

# 🖲 Navigation Buttons (Instead of Sidebar)
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    if st.button("🏠 Home"):
        st.session_state.page = "Home"
with col2:
    if st.button("📁 Upload Data"):
        st.session_state.page = "Upload"
with col3:
    if st.button("🧹 Clean Data"):
        st.session_state.page = "Clean"
with col4:
    if st.button("📊 Visualize"):
        st.session_state.page = "Visualize"
with col5:
    if st.button("🤖 Predict"):
        st.session_state.page = "Predict"

# 📂 Global Variables
uploaded_file = None
df = None

# 📁 Upload Section
def upload_section():
    st.header("📁 Upload Your Dataset")
    st.write("Upload a CSV, Excel, or JSON file. Make sure your data has columns and rows!")

    global df

    uploaded_file = st.file_uploader("Choose your data file", type=["csv", "xlsx", "json"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
            elif uploaded_file.name.endswith('.json'):
                df = pd.read_json(uploaded_file)

            st.success("✅ File Uploaded Successfully!")
            st.write("### Preview of Data:")
            st.dataframe(df)

        except Exception as e:
            st.error(f"⚠️ Error: {e}")

# 🧹 Clean Section
def clean_section():
    st.header("🧹 Clean Your Dataset")
    st.write("Remove missing values and duplicate rows to improve your data quality.")

    global df

    if df is not None:
        st.write("### Current Data:")
        st.dataframe(df)

        if st.button("Remove Missing Values (NaNs)"):
            df.dropna(inplace=True)
            st.success("✅ Missing values removed successfully!")
            st.dataframe(df)

        if st.button("Remove Duplicate Rows"):
            df.drop_duplicates(inplace=True)
            st.success("✅ Duplicates removed successfully!")
            st.dataframe(df)
    else:
        st.warning("⚠️ Please upload a dataset first.")

# 📊 Visualize Section
def visualize_section():
    st.header("📊 Data Visualization Center")
    st.write("Create amazing charts easily by selecting columns and graph type!")

    global df

    if df is not None:
        numeric_columns = df.select_dtypes(include=np.number).columns.tolist()

        if numeric_columns:
            st.write("### Select Variables:")
            x_axis = st.selectbox("Select X-Axis", numeric_columns)
            y_axis = st.selectbox("Select Y-Axis", numeric_columns)

            chart_type = st.selectbox("Choose Chart Type", ["Line Chart", "Bar Chart", "Histogram", "Scatter Plot", "Pictogram (Fake)", "Heatmap"])

            if st.button("Generate Chart"):
                plt.figure(figsize=(10, 6))
                if chart_type == "Line Chart":
                    plt.plot(df[x_axis], df[y_axis])
                    plt.title(f"{y_axis} over {x_axis}")
                    plt.xlabel(x_axis)
                    plt.ylabel(y_axis)
                elif chart_type == "Bar Chart":
                    plt.bar(df[x_axis], df[y_axis])
                    plt.title(f"{y_axis} per {x_axis}")
                    plt.xlabel(x_axis)
                    plt.ylabel(y_axis)
                elif chart_type == "Histogram":
                    plt.hist(df[x_axis], bins=20, color='skyblue', edgecolor='black')
                    plt.title(f"Distribution of {x_axis}")
                    plt.xlabel(x_axis)
                elif chart_type == "Scatter Plot":
                    plt.scatter(df[x_axis], df[y_axis], color='red')
                    plt.title(f"{y_axis} vs {x_axis}")
                    plt.xlabel(x_axis)
                    plt.ylabel(y_axis)
                elif chart_type == "Pictogram (Fake)":
                    plt.bar(df[x_axis], df[y_axis], color='green')
                    plt.title(f"Pictogram Style: {y_axis} per {x_axis}")
                    plt.xlabel(x_axis)
                    plt.ylabel(y_axis)
                elif chart_type == "Heatmap":
                    corr = df.corr()
                    sns.heatmap(corr, annot=True, cmap='coolwarm')
                    plt.title("Heatmap of Correlation Matrix")

                plt.grid(True)
                st.pyplot(plt.gcf())
        else:
            st.warning("⚠️ No numeric columns found for visualization.")
    else:
        st.warning("⚠️ Please upload a dataset first.")

# 🤖 Predict Section
def predict_section():
    st.header("🤖 Predict Future Values (AI)")
    st.write("Use Linear Regression to predict future values based on your data!")

    global df

    if df is not None:
        numeric_columns = df.select_dtypes(include=np.number).columns.tolist()

        if numeric_columns:
            st.write("### Select Feature and Target Variables:")
            feature = st.selectbox("Select Feature (X)", numeric_columns)
            target = st.selectbox("Select Target (Y)", numeric_columns)

            if st.button("Train Model"):
                X = df[[feature]]
                y = df[target]

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                model = LinearRegression()
                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)

                st.success("✅ Model Trained Successfully!")
                st.write(f"📈 Mean Squared Error (lower is better): {mse:.2f}")

                new_value = st.number_input(f"Enter {feature} value to predict {target}:")

                if st.button("Predict"):
                    prediction = model.predict(np.array([[new_value]]))
                    st.success(f"📢 Predicted {target}: {prediction[0]:.2f}")
        else:
            st.warning("⚠️ No numeric columns found for training model.")
    else:
        st.warning("⚠️ Please upload a dataset first.")

# 🔄 Page Routing
if st.session_state.page == "Home":
    st.header("🏠 Home Page")
    st.write("""
    Welcome to the **Math Graph and Data Analysis App**! 🚀

    **Features of this app:**
    - 📂 Upload your dataset
    - 🧹 Clean missing or duplicate data
    - 📊 Create powerful charts easily
    - 🤖 Predict future values using AI (Linear Regression)

    ---
    """)
elif st.session_state.page == "Upload":
    upload_section()
elif st.session_state.page == "Clean":
    clean_section()
elif st.session_state.page == "Visualize":
    visualize_section()
elif st.session_state.page == "Predict":
    predict_section()
