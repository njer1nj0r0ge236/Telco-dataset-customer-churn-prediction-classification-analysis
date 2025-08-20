import streamlit as st
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px



# --- Load the model ---
@st.cache_resource
def load_model():
    with open('final_customer_churn_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

# --- Page Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Data", "Predict", "Dashboard", "History"])

# --- Home Page ---
if page == "Home":
    st.title("Telco Customer Churn Prediction App")
    st.write(
        """
        Welcome! 👋  
        This app predicts customer churn based on various features such as gender, contract method, service usage.  
        
        ### 📌 Overview
        - Understand customer behavior and identify churn risks.  
        - Gain insights from predictive analytics.  
        - Explore different features through interactive visualizations.  

        ### 🔗 Useful Links
        - [GitHub Repository](https://github.com/njer1nj0r0ge236/Telco-dataset-customer-churn-prediction-classification-analysis)
        - [Medium Article](https://medium.com/@njorogediana236/customer-churn-prediction-using-classification-analysis-telco-dataset-c568d1c1babd)
        """
    )
    st.image(r"C:\Users\USER\Desktop\streamlit\Image Aug 20, 2025, 01_06_30 AM.png")


# --- Data Page ---
elif page == "Data":
    st.title("Data Exploration")
    st.write("This page shows the raw datasets used for training and testing the model.")

    # Helper function to load and display datasets
    @st.cache_data
    def load_data():
        try:
            df1 = pd.read_excel('LP2_Telco_churn_first_3000.xlsx')
        except FileNotFoundError:
            df1 = None

        try:
            df2 = pd.read_csv('LP2_Telco-churn-second-2000.csv')
        except FileNotFoundError:
            df2 = None

        try:
            df3 = pd.read_csv('Telco-churn-last-2000(Sheet1).csv')
        except FileNotFoundError:
            df3 = None

        return df1, df2, df3

    df1, df2, df3 = load_data()

    # Organize in tabs
    tabs = st.tabs(["First Dataset (3000)", "Second Dataset (2000)", "Test Dataset"])

    for i, (df, name) in enumerate(zip([df1, df2, df3],
                                       ["First Dataset (3000)", "Second Dataset (2000)", "Test Dataset"])):
        with tabs[i]:
            if df is not None:
                st.subheader(name)
                st.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
                st.write("Columns:", list(df.columns))

                # Preview
                st.dataframe(df.head(10))

                # Expand to see full dataset
                with st.expander("See full dataset"):
                    st.dataframe(df)

                # Missing values check
                if st.checkbox(f"Show missing values in {name}"):
                    st.write(df.isnull().sum())

                # Download option
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label=f"Download {name} as CSV",
                    data=csv,
                    file_name=f"{name.replace(' ', '_')}.csv",
                    mime="text/csv",
                )
            else:
                st.error(f"{name} not found. Please upload the file.")


# --- Predict Page ---
elif page == "Predict":
    st.title("Make a Prediction")
    st.write("Enter the customer details below to predict if they will churn. 0- No, 1- Yes")
    # Replicate the data types and column names from your training data
    # (e.g., SeniorCitizen, Partner, Dependents)
    SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
    Partner = st.selectbox("Partner", [0, 1])
    Dependents = st.selectbox("Dependents", [0, 1])
    tenure = st.number_input("Tenure (months)", value=0, min_value=0, max_value=72)
    PhoneService = st.selectbox("Phone Service", [0, 1])
    MultipleLines = st.selectbox("Multiple Lines", [0, 1])
    OnlineSecurity = st.selectbox("Online Security", [0, 1])
    OnlineBackup = st.selectbox("Online Backup", [0, 1])
    DeviceProtection = st.selectbox("Device Protection", [0, 1])
    TechSupport = st.selectbox("Tech Support", [0, 1])
    StreamingTV = st.selectbox("Streaming TV", [0, 1])
    StreamingMovies = st.selectbox("Streaming Movies", [0, 1])
    PaperlessBilling = st.selectbox("Paperless Billing", [0, 1])
    MonthlyCharges = st.number_input("Monthly Charges", value=0.0)
    TotalCharges = st.number_input("Total Charges", value=0.0)
    gender_Female = st.selectbox("Gender Female", [0, 1])
    gender_Male = st.selectbox("Gender Male", [0, 1])
    InternetService_DSL = st.selectbox("Internet Service DLS", [0, 1])
    InternetService_Fiber_optic = st.selectbox("Internet Service Fiber Optic", [0, 1])
    InternetService_No = st.selectbox("Internet Service No", [0, 1])
    Contract_Month_to_month = st.selectbox("Contract Month-to-Month", [0, 1])
    Contract_One_year = st.selectbox("Contract One Year", [0, 1])
    Contract_Two_year = st.selectbox("Contract Two Year", [0, 1])
    PaymentMethod_Credit_card_automatic = st.selectbox("Payment Method Credit Card (automatic)", [0, 1])
    PaymentMethod_Electronic_check = st.selectbox("Payment Method Electronic Check", [0, 1])
    PaymentMethod_Mailed_check = st.selectbox("Payment Method Mailed Check", [0, 1])
    
    if st.button("Predict"):
        # Create a DataFrame from user inputs
        input_data = pd.DataFrame([[
            SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines, OnlineSecurity,
            OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies,
            PaperlessBilling, MonthlyCharges, TotalCharges, gender_Male,
            InternetService_Fiber_optic, InternetService_No, Contract_One_year,
            Contract_Two_year, PaymentMethod_Credit_card_automatic,
            PaymentMethod_Electronic_check, PaymentMethod_Mailed_check
        ]], columns=[
            'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService',
            'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling',
            'MonthlyCharges', 'TotalCharges', 'gender_Male',
            'InternetService_Fiber optic', 'InternetService_No', 'Contract_One year',
            'Contract_Two year', 'PaymentMethod_Credit card (automatic)',
            'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check'
        ])
        
        # Ensure column order matches the training data
        cols_ordered = model.feature_names_in_
        input_data = input_data[cols_ordered]

        # Make prediction
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)

        st.subheader("Prediction Result")
        if prediction[0] == 1:
            st.error(f"Prediction: This customer is likely to churn. (Probability: {prediction_proba[0][1]:.2f})")
        else:
            st.success(f"Prediction: This customer is likely to stay. (Probability: {prediction_proba[0][0]:.2f})")

            # Log the prediction to a CSV file
        timestamp = pd.Timestamp.now()
        prediction_log = pd.DataFrame({
            'timestamp': [timestamp],
            'prediction': [int(prediction[0])],
            'probability': [prediction_proba],
            # Add other relevant input features here if you want them in the log
            'tenure': [tenure],
            'MonthlyCharges': [MonthlyCharges]
        })

        # Append to a CSV file. Create the file if it doesn't exist.
        if not os.path.exists('prediction_history.csv'):
            prediction_log.to_csv('prediction_history.csv', index=False)
        else:
            prediction_log.to_csv('prediction_history.csv', mode='a', header=False, index=False)



# --- Dashboard Page ---
elif page == "Dashboard":
    st.title("📊 Telco Customer Churn Dashboard")
    st.write("An overview of key insights from the data and model.")

    # Check if the combined data file exists
    if not os.path.exists('df_combined.csv'):
        st.error("Combined data file 'df_combined.csv' not found. Please ensure it is in the same directory.")
    else:
        # Load the combined dataframe directly
        df_combined = pd.read_csv('df_combined.csv')

        # Convert 'TotalCharges' to numeric and handle missing values
        df_combined['TotalCharges'] = pd.to_numeric(df_combined['TotalCharges'], errors='coerce')
        df_combined['TotalCharges'] = df_combined['TotalCharges'].fillna(0)

        # Ensure 'Churn' column is in integer format
        df_combined['Churn'] = df_combined['Churn'].astype(int)

        # --- Sidebar Filters ---
        st.sidebar.header("🔎 Filter Options")
        selected_contract = st.sidebar.multiselect(
            "Select Contract Type:",
            options=df_combined['Contract'].unique(),
            default=df_combined['Contract'].unique()
        )

        selected_gender = st.sidebar.multiselect(
            "Select Gender:",
            options=df_combined['gender'].unique(),
            default=df_combined['gender'].unique()
        )

        tenure_range = st.sidebar.slider(
            "Select Tenure Range:",
            0, int(df_combined['tenure'].max()), (0, 50)
        )


        # --- EDA Section ---
        st.header("📈 Exploratory Data Analysis (EDA)")

        # Interactive churn distribution
        st.subheader("Customer Churn Distribution")
        fig = px.histogram(
            df_combined, x="Churn", color="Churn", barmode="group",
            labels={"Churn": "Customer Churn"}, title="Distribution of Churn"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Interactive churn by contract
        st.subheader("Churn Rate by Contract Type")
        fig = px.bar(
            df_combined, x="Contract", y="Churn", color="Contract",
            labels={"Churn": "Churn Rate"}, title="Churn Rate by Contract"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # --- KPI Section ---
        st.header("📊 Key Performance Indicators (KPIs)")

        kpi_option = st.selectbox("Choose KPI:", ["Overall Churn Rate", "Average CLV"])

        if kpi_option == "Overall Churn Rate":
            total_customers = len(df_combined)
            churned_customers = df_combined['Churn'].sum()
            overall_churn_rate = (churned_customers / total_customers) * 100 if total_customers > 0 else 0
            st.metric(label="Overall Churn Rate", value=f"{overall_churn_rate:.2f}%")

        elif kpi_option == "Average CLV":
            clv_df = df_combined.groupby('Churn').agg(
                avg_tenure=('tenure', 'mean'),
                avg_monthly_charges=('MonthlyCharges', 'mean')
            )
            clv_df['CLV'] = clv_df['avg_tenure'] * clv_df['avg_monthly_charges']

            col1, col2 = st.columns(2)
            if 1 in clv_df.index:
                col1.metric("Avg CLV (Churned)", f"${clv_df.loc[1, 'CLV']:.2f}")
            else:
                col1.metric("Avg CLV (Churned)", "N/A")

            if 0 in clv_df.index:
                col2.metric("Avg CLV (Non-Churned)", f"${clv_df.loc[0, 'CLV']:.2f}")
            else:
                col2.metric("Avg CLV (Non-Churned)", "N/A")

        st.markdown("---")

        # --- Download Option ---
        st.subheader("📂 Download Data")
        st.download_button(
            "Download Filtered Data",
            data=df_combined.to_csv(index=False).encode('utf-8'),
            file_name="churn_data.csv",
            mime="text/csv"
        )

        # --- Extra Insight ---
        st.markdown("---")
        st.subheader("📌 Insights")
        st.write("""
        - Customers with **month-to-month contracts** tend to churn more.  
        - **Higher tenure** generally correlates with lower churn.  
        - **Gender alone** is not a strong churn predictor.  
        - CLV differences highlight revenue impact of churn.  
        """)


# --- History Page ---
elif page == "History":
    st.title("Prediction History")
    st.write("View past predictions and their details.")
    # You could store predictions in a CSV and load them here
     # Check if the history file exists
    if os.path.exists('prediction_history.csv'):
        # Load the prediction history from the CSV file
        history_df = pd.read_csv('prediction_history.csv')
        
        # Display the history in a data table
        st.subheader("Past Predictions")
        st.dataframe(history_df.tail(10)) # Show the last 10 predictions
        
    else:
        st.info("No prediction history found yet. Make some predictions on the 'Predict' page to view a history.")