# 📊 Telco Customer Churn Prediction  

## 📝 Project Scenario  
Customer retention is a critical factor for business growth, especially in competitive industries like telecommunications. With machine learning classification models, businesses can perform **churn analysis** to predict whether a customer is likely to leave.  

This project leverages real-world telecom customer data to build, evaluate, and deploy a **churn prediction system** through an interactive **Streamlit web app**.  

---

## 📂 Datasets  

This project uses **three datasets**:  

### 1. First Dataset (3000 records)  
Stored in a remote SQL Server database:  
- **Server**: `dap-projects-database.database.windows.net`  
- **User**: `LP2_project`  
- **Password**: `Stat$AndD@t@Rul3`  
- **Database**: `dapDB`  
- **Table**: `dbo.LP2_Telco_churn_first_3000`  
> 🔒 _Read-only access_  

### 2. Second Dataset (2000 records)  
- CSV file: `LP2_Telco-churn-second-2000.csv` (from GitHub repo).  

### 3. Testing Dataset (2000 records)  
- Excel file: `Telco-churn-last-2000.xlsx` (from OneDrive).  
- Used exclusively for **final model testing**.  

> 📌 Note: All datasets were downloaded locally in .csv and .xlsx formats then uploaded into **Google Colab**.  

---

## 🚀 Project Objectives  

- Build an end-to-end ML pipeline for **churn prediction**  
- Perform **data preprocessing** (handle missing values, encode categorical variables, scale features)  
- Conduct **EDA** to uncover customer behavior insights  
- Train and evaluate multiple ML models  
- Optimize models with **hyperparameter tuning**  
- Deploy an **interactive Streamlit web app**  
- Share the project via **GitHub & Streamlit Cloud**  

---

## 🛠️ Tools & Technologies  

- **Python**: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn  
- **Google Colab**: for model building & experimentation  
- **Streamlit**: for deployment & interactive dashboards  
- **SQL Server**: acessing the First Dataset (3000 records)
- **GitHub**: version control & collaboration  

---

## ⚙️ How It Works  

1. Load & preprocess datasets (CSV + Excel).  
2. Encode categorical variables & scale numerical features.  
3. Train models including:  
   - Logistic Regression  
   - Random Forest  
   - XGBoost Classifier  
4. Evaluate models with **accuracy, precision, recall, F1-score**.  
5. Deploy best model in a **Streamlit multipage app**.  

---

## 🚀 Streamlit App Features  

The deployed application contains:  

- **🏠 Home Page** – Overview of the project & links (GitHub, LinkedIn, Medium).  
- **📂 Data Page** – Displays sample/raw data, categorical & numeric features.  
- **📊 Dashboard Page** –  
  - EDA dashboard: customer churn insights  
  - KPI dashboard: business metrics  
- **🤖 Predict Page** –  
  - Collects user input 
  - Loads trained ML model  
  - Outputs churn prediction + probability  
- **🕒 History Page** – Displays previous predictions with timestamps & inputs.  



---

## 📊 Model Performance  

The final selected model achieved:  

- **Accuracy**: 0.8067  
- **Precision**: o.6818 
- **Recall**: 0.5455 
- **F1-score**: 0.6061   

---

## ⚙️ Installation & Usage  
```bash
Clone the repository:
git clone https://github.com/your-username/telco-churn-prediction.git
cd telco-churn-prediction

### Create a virtual environment & install dependencies:
pip install -r requirements.txt

### Run the Streamlit app:
streamlit run app/app.py
```

## 🌍 Deployment
The project is deployed on Streamlit Cloud:
👉 [Live App](http://localhost:8501/#telco-customer-churn-prediction-app)

## 👩‍💻 Author  
Developed by [Marydiana Njoroge](https://marydiananjorogeportfolio.vercel.app/)  
💼 [LinkedIn](https://www.linkedin.com/in/marydiana-njoroge-41b236244/)  
🐙 [GitHub](https://github.com/njer1nj0r0ge236)  
✍️ [Medium](https://medium.com/@njorogediana236)  


## 📌 License
This project is licensed under the [MIT License](./LICENSE) – see the LICENSE
 file for details.
