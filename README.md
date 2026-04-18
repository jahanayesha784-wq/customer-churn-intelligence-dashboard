# 📉 Customer Churn Prediction Dashboard

https://customer-churn-intelligence-dashboard-zthdwnypalbprdfx7oern8.streamlit.app/

An **end-to-end Machine Learning project** that predicts whether a telecom customer is likely to **churn (leave the service)** using a trained model and an **interactive Streamlit dashboard**.

This project demonstrates the full **data science workflow**, including **data preprocessing, model training, evaluation, and deployment** as a web application.

---

# 🚀 Project Overview

Customer churn prediction helps businesses **identify customers who are likely to cancel their service**.
By predicting churn early, companies can take **preventive actions** like promotions, discounts, or support improvements.

This project uses a **Random Forest machine learning model** trained on a telecom dataset and deployed through a **Streamlit interactive dashboard**.

---

# ✨ Features

* Data preprocessing pipeline
* Missing value handling
* One-hot encoding for categorical variables
* Feature scaling using StandardScaler
* Random Forest classification model
* Hyperparameter tuning with GridSearchCV
* Cross-validation for model reliability
* ROC-AUC evaluation metric
* Confusion matrix visualization
* Feature importance analysis
* Interactive Streamlit dashboard
* Business insights for churn risk
* Probability-based churn prediction
* Single-customer prediction system

---

# 🧠 Machine Learning Workflow

The project follows a **complete ML lifecycle**:

1. Data loading and cleaning
2. Feature preprocessing
3. Train-test splitting
4. Pipeline creation
5. Model training
6. Hyperparameter tuning
7. Cross-validation
8. Model evaluation
9. Feature importance analysis
10. Model saving
11. Deployment using Streamlit

---

# 📊 Dataset

Dataset used: **Telco Customer Churn Dataset**

The dataset contains information about telecom customers including:

* Customer demographics
* Service subscriptions
* Account information
* Contract type
* Payment methods
* Charges
* Churn status

### Example features

| Feature         | Description                       |
| --------------- | --------------------------------- |
| gender          | Customer gender                   |
| tenure          | Number of months with the company |
| InternetService | Type of internet service          |
| Contract        | Contract type                     |
| MonthlyCharges  | Monthly bill                      |
| TotalCharges    | Total amount paid                 |
| Churn           | Whether the customer left         |

---

# 🏗 Project Structure

```
customer_churn_prediction/
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── models/
│   ├── churn_advanced_model.pkl
│   └── results.txt
│
├── plots/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── feature_importance.png
│
├── .streamlit/
│   └── config.toml
│
├── churn_advanced.py
├── predict_customer.py
├── app.py
├── requirements.txt
└── README.md
```

---

# ⚙️ Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/customer-churn-prediction.git
cd customer-churn-prediction
```

---

### 2️⃣ Create virtual environment

```bash
python3 -m venv venv
```

Activate it:

Linux / Kali / Mac:

```bash
source venv/bin/activate
```

Windows:

```bash
venv\Scripts\activate
```

---

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

# 🧪 Running the Project

## Step 1 — Train the Machine Learning Model

```bash
python churn_advanced.py
```

This will:

* preprocess the dataset
* train the model
* tune hyperparameters
* evaluate performance
* save trained model
* generate plots

Generated files:

```
models/churn_advanced_model.pkl
models/results.txt
plots/confusion_matrix.png
plots/roc_curve.png
plots/feature_importance.png
```

---

## Step 2 — Test Single Customer Prediction

```bash
python predict_customer.py
```

Example output:

```
Predicted Class: Churn
Churn Probability: 82.15 %
```

---

## Step 3 — Run the Dashboard

```bash
streamlit run app.py
```

Open the browser at:

```
http://localhost:8501
```

---

# 📊 Dashboard Features

The Streamlit dashboard allows users to:

* Enter customer details
* Predict churn probability
* View churn risk level
* Display model performance plots
* Get business insights about churn drivers

### Dashboard Sections

1️⃣ Customer Input Form
2️⃣ Churn Probability Prediction
3️⃣ Risk Level Indicator
4️⃣ Business Interpretation
5️⃣ Model Performance Visualization

---

# 📈 Model Evaluation Metrics

The model performance is evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC score
* Confusion Matrix

These metrics help measure how well the model identifies churn customers.

---

# 🔍 Feature Importance

The model identifies which features influence churn most.

Examples:

* Contract type
* Monthly charges
* Tenure
* Internet service type
* Tech support availability

Understanding these helps businesses improve **customer retention strategies**.

---

# 💻 Tech Stack

| Category             | Tools               |
| -------------------- | ------------------- |
| Programming Language | Python              |
| Data Processing      | Pandas, NumPy       |
| Visualization        | Matplotlib, Seaborn |
| Machine Learning     | Scikit-learn        |
| Model Persistence    | Joblib              |
| Web Dashboard        | Streamlit           |
| Environment          | Virtualenv          |

---

# 🧩 Future Improvements

Possible enhancements:

* Add **XGBoost / LightGBM model comparison**
* Implement **SHAP explainability**
* Add **CSV batch prediction**
* Deploy dashboard to **Streamlit Cloud**
* Add **customer segmentation**
* Implement **REST API for predictions**

---

# 🎓 Learning Outcomes

This project demonstrates:

* End-to-end machine learning pipeline
* Feature engineering
* Model tuning and evaluation
* Data visualization
* Model deployment with Streamlit
* Building portfolio-ready data science projects

---

# 🤝 Contributing

Contributions are welcome.

Steps:

1. Fork the repository
2. Create a new branch
3. Commit your changes
4. Submit a pull request

---

⭐ If you like this project, please give it a star on GitHub!
