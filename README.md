# 📊 Mobile App User Classification using K-Nearest Neighbors (KNN)

## 📌 Project Overview
This project focuses on building a Machine Learning model to classify mobile app users as **subscribed (premium)** or **not subscribed** based on their interaction behavior, session activity, and device usage patterns.

The project follows the complete Machine Learning pipeline, including:
- Data cleaning and preprocessing
- Feature engineering
- Data visualization
- Model training using K-Nearest Neighbors (KNN)
- Model evaluation and comparison

---

## 📂 Dataset

**Dataset Name:** User Mobile App Interaction Data  
**Source:** Kaggle  

🔗 https://www.kaggle.com/datasets/mohamedmoslemani/user-mobile-app-interaction-data

### 📊 Dataset Description
The dataset contains **100,000+ event-level records** simulating user interactions in a mobile application. It includes:

- **User Information:** User ID, Age, Subscription Status, Push Enabled  
- **Session Data:** Session ID, Session Duration, Timestamp  
- **Device Details:** OS, Device Model, Screen Resolution  
- **Technical Metrics:** Battery Level, Memory Usage  
- **Network Info:** Network Type (WiFi, 4G, 5G, etc.)  
- **Interaction Data:** Event Type, Event Target, Event Value  

⚠️ The dataset includes **intentional noise**, such as:
- Missing values  
- Typos and corrupted entries  
- Random string injections  

---

## 🎯 Problem Statement
Predict whether a user is a **premium subscriber (1)** or **not (0)** using behavioral and technical features.

---

## 🛠️ Project Pipeline

### 1. Data Loading
A custom data loader class was implemented to extract and load the dataset efficiently.

---

### 2. Data Cleaning
- Handled missing values
- Removed corrupted entries
- Normalized inconsistent values (especially target variable)
- Converted columns to appropriate data types

---

### 3. Feature Engineering
Since the dataset is event-level, it was aggregated into **user-level features**:

Final features used:
- total_sessions  
- total_events  
- active_days  
- avg_session_duration  
- avg_event_value  
- avg_battery_level  
- avg_memory_usage_mb  
- user_age  

Target:
- subscription_status (0 or 1)

---

### 4. Data Preprocessing
- Missing values handled using **median imputation**
- Feature scaling applied using **StandardScaler**
- Ensured all features are on the same scale (important for KNN)

---

### 5. Data Visualization
Various plots were generated to explore patterns and relationships:
- Class distribution  
- Correlation heatmap  
- Feature distributions  
- Boxplots by class  
- Scatter plots  
- PCA visualization  

---

### 6. Model Training (KNN)
K-Nearest Neighbors was used with different values of K:
- K = 5  
- K = 11  
- K = 21  

---

### 7. Model Evaluation
Each model was evaluated using:
- Accuracy  
- Precision  
- Recall  
- Confusion Matrix  
- Misclassification Error  

Visualization of results:
- Accuracy vs K  
- Precision & Recall comparison  
- Confusion matrices  

---

## 📈 Results
- Lower K values showed higher training accuracy (possible overfitting)
- Higher K values provided smoother decision boundaries
- The best K was selected based on test performance and balanced metrics

---

## 📦 Project Structure

'''
project/
    │
    ├── venv/
    script/
        ├── __init__.py
        ├── data_loader.py
        ├── feature_builder.py
        ├── preprocessor.py
        ├── visualizer.py
    ├── main.py
    ├── .gitignore
    ├── dataset_file.zip
    ├── requirements.txt
    ├── visual_refs/
    └── README.md
'''

---

## ⚙️ Technologies Used
- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Matplotlib  
- Seaborn  

---

## 🚀 How to Run

1. Clone the repository:

'git clone <your-repo-link>'

2. Install dependencies:

'pip install -r requirements.txt'

3. Run the main script:

'python main.py'

---

## 📌 Key Learnings
- Importance of data cleaning in noisy datasets  
- Feature engineering from event-level to user-level data  
- Impact of feature scaling on KNN performance  
- Bias-variance tradeoff through different K values  

---

## 📢 Future Improvements
- Use real-world datasets instead of synthetic data  
- Try advanced models (Random Forest, XGBoost)  
- Hyperparameter tuning using cross-validation  
- Feature selection techniques  

---

## 👨‍💻 Author
Ishaan Pathak  
Master’s in Computer Science  
Lawrence Technological University  

---