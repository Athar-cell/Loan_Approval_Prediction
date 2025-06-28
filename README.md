# Loan_Approval_Prediction
# 💰 Loan Approval Prediction using Machine Learning

This project predicts whether a loan application will be **approved or rejected** based on applicant details using a **Logistic Regression** model.

---

## 🎯 Objective

To automate loan status prediction by training a machine learning model on historical applicant data.

---

## 📂 Dataset

- **Source**: [Kaggle – Loan Prediction Dataset](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset)
- **Records:** 614 rows × 13 columns
- **Target Variable:** `Loan_Status`  
  - `Y` → Approved  
  - `N` → Not Approved

---

## 🛠️ Tools & Libraries Used

- Python  
- pandas, numpy  
- scikit-learn (LabelEncoder, train_test_split, LogisticRegression, accuracy_score)

---

## ⚙️ Workflow Steps

1. **Data Loading**
   - Loaded the dataset using pandas
   - Explored shape, missing values, class distribution

2. **Data Preprocessing**
   - Handled missing values (LoanAmount, Self_Employed, etc.)
   - Label Encoding for categorical columns (Gender, Education, etc.)
   - Converted target variable to binary (Y → 1, N → 0)

3. **Model Training**
   - Split data into training and test sets (80/20)
   - Trained Logistic Regression model on training data

4. **Model Evaluation**
   - Evaluated using `accuracy_score` on test data

---

## ✅ Results

- **Model Used**: Logistic Regression  
- **Test Accuracy**: 🌟 **XX.XX%** *(replace with your actual score)*

---

## 📁 Files Included

- `Project-4_Loan_approval_Prediction.ipynb` – Jupyter Notebook    
- `README.md` – This documentation file

---

## 🧠 What I Learned

- Handling missing and categorical data  
- Encoding and preparing real-world financial datasets  
- Training a binary classifier using Logistic Regression  
- Evaluating predictions with accuracy score

---

## 🔮 Future Improvements

- Try more advanced models: Random Forest, XGBoost  
- Add confusion matrix, F1-score, ROC curve  
- Build a **Streamlit form** to predict loan approval based on user input  
- Add explainability using SHAP or LIME

---

## 🤝 Connect with Me

- GitHub: [github.com/yourusername](https://github.com/Athar-cell)  

---

> 🚀 Built with ❤️ while exploring machine learning and real-world financial data
