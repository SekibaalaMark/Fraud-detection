# 💳 Credit Card Fraud Detection using KNN Classifier

This repository contains a Python implementation of a **K-Nearest Neighbors (KNN)** machine learning model to detect fraudulent transactions in a highly imbalanced dataset.

---

## 📌 Project Overview

The goal of this project is to classify credit card transactions as either:

- **Fraudulent (1)**
- **Genuine (0)**

Due to the imbalance in the dataset (very few fraud cases), the project emphasizes:

- Data preprocessing  
- Feature scaling  
- Reliable evaluation metrics beyond accuracy  

---

## 🛠️ Data Preprocessing & Pipeline

The following steps were applied to improve model performance:

### 1. Handling Missing Values
- Missing values in the `Class` column were filled with **0 (non-fraud)**  

### 2. Stratified Train-Test Split
- Used a **70/30 split** with `stratify=y`  
- Ensures equal representation of fraud cases in both training and testing datasets  

### 3. Feature Scaling
- Applied **StandardScaler**  
- Standardizes features to:  
  - Mean = 0  
  - Variance = 1  
- Important because KNN relies on distance calculations  

---

## 🧪 Hyperparameter Tuning

- Used **5-Fold Cross-Validation**  
- Tested values of **K from 1 to 30**  

### Results:
- Automatically selects the **optimal K**  
- Generates a plot showing:  
  - Number of neighbors vs accuracy  
  - Helps identify model stability  

---

## 📊 Evaluation Metrics

Because of class imbalance, multiple metrics were used:

- **Accuracy** – Overall correctness  
- **Precision (Weighted)** – Measures false positives  
- **Recall (Weighted)** – Measures false negatives  
- **F1 Score** – Balance between precision and recall  
- **Confusion Matrix** – Shows:  
  - True Positives  
  - True Negatives  
  - False Positives  
  - False Negatives  

---

## 🚀 How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/sekibaala-mark/Fraud-detection.git
cd Fraud-detection


