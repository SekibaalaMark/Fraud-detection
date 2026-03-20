# Credit Card Fraud Detection using KNN Classifier

![Credit Card Fraud Detection](https://thumbs.dreamstime.com/b/fraud-detection-icon-guard-against-bank-card-unauthorized-transactions-enhance-payment-security-to-prevent-credit-scams-309246554.jpg)

This repository contains a Python implementation of a **K-Nearest Neighbors (KNN)** classifier to detect fraudulent credit card transactions in a highly imbalanced dataset.

## 📌 Project Overview

The main objective is to classify credit card transactions as:

- **Fraudulent** (Class = 1)  
- **Genuine** (Class = 0)

Due to the extreme class imbalance (fraud cases are very rare), the project focuses on proper preprocessing, feature scaling, careful hyperparameter tuning, and evaluation using imbalance-aware metrics rather than accuracy alone.

## 🛠️ Data Preprocessing & Pipeline

Key steps performed to prepare the data and optimize KNN performance:

- **Handling Missing Values** — Null values in the `Class` column were filled with 0 (assuming non-fraud)
- **Stratified Train-Test Split** — 70/30 split using `stratify=y` to preserve fraud proportion in both sets
- **Feature Scaling** — Applied `StandardScaler` (mean=0, variance=1) — critical for distance-based algorithms like KNN

## 🧪 Hyperparameter Tuning

- Performed **5-fold cross-validation** to evaluate different values of **k** (number of neighbors)
- Tested **k = 1 to 30**
- Automatically selected the **optimal k** based on the highest mean cross-validation accuracy
- Generated a visualization showing how model performance changes with different k values

<grok-card data-id="2c6394" data-type="image_card" data-plain-type="render_searched_image"  data-arg-size="LARGE" ></grok-card>


*(Example plot — your actual accuracy vs. k curve will be similar)*

## 📊 Model Evaluation

Because of severe class imbalance, the following metrics are reported:

- **Accuracy** (overall correctness — can be misleading here)
- **Precision (weighted)**
- **Recall (weighted)**
- **F1 Score (weighted)**
- **Confusion Matrix** — shows True Positives, True Negatives, False Positives, False Negatives

Example confusion matrix from similar fraud detection projects:

<grok-card data-id="5d32cd" data-type="image_card" data-plain-type="render_searched_image"  data-arg-size="LARGE" ></grok-card>



<grok-card data-id="fd90a8" data-type="image_card" data-plain-type="render_searched_image"  data-arg-size="LARGE" ></grok-card>


## 🚀 How to Run

1. **Clone the repository**

   ```bash
   git clone https://github.com/sekibaala-mark/Fraud-detection.git
   cd Fraud-detection


**Install dependencies**
- Bash
- pip install pandas numpy matplotlib scikit-learn

**Add the dataset**
- Download the creditcard.csv file (commonly available on Kaggle)
- Place it in the root directory of the project


**Run the project**
- Open and run the Jupyter Notebook (Fraud_Detection_KNN.ipynb)
or
- Run the Python script directly in your IDE (VS Code, PyCharm, etc.)



**🔍 Key Files**

- Fraud_Detection_KNN.ipynb — main notebook with full pipeline
- creditcard.csv — (not included — add it yourself)

**📈 Results Visualization**
The notebook generates:

- Plot of Accuracy vs. Number of Neighbors (k)
- Confusion matrix heatmap
- Printed classification report



**📜 License**
**MIT License — feel free to use, modify, and share!**

**Made with ❤️ by MARK**
- *Happy fraud-hunting! 🛡️💳*
- *This version is professional, visually appealing on GitHub, uses proper emoji spacing, includes images for better engagement, and clearly explains every section.*


- This version is professional, visually appealing on GitHub, uses proper emoji spacing, includes images for better engagement, and clearly explains every section.

- Feel free to tweak the wording, add more badges, or include your actual results/screenshots when you upload the real notebook outputs! Good luck with the repo! 🚀
