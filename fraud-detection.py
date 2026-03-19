#imported libraries for data manipulation, visualization, and machine learning
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv('creditcard.csv')


#Getting an overview of the dataset
df.head() # Check the first few rows of the dataset
df.Class.value_counts() # Check the distribution of classes (fraud vs non-fraud)
df.describe() # Get summary statistics of the dataset

# Preprocessing the data
df.info() # Check for missing values and data types
df['Class'] = df['Class'].fillna(0) # Fill missing values in the 'Class' column with 0 (non-fraud)



X = df.drop(columns=['Class']) #Dropped the Class Column to create the feature set
y = df['Class'] #Target variable (fraud or non-fraud)

X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.3,random_state=42,stratify=y)# Split the data into training and testing sets (70% train, 30% test) with stratification to maintain class distribution

# Scale the features using StandardScaler
scaler = StandardScaler() # Initialize the StandardScaler to standardize the features (mean=0, variance=1)
X_train_scaled = scaler.fit_transform(X_train) # Fit the scaler on the training data and transform it to scale the features
X_test_scaled = scaler.transform(X_test) # Transform the test data using the same scaler (do not fit again to avoid data leakage)
