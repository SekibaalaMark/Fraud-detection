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
