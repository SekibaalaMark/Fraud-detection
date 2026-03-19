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


pd.DataFrame(X_train_scaled).head() # Check the first few rows of the scaled training data to verify that the features have been standardized

# Train a K-Nearest Neighbors (KNN) classifier
knn = KNeighborsClassifier(n_neighbors=5) # Initialize the KNN classifier with 5 neighbors (you can experiment with different values of k to find the best performance)
knn.fit(X_train_scaled,y_train) # Fit the KNN model to the scaled training data (learn the patterns in the training set)
y_pred = knn.predict(X_test_scaled) # Predict the class labels for the scaled test data using the trained KNN model

 

# Evaluate the model's performance using various metrics 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report # Import evaluation metrics from scikit-learn
accuracy = round(accuracy_score(y_test,y_pred),5) # Calculate the accuracy of the model by comparing the true labels (y_test) with the predicted labels (y_pred) and round it to 5 decimal places
precision = round(precision_score(y_test,y_pred,average="weighted"),5) # Calculate the precision of the model, which is the ratio of true positives to the sum of true positives and false positives, using weighted average to account for class imbalance, and round it to 5 decimal places
recall = round(recall_score(y_test,y_pred,average="weighted"),5) # Calculate the recall of the model, which is the ratio of true positives to the sum of true positives and false negatives, using weighted average to account for class imbalance, and round it to 5 decimal places
f1 = round(f1_score(y_test,y_pred),5) # Calculate the F1 score of the model, which is the harmonic mean of precision and recall, by comparing the true labels (y_test) with the predicted labels (y_pred) and round it to 5 decimal places
conf_matrix = confusion_matrix(y_test,y_pred) # Generate the confusion matrix, which is a table that summarizes the performance of the classification model by showing the counts of true positives, true negatives, false positives, and false negatives, by comparing the true labels (y_test) with the predicted labels (y_pred)
print(f"Accuarcy: {accuracy}") # Print the accuracy of the model to the console
print(f"Precision: {precision}") # Print the precision of the model to the console
print(f"Recall: {recall}") # Print the recall of the model to the console
print(f"F1 Score: {f1}") # Print the F1 score of the model to the console
print(conf_matrix) # Print the confusion matrix to the console, which shows the counts of true positives, true negatives, false positives, and false negatives for the model's predictions on the test data


# Perform cross-validation to evaluate the model's performance more robustly
from sklearn.model_selection import cross_val_score # Import the cross_val_score function from scikit-learn to perform cross-validation
k_values = range(1,31) # Define a range of k values (number of neighbors) to evaluate the KNN model's performance across different values of k, from 1 to 30
accuracies = [] # Initialize an empty list to store the mean accuracy scores for each value of k during cross-validation
for k in k_values: # Iterate over each value of k in the defined range to evaluate the KNN model's performance for each number of neighbors
  knn = KNeighborsClassifier(n_neighbors=k) # Initialize the KNN classifier with the current value of k (number of neighbors) to be evaluated
  scores = cross_val_score(knn,X_train_scaled,y_train,cv=5,scoring="accuracy") # Perform 5-fold cross-validation on the training data (X_train_scaled and y_train) using the current KNN model and evaluate the accuracy for each fold, storing the scores in the 'scores' variable
  accuracies.append(scores.mean()) # Calculate the mean accuracy score across the 5 folds for the current value of k and append it to the 'accuracies' list to keep track of the performance of the KNN model for each number of neighbors
  
  
  
optimal_k = k_values[np.argmax(accuracies)] # Find the value of k that corresponds to the highest mean accuracy score in the 'accuracies' list by using np.argmax to get the index of the maximum value and then using that index to retrieve the corresponding k value from the 'k_values' range, which will be the optimal number of neighbors for the KNN model based on cross-validation performance
print(optimal_k) # Print the optimal value of k (number of neighbors) that achieved the highest mean accuracy score during cross-validation to the console, which can be used to train the final KNN model for fraud detection with the best performance.

#plotting the accuracies for different values of k to visualize the performance of the KNN model across different numbers of neighbors
plt.figure(figsize=(10,6)) # Set the figure size for the plot to 10 inches wide and 6 inches tall to ensure that the plot is large enough to clearly display the accuracy scores for different values of k
plt.plot(k_values,accuracies,marker="o") # Create a line plot of the mean accuracy scores (accuracies) for each value of k (k_values) with markers at each point to visualize the performance of the KNN model across different numbers of neighbors
plt.title("KNN Accuracy for Different Values of K") # Set the title of the plot to "KNN Accuracy for Different Values of K" to indicate that the plot shows the accuracy of the KNN model for different numbers of neighbors
plt.xlabel("Number of Neighbors (K)") # Set the x-axis label to "Number of Neighbors (K)" to indicate that the x-axis represents the different values of k (number of neighbors) evaluated in the KNN model
plt.ylabel("Mean Accuracy") # Set the y-axis label to "Mean Accuracy" to indicate that the y-axis represents the mean accuracy scores obtained from cross-validation for each value of k
plt.xticks(k_values) # Set the x-axis ticks to be the values of k (k_values) to ensure that each value of k is clearly marked on the x-axis for better visualization of the accuracy scores for different numbers of neighbors
plt.grid() # Add a grid to the plot for better readability and to help visualize the accuracy scores for different values of k more clearly
plt.show() # Display the plot to visualize the performance of the KNN model across different numbers of neighbors (k) and to identify the optimal k value that achieves the highest mean accuracy score during cross-validation.




# Train the final KNN model using the optimal k value obtained from cross-validation and evaluate its performance on the test set
knn = KNeighborsClassifier(n_neighbors=optimal_k) # Initialize the KNN classifier with the optimal number of neighbors (optimal_k) that was determined from cross-validation to train the final model for fraud detection
knn.fit(X_train_scaled,y_train) # Fit the KNN model to the scaled training data (X_train_scaled and y_train) using the optimal number of neighbors to learn the patterns in the training set for fraud detection
y_pred = knn.predict(X_test_scaled) # Predict the class labels for the scaled test data (X_test_scaled) using the trained KNN model with the optimal number of neighbors to evaluate its performance on unseen data





