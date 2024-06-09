# Predicting-Customer-Churn
![ScreenShot](https://miro.medium.com/v2/resize:fit:720/format:webp/1*47xx1oXuebvYwZeB0OutuA.png)

This project is intended to build and evaluate an Artificial Neural Network (ANN) for predicting customer churn in a bank.

## Overview
The project involves the following steps:
1. **Data Preprocessing**:
   - Load the dataset.
   - Encode categorical features (Gender and Geography) using One-Hot Encoding.
   - Split the dataset into features (`X`) and target (`y`), and then into training and testing sets.
   - Scale the features using `StandardScaler`.

2. **Build the ANN**:
   - Initialize the ANN using `tf.keras.models.Sequential`.
   - Add input and hidden layers with ReLU activation functions.
   - Add the output layer with a sigmoid activation function for binary classification.

3. **Compile and Train the ANN**:
   - Compile the ANN with the Adam optimizer and binary cross-entropy loss function.
   - Train the ANN on the training set for 100 epochs with a batch size of 32.

4. **Evaluate the Model**:
   - Make predictions on the test set and compute the ROC curve and AUC score.
   - Plot the ROC curve.
   - Calculate and print the confusion matrix and various performance metrics (FPR, FNR, sensitivity, specificity, accuracy).
   - Plot the confusion matrix.

5. **Make Predictions for a New Customer**:
   - Define a new customer’s data.
   - Scale the new customer’s data.
   - Predict the probability of the new customer leaving the bank.
   - Print the prediction probability and interpret whether the customer will leave or stay.
