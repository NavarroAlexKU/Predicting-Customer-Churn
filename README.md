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

## Model Results

### ROC Curve
The ROC (Receiver Operating Characteristic) curve is a graphical representation of a model's diagnostic ability. The ROC curve illustrates the trade-off between the true positive rate (sensitivity) and the false positive rate (1-specificity) across different threshold values. The AUC (Area Under the Curve) score, which ranges from 0 to 1, indicates the model's ability to distinguish between classes. An AUC score of 0.87 suggests that the model performs well in predicting customer churn.

![ROC Curve](https://raw.githubusercontent.com/NavarroAlexKU/Predicting-Customer-Churn/main/ROC%20Plot.jfif)

### Confusion Matrix and Metrics
The confusion matrix provides a summary of the prediction results on a classification problem. It shows the number of true positives (TP), true negatives (TN), false positives (FP), and false negatives (FN). From the confusion matrix, we can calculate several performance metrics:

- **False Positive Rate (FPR)**: 4.51%
- **False Negative Rate (FNR)**: 52.10%
- **Overall Error Rate**: 14.15%
- **Sensitivity (Recall)**: 47.90%
- **Specificity**: 95.49%
- **Accuracy**: 85.85%

![Confusion Matrix](https://raw.githubusercontent.com/NavarroAlexKU/Predicting-Customer-Churn/main/confusion%20matrix.jfif)

**Classification Report:**
          precision    recall  f1-score   support

       0       0.88      0.95      0.91      1595
       1       0.73      0.48      0.58       405

       accuracy                           0.86      2000


### Explanation:
- **False Positive Rate (FPR)** is the proportion of negative instances (no churn) that were incorrectly classified as positive (churn). In this case, it is 4.51%.
- **False Negative Rate (FNR)** is the proportion of positive instances (churn) that were incorrectly classified as negative (no churn). In this case, it is 52.10%.
- **Overall Error Rate** is the proportion of all instances that were incorrectly classified. In this case, it is 14.15%.
- **Sensitivity (Recall)** is the proportion of actual positives (churn) that were correctly identified. In this case, it is 47.90%. This indicates how well the model identifies true positives.
- **Specificity** is the proportion of actual negatives (no churn) that were correctly identified. In this case, it is 95.49%. This indicates how well the model identifies true negatives.
- **Accuracy** is the proportion of correctly classified instances (both true positives and true negatives) out of the total number of instances. In this case, it is 85.85%.

## Predicting Customer Bank Churn

Here is an example of how to use the trained model to predict whether a new customer will leave the bank:

```python
# Define the new customer's data
new_customer = {
    'CreditScore': 600,
    'Age': 40,
    'Tenure': 3,
    'Balance': 60000,
    'NumOfProducts': 2,
    'HasCrCard': 1,  # Yes -> 1
    'IsActiveMember': 1,  # Yes -> 1
    'EstimatedSalary': 50000,
    'Gender_Female': 0,  # Male -> 0, Female -> 0
    'Gender_Male': 1,    # Male -> 1, Female -> 0
    'Geography_France': 1,  # France -> 1, Germany -> 0, Spain -> 0
    'Geography_Germany': 0,
    'Geography_Spain': 0
}

# Convert the new customer's data to a DataFrame
new_customer_df = pd.DataFrame(new_customer, index=[0])

# Ensure the new data has the same structure as the training data
print("New customer data (before scaling):")
print(new_customer_df)

# Apply the same scaling transformation
new_customer_scaled = sc.transform(new_customer_df)

# Make the prediction
new_customer_pred_proba = ann.predict(new_customer_scaled)
new_customer_pred = (new_customer_pred_proba > 0.5).astype(int)

# Convert prediction probability to percentage
new_customer_pred_proba_percentage = new_customer_pred_proba[0][0] * 100

# Print the prediction result
print(f"Prediction probability: {new_customer_pred_proba_percentage:.2f}%")
print(f"Prediction (0 = No, 1 = Yes): {new_customer_pred[0][0]}")

# Interpret the prediction
if new_customer_pred[0][0] == 1:
    print("The model predicts that the customer will leave the bank.")
else:
    print("The model predicts that the customer will stay with the bank.")
