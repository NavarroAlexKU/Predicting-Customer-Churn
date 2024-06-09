#!/usr/bin/env python
# coding: utf-8

# # Artificial Neural Network

# ### Importing the libraries

# In[1]:


# Import python packages:
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import tensorflow as tf
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')


# In[2]:


# check verison of tensorflow:
tf.__version__


# ## Part 1 - Data Preprocessing

# ### Importing the dataset

# In[3]:


dataset = pd.read_csv(r'Churn_Modelling.csv')
# check first 5 rows of data:
dataset.head()


# ### Encoding categorical data

# In[4]:


# One-Hot Encoding for Gender and Geography
one_hot_encoder = OneHotEncoder(sparse=False)  # drop='first' to avoid multicollinearity
encoded_columns = one_hot_encoder.fit_transform(dataset[['Gender', 'Geography']]).astype(int)  # Convert to integers

# Creating a DataFrame for the one-hot encoded columns
encoded_columns_df = pd.DataFrame(encoded_columns, columns=one_hot_encoder.get_feature_names_out(['Gender', 'Geography']))

# Concatenating the one-hot encoded columns with the original DataFrame
dataset = pd.concat([dataset, encoded_columns_df], axis=1).drop(['Gender', 'Geography'], axis=1)
dataset.head()


# In[5]:


dataset.columns


# ### Splitting the dataset into the Training set and Test set

# In[6]:


# Create X and Y variable:
X = dataset[[
     'CreditScore', 'Age', 'Tenure',
       'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember',
       'EstimatedSalary', 'Gender_Female', 'Gender_Male',
       'Geography_France', 'Geography_Germany', 'Geography_Spain'
]]

y = dataset['Exited']

# Train Test Split Data:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)


# ### Feature Scaling

# In[7]:


# instantiate scaler:
sc = StandardScaler()

# scale X_train and X_Test:
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


# ## Part 2 - Building the ANN

# ### Initializing the ANN

# In[8]:


# create artifical neural network:
ann = tf.keras.models.Sequential()


# ### Adding the input layer and the first hidden layer

# In[9]:


ann.add(
    tf.keras.layers.Dense(
        units = 6,
        activation = 'relu'
        )
)


# ### Adding the second hidden layer

# In[10]:


ann.add(
    tf.keras.layers.Dense(
        units = 6,
        activation = 'relu'
        )
)


# ### Adding the output layer

# In[11]:


ann.add(
    tf.keras.layers.Dense(
        # set to one since we're predicting binary variable:
        units = 1,
        # set to sigmoid for probabilites:
        activation = 'sigmoid'
        )
)


# ## Part 3 - Training the ANN

# ### Compiling the ANN

# In[12]:


ann.compile(
    optimizer = 'adam',
    loss = 'binary_crossentropy',
    metrics = ['accuracy']
)


# ### Training the ANN on the Training set

# In[13]:


ann.fit(
    X_train,
    y_train,
    batch_size = 32,
    epochs = 100
)


# ## Part 4 - Making the predictions and evaluating the model

# In[14]:


y_pred_proba = ann.predict(X_test)  # Predict probabilities

# compute ROC curve and AUC score:
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


# ### Making the Confusion Matrix

# In[15]:


# Make predictions
y_pred = ann.predict(X_test)
y_pred_classes = (y_pred > 0.5).astype(int)  # Assuming a threshold of 0.5

# Compute the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_classes)

# Calculate metrics from the confusion matrix
TN, FP, FN, TP = conf_matrix.ravel()
FPR = (FP / (FP + TN)) * 100
FNR = (FN / (FN + TP)) * 100
overall_error_rate = ((FP + FN) / (FP + FN + TP + TN)) * 100
sensitivity = (TP / (TP + FN)) * 100  # Sensitivity (Recall)
specificity = (TN / (TN + FP)) * 100  # Specificity

# Calculate accuracy using accuracy_score
accuracy = accuracy_score(y_test, y_pred_classes) * 100

# Print the confusion matrix
print("Confusion Matrix:")
print(conf_matrix)

# Print the metrics in percentages
print("\nMetrics:")
print(f"False Positive Rate (FPR): {FPR:.2f}%")
print(f"False Negative Rate (FNR): {FNR:.2f}%")
print(f"Overall Error Rate: {overall_error_rate:.2f}%")
print(f"Sensitivity (Recall): {sensitivity:.2f}%")
print(f"Specificity: {specificity:.2f}%")
print(f"Accuracy: {accuracy:.2f}%")

# Detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_classes))

# Explanation of metrics
print("\nExplanation:")
print(f"False Positive Rate (FPR) is the proportion of negative instances (no churn) that were incorrectly classified as positive (churn). In this case, it is {FPR:.2f}%.")
print(f"False Negative Rate (FNR) is the proportion of positive instances (churn) that were incorrectly classified as negative (no churn). In this case, it is {FNR:.2f}%.")
print(f"Overall Error Rate is the proportion of all instances that were incorrectly classified. In this case, it is {overall_error_rate:.2f}%.")
print(f"Sensitivity (Recall) is the proportion of actual positives (churn) that were correctly identified. In this case, it is {sensitivity:.2f}%. This indicates how well the model identifies true positives.")
print(f"Specificity is the proportion of actual negatives (no churn) that were correctly identified. In this case, it is {specificity:.2f}%. This indicates how well the model identifies true negatives.")
print(f"Accuracy is the proportion of correctly classified instances (both true positives and true negatives) out of the total number of instances. In this case, it is {accuracy:.2f}%.")

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted No', 'Predicted Yes'], yticklabels=['Actual No', 'Actual Yes'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# ### Compute probability of a customer leaving the bank with specific features:

# In[16]:


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

