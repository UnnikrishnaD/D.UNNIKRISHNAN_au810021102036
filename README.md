# Heart-Disease-Prediction
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

# 1. Data Collection
# Assuming 'heart_disease_data.csv' contains the dataset
data = pd.read_csv('heart_disease_data.csv')

# 2. Data Preprocessing
X = data.drop('target', axis=1)
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Logistic Regression Model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# 4. Model Evaluation
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_pred))

# 5. Interpretation and Prediction
# Coefficients of the model
coefficients = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_[0]})
print(coefficients)

# Predicting for new data
new_data = [[65, 1, 140, 240, 1, 0, 1, 165, 0, 2.5, 1, 3, 1]]
new_data_scaled = scaler.transform(new_data)
prediction = model.predict_proba(new_data_scaled)
print("Probability of Heart Disease:", prediction[0][1])
  
