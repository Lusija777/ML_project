import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


red_wine = pd.read_csv("wine+quality/winequality-red.csv", sep=';')
white_wine = pd.read_csv("wine+quality/winequality-white.csv", sep=';')

# Combine datasets
wine_data = pd.concat([red_wine, white_wine], axis=0).reset_index(drop=True)

# Define wine quality categories
wine_data['quality_category'] = pd.cut(wine_data['quality'], bins=[-1, 3, 5, 7, 10], labels=['Very Bad', 'Average', 'Good', 'Excellent'])

X = wine_data.drop(columns=['quality', 'quality_category', 'alcohol'])  # Features
y = wine_data['quality_category']  # Target categories

print(wine_data['quality_category'].value_counts())

# Split into training (70%), validation (20%), and testing (10%) sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=1/3, random_state=42, stratify=y_temp)

clf = GradientBoostingClassifier(random_state=42, n_estimators=100)
clf.fit(X_train, y_train)

y_val_pred = clf.predict(X_val)

print("Validation Classification Report:")
print(classification_report(y_val, y_val_pred))

print("Validation Confusion Matrix:")
val_cm = confusion_matrix(y_val, y_val_pred, labels=['Very Bad', 'Average', 'Good', 'Excellent'])
print(val_cm)

plt.figure(figsize=(8, 6))
sns.heatmap(val_cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Very Bad', 'Average', 'Good', 'Excellent'], yticklabels=['Very Bad', 'Average', 'Good', 'Excellent'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Validation Confusion Matrix')
plt.show()

# Evaluation on Test Set
y_test_pred = clf.predict(X_test)

print("Test Classification Report:")
print(classification_report(y_test, y_test_pred))

print("Test Confusion Matrix:")
test_cm = confusion_matrix(y_test, y_test_pred, labels=['Very Bad', 'Average', 'Good', 'Excellent'])
print(test_cm)

plt.figure(figsize=(8, 6))
sns.heatmap(test_cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Very Bad', 'Average', 'Good', 'Excellent'], yticklabels=['Very Bad', 'Average', 'Good', 'Excellent'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Test Confusion Matrix')
plt.show()

# Feature Importance
importances = clf.feature_importances_
feature_names = X.columns

plt.figure(figsize=(12, 6))
plt.barh(feature_names, importances, color='skyblue')
plt.xlabel('Importance')
plt.title('Feature Importances in Gradient Boosting')
plt.show()

# Calculate percentage misclassification error
val_misclassification_error = np.mean(y_val != y_val_pred) * 100
print(f"Validation Misclassification Error: {val_misclassification_error:.2f}%")

test_misclassification_error = np.mean(y_test != y_test_pred) * 100
print(f"Test Misclassification Error: {test_misclassification_error:.2f}%")

