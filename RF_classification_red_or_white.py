import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

red_wine = pd.read_csv("wine+quality/winequality-red.csv", sep=';')
white_wine = pd.read_csv("wine+quality/winequality-white.csv", sep=';')

# new column to distinguish red (1) and white (0) wines
red_wine['type'] = 1
white_wine['type'] = 0

# Combine datasets
wine_data = pd.concat([red_wine, white_wine], axis=0).reset_index(drop=True)

X = wine_data.drop(columns=['type'])  # Features
y = wine_data['type']  # Target (1 = red, 0 = white)

# Split into training (70%), validation (20%), and testing (10%) sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=1/3, random_state=42, stratify=y_temp)

clf = RandomForestClassifier(random_state=42, n_estimators=100)
clf.fit(X_train, y_train)

y_val_pred = clf.predict(X_val)

print("Validation Classification Report:")
print(classification_report(y_val, y_val_pred))

print("Validation Confusion Matrix:")
val_cm = confusion_matrix(y_val, y_val_pred)
print(val_cm)

# Visualize Validation Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(val_cm, annot=True, fmt='d', cmap='Blues', xticklabels=['White', 'Red'], yticklabels=['White', 'Red'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Validation Confusion Matrix')
plt.show()

# Evaluation on Test Set
y_test_pred = clf.predict(X_test)

print("Test Classification Report:")
print(classification_report(y_test, y_test_pred))

print("Test Confusion Matrix:")
test_cm = confusion_matrix(y_test, y_test_pred)
print(test_cm)

plt.figure(figsize=(8, 6))
sns.heatmap(test_cm, annot=True, fmt='d', cmap='Blues', xticklabels=['White', 'Red'], yticklabels=['White', 'Red'])
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
plt.title('Feature Importances in Random Forest')
plt.show()
