import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_selection import SelectFromModel
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

# Feature Selection using Gradient Boosting
feature_selector = GradientBoostingClassifier(random_state=42, n_estimators=100)
feature_selector.fit(X_train, y_train)

# Select important features
sfm = SelectFromModel(feature_selector, prefit=True)
X_train_selected = sfm.transform(X_train)
X_val_selected = sfm.transform(X_val)
X_test_selected = sfm.transform(X_test)

selected_features = X.columns[sfm.get_support()]
print("Selected Features:", selected_features)

# Train a Gradient Boosting Classifier
clf = GradientBoostingClassifier(random_state=42, n_estimators=100)
clf.fit(X_train_selected, y_train)

y_val_pred = clf.predict(X_val_selected)

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

# Final Evaluation on Test Set
y_test_pred = clf.predict(X_test_selected)

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

plt.figure(figsize=(12, 6))
plt.barh(selected_features, importances, color='skyblue')
plt.xlabel('Importance')
plt.title('Feature Importances in Gradient Boosting')
plt.show()
