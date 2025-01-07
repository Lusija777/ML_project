import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from xgboost.sklearn import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import accuracy_score


red_wine = pd.read_csv("wine+quality/winequality-red.csv", sep=';')
white_wine = pd.read_csv("wine+quality/winequality-white.csv", sep=';')

# Combine datasets
wine_data = pd.concat([red_wine, white_wine], axis=0).reset_index(drop=True)

# Define wine quality categories
wine_data['quality_category'] = pd.cut(wine_data['quality'], bins=[-1, 3, 5, 7, 10], labels=['Very Bad', 'Average', 'Good', 'Excellent'])

X = wine_data.drop(columns=['quality', 'quality_category'])  # Features
y = wine_data['quality_category']  # Target categories

print(wine_data['quality_category'].value_counts())

# Split into training (70%), validation (20%), and testing (10%) sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=1/3, random_state=42, stratify=y_temp)

smote = SMOTE(random_state=42, sampling_strategy='auto', k_neighbors=5)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

label_encoder = LabelEncoder()

# Fit and transform the training, validation, and test labels
y_train_encoded = label_encoder.fit_transform(y_train_resampled)
y_val_encoded = label_encoder.transform(y_val)
y_test_encoded = label_encoder.transform(y_test)

# Define the hyperparameter grid
param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 6, 9]
}

best_score = 0
best_params = None

# Iterate over parameter grid
for n_estimators in param_grid['n_estimators']:
    for learning_rate in param_grid['learning_rate']:
        for max_depth in param_grid['max_depth']:
            # Create model with current hyperparameters
            model = XGBClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                random_state=42,
            )
            # Fit model
            model.fit(X_train_resampled, y_train_encoded)

            # Evaluate model on validation set
            y_val_pred_encoded = model.predict(X_val)
            y_val_pred = label_encoder.inverse_transform(y_val_pred_encoded)

            score = accuracy_score(y_val, y_val_pred)

            # Update best score and parameters
            if score > best_score:
                best_score = score
                best_params = {
                    'n_estimators': n_estimators,
                    'learning_rate': learning_rate,
                    'max_depth': max_depth,
                }

print("Best parameters:", best_params)
print("Best validation accuracy:", best_score)


model = XGBClassifier(random_state=42, n_estimators=200, learning_rate=0.1, max_depth=9)

model.fit(X_train_resampled, y_train_encoded)

y_val_pred_encoded = model.predict(X_val)
y_test_pred_encoded = model.predict(X_test)

# Convert predictions back to original labels
y_val_pred = label_encoder.inverse_transform(y_val_pred_encoded)
y_test_pred = label_encoder.inverse_transform(y_test_pred_encoded)

print("Validation Classification Report after SMOTE:")
print(classification_report(y_val, y_val_pred))

print("Test Classification Report after SMOTE:")
print(classification_report(y_test, y_test_pred))

val_cm_resampled = confusion_matrix(y_val, y_val_pred, labels=['Very Bad', 'Average', 'Good', 'Excellent'])
test_cm_resampled = confusion_matrix(y_test, y_test_pred, labels=['Very Bad', 'Average', 'Good', 'Excellent'])

plt.figure(figsize=(8, 6))
sns.heatmap(val_cm_resampled, annot=True, fmt='d', cmap='Blues', xticklabels=['Very Bad', 'Average', 'Good', 'Excellent'], yticklabels=['Very Bad', 'Average', 'Good', 'Excellent'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Validation Confusion Matrix After SMOTE')
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(test_cm_resampled, annot=True, fmt='d', cmap='Blues', xticklabels=['Very Bad', 'Average', 'Good', 'Excellent'], yticklabels=['Very Bad', 'Average', 'Good', 'Excellent'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Test Confusion Matrix After SMOTE')
plt.show()

# Feature Importance
importances = model.feature_importances_
feature_names = X.columns

plt.figure(figsize=(12, 6))
plt.barh(feature_names, importances, color='skyblue')
plt.xlabel('Importance')
plt.title('Feature Importances in XGboost')
plt.show()

# Calculate percentage misclassification error
val_misclassification_error = np.mean(y_val != y_val_pred) * 100
print(f"Validation Misclassification Error: {val_misclassification_error:.2f}%")

test_misclassification_error = np.mean(y_test != y_test_pred) * 100
print(f"Test Misclassification Error: {test_misclassification_error:.2f}%")

