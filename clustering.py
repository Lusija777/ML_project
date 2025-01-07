import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier

red_wine = pd.read_csv("wine+quality/winequality-red.csv", sep=';')
white_wine = pd.read_csv("wine+quality/winequality-white.csv", sep=';')

# Combine datasets
wine_data = pd.concat([red_wine, white_wine], axis=0).reset_index(drop=True)

X = wine_data.drop(columns=['quality'])
y = wine_data['quality']

# Split the data into labeled and unlabeled sets
X_train, X_unlabeled, y_train, y_unlabeled = train_test_split(X, y, test_size=0.8, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_unlabeled_scaled = scaler.transform(X_unlabeled)

# Apply K-means clustering
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans_labels = kmeans.fit_predict(X_unlabeled_scaled)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_unlabeled_scaled[:, 0], y=X_unlabeled_scaled[:, 1], hue=kmeans_labels, palette='Set2')
plt.title("K-means Clustering")
plt.show()

# Assign pseudo-labels to the unlabeled data
X_combined = pd.concat([X_train, X_unlabeled])
y_combined = pd.concat([y_train, pd.Series(kmeans_labels)])  # Using K-means labels as pseudo-labels

# Train a classifier (Random Forest)
clf = RandomForestClassifier(random_state=42)
clf.fit(X_combined, y_combined)

# Evaluate the model on the labeled data
y_pred = clf.predict(X_train)
print("Classification Report before Semi-supervised Learning:")
print(classification_report(y_train, y_pred, zero_division=1))

# Evaluate the model on the test set, unlabeled
y_test_pred = clf.predict(X_unlabeled)
print("Classification Report after Semi-supervised Learning (using pseudo-labels):")
print(classification_report(y_unlabeled, y_test_pred, zero_division=1))

# Use t-SNE to reduce dimensionality for visualization
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_unlabeled_scaled)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=kmeans_labels, palette='Set2')
plt.title("t-SNE Visualization of K-means Clusters")
plt.show()