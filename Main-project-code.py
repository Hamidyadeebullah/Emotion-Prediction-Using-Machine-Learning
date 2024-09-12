import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression  # Changed from LinearRegression to LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import sys

# Get file path from command line argument or default value
file_path = sys.argv[1] if len(sys.argv) > 1 else "D:/Data Mining Project/Dataset_study4.csv"

# Load the dataset
df = pd.read_csv(file_path)

# here we explore the data for like any missing value
print(df.head())  # Display a few rows of the dataset
print(df.info())
print(df.describe())

# we do data cleaning for the missing values
print("Missing Values:\n", df.isnull().sum())

# Pairwise scatterplot for feature visualization
#sns.pairplot(df, hue='Label')
#plt.suptitle("Pairwise Scatterplot of Features")
#plt.show()

# Feature Selection
X = df.drop("Label", axis=1)
y = df["Label"]

# SelectKBest feature selection
selector = SelectKBest(f_classif, k=10)
X_new = selector.fit_transform(X, y)

# Print selected features and their scores
selected_features = X.columns[selector.get_support()]
feature_scores = selector.scores_
feature_ranking = selector.pvalues_
print("Selected Features and Scores:")
for feature, score, rank in zip(selected_features, feature_scores, feature_ranking):
    print(f"{feature}: {score} (p-value: {rank})")

# PCA for dimensionality reduction
pca = PCA(n_components=5)
X_pca = pca.fit_transform(X_new)

# Classification (Logistic Regression as an example)
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42, stratify=y)

# Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build a Logistic Regression Classifier with cross-validation
clf_lr = LogisticRegression(max_iter=1000, random_state=42)

# Use StratifiedKFold for cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores_lr = cross_val_score(clf_lr, X_train_scaled, y_train, cv=cv, scoring='accuracy')
print("Cross-Validation Mean Accuracy (Logistic Regression):", cv_scores_lr.mean(), "±", cv_scores_lr.std())

# Train the logistic regression model
clf_lr.fit(X_train_scaled, y_train)

# Predictions and Evaluation for Logistic Regression
y_pred_lr = clf_lr.predict(X_test_scaled)
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))
print("Classification Report:\n", classification_report(y_test, y_pred_lr, zero_division=1))

# Classification (Support Vector Classifier as an example)
clf_svm = SVC(kernel='linear', C=1)

# Use StratifiedKFold for cross-validation
cv_svm = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores_svm = cross_val_score(clf_svm, X_train_scaled, y_train, cv=cv_svm, scoring='accuracy')
print("SVM Cross-Validation Mean Accuracy:", cv_scores_svm.mean(), "±", cv_scores_svm.std())

# Train the SVM model
clf_svm.fit(X_train_scaled, y_train)

# Predictions and Evaluation for SVM
y_pred_svm = clf_svm.predict(X_test_scaled)
print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_svm))
print("Classification Report:\n", classification_report(y_test, y_pred_svm, zero_division=1))

final_accuracy_lr = accuracy_score(y_test, y_pred_lr)
final_accuracy_svm = accuracy_score(y_test, y_pred_svm)

print(f"Final Logistic Regression Accuracy: {final_accuracy_lr}")
print(f"Final SVM Accuracy: {final_accuracy_svm}")

final_model = clf_lr if final_accuracy_lr > final_accuracy_svm else clf_svm

final_y_pred = final_model.predict(X_test_scaled)
print("Final Accuracy:", accuracy_score(y_test, final_y_pred))
print("Final Confusion Matrix:\n", confusion_matrix(y_test, final_y_pred))
print("Final Classification Report:\n", classification_report(y_test, final_y_pred, zero_division=1))
