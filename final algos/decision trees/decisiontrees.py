import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Load the dataset
data = pd.read_csv('filtered_data.csv')

# Prepare the feature matrix (X) and the target variable (y)
X = data.drop(columns=['class_bot', 'id'])
y = data['class_bot']

# Balance the dataset by oversampling the minority class
sampler = RandomOverSampler(random_state=42)
X_resampled, y_resampled = sampler.fit_resample(X, y)

# Create a Random Forest model
model = RandomForestClassifier(random_state=42)

# Implement stratified k-fold cross-validation
skf = StratifiedKFold(n_splits=5)
cross_val_scores = []
for train_index, test_index in skf.split(X_resampled, y_resampled):
    X_train, X_test = X_resampled.iloc[train_index], X_resampled.iloc[test_index]
    y_train, y_test = y_resampled.iloc[train_index], y_resampled.iloc[test_index]

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    cross_val_scores.append(accuracy_score(y_test, predictions))

# Calculate average cross-validation score
average_cv_score = np.mean(cross_val_scores)

# Generate a classification report and confusion matrix for the last fold
report = classification_report(y_test, predictions)
conf_matrix = confusion_matrix(y_test, predictions)

# Print the results
print(f"Cross-Validation Scores: {cross_val_scores}")
print(f"Average CV Score: {average_cv_score}")
print("Classification Report:")
print(report)
print("Confusion Matrix:")
print(conf_matrix)