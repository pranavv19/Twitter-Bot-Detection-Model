import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE  # Assuming you have imbalanced classes

# Load the dataset from the CSV file
data = pd.read_csv('dataset1.csv')

# Split the dataset into features (X) and target variable (y)
X = data.iloc[:, 2:]  # Exclude 'class_bot' and 'id' columns as they are not features
y = data['class_bot']

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE for oversampling (assuming imbalanced classes)
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Create and train the Random Forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=4, min_samples_split=2, min_samples_leaf=2, max_depth=5)
model.fit(X_resampled, y_resampled)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Print the results
print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(report)

# Cross-validation scores for better estimate
cross_val_scores = cross_val_score(model, X, y, cv=5)
print("Cross-Validation Scores:", cross_val_scores)
