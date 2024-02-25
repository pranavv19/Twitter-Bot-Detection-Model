import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset from the CSV file
data = pd.read_csv('dataset1.csv')

# Split the dataset into features (X) and target variable (y)
X = data.iloc[:, 2:]  # Exclude 'class_bot' and 'id' columns as they are not features
y = data['class_bot']

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the SVM classifier
svm_classifier = SVC(kernel='rbf',C=0.0035)  # You can choose different kernel types (e.g., 'linear', 'rbf', 'poly')
svm_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = svm_classifier.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(report)
