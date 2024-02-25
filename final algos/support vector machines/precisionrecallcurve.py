import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

# Load the dataset from the CSV file
data = pd.read_csv('dataset1.csv')

# Split the dataset into features (X) and the target variable (y)
X = data.iloc[:, 2:]  # Exclude 'class_bot' and 'id' columns as they are not features
y = data['class_bot']

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the SVM classifier
svm_classifier = SVC(kernel='rbf', C=0.0035)
svm_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = svm_classifier.decision_function(X_test)  # Use decision_function to get scores for PR curve

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred > 0)
report = classification_report(y_test, y_pred > 0)
average_precision = average_precision_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(report)
print(f"Average Precision Score: {average_precision:.2f}")

# Compute Precision-Recall curve
precision, recall, thresholds = precision_recall_curve(y_test, y_pred)

# Plot the Precision-Recall curve
plt.figure()
plt.step(recall, precision, color='b', where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.show()
