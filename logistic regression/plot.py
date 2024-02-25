import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('filtered_data.csv')

# Define features and target variable
X = data.drop(columns=['class_bot', 'id'])
y = data['class_bot']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Logistic Regression model with regularization
model = LogisticRegression(C=0.0007)  # You can adjust C to control the strength of regularization
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate accuracy and generate a classification report
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Print the results
print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(report)

# Create a bar chart to visualize the distribution of predicted classes
classes = ['Legitimate Users', 'Bots']
predicted_counts = [len(y_pred[y_pred == 0]), len(y_pred[y_pred == 1])]

plt.bar(classes, predicted_counts, color=['blue', 'red'])
plt.xlabel('Class')
plt.ylabel('Count')
plt.title('Distribution of Predicted Classes')
plt.show()
