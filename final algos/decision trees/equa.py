import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler

# Load the dataset
data = pd.read_csv('filtered_data.csv', header=0, delimiter=',')

# Split the dataset into features and target variable
X = data.iloc[:, 2:]  # Exclude 'class_bot' and 'id' columns
y = data['class_bot']

# Perform undersampling
rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X, y)

# Create and adjust the Decision Tree model
model = DecisionTreeClassifier(max_depth=4, min_samples_split=20, min_samples_leaf=10, random_state=42)

# Evaluate the model using cross-validation
cross_val_scores = cross_val_score(model, X_resampled, y_resampled, cv=5)

# Split the resampled data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions and evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Print the results
print(f"Cross-Validation Scores: {cross_val_scores}")
print(f"Average CV Score: {cross_val_scores.mean()}")
print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(report)

# Visualization of predicted classes
classes = ['Legitimate Users', 'Bots']
predicted_counts = [len(y_pred[y_pred == 0]), len(y_pred[y_pred == 1])]
plt.bar(classes, predicted_counts, color=['blue', 'red'])
plt.xlabel('Class')
plt.ylabel('Count')
plt.title('Distribution of Predicted Classes')
plt.show()