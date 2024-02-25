import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt

# Load the dataset from the CSV file
data = pd.read_csv('dataset1.csv', header=0, delimiter=',')

# Split the dataset into features (X) and target variable (y)
X = data.iloc[:, 2:]  # Exclude 'class_bot' and 'id' columns as they are not features
y = data['class_bot']

# Perform undersampling to balance the dataset
rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X, y)

# Split the resampled data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Create and train the Gaussian Naive Bayes model with Laplace smoothing
model = GaussianNB(var_smoothing=1e+1)  # Adjust the smoothing parameter as needed
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

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
