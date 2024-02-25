import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import RandomOverSampler

# Load the dataset
data = pd.read_csv('filtered_data.csv')

# Define features and target variable
X = data.drop(columns=['class_bot', 'id'])
y = data['class_bot']

# Balance the dataset by oversampling the minority class
sampler = RandomOverSampler(random_state=42)
X_resampled, y_resampled = sampler.fit_resample(X, y)

# Split the resampled dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Create and train the Decision Tree model with additional tuning
model = DecisionTreeClassifier(max_depth=2, min_samples_split=1205, min_samples_leaf=715, random_state=42)
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
