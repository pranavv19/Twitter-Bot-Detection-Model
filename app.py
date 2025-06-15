from flask import Flask, render_template, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

app = Flask(__name__)

def get_decision_tree_results():
    data = pd.read_csv('decision trees/filtered_data.csv')
    X = data.drop(columns=['class_bot', 'id'])
    y = data['class_bot']
    sampler = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = sampler.fit_resample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
    model = DecisionTreeClassifier(max_depth=2, min_samples_split=1205, min_samples_leaf=715, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1]
    return get_metrics(y_test, y_pred, y_score)

def get_logistic_regression_results():
    data = pd.read_csv('logistic regression/filtered_data.csv')
    X = data.drop(columns=['class_bot', 'id'])
    y = data['class_bot']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(C=0.0007)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1]
    return get_metrics(y_test, y_pred, y_score)

def get_random_forest_results():
    data = pd.read_csv('random forest/dataset1.csv')
    X = data.drop(columns=['class_bot', 'id'])
    y = data['class_bot']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1]
    return get_metrics(y_test, y_pred, y_score)

def get_naive_bayes_results():
    data = pd.read_csv('Naive bayes/dataset1.csv')
    X = data.drop(columns=['class_bot', 'id'])
    y = data['class_bot']
    rus = RandomUnderSampler(random_state=42)
    X_resampled, y_resampled = rus.fit_resample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
    model = GaussianNB(var_smoothing=1e+1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1]
    return get_metrics(y_test, y_pred, y_score)

def get_svm_results():
    data = pd.read_csv('support vector machines/dataset1.csv')
    X = data.drop(columns=['class_bot', 'id'])
    y = data['class_bot']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = SVC(kernel='rbf', C=0.0035, probability=True)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1]
    return get_metrics(y_test, y_pred, y_score)

def get_metrics(y_test, y_pred, y_score):
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred).tolist()
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(y_test, y_score)
    avg_precision = average_precision_score(y_test, y_score)
    return {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': conf_matrix,
        'roc_curve': {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'auc': roc_auc
        },
        'pr_curve': {
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'avg_precision': avg_precision
        }
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/results')
def get_results():
    results = {
        'Decision Tree': get_decision_tree_results(),
        'Random Forest': get_random_forest_results(),
        'Naive Bayes': get_naive_bayes_results(),
        'Logistic Regression': get_logistic_regression_results(),
        'SVM': get_svm_results()
    }
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True) 