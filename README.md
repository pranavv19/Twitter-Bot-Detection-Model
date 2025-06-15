# Twitter Bot Detection Model

This project is a web application for analyzing and visualizing the results of various machine learning models used to detect Twitter bots. It provides a modern UI to compare model performance, confusion matrices, ROC and Precision-Recall curves.

## Features
- Decision Tree, Random Forest, Logistic Regression, Naive Bayes, and SVM models
- Automatic data preprocessing as per original scripts
- Interactive web UI with model selection and visualizations
- Unified results for easy comparison

## Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd Twitter-Bot-Detection-Model
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python app.py
   ```

4. **Open your browser** and go to [http://localhost:5000](http://localhost:5000)

## Folder Structure
- `app.py` - Main Flask backend
- `templates/` - HTML templates for the UI
- `decision trees/`, `random forest/`, `logistic regression/`, `Naive bayes/`, `support vector machines/` - Model scripts and datasets

## Demo Video
A demo video showing the application in action is available below:

[![Demo Video](demo-thumbnail.png)](https://your-demo-video-link)

*Replace the link above with your actual video URL and add a thumbnail image as `demo-thumbnail.png` if desired.*

## Notes
- Ensure all required datasets are present in their respective folders.
- If you encounter import errors, make sure your Python environment is activated and all dependencies are installed.

---
Feel free to contribute or raise issues!
