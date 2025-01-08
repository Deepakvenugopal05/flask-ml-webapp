# Flask ML WebApp

A Flask-based web application for training and deploying machine learning models. This app supports both regression and classification tasks and provides endpoints for training models with custom datasets, making predictions, and visualizing results.

---

## Features

- Train a regression model using RandomForestRegressor.
- Train a classification model using RandomForestClassifier or MultiOutputClassifier for multi-target classification.
- Load datasets via CSV files for training.
- Predict outputs for new data using trained models.
- Save and load trained models and scalers.
- Preprocess datasets with standard scaling and label encoding for categorical columns.

---

## Tech Stack

- **Backend**: Flask
- **Machine Learning**: Scikit-learn
- **Frontend**: HTML with Tailwind CSS
- **Utilities**: Joblib for model persistence, Pandas for data handling.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/flask-ml-webapp.git
   cd flask-ml-webapp
   
Create a virtual environment:

bash
`python3 -m venv venv`
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
Install dependencies:

bash
`pip install -r requirements.txt`


Run the application:
bash
`python app.py`
