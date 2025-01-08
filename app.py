from flask import Flask, jsonify, request, render_template
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.multioutput import MultiOutputClassifier
import pandas as pd
import joblib


app = Flask(__name__)

model = None

@app.route('/',methods=['GET'])
def index():
    return render_template('index.html')

def load_regression_data():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    # file input
    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and file.filename.endswith('.csv'):
        data = pd.read_csv(file)
        return data
    else:
         return jsonify({'error': 'Invalid file format. Only CSV files are allowed.'}), 400


@app.route('/train_regression', methods=['POST'])
def train_regression_model():
    global regression_model, feature_scaler
    try:
        # Load data
        data = load_regression_data()
        print(type(data))
        
        request_data = request.form.to_dict()
        target_columns = request_data.get('target_columns')
        feature_columns = request_data.get('feature_columns')

        if not target_columns or not feature_columns:
            return jsonify({'error': 'Target and feature columns must be provided'}), 400

        feature_columns = feature_columns.split(',')
        target_columns = target_columns.split(',')

        # Scale features
        feature_scaler = StandardScaler()
        X = data[feature_columns]
        y = data[target_columns]
        X_scaled = feature_scaler.fit_transform(X)

        # Train model
        regression_model = RandomForestRegressor(n_estimators=100, random_state=42)
        regression_model.fit(X_scaled, y)

        # Save scalers for features and targets
        joblib.dump(regression_model, 'regression_model.pkl')
        joblib.dump(feature_scaler, 'feature_scaler.pkl')

        return jsonify({'message': 'Regression model trained and saved successfully',"accuracy": regression_model.score(X_scaled, y)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_regression', methods=['POST'])
def predict_regression():
    try:
        # load the model and scaler
        regression_model = joblib.load('regression_model.pkl')
        feature_scaler = joblib.load('feature_scaler.pkl')

        # Get input data 
        data = request.get_json(force=True)
        input_data = pd.DataFrame([data])

        # scaling the input 
        input_scaled = feature_scaler.transform(input_data)

        predictions = regression_model.predict(input_scaled)

        prediction_list = predictions.tolist()

        return jsonify({'prediction': prediction_list})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Classification
def load_classification_data():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    # File input
    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and file.filename.endswith('.csv'):
        data = pd.read_csv(file)
        return data
    else:
         return jsonify({'error': 'Invalid file format. Only CSV files are allowed.'}), 400

@app.route('/train_classification', methods=['POST'])
def train_classification_model():
    global classification_model, feature_scaler
    try:
        data = load_classification_data()
        cols = data.columns

        # Detect categorical columns 
        categorical_columns = [col for col in cols if data[col].dtype == 'object' or len(data[col].unique()) < 10]

        # LabelEncoder for each categorical column
        label_encoders = {}
        for col in categorical_columns:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
            label_encoders[col] = le

        # Save the label encoders
        joblib.dump(label_encoders, 'label_encoders.pkl')

        # Extract target and feature columns from the request
        request_data = request.form.to_dict()
        target_columns = request_data.get('target_columns')
        feature_columns = request_data.get('feature_columns')

        if not target_columns or not feature_columns:
            return jsonify({'error': 'Target and feature columns must be provided'}), 400

        feature_columns = feature_columns.split(',')
        target_columns = target_columns.split(',')

        # Scale features
        feature_scaler = StandardScaler()
        X = data[feature_columns]
        y = data[target_columns]
        X_scaled = feature_scaler.fit_transform(X)

        # Train model
        if len(target_columns) > 1:
            base_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
            classification_model = MultiOutputClassifier(base_classifier)
        else:
            classification_model = RandomForestClassifier(n_estimators=100, random_state=42)

        classification_model.fit(X_scaled, y)

        # Save model and scaler
        joblib.dump(classification_model, 'classification_model.pkl')
        joblib.dump(feature_scaler, 'feature_scaler.pkl')

        accuracy = classification_model.score(X_scaled, y)
        return jsonify({'message': 'Classification model trained and saved successfully', "accuracy": accuracy})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict_classification', methods=['POST'])
def predict_classification():
    try:
        # load the train file
        classification_model = joblib.load('classification_model.pkl')
        feature_scaler = joblib.load('feature_scaler.pkl')
        label_encoders = joblib.load('label_encoders.pkl')

        data = request.get_json(force=True)
        input_data = pd.DataFrame([data])

        # Catogories the columns
        categorical_columns = [col for col in input_data.columns if input_data[col].dtype == 'object' or len(input_data[col].unique()) < 10]

        for col in categorical_columns:
            if col in label_encoders:
                input_data[col] = label_encoders[col].transform(input_data[col])

        input_scaled = feature_scaler.transform(input_data)

        # Predicting the data
        predictions = classification_model.predict(input_scaled)

        prediction_list = predictions.tolist()

        return jsonify({'prediction': prediction_list})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Load Iris dataset
def load_data():
    data = load_iris()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    print(df.head())
    return df

@app.route('/train', methods=['POST'])
def train_model():
    global model

    try:
        df = load_data()
        X = df.drop('target', axis=1)
        y = df['target']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Save the model to a file
        joblib.dump(model, 'classification_model.pkl')

        return jsonify({'message': 'Model trained and saved successfully', "accuracy": accuracy})
    except Exception as e:
        return jsonify({'error': str(e)})
    
@app.route('/predict', methods=['POST'])
def predict():
    try:
        model = joblib.load('classification_model.pkl')

        data = request.get_json(force=True)
        print("Input data:", data)

        input_data = pd.DataFrame([data])
        print("Input data after conversion:", input_data)

        feature_rename_map = {
            'sepal_length': 'sepal length (cm)',
            'sepal_width': 'sepal width (cm)',
            'petal_length': 'petal length (cm)',
            'petal_width': 'petal width (cm)'
        }

        input_data = input_data.rename(columns=feature_rename_map)

        input_data = input_data.apply(pd.to_numeric)

        expected_features = load_iris().feature_names

        if list(input_data.columns) != expected_features:
            return jsonify({'error': f'Invalid input data. Expected features: {expected_features}'}), 400

        prediction = int(model.predict(input_data)[0])

        results = {
            0: 'setosa',
            1: 'versicolor',
            2: 'virginica'
        }

        return jsonify({'prediction': results[prediction]})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500



@app.route('/testing_query_param', methods=['GET'])
def query_param():
    name = request.args.get("name")

    if name=="regresion":
        a,b= 10,29
        return jsonify({"a":a,"b":b})
    elif name =="classification":
        a,b= 50,49
        return jsonify({"a":a,"b":b})
    else:
        return jsonify({'error': 'Invalid query parameter'}), 400



if __name__ == '__main__':
    app.run(debug=True)
