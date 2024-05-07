from flask import Flask, render_template, request

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
import pickle

app = Flask(__name__, template_folder='templates')

# Load the dataset
balanced_data = pd.read_csv('bank.csv')
X = balanced_data.drop(columns=['target'])
y = balanced_data['target']
print(X.count())

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the classifier pipeline
clf = BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=300, random_state=42)

# Train the classifier
clf.fit(X_train, y_train)

# with open('trained_model.pkl.pkl', 'rb') as file:
#     random_forest_model = pickle.load(file)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the form data
        age = int(request.form['age'])
        job = request.form['job']
        marital = request.form['marital']
        education = request.form['education']
        credit = request.form.get('credit') == 'on'
        balance = int(request.form['balance'])
        housing = request.form.get('housing') == 'on'
        loan = request.form.get('loan') == 'on'
        month = request.form['month']
        duration = int(request.form['duration'])
        campaign = int(request.form['campaign'])
        pdays = int(request.form['pdays'])
        previous = int(request.form['previous'])

        def switch_job(job):
            job_mapping = {'Admin': 0, 'Blue-collar': 1, 'Entrepreneur': 2, 'Housemaid': 3,
                           'Management': 4, 'Retired': 5, 'Self-employed': 6, 'Services': 7,
                           'Student': 8, 'Technician': 9, 'Unemployed': 10, 'Unknown': 11}
            return job_mapping.get(job, 11)  # Return 11 if job not found

        def switch_marital(marital):
            marital_mapping = {'Divorced': 0, 'Married': 1, 'Single': 2}
            return marital_mapping.get(marital, 2)  # Return 2 if marital not found

        def switch_education(edu):
            education_mapping = {'Primary': 0, 'Secondary': 1, 'Tertiary': 2, 'Unknown': 3
                                 }
            return education_mapping.get(edu, 3)  # Return 3 if education not found

        def switch_YN(opt):
            return int(opt)  # Convert True/False to 1/0 directly

        def switch_month(mm):
            month_mapping = {'April': 0, 'August': 1, 'December': 2, 'February': 3,
                             'January': 4, 'July': 5, 'June': 6, 'March': 7,
                             'May': 8, 'November': 9, 'October': 10, 'September': 11}
            return month_mapping.get(mm, 11)  # Return 11 if month not found

        # Usage of switch functions
        job = switch_job(job)
        marital = switch_marital(marital)
        education = switch_education(education)
        credit = switch_YN(credit)
        housing = switch_YN(housing)
        loan = switch_YN(loan)
        month = switch_month(month)

        # Preprocess the input data
        input_data = pd.DataFrame({
            'ordinalencoder__V2': [job],
            'ordinalencoder__V3': [marital],
            'ordinalencoder__V4': [education],
            'ordinalencoder__V5': [credit],
            'ordinalencoder__V7': [housing],
            'ordinalencoder__V8': [loan],
            'ordinalencoder__V11': [month],
            'remainder__V1': [age],
            'remainder__V6': [balance],
            'remainder__V12': [duration],
            'remainder__V13': [campaign],
            'remainder__V14': [pdays],
            'remainder__V15': [previous]
        })

        # Make prediction
        prediction = clf.predict(input_data)

        prediction_label = "YES" if prediction[0] == 2 else "NO"

        return render_template('predict.html', prediction=prediction_label)


if __name__ == '__main__':
    app.run(debug=True)
