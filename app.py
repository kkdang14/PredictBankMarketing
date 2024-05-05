from flask import Flask, render_template, request

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier


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
            if job == 'Admin':
                return 0
            elif job == 'Blue-collar ':
                return 1
            elif job == 'Entrepreneur':
                return 2
            elif job == 'Housemaid':
                return 3
            elif job == 'Management':
                return 4
            elif job == 'Retired':
                return 5
            elif job == 'Self-employed':
                return 6
            elif job == 'Services':
                return 7
            elif job == 'Student':
                return 8
            elif job == 'Technician':
                return 9
            elif job == 'Unemployed':
                return 10
            elif job == 'Unknown':
                return 11
        def switch_marital(marital):
            if marital == 'Divorced':
                return 0
            elif marital == 'Married':
                return 1
            elif marital == 'Single':
                return 2
        def switch_education(edu):
            if edu == 'Primary':
                return 0
            elif edu == 'Secondary':
                return 1
            elif edu == 'Tertiary':
                return 2
            elif edu == 'Unknown':
                return 3
        def switch_YN(opt):
            if opt == True:
                return 1
            else: return 0
        def swtich_month(mm):
            if mm == 'April':
                return 0
            elif mm == 'August':
                return 1
            elif mm == 'December':
                return 2
            elif mm == 'February':
                return 3
            elif mm == 'January':
                return 4
            elif mm == 'July':
                return 5
            elif mm == 'June':
                return 6
            elif mm == 'March':
                return 7
            elif mm == 'May':
                return 8
            elif mm == 'November':
                return 9
            elif mm == 'October':
                return 10
            elif mm == 'September':
                return 11

        job = switch_job(job)
        marital = switch_marital(marital)
        education = switch_education(education)
        credit = switch_YN(credit)
        housing = switch_YN(housing)
        loan = switch_YN(loan)
        month = swtich_month(month)

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
