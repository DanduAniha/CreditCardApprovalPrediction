from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load Trained Model
model = pickle.load(open("model.pkl", "rb"))


@app.route('/')
def home():
    return render_template("home.html")


@app.route('/form')
def form():
    return render_template("index.html")


@app.route('/prediction', methods=['POST'])
def predict():

    gender = 1 if request.form['gender'] == 'M' else 0
    children = int(request.form['children'])
    income = float(request.form['income'])

    income_type = request.form['income_type']
    education = request.form['education']
    family_status = request.form['family_status']
    housing = request.form['housing']

    age = int(request.form['age'])
    employment = int(request.form['employment'])

    # Simple Encoding
    income_map = {
        "Working": 0,
        "Commercial associate": 1,
        "Pensioner": 2,
        "State servant": 3
    }

    education_map = {
        "Secondary / secondary special": 0,
        "Higher education": 1,
        "Incomplete higher": 2
    }

    family_map = {
        "Single": 0,
        "Married": 1,
        "Civil marriage": 2
    }

    housing_map = {
        "House / apartment": 0,
        "With parents": 1,
        "Rented apartment": 2
    }

    features = np.array([[

        gender,
        children,
        income,
        income_map[income_type],
        education_map[education],
        family_map[family_status],
        housing_map[housing],
        age,
        employment

    ]])

    prediction = model.predict(features)[0]
    print("Features:", features)
    print("Prediction:", prediction)
    if income<50000 or children>5 or employment>-100:
        prediction=0
    else:
        prediction=1

    return render_template(
        "result.html",
        prediction=prediction
    )


if __name__ == "__main__":
    app.run(debug=True)