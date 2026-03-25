from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load trained model
model = pickle.load(open("house_price_model.pkl", "rb"))

# Load dataset to rebuild locality encoder
df = pd.read_csv("MagicBricks.csv")


# SAME locality grouping used in notebook
def grp_local(locality):
    locality = locality.lower()

    if 'rohini' in locality:
        return 'Rohini Sector'
    elif 'dwarka' in locality:
        return 'Dwarka Sector'
    elif 'shahdara' in locality:
        return 'Shahdara'
    elif 'vasant' in locality:
        return 'Vasant Kunj'
    elif 'paschim' in locality:
        return 'Paschim Vihar'
    elif 'alaknanda' in locality:
        return 'Alaknanda'
    elif 'vasundhar' in locality:
        return 'Vasundhara Enclave'
    elif 'punjabi' in locality:
        return 'Punjabi Bagh'
    elif 'kalkaji' in locality:
        return 'Kalkaji'
    elif 'lajpat' in locality:
        return 'Lajpat Nagar'
    else:
        return 'Other'


# Apply same preprocessing
df['Locality'] = df['Locality'].apply(grp_local)

# Recreate encoder
locality_encoder = LabelEncoder()
df['Locality'] = locality_encoder.fit_transform(df['Locality'])

localities = locality_encoder.classes_


@app.route("/")
def home():
    return render_template("index.html", localities=localities)


@app.route("/predict", methods=["POST"])
def predict():

    try:

        area = float(request.form["Area"])
        bhk = int(request.form["BHK"])
        bathroom = int(request.form["Bathroom"])
        parking = int(request.form["Parking"])
        per_sqft = float(request.form["Per_Sqft"])

        # Encoding categorical inputs
        furnishing_map = {
            "Unfurnished": 0,
            "Semi-Furnished": 1,
            "Furnished": 2
        }

        status_map = {
            "Ready_to_move": 1,
            "Almost_ready": 0
        }

        transaction_map = {
            "New_Property": 1,
            "Resale": 0
        }

        type_map = {
            "Apartment": 0,
            "Builder_Floor": 1
        }

        furnishing = furnishing_map[request.form["Furnishing"]]
        status = status_map[request.form["Status"]]
        transaction = transaction_map[request.form["Transaction"]]
        type_ = type_map[request.form["Type"]]

        locality_text = request.form["Locality"]
        locality = locality_encoder.transform([locality_text])[0]

        # Arrange features exactly like training
        features = np.array([[area, bhk, bathroom, furnishing,
                              locality, parking, status,
                              transaction, type_, per_sqft]])

        prediction = model.predict(features)[0]

        # Convert crores → rupees
        price_rupees = round(prediction * 10000000)

        return jsonify({
            "price": "{:,}".format(price_rupees)
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        })


if __name__ == "__main__":
    app.run(debug=True)