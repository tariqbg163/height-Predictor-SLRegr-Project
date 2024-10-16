from flask import Flask , render_template , request , jsonify
import numpy as np
import pandas as pd
import pickle


with open("models/scaler.pkl" , 'rb') as f:
    scaler_model = pickle.load(f)

with open("models/Regressor.pkl" , 'rb') as f:
    li_model = pickle.load(f)  # linear_model


app = Flask(__name__)

@app.route("/")
def homePage():
    return render_template("index.html")


@app.route("/predictdata" , methods=["GET" , "POST"])
def predict_datapoint():
    if request.method == "POST":
        Weight = float(request.form.get("Weight"))

        new_scaled_data = scaler_model.transform([[Weight]])
        result = li_model.predict(new_scaled_data)

        return render_template("home.html" , result = result[0])
    
    else:
         return render_template("home.html")


if __name__ == "__main__":
    app.run(host= "0.0.0.0")
