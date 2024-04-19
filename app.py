from flask import Flask, render_template, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from chat import get_response
import logging
import nltk
import pandas as pd
import torch

app = Flask(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

df_intent1 = pd.read_csv("csv_data/agamatrix1.csv")
df_intent2 = pd.read_csv("csv_data/Book2.csv")
df_intent3 = pd.read_csv("csv_data/diabetes1.csv")
df_intent4 = pd.read_csv("csv_data/diabetes2.csv")
df_intent5 = pd.read_csv("csv_data/diabetes5.csv")
df_intent6 = pd.read_csv("csv_data/eatingwell.csv")
df_intent7 = pd.read_csv("csv_data/gleneagles1.csv")
df_intent8 = pd.read_csv("csv_data/doctorqstns.csv")
df_intent9 = pd.read_csv("csv_data/Questions.csv")

@app.get("/")
def index_get():
    return render_template("base.html")


@app.post("/predict")
def predict():
    text = request.get_json().get("message")
    #todo check if text is valid
    response = get_response(text)
    message = {"answer": response}
    return jsonify(message)

if __name__ =="__main__":
    app.run(debug=True)

@app.route("/")
def index():
    return render_template("base.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    message = data.get("message")

    if not message:
        return jsonify({"error": "Invalid input"}), 400

    response = get_response(message)
    return jsonify({"answer": response})

if __name__ == "__main__":
    app.run(port=5001)