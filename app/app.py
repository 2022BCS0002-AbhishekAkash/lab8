from flask import Flask, request, jsonify
import pandas as pd
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

@app.route("/")
def home():
    return "California Housing Model API"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    return jsonify({"prediction": "Model deployed successfully"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)