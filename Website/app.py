from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import nltk 
import pickle
from flask import Flask, render_template, g, jsonify, request ,abort
import pandas as pd


app = Flask(__name__)

# Load the model
nltk.download("stopwords")

filename = './model1.sav'
loaded_model = pickle.load(open(filename, 'rb'))
arb_stopwords = set(nltk.corpus.stopwords.words("arabic"))
vectorizer = TfidfVectorizer(stop_words=arb_stopwords)
df = pd.read_csv("./clean_data.csv")
documents = df.Text.copy()
X = vectorizer.fit_transform(documents)

classes = ["Health", "Health","Politics", "Sports", "Economy"]
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    X = vectorizer.transform([request.form["content"]])
    prediction = loaded_model.predict(X)    
    data = {"result": classes[prediction[0]]}

    return data

"""
Error handlers
"""
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "success": False,
        "error": 404,
        "message": "We couldn't find what you are looking for!"
    }), 404


@app.errorhandler(400)
def bad_request(error):
    return jsonify({
        "success": False,
        "error": 400,
        "message": "Your request is not well formated!"
    }), 400


@app.errorhandler(500)
def unprocessable_entity(error):
    return jsonify({
        "success": False,
        "message": "Sorry, we coludn't proccess your request. ERROR during prediction"
    }), 500

if __name__ == "__main__":
    app.run(debug=True)

