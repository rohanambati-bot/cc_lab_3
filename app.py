from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

# Load ML model and vectorizer
model = joblib.load("spam_classifier_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    result = ""

    if request.method == "POST":
        message = request.form["message"]
        data = vectorizer.transform([message])
        prediction = model.predict(data)[0]

        if prediction == 1:
            result = "SPAM"
        else:
            result = "NOT SPAM"

    return render_template("index.html", result=result)

# IMPORTANT for Docker + low-resource VM
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=False)

