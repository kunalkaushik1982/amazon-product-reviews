from flask import Flask, render_template,request, redirect, url_for
import mlflow
from mlflow.tracking import MlflowClient
from preprocessing_utility import normalize_text
import dagshub
import pickle

mlflow.set_tracking_uri('https://dagshub.com/kunalkaushik1982/amazon-product-reviews.mlflow')
dagshub.init(repo_owner='kunalkaushik1982', repo_name='amazon-product-reviews', mlflow=True)

app = Flask(__name__)

# In-memory list to store reviews and reactions
history = []

# load model from model registry
def get_latest_model_version(model_name):
    client = MlflowClient()
    latest_version = client.get_latest_versions(model_name, stages=["Production"])
    if not latest_version:
        latest_version = client.get_latest_versions(model_name, stages=["Staging"])
    return latest_version[0].version if latest_version else None

model_name = "my_model"
model_version = get_latest_model_version(model_name)

model_uri = f'models:/{model_name}/{model_version}'
model = mlflow.pyfunc.load_model(model_uri)

vectorizer = pickle.load(open('models/vectorizer.pkl','rb'))



@app.route('/')
def home():
    #return render_template('index.html',result=None)
    return render_template('index.html', result=None, text=None, history=history)

@app.route('/predict', methods=['POST'])
def predict():

    text = request.form['text']

    # clean
    text = normalize_text(text)

    # bow
    features = vectorizer.transform([text])

    # prediction
    result = model.predict(features)

    # Map numerical result to user-friendly string
    result_str = 'Sad' if result[0] == 1 else 'Happy'

    # Add to history list (store review text and result)
    history.append({'text': text, 'reaction': result_str})

    # Pass the history, result, and text to be displayed
    return render_template('index.html', result=result_str, text=text, history=history)

app.run(debug=True)