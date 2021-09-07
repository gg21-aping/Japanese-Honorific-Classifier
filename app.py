import re
import flask
from flask import Flask, render_template, request,jsonify
import pickle
import numpy as np
import nagisa

app = Flask(__name__)

def tokenizer_jp(text):
  words = nagisa.filter(text, filter_postags=['助詞'])
  return words.words

model = pickle.load(open('model.pkl','rb'))
vectorizer = pickle.load(open('vectorizer.pkl','rb'))

@app.route('/', methods=['GET'])
def home():
  return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
  if request.method == 'POST':
    data = request.form["sentence"]
    input_text = [data]
    input_text[0] = input_text[0].replace("。","").replace("、","")
    input_text[0] = re.sub(r"\d+", "", input_text[0])

    input_vector = vectorizer.transform(input_text)
    prediction = model.predict(input_vector)

  if prediction == 0:
    return render_template("index.html", prediction_text=f'"{data}" is INFORMAL!')
  elif prediction == 1:
    return render_template("index.html", prediction_text=f'"{data}" is POLITE!')
  else:
    return render_template('index.html', prediction_text=f'"{data}" is FORMAL!')


app.run()
