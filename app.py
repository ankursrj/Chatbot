from flask import Flask, request, jsonify, render_template
import numpy as np
import nltk
from nltk.stem.lancaster import LancasterStemmer
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import json
import random
import os

# Initialize Flask app
app = Flask(__name__)

# Load the data
with open("intents.json") as file:
    data = json.load(file)

# Initialize stemmer
stemmer = LancasterStemmer()

# Preprocessing steps
words = []
labels = []
docs_x = []
docs_y = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent["tag"])

    if intent["tag"] not in labels:
        labels.append(intent["tag"])

words = [stemmer.stem(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))

labels = sorted(labels)

training = []
output = []

out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag = []

    wrds = [stemmer.stem(w) for w in doc]

    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)
    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)

training = np.array(training)
output = np.array(output)

# Ensure model is compiled before loading weights
model = Sequential([
    Dense(128, input_shape=(len(training[0]),), activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(len(output[0]), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Check if the weights file exists
if os.path.isfile("model.weights.h5"):
    model.load_weights("model.weights.h5")
else:
    print("Model weights file not found. Please ensure 'model.weights.h5' is in the correct directory.")
    exit(1)

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]
    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    return np.array(bag)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.content_type != 'application/json':
        return jsonify({"response": "Invalid request content type"}), 400

    data = request.get_json()
    message = data.get('message')
    if not message:
        return jsonify({"response": "Message is required"}), 400

    # Load the intents from the intents.json file
    with open("intents.json") as file:
        intents = json.load(file)["intents"]

    results = model.predict(np.array([bag_of_words(message, words)]))[0]
    results_index = np.argmax(results)
    tag = labels[results_index]

    if results[results_index] > 0.5:
        for intent in intents:
            if intent['tag'] == tag:
                responses = intent['responses']
        response = random.choice(responses)
    else:
        response = "I didn't get that, try again"

    return jsonify({'response': response})

if __name__ == "__main__":
    app.run(port=5000, debug=True)