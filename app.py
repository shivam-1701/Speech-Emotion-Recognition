from flask import Flask, redirect
import numpy as np
from flask import Flask, request, render_template
import pickle
import json
import tensorflow as tf
from tensorflow import keras
import librosa
import pandas as pd
from keras.models import Sequential, Model, model_from_json

app = Flask(__name__)
json_file = open("C:\\Users\\prade\\PycharmProjects\\SER\\venv\\model_json.json", 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("C:\\Users\\prade\\PycharmProjects\\SER\\venv\\Emotion_Model.h5")

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/upload', methods=["GET","POST"])
def upload():
    prediction=""
    if request.method =="POST":
        print("FORM DATA RECIEVED")

        if "file" not in request.files:
            return redirect(request.url)

        file = request.files["file"]

        if file.filename == "":
            return redirect(request.url)

        if file:
            audio_file = file
            X, sample_rate = librosa.load(audio_file
                                          , res_type='kaiser_fast'
                                          , duration=2.5
                                          , sr=44100
                                          , offset=0.5
                                          )
            sample_rate = np.array(sample_rate)
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13), axis=0)
            newdf = pd.DataFrame(data=mfccs).T
            print(newdf)

            # Apply predictions
            newdf = np.expand_dims(newdf, axis=2)
            newpred = loaded_model.predict(newdf,
                                           batch_size=16,
                                           verbose=1)

            print(newpred)

            filename = r"C:\Users\prade\PycharmProjects\SER\venv\labels"
            infile = open(filename, 'rb')
            lb = pickle.load(infile)
            infile.close()

            # Get the final predicted label
            final = newpred.argmax(axis=1)
            final = final.astype(int).flatten()
            prediction = (lb.inverse_transform((final)))
            print(prediction)  # emo(final) #gender(final)

    return render_template('upload.html', transcript=prediction)


if __name__ == "__main__":
    app.run(debug=True, threaded = True)