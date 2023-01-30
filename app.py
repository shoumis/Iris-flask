
import pickle
import numpy as np
from flask import Flask , render_template, request

model = pickle.load(open("iris_svm_model.pkl","rb"))

app = Flask(__name__)

@app.route("/")

def home():
    return render_template("index_iris.html")


@app.route("/predict",methods = ['POST'])
def predict():
    if request.method=="POST":
        init_features = [float(x) for x in request.form.values()]
        final_features = [np.array(init_features)]
        pred = model.predict(final_features)
        if pred == "Iris-setosa":
            prediction = "Classified to Iris-setosa species"
        elif pred == "Iris-versicolor":
            prediction = "Classified to Iris-versicolor species "
        else:
            prediction = "Classified to Iris-virginica species"
        return render_template("index_iris.html",prediction_text = "Predicited class: {}".format(prediction))


if __name__ == "__main__":
    app.run(debug=True)
