import numpy as np
from flask import Flask,jsonify,render_template,request
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=["POST"])
def predict():
    int_features=[[int(x) for x in request.form.values()]]
    print(int_features)
    final_features=np.array(int_features)
    prediction=model.predict(final_features)
    output=prediction
    
    return render_template('index.html',prediction_text='Prediction is {}'.format(output))

@app.route('/predict_api/<int:feature1>/<int:feature2>/<int:feature3>/<int:feature4>/<int:feature5>', methods=['GET'])
def predict_api(feature1, feature2, feature3, feature4, feature5):
    # Convert the input features to a numpy array
    input_data = np.array([[feature1, feature2, feature3, feature4, feature5]])

    # Make the prediction using the model
    prediction = model.predict(input_data)

    output = np.round(prediction[0]).astype(int).tolist()

    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)