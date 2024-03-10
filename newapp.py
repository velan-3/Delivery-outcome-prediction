import numpy as np
from flask import Flask,jsonify,render_template,request
import pickle
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        age = request.form['age']
        delivery_number = request.form['deliveryNumber']
        delivery_time = request.form['deliveryRadio']
        blood_pressure = request.form['bpRadio']
        heart_problem = request.form['heartRadio']
        
        #for premature,early,latecomer
        if delivery_time == 'Timely' :
            delivery_time = 2
        elif delivery_time == 'Premature':
            delivery_time= 1
        else:
            delivery_time = 0
            
        # for high low   
        if blood_pressure == 'High':
            blood_pressure=0
        elif blood_pressure == 'Normal':
            blood_pressure =2
        else:
            blood_pressure=1
            
        #apt or inapt
        if heart_problem == 'apt':
            heart_problem=0
        else:
            heart_problem=1    
        

        # Preprocess data if needed (e.g., convert to numerical format)
        # ...

        # Call your ML model for prediction
        prediction = model.predict([[age, delivery_number, delivery_time, blood_pressure, heart_problem]])
        print(prediction)
        prediction.astype(int)

        return render_template('index.html', prediction=prediction)
    #else:
        #return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
