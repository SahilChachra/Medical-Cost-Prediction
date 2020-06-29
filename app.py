from flask import Flask, request, url_for, redirect, render_template
import pickle

import numpy as np

app = Flask(__name__, template_folder='./templates', static_folder='./static')

Pkl_Filename = "rf_tuned.pkl" 
with open(Pkl_Filename, 'rb') as file:  
    model = pickle.load(file)
@app.route('/')

def hello_world():
    return render_template('home.html')

@app.route('/predict', methods=['POST','GET'])
def predict():
    features = [int(x) for x in request.form.values()]

    print(features)
    final = np.array(features).reshape((1,6))
    print(final)
    pred = model.predict(final)[0]
    print(pred)

    
    if pred < 0:
        return render_template('op.html', pred='Error calculating Amount!')
    else:
        return render_template('op.html', pred='Expected amount is {0:.3f}'.format(pred))

if __name__ == '__main__':
    app.run(debug=True)