from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load model
model_path = os.path.join(os.path.dirname(__file__), 'knn_model.pkl')
with open(model_path, 'rb') as f:
    model = pickle.load(f)

target_names = ['setosa', 'versicolor', 'virginica']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Mengambil input
        val_sl = request.form['sepal_l']
        val_sw = request.form['sepal_w']
        val_pl = request.form['petal_l']
        val_pw = request.form['petal_w']
        
        # Konversi ke float untuk model
        input_data = np.array([[float(val_sl), float(val_sw), float(val_pl), float(val_pw)]])
        
        # Prediksi
        prediction = model.predict(input_data)
        result = target_names[prediction[0]]
        
        # Kirim balik nilai input dan hasil prediksi ke template
        return render_template('index.html', 
                               prediction=result, 
                               sl=val_sl, sw=val_sw, pl=val_pl, pw=val_pw)
    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
