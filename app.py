from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load model dari file pickle
# Pastikan 'model.pkl' ada di folder project
model_path = 'knn_model.pkl'
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# Target names sesuai urutan dataset Iris
target_names = ['setosa', 'versicolor', 'virginica']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Mengambil input dari form
        sepal_l = float(request.form['sepal_l'])
        sepal_w = float(request.form['sepal_w'])
        petal_l = float(request.form['petal_l'])
        petal_w = float(request.form['petal_w'])
        
        input_data = np.array([[sepal_l, sepal_w, petal_l, petal_w]])
        
        # Prediksi menggunakan model yang sudah di-load
        prediction = model.predict(input_data)
        result = target_names[prediction[0]]
        
        return render_template('index.html', prediction=result)
    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
