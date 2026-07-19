from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load model dari file pickle
model_path = os.path.join(os.path.dirname(__file__), 'knn_model.pkl')
with open(model_path, 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Mengambil input dari form dan langsung menjadikannya float
        val_sl = float(request.form['sepal_l'])
        val_sw = float(request.form['sepal_w'])
        val_pl = float(request.form['petal_l'])
        val_pw = float(request.form['petal_w'])
        
        # Konversi ke array 2D yang dibutuhkan sklearn
        input_data = np.array([[val_sl, val_sw, val_pl, val_pw]])
        
        # Lakukan prediksi
        prediction = model.predict(input_data)
        hasil_prediksi = prediction[0]
        
        # Logika sinkronisasi: 
        # Jika model mengeluarkan angka (0,1,2), ubah jadi teks.
        # Jika model mengeluarkan teks ('setosa', dll), langsung gunakan teksnya.
        target_names = ['Setosa', 'Versicolor', 'Virginica']
        if isinstance(hasil_prediksi, (int, np.integer)):
            result = target_names[int(hasil_prediksi)]
        else:
            result = str(hasil_prediksi).capitalize()
        
        # Kembalikan ke halaman dengan membawa hasil dan nilai input sebelumnya
        return render_template('index.html', 
                               prediction=result, 
                               sl=val_sl, sw=val_sw, pl=val_pl, pw=val_pw)
                               
    except Exception as e:
        # Jika error, tangkap input yang gagal agar tidak hilang di form
        return render_template('index.html', error=str(e),
                               sl=request.form.get('sepal_l', ''),
                               sw=request.form.get('sepal_w', ''),
                               pl=request.form.get('petal_l', ''),
                               pw=request.form.get('petal_w', ''))

if __name__ == '__main__':
    app.run(debug=True)
