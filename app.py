from flask import Flask, render_template, request, jsonify
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

app = Flask(__name__)

# Training Model
iris = load_iris()
model = KNeighborsClassifier(n_neighbors=3)
model.fit(iris.data, iris.target)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Mengambil input dari form (mengubah string ke float)
        sepal_l = float(request.form['sepal_l'])
        sepal_w = float(request.form['sepal_w'])
        petal_l = float(request.form['petal_l'])
        petal_w = float(request.form['petal_w'])
        
        input_data = np.array([[sepal_l, sepal_w, petal_l, petal_w]])
        prediction = model.predict(input_data)
        result = iris.target_names[prediction[0]]
        
        return render_template('index.html', prediction=result)
    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
