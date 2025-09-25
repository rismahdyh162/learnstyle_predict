from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load model dari file .sav
model = pickle.load(open('learnstylemodel.sav', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')  

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        gender_input = request.form.get('gender', '').lower()
        age = float(request.form.get('age', 0))

        # One-hot encoding manual untuk Gender
        if gender_input == 'male':
            gender_female, gender_male = 0, 1
        elif gender_input == 'female':
            gender_female, gender_male = 1, 0
        else:
            gender_female, gender_male = 0, 0  # default jika tidak valid

        try:
            q1 = float(request.form.get('q1', 0))
            q2 = float(request.form.get('q2', 0))
            q3 = float(request.form.get('q3', 0))
            q4 = float(request.form.get('q4', 0))
            q5 = float(request.form.get('q5', 0))
            q6 = float(request.form.get('q6', 0))
            q7 = float(request.form.get('q7', 0))
            q8 = float(request.form.get('q8', 0))
            q9 = float(request.form.get('q9', 0))
            q10 = float(request.form.get('q10', 0))
            q11 = float(request.form.get('q11', 0))
            q12 = float(request.form.get('q12', 0))
            q13 = float(request.form.get('q13', 0))
            q14 = float(request.form.get('q14', 0))
        except ValueError:
            return render_template('index.html', hasil="Isi semua pertanyaan dengan benar.")

        # Susun sesuai urutan fitur yang dipakai saat training 
        data = np.array([[age, q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11, q12, q13, q14, gender_female, gender_male]])


        # Prediksi
        prediction = model.predict(data)
        predicted_class = prediction[0]

        # Mapping hasil
        label_mapping = {
            0: "Visual",
            1: "Auditori",
            2: "Kinestetik"
        }
        hasil_akhir = label_mapping.get(predicted_class, "Tidak diketahui")

        # Tampilkan hasil di halaman yang sama
        return render_template('index.html', hasil=hasil_akhir)
    
        expected_features = 17
        if data.shape[1] != expected_features:
            return render_template('index.html', hasil="Jumlah fitur input tidak sesuai.")
    

if __name__ == '__main__':
    app.run(debug=True)
