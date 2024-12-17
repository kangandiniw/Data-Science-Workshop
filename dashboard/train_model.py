import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, classification_report
import joblib

# Baca dataset
data = pd.read_csv("data/breast-cancer-wisconsin.data")

# Ganti nama kolom sesuai dengan data breast cancer
data.columns = ['ID', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 
                'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 
                'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class']

# Menghapus kolom 'ID' dan mengatasi nilai yang hilang
data = data.drop(columns=['ID'])
data = data.replace('?', np.NaN)
data = data.dropna()

# Memisahkan fitur dan label
X = data.drop(columns=['Class'])
y = data['Class'].astype(int)

# Bagi data menjadi data pelatihan dan pengujian
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Inisialisasi model
model = GaussianNB()

# Latih model
model.fit(X_train, y_train)

# Lakukan prediksi pada data pengujian
y_pred = model.predict(X_test)

# Evaluasi model
accuracy = round(accuracy_score(y_test, y_pred), 3)
recall = round(recall_score(y_test, y_pred, average='weighted'), 3)
f1 = round(f1_score(y_test, y_pred, average='weighted'), 3)
precision = round(precision_score(y_test, y_pred, average='weighted'), 3)

print("\nModel Evaluation:")
print(f"Accuracy: {accuracy}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"Precision: {precision}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Simpan model
joblib.dump(model, "model/my_model.pkl")
