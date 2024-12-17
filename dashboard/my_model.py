# my_model.py

import joblib
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

class MyModel:
    def __init__(self, model_path=None):
        if model_path:
            self.model = joblib.load(model_path)
        else:
            self.model = GaussianNB()

    def train(self, X, y, test_size=0.3, random_state=42):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        self.evaluate(y_test, y_pred)

    def evaluate(self, y_test, y_pred):
        accuracy = round(accuracy_score(y_test, y_pred), 3)

        print("\nModel Evaluation:")
        print(f"Accuracy: {accuracy}")

    def predict(self, input_data):
        prediction = self.model.predict(input_data)
        return prediction

    def save(self, model_path):
        joblib.dump(self.model, model_path)
