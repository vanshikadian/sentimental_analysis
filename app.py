from flask import Flask, request, jsonify
import joblib
import pickle

app = Flask(__name__)

# Load your model and other necessary components
model = pickle.load(open('sentimental_analysis_models/model_xgb.pkl', 'rb'))
vectorizer = pickle.load(open('sentimental_analysis_models/countVectorizer.pkl', 'rb'))
scaler = pickle.load(open('sentimental_analysis_models/scaler.pkl', 'rb'))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    review = data['review']
    review_vectorized = vectorizer.transform([review])
    review_scaled = scaler.transform(review_vectorized.toarray())
    prediction = model.predict(review_scaled)
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
