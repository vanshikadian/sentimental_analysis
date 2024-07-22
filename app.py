import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK stopwords and other necessary datasets
nltk.download()

# Load your model and other necessary components
try:
    model = pickle.load(open('sentimental_analysis_models/xgb.pkl', 'rb'))
    vectorizer = pickle.load(open('sentimental_analysis_models/countVectorizer.pkl', 'rb'))
    scaler = pickle.load(open('sentimental_analysis_models/scaler.pkl', 'rb'))
except Exception as e:
    st.error(f"Error loading model or components: {e}")

# Function to preprocess text
def preprocess_text(text):
    wordnet = WordNetLemmatizer()
    review = re.sub('[^a-zA-Z]', ' ', text)
    review = review.lower()
    review = review.split()
    review = [wordnet.lemmatize(word) for word in review if word not in set(stopwords.words('english'))]
    return ' '.join(review)

# Streamlit application
st.title('Sentiment Analysis of Amazon Reviews')

review_text = st.text_area('Enter a review:')
if st.button('Predict'):
    if review_text:
        try:
            # Preprocess the input review
            processed_text = preprocess_text(review_text)
            st.write(f"Processed text: {processed_text}")
            review_vectorized = vectorizer.transform([processed_text])
            st.write(f"Vectorized text: {review_vectorized.toarray()}")
            review_scaled = scaler.transform(review_vectorized.toarray())
            st.write(f"Scaled text: {review_scaled}")
            prediction = model.predict(review_scaled)
            
            result = 'Positive' if prediction[0] == 1 else 'Negative'
            st.write(f'Sentiment: {result}')
        except Exception as e:
            st.error(f"Error making prediction: {e}")
    else:
        st.write('Please enter a review text.')
