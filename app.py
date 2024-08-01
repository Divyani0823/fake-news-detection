import streamlit as st
import pandas as pd
import re
import pickle
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Load the pre-trained model and vectorizer
with open('logistic_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Stemming function
ps = PorterStemmer()
def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [ps.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

# Streamlit interface
st.title('Fake News Detection')

input_text = st.text_area('Enter the news text')

if st.button('Predict'):
    if input_text:
        processed_text = stemming(input_text)
        vectorized_input = vectorizer.transform([processed_text])
        prediction = model.predict(vectorized_input)

        if prediction[0] == 1:
            st.write('Fake news')
        else:
            st.write('Real news')
    else:
        st.write('Please enter some text to classify.')

