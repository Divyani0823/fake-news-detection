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

# import streamlit as st
# import numpy as np
# import pandas as pd
# import re
# from transformers import GPT2Tokenizer, GPT2LMHeadModel
# import torch

# # Load pre-trained GPT-2 model and tokenizer
# tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# model = GPT2LMHeadModel.from_pretrained('gpt2')

# def generate_text(prompt, max_length=50):
#     inputs = tokenizer.encode(prompt, return_tensors='pt')
#     outputs = model.generate(inputs, max_length=max_length, num_return_sequences=1)
#     text = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return text

# # Example usage
# prompt = "Detecting fake news involves analyzing the credibility of sources and verifying the facts presented. A model trained on various news articles can be used to classify news as fake or real."
# generated_text = generate_text(prompt)
# print("Generated Text:", generated_text)

# # If you want to evaluate or use this for further purposes, you can integrate it as needed


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

