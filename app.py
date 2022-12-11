from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
import numpy as np
import pickle
import streamlit as st
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)


model_load = keras.models.load_model('final_model.h5')

def generate_text_seq(model,tokenizer,text_seq_length,seed_text,n_words):
    text=[]
    #n_words=how many words i need to generate
    
    for _ in range(n_words):
        encoded=tokenizer.texts_to_sequences([seed_text])[0]
        encoded=pad_sequences([encoded],maxlen=text_seq_length,truncating='pre')
        
        y_predict=model.predict(encoded)
        
        
        predicted_word=''
        
        for word ,index in tokenizer.word_index.items():
            if index==np.argmax(y_predict):
                predicted_word=word
                break
        seed_text=seed_text+ ' '+predicted_word
        text.append(predicted_word)
    return ' '.join(text)
st.title('Auto-Text Generation ! ! ')
st.info("This application aims to auto-generate the text from William shakespear Sonnets. Write a simple sentence (not more than 50) words . it will Autocomplete the sentence")
input_text = st.text_area('Enter Text Below (maximum 50 words):', height=100)
submit=st.button('Generate')
if submit:
    st.subheader("Generated Text :")
    with st.spinner(text="This may take a moment..."):
        output=generate_text_seq(model_load,tokenizer,50,input_text,100)
    st.text_area(label ="",value=output, height = 500)
