import streamlit as st
import pickle
import nltk
import pandas as pd
import time

from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

def preprocess(text):
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens]
    tokens = [word for word in tokens if word.isalnum()]
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    tokens = [stemmer.stem(word) for word in tokens]

    return ' '.join(tokens)

with open('model/dataqb_svm.model', 'rb') as f:
    model = pickle.load(f)
with open('model/vectorizer.model', 'rb') as f:
    vectorizer = pickle.load(f)

st.header("Analisis Sentimen Aplikasi QuranBest")

with st.form("my_form"):
    st.write("Form Sentimen Analisis")
    input_text = st.text_input("Masukan Teks")

    submitted = st.form_submit_button("Submit")
    if submitted:
        preprocessed_text = preprocess(input_text)
        vectorized_text = vectorizer.transform([preprocessed_text]).toarray()
        result = model.predict(vectorized_text)
        st.text(f"Sentimen : {result[0]}")  

with st.form("form_file"):
    st.write("Form Sentimen Analisis File .CSV")
    uploaded_file = st.file_uploader("Choose a file")
    submitted = st.form_submit_button("Submit")
    if submitted:
        if uploaded_file is not None:
            # To read file as bytes:
            df_test = pd.read_csv(uploaded_file)

            # st.header("Data File CSV")

            # st.write(df_test)
            with st.spinner('Wait for it...'):
                text_column = 'content'

                df_test['Hasil Prediksi'] = ""

                # Loop melalui setiap baris dan lakukan prediksi
                for index, row in df_test.iterrows():
                    text_to_predict = row[text_column]

                    preprocessed_text = preprocess(text_to_predict)
                    vectorized_text = vectorizer.transform([preprocessed_text]).toarray()
                    result = model.predict(vectorized_text)

                    # Menyimpan hasil prediksi pada kolom 'predicted_sentiment'
                    df_test.at[index, 'Hasil Prediksi'] = result
                
                st.success('Done!')
                st.header("Hasil Sentimen")
                st.write(df_test)

      
