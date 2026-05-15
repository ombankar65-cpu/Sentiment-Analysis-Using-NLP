import streamlit as st
import pickle
import time
import pandas as pd
import plotly.express as px
from streamlit_lottie import st_lottie
import requests

# Page Configuration
st.set_page_config(page_title="Sentiment Analyzer", page_icon="✨", layout="centered")

# Function to load Lottie animations
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Load Assets
lottie_anime = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_v7v9mjat.json")
model = pickle.load(open('nlp_model.pkl', 'rb'))
# Replace 'vectorizer.pkl' with your actual vectorizer file name
cv = pickle.load(open('vectorizer.pkl', 'rb')) 

# App Header
st_lottie(lottie_anime, height=200, key="coding")
st.title("✨ Review Sentiment Analyzer")
st.markdown("Analyze the tone of your customer reviews with high precision.")

# Input Section
st.subheader("Enter your review below:")
user_input = st.text_area("Write something...", placeholder="Type here...", height=150)

if st.button("Analyze Sentiment"):
    if user_input:
        with st.spinner('Analyzing...'):
            time.sleep(1) # Visual effect
            
            # Transformation and Prediction
            data = [user_input]
            vect = cv.transform(data).toarray()
            prediction = model.predict(vect)
            prob = model.predict_proba(vect)

            # Display Results
            st.divider()
            sentiment = prediction[0]
            
            if sentiment == 'positive':
                st.balloons()
                st.success(f"### The Sentiment is: **{sentiment.upper()}** 😊")
            else:
                st.error(f"### The Sentiment is: **{sentiment.upper()}** 😠")

            # Visualization
            st.subheader("Confidence Score")
            df_prob = pd.DataFrame(prob, columns=model.classes_)
            fig = px.bar(df_prob.T, orientation='h', labels={'index': 'Sentiment', 'value': 'Probability'},
                         color=df_prob.T.index, color_discrete_map={'positive': 'green', 'negative': 'red'})
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Please enter a review to analyze.")
