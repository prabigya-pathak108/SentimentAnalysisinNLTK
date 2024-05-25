import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
#import sklearn

current_dir = os.getcwd()


# Construct the relative path to the file

# image_path = os.path.join(current_dir,r"dataset\car_image.png")
# csv_path = os.path.join(current_dir,r"dataset\cleaned_car_price_csv.csv")
# model_path_a = os.path.join(current_dir,r"code\saved_model_for_car_price.pkl")

image_path = "sentiment.jpg"


options=[]

sentences = [
    "I am extremely happy with my new car; it drives smoothly and looks fantastic.",
    "I felt really disappointed with my performance in the meeting today.",
    "My vacation was absolutely wonderful; I enjoyed every single day.",
    "I am very frustrated with my computer; it keeps crashing and losing my work.",
    "I am proud of my achievements this year; I've worked hard and it paid off.",
    "I can't stand my new job; the work environment is toxic and stressful.",
    "I love spending time with my family; they always make me feel happy and supported.",
    "I am unhappy with my current living situation; the neighbors are too noisy.",
    "My health has improved significantly, and I feel better than ever.",
    "I regret my decision to move to this city; I feel lonely and out of place.",
    "I am excited about my upcoming project; it's going to be a great opportunity.",
    "I am worried about my financial situation; my expenses are too high.",
    "I feel confident in my abilities; I know I can handle any challenge.",
    "I am dissatisfied with my internet service; it's slow and unreliable.",
    "I am feeling very stressed about my upcoming exams.",
    "I am thrilled with my new hobby; it brings me so much joy.",
]
st.set_page_config(page_title="Sentiment Analysis",page_icon="🙂",layout="centered")

st.markdown("<div style='background-color:#219C90; border-radius:50px; align-items:center; justify-content: center;'><h1 style='text-align:center; color:white;'>Sentiment Analysis</h1></div>",unsafe_allow_html=True)

st.markdown("<h4 style='text-align:center; color:black;'>Analyse the Sentiment of Person From his chat</h4>",unsafe_allow_html=True)

#Styling Streamlit Web App
col1 , col2 = st.columns(2)

with col1:
    st.write("")
    st.image(image=image_path,use_column_width=True,caption="Here comes Sentiment Analysis")


with col2:
    user_input = st.selectbox(label="Select The car model",options=sentences,placeholder="Select",index=None)
    col3,col4 = st.columns(2)
    
    pred = st.button("Predict",use_container_width=True)


if pred:
    #prediction = model.predict(df1)[0]
    st.success(f"Predicted Price of Your Car is : ₹", icon="✅")

    # Pie chart
    labels = ['Confidence Level', 'Other']
    sizes = [0.4, 0.6]
    colors = ['#ff9999','#66b3ff']
    explode = (0.1, 0)  # explode 1st slice
    
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    st.pyplot(fig1)