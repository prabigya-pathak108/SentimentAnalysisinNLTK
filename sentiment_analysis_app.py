import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
#import sklearn

import nltk
#import enchant   #for spelling correction and checking
from nltk.metrics import edit_distance  # to find the case where spelling correction is needed

from nltk.corpus import wordnet as wn,stopwords
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from collections import defaultdict

import re
current_dir = os.getcwd()
with open('enchant_dict.pkl', 'rb') as f:
    d = pickle.load(f)
    f.close()



# Construct the relative path to the file

# image_path = os.path.join(current_dir,r"dataset\car_image.png")
# csv_path = os.path.join(current_dir,r"dataset\cleaned_car_price_csv.csv")
# model_path_a = os.path.join(current_dir,r"code\saved_model_for_car_price.pkl")

image_path = "sentiment.jpg"
# Load the saved model
with open('sentiment_analysis_model.pkl', 'rb') as f:
    model = pickle.load(f)
    f.close()

# Load the CountVectorizer
with open('cv.pkl', 'rb') as f:
    cv = pickle.load(f)
    f.close()

#Lets define some function for replacement of common sentece use cases
replacement_patterns = [
 (r'won\'t', 'will not'),
 (r'can\'t', 'cannot'),
 (r'i\'m', 'i am'),
 (r'', ''),
 (r'wanna', 'want'),
 (r'gonna', 'going to'),
 (r'ain\'t', 'is not'),
 (r'(\w+)\'ll', '\g<1> will'),
 (r'(\w+)n\'t', '\g<1> not'),
 (r'(\w+)\'ve', '\g<1> have'),
 (r'(\w+)\'s', '\g<1> is'),
 (r'(\w+)\'re', '\g<1> are'),
 (r'(\w+)\'d', '\g<1> would')
]
patterns = [(re.compile(regex), repl) for (regex, repl) in replacement_patterns]

def replace_function(text):
    s = text
    for (pattern, repl) in patterns:
        s = re.sub(pattern, repl, s)
    return s




#the following function remove stopwords
def remove_stopwords(text):
    stopwords_list=stopwords.words("english")
    text_without_stopword=""
    for i in str(text).split():
        if i not in stopwords_list:
            text_without_stopword=text_without_stopword+" "+str(i).lower()
    return text_without_stopword.strip()




#the following function is used for spelling checking and correction
def correct_spellings_all(text):
    words = text.split()
    corrected_words = []
    for word in words:
        if d.check(word):
            corrected_words.append(word)
        else:
            suggestions = d.suggest(word)
            if suggestions:
                if (edit_distance(word,suggestions[0])>1):
                    corrected_words.append(suggestions[0])
                else:
                    corrected_words.append(word)
            else:
                corrected_words.append(word)
    return ' '.join(corrected_words)




#the follwing function is used for lammetizing by finding the POS
tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV

lemmatizer = WordNetLemmatizer()


def lemmatize_sentence(text):
    words = nltk.word_tokenize(text)
    corrected_words = []
    for token, tag in pos_tag(words):
        lemma = lemmatizer.lemmatize(token, tag_map[tag[0]])
        corrected_words.append(lemma)
    return ' '.join(corrected_words)

def preprocess_text(text):
    # Apply preprocessing steps to the text
    processed_text = replace_function(text)
    processed_text = remove_stopwords(processed_text)
    processed_text = correct_spellings_all(processed_text)
    processed_text = lemmatize_sentence(processed_text)
    return processed_text

def expression_check(prediction_input):
    if prediction_input == 0:
        return "It has Negative Sentiment."
    elif prediction_input == 1:
        return "It has Positive Sentiment."
    else:
        return "Invalid Statement."

def predict_from_user_input(user_input, model, cv):
    # Preprocess the user input
    processed_input = preprocess_text(user_input)
    
    # Transform the preprocessed input into numerical features
    input_data = cv.transform([processed_input])
    
    # Make predictions using the trained model
    predicted_class = model.predict(input_data)

    predicted_probabilities = model.predict_proba(input_data)
    predicted_class_index = np.argmax(predicted_probabilities)
    #print(probability_predicted,confidence)
    
    prediction_msg =expression_check(predicted_class)

    return predicted_probabilities,prediction_msg

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
    if (user_input is None):
        st.error("Please, Select At least One Sentence.",icon="📝")
    else:
        #prediction = model.predict(df1)[0]
        probabilities,result = predict_from_user_input(user_input, model, cv)
        st.text_area("Original Sentence", value=user_input, height=50)
        
        
        if "Negative Sentiment." in result:
            st.error(result,icon="🚨")
        else:
            st.success(result, icon="✅")

        # Pie chart
        labels = ['Positive', 'Negative']
        sizes = [probabilities[0][1], probabilities[0][0]]
        colors = ['#ff9999','#66b3ff']
        explode = (0.1, 0)  # explode 1st slice
        
        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
                shadow=True, startangle=90)
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        st.pyplot(fig1)