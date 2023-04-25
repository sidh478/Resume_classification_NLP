import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import re
import nltk
import string
from nltk.corpus import stopwords
from wordcloud import WordCloud
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import aspose.words as aw

# Define a function to preprocess the resume text
def preprocess_resume_text(resume_text):
    # Convert to lowercase
    resume_text = resume_text.lower()
    # Remove numbers and special characters
    resume_text = re.sub(r'\d+', '', resume_text)
    resume_text = re.sub(r'[^\w\s]', '', resume_text)
    # Tokenize the text
    words = nltk.word_tokenize(resume_text)
    # Remove stop words
    words = [word for word in words if word not in stopwords.words('english')]
    # Lemmatize the words
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    # Join the words back together
    resume_text = ' '.join(words)
    return resume_text
def word_cloud(resumes):
    oneSetOfStopWords = set(stopwords.words('english')+['``',"''"])
    totalWords =[]
    cleanedSentences = ""
    for records in resumes:
        cleanedSentences += records
        requiredWords = nltk.word_tokenize(records)
        for word in requiredWords:
            if word not in oneSetOfStopWords and word not in string.punctuation:
                totalWords.append(word)
    wordfreqdist = nltk.FreqDist(totalWords)
    mostcommon = wordfreqdist.most_common(100)

    wn = WordNetLemmatizer() 
    lem_words=[]
    for word in wordfreqdist:
        word=wn.lemmatize(word)
        lem_words.append(word) 

    wc = WordCloud().generate(cleanedSentences)
    fig1=plt.figure(figsize=(10,10))
    plt.imshow(wc, interpolation='bilinear')
    st.pyplot(fig1)


    
# Load the model and vectorizer from the pkl files
model = pickle.load(open(r'C:\Users\harsh\Downloads\Resume_Classification_NLP\trained_model.sav', 'rb'))
vectorizer =pickle.load(open(r'C:\Users\harsh\Downloads\Resume_Classification_NLP\tfidf_vectorizer.sav', 'rb'))

# Define the resume_dic mapping dictionary
resume_dict={0:'PeopleSoft Resume',1:'React JS Developer Resume',2:'SQL Developer Lightning Insight Resume',3:'Workday Resume'}

# Define the Streamlit app
st.title('Resume Classification')

# Upload the resume file
resume_file = st.file_uploader('Upload your resume', type=['pdf', 'docx', 'txt'])

if resume_file is not None:
    # Read the resume file
    resume_text=aw.Document(resume_file)
    resume_text=resume_text.get_text().strip()
    # Preprocess the resume text
    resume_text = preprocess_resume_text(resume_text)
    #Word cloud the resume text
    wordcloud=word_cloud(resume_text) 
    # Vectorize the resume text
    resume_vector = vectorizer.transform([resume_text])
    # Make a prediction using the model
    prediction = model.predict(resume_vector)
    
    # Map the prediction to the corresponding resume type using the dictionary
    resume_type = resume_dict[prediction[0]]
    # Display the prediction to the user
    st.write('The resume is best suited for:', resume_type)
    