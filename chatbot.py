# To create an end-to-end chatbot using Python, we follow the steps mentioned below:

# 1. Define Intents
# 2. Create training data
# 3. Train the chatbot
# 4. Build the chatbot
# 5. Test the chatbot
# 6. Deploy the chatbot

import os
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

# 1. ssl._create_default_https_context = ssl._create_unverified_context:
# Normally, when the Python code tries to download data from the internet (like the NLTK data), it checks if the website's SSL certificate is valid to ensure the connection is secure.
# This line changes the default behavior, so Python skips this certificate check. It's like telling Python, "Don't worry if the website's certificate isn't verified; just download the data anyway."

# 2. nltk.data.path.append(os.path.abspath("nltk_data")):
# NLTK (Natural Language Toolkit) uses a directory to store the data it needs (like language models).
# This line tells NLTK to also look in a specific folder, named "nltk_data" in your current directory, when it needs to find or save data.

# 3. nltk.download('punkt'):
# This command downloads the "punkt" tokenizer model, which is used to break down text into sentences or words. It's a basic tool in NLTK for text processing.

# In short, these lines allow your code to download and use NLTK resources without worrying about SSL verification issues and store those resources in a specific folder.


intents = [
    {
        "tag": "greeting",
        "patterns": ["Hi", "Hello", "Hey", "How are you", "What's up"],
        "responses": ["Hi there", "Hello", "Hey", "I'm fine, thank you", "Nothing much"]
    },
    {
        "tag": "goodbye",
        "patterns": ["Bye", "See you later", "Goodbye", "Take care"],
        "responses": ["Goodbye", "See you later", "Take care"]
    },
    {
        "tag": "thanks",
        "patterns": ["Thank you", "Thanks", "Thanks a lot", "I appreciate it"],
        "responses": ["You're welcome", "No problem", "Glad I could help"]
    },
    {
        "tag": "about",
        "patterns": ["What can you do", "Who are you", "What are you", "What is your purpose"],
        "responses": ["I am a chatbot", "My purpose is to assist you", "I can answer questions and provide assistance"]
    },
    {
        "tag": "help",
        "patterns": ["Help", "I need help", "Can you help me", "What should I do"],
        "responses": ["Sure, what do you need help with?", "I'm here to help. What's the problem?", "How can I assist you?"]
    },
    {
        "tag": "age",
        "patterns": ["How old are you", "What's your age"],
        "responses": ["I don't have an age. I'm a chatbot.", "I was just born in the digital world.", "Age is just a number for me."]
    },
    {
        "tag": "weather",
        "patterns": ["What's the weather like", "How's the weather today"],
        "responses": ["I'm sorry, I cannot provide real-time weather information.", "You can check the weather on a weather app or website."]
    },
    {
        "tag": "budget",
        "patterns": ["How can I make a budget", "What's a good budgeting strategy", "How do I create a budget"],
        "responses": ["To make a budget, start by tracking your income and expenses. Then, allocate your income towards essential expenses like rent, food, and bills. Next, allocate some of your income towards savings and debt repayment. Finally, allocate the remainder of your income towards discretionary expenses like entertainment and hobbies.", "A good budgeting strategy is to use the 50/30/20 rule. This means allocating 50% of your income towards essential expenses, 30% towards discretionary expenses, and 20% towards savings and debt repayment.", "To create a budget, start by setting financial goals for yourself. Then, track your income and expenses for a few months to get a sense of where your money is going. Next, create a budget by allocating your income towards essential expenses, savings and debt repayment, and discretionary expenses."]
    },
    {
        "tag": "credit_score",
        "patterns": ["What is a credit score", "How do I check my credit score", "How can I improve my credit score"],
        "responses": ["A credit score is a number that represents your creditworthiness. It is based on your credit history and is used by lenders to determine whether or not to lend you money. The higher your credit score, the more likely you are to be approved for credit.", "You can check your credit score for free on several websites such as Credit Karma and Credit Sesame."]
    }
]

vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=10000)

# TfidfVectorizer(): This tool converts text into numbers (specifically, TF-IDF scores) that a machine learning model can understand. 
# TF-IDF stands for "Term Frequency-Inverse Document Frequency," which gives importance to words that are common in a document but rare across different documents.
# LogisticRegression(): This is the machine learning model we're using to classify text. It tries to predict which category (or "tag") a piece of text belongs to based on the patterns it sees during training.
# `random_state=0` ensures the results are consistent each time you run the code, and `max_iter=10000` sets the maximum number of times the algorithm will run to find the best model.


tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

# What Happens Here:
# Tokenization: The TfidfVectorizer first splits each sentence into individual words (tokens).
# Vocabulary Building: It creates a vocabulary (list of unique words) from the entire patterns list.
# Example-["hello", "how", "are", "you", "goodbye", "see", "later", "good", "morning"]
# TF-IDF Calculation: The vectorizer then calculates the TF-IDF score for each word in each sentence.
# This score reflects how important a word is in a sentence relative to its occurrence in other sentences.

def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return response
        
counter = 0

def main():
    global counter
    st.title("Purav's Chatbot")
    st.write("Welcome to the chatbot. Please type a message and press Enter to start the conversation.")

    counter += 1
    user_input = st.text_input("You:", key=f"user_input_{counter}")

    if user_input:
        response = chatbot(user_input)
        st.text_area("Chatbot:", value=response, height=100, max_chars=None, key=f"chatbot_response_{counter}")

        if response.lower() in ['goodbye', 'bye']:
            st.write("Thank you for chatting with me. Have a great day!")
            st.stop()

if __name__ == '__main__':
    main()