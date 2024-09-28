import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Expanded synthetic dataset
data = {
    'text': [
        'Free entry in 2 a wkly comp!', 'Hey, how are you?', 'Congratulations! You won a prize!',
        'Meet me at 5 PM', 'Free tickets to the event!', 'Call me when you can',
        'Urgent! Call now for free cash', 'Let’s grab dinner', 'Free holiday just for you!',
        'Limited time offer!', 'Your account has been compromised', 'Can we meet tomorrow?',
        'Click here for a free gift', 'I’ll be there in 10 mins', 'Congratulations on your prize!',
        'Win a car now by clicking', 'Let’s have a meeting at 3', 'You have won a lottery!',
        'Can we reschedule our call?', 'Don’t miss this free opportunity', 'Your order is ready',
        'Claim your free tickets now', 'Good morning, how was your day?', 'Win an iPhone! Just click',
        'How about coffee later?', 'Get a free iPad today!', 'Please review your account balance',
        'You are eligible for a free service', 'I’ll call you later', 'Limited time sale on all items!',
        'Special discount just for you!', 'Exclusive deal on new products', 'Update your bank details urgently',
        'You have been selected for a prize!', 'Order your free samples now', 'Join us for free today!',
        'This is the last chance to win!', 'Verify your identity to receive your gift', 'Your credit card details are needed',
        'Win cash prizes instantly!', 'Register today for a free consultation!'
    ],
    'label': [
        'spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam',
        'spam', 'spam', 'ham', 'spam', 'ham', 'spam', 'spam', 'ham', 'spam',
        'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'spam',
        'ham', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam',
        'spam', 'spam', 'spam', 'spam', 'spam'
    ]
}

# Load data into a pandas DataFrame
df = pd.DataFrame(data)

# Split dataset into features (text) and target (label)
X = df['text']  # Text messages
y = df['label']  # Labels ('spam' or 'ham')

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Convert text data into TF-IDF features
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Define multiple models to evaluate
models = {
    'Naive Bayes': MultinomialNB(),
    'Support Vector Machine (SVM)': SVC(kernel='linear', probability=True),
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'K-Nearest Neighbors (KNN)': KNeighborsClassifier(n_neighbors=5)
}

# Train the models
trained_models = {}
for model_name, model in models.items():
    model.fit(X_train_tfidf, y_train)
    trained_models[model_name] = model

# Streamlit UI
st.title("Email Spam Classification")
st.write("Enter the email text below:")

email_text = st.text_area("Email Text", "")

selected_model = st.selectbox("Select Model", list(trained_models.keys()))

if st.button("Predict"):
    if email_text:
        # Preprocess input text
        input_tfidf = tfidf_vectorizer.transform([email_text])
        
        # Make prediction
        model = trained_models[selected_model]
        prediction = model.predict(input_tfidf)
        prediction_proba = model.predict_proba(input_tfidf)
        
        st.write(f"Prediction: {prediction[0]}")
        st.write(f"Prediction Probabilities: {dict(zip(model.classes_, prediction_proba[0]))}")
    else:
        st.warning("Please enter some email text.")
