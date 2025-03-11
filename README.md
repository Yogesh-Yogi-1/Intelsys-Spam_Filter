# TITLE:
## FilterFlux

# BRIEF EXPLANATION ABOUT THE WORKING MODEL:
(include a very short explanation of your won on how your model works)
(2 to 3 lines is enough)

# ML ALGORITHM USED:
NAIVE BAYES
Naive Bayes is a family of probabilistic algorithms based on Bayes' Theorem, often used for classification tasks. It's called "naive" because it assumes that all features (or attributes) are independent, which is a simplifying assumption that is rarely true in real-world data. Despite this assumption, Naive Bayes classifiers often perform surprisingly well in practice, particularly for text classification tasks like spam filtering, sentiment analysis, and document categorization.
# TECHNOLOGY USED FOR BUILDING THE FRONT-END:
STREAMLIT
Streamlit is an open-source Python library that allows developers to easily create and share interactive web applications for machine learning and data science projects. It’s particularly popular among data scientists and machine learning practitioners because of its simplicity, speed, and focus on making data visualization and model interaction seamless.

# STEP-BY-STEP EXPLANATION OF THE CODE:
## Model code explanation:
### 1.Loading and Preprocessing Data:

First up, we load the dataset. This dataset contains email messages and their corresponding labels—either "spam" or "ham".
We use pandas to read the CSV file where all the emails are stored.
We split the dataset into two parts: the email content (X) and the spam/ham labels (y). We’ll use these to train the model.

### 2.Training the Model:

We then move on to turning all those email texts into numbers because computers like numbers, not text!
CountVectorizer is used here to break down the text and count how many times each word appears in the email. It also removes common words like "the", "is", "in" (these are called stop words).
Now that we’ve got a nice matrix of numbers representing our emails, we feed this data into Naive Bayes, which is a simple but powerful machine learning algorithm that works really well for text classification.
The model learns how to recognize patterns, like which words are more likely to appear in spam vs. ham.

### 3.Evaluating the Model:

Once the model is trained, we need to check how well it’s doing. So, we test it using a separate set of emails that the model hasn’t seen before.
We calculate the accuracy (how often it gets things right), show a confusion matrix (basically a fancy table showing where it’s making mistakes), and print out a classification report (telling us how well it does on both spam and ham emails).

### 4.Saving the Model:

Once we’re happy with how the model performs, we save it (along with the CountVectorizer) using joblib. This is like saving your progress in a video game—so the next time you want to use it, you don’t have to train it again from scratch.

### 5.Classifying New Emails:

The fun part! After the model is saved, we can start classifying new emails.
The user can type in an email, and the program will use the saved model to predict if it's spam or ham.
The email gets transformed (just like in the training phase), and the model spits out the result—spam or ham.
This keeps going until the user types “exit” to stop the program.

### Example Walkthrough:
Imagine you have an email saying:

"Congratulations, you've won a lottery!"

When you input this into the program, it runs through these steps:

The email text is passed through CountVectorizer, which converts it into a bunch of numbers based on word frequency.
The trained Naive Bayes model looks at these numbers and compares them to what it learned from the training data.
The model predicts: spam! (Because words like "won" and "lottery" are often associated with spam messages).
Now, let’s say you enter a regular email like:

"Meeting at 3 PM in the conference room."

The process is the same, but this time the model is likely to predict ham, since it doesn’t contain any keywords typically found in spam emails.

### Why It’s Cool:

Speed: Once the model is trained, it’s super quick at making predictions on new emails.

Simplicity: The Naive Bayes algorithm is straightforward but surprisingly effective for text classification tasks.

Real-World Use: This is the kind of thing you’d see in email systems to filter out spam automatically, so you don’t have to wade through all that junk.

### Final Thought:
So, to sum it up: the program trains a machine learning model to classify emails, saves that model, and then lets you interact with it to classify new emails. It’s efficient, easy to set up, and can be applied to many other kinds of text classification tasks, like detecting sentiment in reviews or categorizing news articles.

__________________________________________________________________________________________________________________________

## User code explanation:
### 1. Importing Libraries
```
import streamlit as st
import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
```

Streamlit (st): Used to create the web interface for the application.

joblib: Used to load the pre-trained model and vectorizer from disk.

pandas: Helps in handling and reading CSV files (in case users upload email datasets).

CountVectorizer: From scikit-learn, used to convert text data into a format the machine learning model can understand (i.e., numerical vectors).

### 2. Load the Pre-trained Model and Vectorizer
```
def load_model_and_vectorizer():
    model = joblib.load('spam_classifier_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    return model, vectorizer
```

load_model_and_vectorizer: This function loads the pre-trained Naive Bayes model and vectorizer from disk using joblib.

spam_classifier_model.pkl: The trained Naive Bayes model.

vectorizer.pkl: The vectorizer used during training (usually a CountVectorizer that converts email text into numerical features).

### 3. Classify a Single Email
```
def classify_email(model, vectorizer, email_content):
    email_vec = vectorizer.transform([email_content])
    prediction = model.predict(email_vec)
    return prediction[0]
```

classify_email:

This function takes the email content, converts it into numerical vectors using the vectorizer, and then passes it to the model to predict whether the email is spam or ham.

The prediction (either 'spam' or 'ham') is returned.

### 4. Streamlit UI
```
def main():
    # Set up the page title
    st.title("Spam/Ham Email Classifier")

    # Provide a description
    st.write("""
    This is a simple email classifier that uses a Naive Bayes model to classify emails as either **Spam** or **Ham**.
    Enter the email content below, and the model will classify it for you.
    """)
```

The Streamlit UI starts with setting a title for the page and provides a brief description about the app. This is displayed to the user when they open the web interface.

### 5. Email Content Input Field
```
email_content = st.text_area("Enter the email content here:")
```

st.text_area: This is a text area widget in Streamlit that allows the user to enter the content of an email for classification.

Users can type or paste an email's content into this text box.

### 6. Classify Button
```
if st.button("Classify Email"):
    if email_content:
        result = classify_email(model, vectorizer, email_content)
        if result == 'spam':
            st.write("**The email is classified as: Spam**")
        else:
            st.write("**The email is classified as: Ham**")
    else:
        st.write("Please enter some email content to classify.")
```

st.button("Classify Email"): This creates a button that the user can click to classify the email.

When the button is clicked, the app:

1. Checks if the user has entered email content.

2. If content is provided, it calls classify_email with the model and vectorizer to predict the email's classification.
  
3. The result is displayed to the user, telling them if the email is spam or ham.

4. If no content is entered, the app asks the user to provide email content.

### 7. File Upload Option
```
uploaded_file = st.file_uploader("Or upload a file containing emails", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    if 'Message' in data.columns:
        st.write("Classifying emails in the uploaded file...")
        for email in data['Message']:
            result = classify_email(model, vectorizer, email)
            st.write(f"Email: {email}")
            if result == 'spam':
                st.write("**Classified as: Spam**")
            else:
                st.write("**Classified as: Ham**")
    else:
        st.write("Uploaded file does not contain a 'Message' column.")
```

st.file_uploader: This creates a file upload button that allows users to upload a CSV file. The file must be in CSV format, and it is assumed that the emails are contained in a column named Message.

After the user uploads a file:

1. The app reads the file into a pandas DataFrame using pd.read_csv.

2. If the file contains a column named Message (which is expected to contain the email text), the app iterates through all emails in that column and classifies them as spam or ham.

3. For each email, the app displays the content of the email and the classification result.

4. If the file doesn’t have a Message column, it shows a warning.

### 8. Running the App
```
if __name__ == "__main__":
    main()
```

This is the entry point to the program. When the script is run, it calls the main() function, which starts the Streamlit app and displays the UI.

### Final Flow:

#### User Interaction:

The user enters email content manually or uploads a file with multiple emails.

They click the "Classify Email" button to get a spam/ham prediction for the email content or emails in the file.

#### Model Prediction:

The app uses the pre-trained Naive Bayes model to classify each email and shows the result (either spam or ham) on the web interface.

### Summary:

Spam/Ham Email Classifier UI is a Streamlit-based web app that allows users to classify emails as spam or ham using a pre-trained Naive Bayes model.

Users can input a single email or upload a CSV file containing multiple emails.

The app will predict the classification of each email and display the result to the user.

# OUTPUT:
![Screenshot 2025-01-22 205931](https://github.com/user-attachments/assets/26ace7e6-a99e-4c52-b4fb-5b58763c24ff)

# RESULT:
i.Accuracy: 98.5% — The model is doing a great job, correctly predicting spam and ham emails almost 99% of the time.

ii.Confusion Matrix: The majority of spam emails are correctly identified as spam, and most ham emails are accurately classified as ham.

iii.Classification Report: Precision and recall for both spam and ham are high, with the model's F1-scores reflecting its solid performance in both categories.

This is the basic result you’d get when you run the classifier and evaluate it on a dataset. It's a strong result if you’re working with real-world email data—98.5% accuracy is a good performance for spam classification!
