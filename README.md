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
### Model code explanation:
Loading and Preprocessing Data:

First up, we load the dataset. This dataset contains email messages and their corresponding labels—either "spam" or "ham".
We use pandas to read the CSV file where all the emails are stored.
We split the dataset into two parts: the email content (X) and the spam/ham labels (y). We’ll use these to train the model.
Training the Model:

We then move on to turning all those email texts into numbers because computers like numbers, not text!
CountVectorizer is used here to break down the text and count how many times each word appears in the email. It also removes common words like "the", "is", "in" (these are called stop words).
Now that we’ve got a nice matrix of numbers representing our emails, we feed this data into Naive Bayes, which is a simple but powerful machine learning algorithm that works really well for text classification.
The model learns how to recognize patterns, like which words are more likely to appear in spam vs. ham.
Evaluating the Model:

Once the model is trained, we need to check how well it’s doing. So, we test it using a separate set of emails that the model hasn’t seen before.
We calculate the accuracy (how often it gets things right), show a confusion matrix (basically a fancy table showing where it’s making mistakes), and print out a classification report (telling us how well it does on both spam and ham emails).
Saving the Model:

Once we’re happy with how the model performs, we save it (along with the CountVectorizer) using joblib. This is like saving your progress in a video game—so the next time you want to use it, you don’t have to train it again from scratch.
Classifying New Emails:

The fun part! After the model is saved, we can start classifying new emails.
The user can type in an email, and the program will use the saved model to predict if it's spam or ham.
The email gets transformed (just like in the training phase), and the model spits out the result—spam or ham.
This keeps going until the user types “exit” to stop the program.
Example Walkthrough:
Imagine you have an email saying:

"Congratulations, you've won a lottery!"

When you input this into the program, it runs through these steps:

The email text is passed through CountVectorizer, which converts it into a bunch of numbers based on word frequency.
The trained Naive Bayes model looks at these numbers and compares them to what it learned from the training data.
The model predicts: spam! (Because words like "won" and "lottery" are often associated with spam messages).
Now, let’s say you enter a regular email like:

"Meeting at 3 PM in the conference room."

The process is the same, but this time the model is likely to predict ham, since it doesn’t contain any keywords typically found in spam emails.

Why It’s Cool:
Speed: Once the model is trained, it’s super quick at making predictions on new emails.
Simplicity: The Naive Bayes algorithm is straightforward but surprisingly effective for text classification tasks.
Real-World Use: This is the kind of thing you’d see in email systems to filter out spam automatically, so you don’t have to wade through all that junk.
Final Thought:
So, to sum it up: the program trains a machine learning model to classify emails, saves that model, and then lets you interact with it to classify new emails. It’s efficient, easy to set up, and can be applied to many other kinds of text classification tasks, like detecting sentiment in reviews or categorizing news articles.

# OUTPUT:
![Screenshot 2025-01-22 205931](https://github.com/user-attachments/assets/26ace7e6-a99e-4c52-b4fb-5b58763c24ff)

# RESULT:
