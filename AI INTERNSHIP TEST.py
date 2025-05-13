#Section A: Python & Data Manipulation

#Q1. Data Cleanup & Summary
#You are given a CSV file: student_scores.csv with columns:
#Name, Math, Science, English, Gender
#Write a function that:
#• Fills any missing numeric values with the mean of the column.
#• Converts Gender into binary values.
#• Returns a summary DataFrame showing average scores per gender.

# ANSWER: Steps:
#         a)Read the CSV file: We'll use pandas to load the CSV file.
#         b)Fill missing numeric values with the mean: We'll use pandas' fillna() method to fill missing values.
#         c)Convert Gender into binary values: We'll convert "Male" to 1 and "Female" to 0 (or vice versa).
#         d)Compute the summary DataFrame: We will group by Gender and calculate the mean for each numeric column.

import pandas as pd

def process_student_scores(file_path):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(file_path)

    # Step 1: Fill missing numeric values with the mean of each column
    numeric_columns = ['Math', 'Science', 'English']
    for col in numeric_columns:
        df[col].fillna(df[col].mean(), inplace=True)

    # Step 2: Convert Gender into binary values
    # Assuming 'Male' -> 1 and 'Female' -> 0
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})

    # Step 3: Compute average scores per gender
    gender_summary = df.groupby('Gender')[['Math', 'Science', 'English']].mean()

    return gender_summary

# Example usage:
file_path = 'student_scores.csv'  # Replace with your file path
summary = process_student_scores(file_path)
print(summary)




#Q2. Dictionary-Based Stats
#Write a function that takes a dictionary of the form:
#{
#"user_1": [80, 90, 85],
#"user_2": [60, 65, 70],
#...
#}
#• Average score
#• Min score
#• Max score

# ANSWER: 
def process_scores(scores_dict):
    # Create a new dictionary to store the results
    result = {}

    # Iterate through the input dictionary
    for user, scores in scores_dict.items():
        # Compute the average, min, and max scores
        avg_score = sum(scores) / len(scores)  # Average score
        min_score = min(scores)                # Min score
        max_score = max(scores)                # Max score

        # Store the results for each user
        result[user] = {"Average": avg_score, "Min": min_score, "Max": max_score}

    return result


  
#Section B: Machine Learning
  
#Q3. Classifier on Iris
#• Load the Iris dataset from sklearn.datasets.
#• Train a Decision Tree classifier.
#• Split the data (80-20).
#• Predict and print accuracy on the test set.
#• Plot a confusion matrix using matplotlib or seaborn.

#ANSWER: 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Step 1: Load the Iris dataset
iris = load_iris()
X = iris.data  # Features
y = iris.target  # Target labels

# Step 2: Split the data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train a Decision Tree classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Step 4: Predict on the test set
y_pred = clf.predict(X_test)

# Step 5: Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Step 6: Plot the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot using seaborn's heatmap for a better visual representation
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()


#Q4. Simple Regression
#You are given a CSV simple_housing.csv with:
#area, bedrooms, price
#Build a linear regression model:
#• Predict price using the other columns.
#• Evaluate it using Mean Absolute Error (MAE).
#• Plot actual vs. predicted prices on a scatter plot

#ANSWER: 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Step 1: Load the CSV data
df = pd.read_csv('simple_housing.csv')

# Step 2: Preprocess the data (check for missing values)
print(df.isnull().sum())  # Check for missing values
df = df.dropna()  # Drop rows with missing values if any

# Step 3: Define features and target variable
X = df[['area', 'bedrooms']]  # Features
y = df['price']  # Target variable

# Step 4: Split the data into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 6: Make predictions on the test set
y_pred = model.predict(X_test)

# Step 7: Evaluate the model using Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error (MAE): {mae:.2f}")

# Step 8: Plot actual vs predicted prices on a scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', linewidth=2)
plt.title('Actual vs Predicted Prices')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.show()



#Section C: AI/ML Application & Thinking

#Q5. Conceptual

#1. What is overfitting in machine learning?
#ANSWER: Overfitting in machine learning occurs when a model learns the training data too well, including its noise and outliers, leading to poor performance on new, unseen data. 
#        Essentially, the model memorizes the training set rather than generalizing the underlying patterns. 
#        This results in high accuracy on the training data but low accuracy on the test or validation data. 


#2. When would you use a decision tree over logistic regression?
#ANSWER: Decision trees are often preferred over logistic regression when dealing with complex, non-linear relationships between variables, especially when the data contains categorical features or outliers.
#        Logistic regression is generally better when the relationship between variables is believed to be linear or can be easily transformed into a linear form.


#3. Explain the train-test split and why it’s important.
#ANSWER: The train-test split is a technique used in machine learning to evaluate the performance of a model by dividing a dataset into two subsets: a training set and a testing set. 
#        The training set is used to train the model, while the testing set is used to evaluate its performance on unseen data, which helps determine how well the model will generalize to new data. 
#        a)Evaluation of Model Generalization:The main reason for using train-test split is to assess how well the model generalizes to new, unseen data. 
#        b)Preventing Overfitting: If a model is trained on all the available data, it might become overly specialized to the training set and not perform well on new data.  
#                                  By holding out a portion of the data for testing, we can identify and mitigate overfitting. 
#        c)Objective Performance Evaluation:Using a separate test set allows for a more objective evaluation of the model's performance. 
#        d)Comparing Different Models: Train-test split can be used to compare the performance of different models or different configurations of the same model.


#4. What’s the purpose of normalization?
#ANSWER: The primary purpose of normalization in machine learning (ML) is to scale numerical features to a common range, typically between 0 and 1. 
#        This process ensures that features with vastly different scales (e.g., height in centimeters and income in dollars) don't disproportionately influence the model's training and performance.


#5. What’s the difference between classification and regression?
#ANSWER: a)CLASSIFICATION: Classification is used when you want to categorize data into different classes or groups. 
#                          For example, classifying emails as “spam” or “not spam” or predicting whether a patient has a certain disease based on their symptoms.
#                          Here are some common types of classification models: Decision Tree Classification, Random Forest Classification, K-Nearest Neighbor 
#        b) REGRESSION: Regression algorithms predict a continuous value based on input data.
#                       This is used when you want to predict numbers such as income, height, weight, or even the probability of something happening (like the chance of rain). 
#                       Some of the most common types of regression are:Simple Linear Regression, Multiple Linear Regression, Polynomial Regression


#Q6. Simple NLP Task – Sentiment Classification
#Use the built-in sklearn.datasets.fetch_20newsgroups:
#• Load only two categories: rec.autos and comp.sys.mac.hardware
#• Use TfidfVectorizer to convert text to vectors.
#• Train a Logistic Regression classifier to predict the category.
#• Print accuracy and show 5 most important words per class.
#ANSWER:

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Step 1: Load the dataset with only the required categories
categories = ['rec.autos', 'comp.sys.mac.hardware']
newsgroups = fetch_20newsgroups(subset='all', categories=categories, remove=('headers', 'footers', 'quotes'))

# Step 2: Preprocess the data with TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english')  # Remove common stop words
X = vectorizer.fit_transform(newsgroups.data)
y = newsgroups.target

# Step 3: Split the data into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train a Logistic Regression classifier
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# Step 5: Predict and evaluate the model
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Step 6: Print the 5 most important words per class
feature_names = np.array(vectorizer.get_feature_names_out())
for i, category in enumerate(newsgroups.target_names):
    top5 = np.argsort(clf.coef_[i])[-5:]  # Get the top 5 words for each class
    print(f"\nTop 5 important words for class '{category}':")
    print(" ".join(feature_names[top5]))
