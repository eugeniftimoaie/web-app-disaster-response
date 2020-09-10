'''
Machine Learning Pipeline for loading data from Sqlite database, splitting database into train and test sets, building a text processing and machine learning pipeline, training and tuning the model, outputting results on the test set and exporting final model as a pickle file.
'''

# import python libraries
import sys
import pandas as pd
from sqlalchemy import create_engine

import re
import nltk
nltk.download(['punkt','stopwords'])
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

import joblib


def load_data(database_filepath):
    """
    Function to load sqlite database

    Args:
        database_filepath (str): path to database file

    Returns:
        X (pd.Series, str): containing messages
        y (pd.DataFrame): containing features (target variables)
        category names (pd.Series, str): list with names of the features (target variables)
    """

    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('FigureEight_data', con = engine)

    # define feature and target variables X, y
    X = df['message']
    Y = df.iloc[:, 4:]

    # category names
    category_names = list(df.columns[4:])

    return X, Y, category_names


def tokenize(text):
    """
    Function to process text data taking following steps:
        1) normalization and punctuation removal: convert to lower case and remove punctuations
        2) tokenization: splitting each sentence into sequence of words
        3) stop words removal: removal of words which do not add a meaning to the sentence
        4) lemmatization: reducting words to their root form

    Args:
        text (str): string with message

    Returns:
        clean_tokens: cleaned tokens of the message with word list
    """

    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # tokenize text and innitiate lemmatizer
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    # remove stopwords
    tokens = [w for w in tokens if w not in stopwords.words('english')]

    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        # lemmatize and remove leading/ trailing white space
        clean_tok = lemmatizer.lemmatize(tok).strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


def build_model():
    """
    Function to create (initialize) a Model with classification results on the categories of the dataset and to optimize entire Workflow (parameter tuning)

    Args:
        none

    Returns:
        cv: cross-validation generator of GridSearchCV over full pipeline showing classification results
    """
    # build pipeline
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer = tokenize)),
    #('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(DecisionTreeClassifier()))
    ])

    # parameters for Grid Search
    parameters = {
                 #'clf__estimator__criterion': ['gini', 'entropy']
                 }

    # initializing a Grid Search
    cv = GridSearchCV(estimator = pipeline, param_grid = parameters)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Function to evaluate model by computing f1_score, precision and recall for each target variable (feature) and as overall score (mean of all target variables (features))

    Args:
        model (function): build_model function with classification results
        X_test (pd.Series, str): containing messages of test set
        y_test (pd.DataFrame): containing features (target variables) of test set
        category_names (pd.Series, str): containing category_names of features (target variables)

    Returns:
        df (pd.DataFrame): containing data of category, f1_score, precision and recall for each target variable (feature)
        print statements with overall f1_score, precision and recall
    """

    # predict on test data
    Y_pred = model.predict(X_test)

    # create empty dataframe for results with columns Category, f1_score, precision and recall
    results = pd.DataFrame(columns = ['category', 'f1_score', 'precision', 'recall'])

    # iterate through y_test columns with target variables (features) for scores of each feature
    i = 0
    for category in Y_test.columns:
        precision, recall, f1_score, support = precision_recall_fscore_support(Y_test[category], Y_pred[:,i], average = 'weighted')
        results.at[i + 1, 'category'] = category
        results.at[i + 1, 'f1_score'] = f1_score
        results.at[i + 1, 'precision'] = precision
        results.at[i + 1, 'recall'] = recall
        i += 1

    # print mean scores of all target variables (features)
    print('Overall f1_score: ', '{:.4}'.format(results['f1_score'].mean()))
    print('Overall precision: ', '{:.4}'.format(results['precision'].mean()))
    print('Overall recall: ', '{:.4}'.format(results['recall'].mean()))
    return results


def save_model(model, model_filepath):
    """
    Function to save trained model into a pickle file

    Args:
        model_filename (str): path to the pickle file

    Returns:
        pickle file with trained model
    """

    joblib.dump(model, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)

        print('Shape of training set: X_train {} | Y_train {}'.format(X_train.shape, Y_train.shape))
        print('Shape of testing set: X_train {} | Y_train {}'.format(X_test.shape, Y_test.shape))

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/drp.db classifier.pkl')


if __name__ == '__main__':
    main()
