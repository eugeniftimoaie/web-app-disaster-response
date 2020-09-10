import json
import plotly
import pandas as pd
import numpy as np

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Pie
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/drp.db')
df = pd.read_sql_table('FigureEight_data', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # distribution of different category
    category = list(df.columns[4:])
    category_counts = []
    for column_name in category:
        category_counts.append(np.sum(df[column_name]))

    category_count = df.drop(columns=['id','message','original','genre']).sum()
    category_names = list(category_count.index)
    category_corr = df.drop(columns=['id','message','original','genre']).corr()

    # create visuals
    graphs = [
        {
            'data': [
                Pie(
                    labels = genre_names,
                    values = genre_counts,
                    hole = 0.25,
                    direction = 'clockwise',
                    marker = {'colors' : ['rgb(152,251,152)', 'rgb(255,222,173)', 'rgb(176,224,230)']},
                    marker_line_color = 'rgb(0,0,0)',
                    marker_line_width = 1,
                    textinfo = 'label + value + percent',
                    textposition = 'outside',
                    showlegend = False,
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
            }
        },
        {
            'data': [
                Bar(
                    x = category,
                    y = category_counts,
                    marker_color = 'rgb(135,206,235)',
                    marker_line_color = 'rgb(0,0,0)',
                    marker_line_width = 1,
                    #text = category_counts,
                    textposition = 'outside',
                    textangle = -90,
                )
            ],

            'layout': {
                'autosize': False,
                'height': 500,
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': 'Count',
                    'ticks': 'outside',
                    'dtick': 5000,
                    #'range': [0, 25000],
                },
                'xaxis': {
                    'title': 'Category',
                    'ticks': 'outside',
                    'tickangle': -45,
                },
                'margin' : {
                    'b': 200,
                },
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results)


def main():
    #app.run(host='0.0.0.0', port=3001, debug=True)
    #app.run(host = '127.0.0.1', port = 5000, debug = True)
    app.run(debug = True)


if __name__ == '__main__':
    main()
