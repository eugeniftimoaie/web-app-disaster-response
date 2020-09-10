'''
ETL Pipeline for loading datasets from csv files, merging data into one dataframe, cleaning data and storing data into Sqlite database.
'''

# import python libraries
import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Function to load 2 csv dataset files and to merge them to 1 dataframe

    Args:
        messages_filepath (str): file with messages data
        categories_filepath (str): file with categories data

    Returns:
        df (pd.DataFrame): merged dataframe of 2 loaded datasets
    """

    # load datasets
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # Print statement with shapes of the dataframes:
    #print('Rows and columns in file with messages:', messages.shape)
    #print('Rows and columns in file with categories :', categories.shape)

    # merge datasets
    df = pd.merge(messages, categories, on = 'id')

    # Print statement with shapes of the dataframes:
    #print('Rows and columns in merged file with messages and categories :', categories.shape)

    return df


def clean_data(df):
    """
    Function to perform data cleaning of loaded dataset using following steps:
        1) create categories dataframe of column with individual category features
        2) select first row of created category dataframe and take this row to extract list of new column names
        3) rename column names of category features
        4) convert category values to just numbers 0 or 1
        5) drop original categories column from dataframe
        6) concatenate original dataframe with transformed categories dataframe
        7) drop duplicates of cleaned dataframe

    Args:
        df (pd.DataFrame): merged dataframe of 2 loaded data sets

    Returns:
        df (pd.DataFrame): cleaned dataframe
    """
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(pat = ';', expand = True)

    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # take row to extract a list of new column names for categories
    category_colnames = row.apply(lambda x: x.split('-')[0])

    # rename the columns of `categories`
    categories.columns = category_colnames

    # convert category values to just numbers 0 or 1
    for column in categories.columns:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.split('-').apply(lambda x: x[1])

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    # drop the original categories column from `df`
    del df['categories']

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], join = 'inner', axis = 1)

    # drop duplicates
    df.drop_duplicates(inplace = True)

    return df


def save_data(df, database_filepath):
    """
    Function to save cleaned dataframe into an sqlite database

    Args:
        df (pd.DataFrame): cleaned dataframe
        database_filename (str): path to the database file

    Returns:
        sqlite database with cleaned dataframe data
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df.to_sql('FigureEight_data', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'drp.db')


if __name__ == '__main__':
    main()
