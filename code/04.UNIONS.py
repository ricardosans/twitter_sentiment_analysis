"""
-----------------------------------------------------------------------------------------------------------------------
PROGRAM #4
-----------------------------------------------------------------------------------------------------------------------
The objective of this code is to unite all the separate crypto tables into a single one.
"""

"""
IMPORT LIBRARIES
"""
# Import psycopg2 for creating and manipulating SQL databases
import psycopg2
# Import pandas for dataframe manipulation
import pandas as pd
# Import Login_access for logging in in our DB
from Login_access import host, port, user, password

"""
MAIN FUNCTION
"""


def main():
    cryptos = pd.read_csv('./crypto_names.txt', sep=',')
    # Create new table for uniting all tables
    dbTable()
    for symbol in cryptos.symbol:
        db_append(symbol)

    # Disconnect
    cur.close()
    conn.close()


"""
DEFINE OTHER FUNCTIONS
"""


def dbTable():
    global cur, conn
    conn = psycopg2.connect(host=host, database="TwitterDB", port=port, user=user, password=password)
    # Create the cursor to execute SQL commands
    cur = conn.cursor()
    # Create table
    command = '''CREATE TABLE crypto_table (
                                tweet_id BIGINT,
                                user_id BIGINT,
                                user_name TEXT,
                                symbol TEXT,
                                tweet TEXT,
                                retweet_count INT,
                                sentiment_score numeric(10,2),
                                outcome TEXT,
                                created_at TIMESTAMP WITH TIME ZONE);'''
    cur.execute(command)
    # Commit changes
    conn.commit()


def db_append(symbol):
    # Create the cursor to execute SQL commands
    cur = conn.cursor()
    # insert tweet information into the TwitterTweet table
    command = '''INSERT INTO crypto_table (tweet_id, user_id, user_name, symbol, tweet, retweet_count, 
                                           sentiment_score, outcome, created_at)
                 (SELECT A.tweet_id,
                         A.user_id,
                         B.user_name, ''' + \
                         "'" + symbol + "'" + ''' AS symbol,
                         A.tweet,
                         A.retweet_count,
                         A.sentiment_score,
                         A.outcome,
                         A.created_at
                  FROM ''' + symbol + ''' AS A
                  LEFT JOIN twitteruser AS B ON A.USER_ID=B.USER_ID);'''
    # Execute the query
    cur.execute(command)
    # Commit changes
    conn.commit()
    # Disconnect
    cur.close()


"""
EXECUTION
"""
if __name__ == "__main__":
    main()
