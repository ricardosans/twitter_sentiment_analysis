"""
-----------------------------------------------------------------------------------------------------------------------
PROGRAM #1
-----------------------------------------------------------------------------------------------------------------------
The objective of this code is to create the table that will contain the retrieved information from the Twitter API.
It will consist of two functions:
    - the first will create two main tables, one for the IDs of our influencers and another for storing ALL of their
      tweets, including non-crypto related ones
    - the second one will create 15 tables for the 15 cryptocurrencies where their tweets will be stored regardless
      of the crypto influencer.
"""


"""
IMPORT LIBRARIES
"""
# Import psycopg2 for creating and manipulating SQL databases
import psycopg2
# Import pandas for dataframe manipulation
import pandas as pd
# Import Login_access for logging in in our recently created DB
from Login_access import host, port, user, password

"""
MAIN FUNCTION
"""
# Main function
def main():
    # Defining global variables
    global conn
    # Connecting to database server
    conn = psycopg2.connect(host=host, database="TwitterDB", port=port, user=user, password=password)
    # Retrieve crypto names and their tickers
    cryptos = pd.read_csv('./crypto_names.txt', sep=',')

    # Execute the creation of the tables
        # Main tables:
    main_tables()
        # Crypto-specific tables:
    for crypto in cryptos.symbol:
        crypto_tables(crypto)


"""
DEFINE OTHER FUNCTIONS
"""
def main_tables():
    # Writing the commands for the creation of these two main tables
    # They will host ALL tweets from all of our "Crypto" influencers
    commands = (  # Table 1
        '''Create Table TwitterUser(User_Id BIGINT PRIMARY KEY, User_Name TEXT);''',
        # Table 2
        '''Create Table TwitterTweet(Tweet_Id BIGINT PRIMARY KEY,
                                             User_Id BIGINT,
                                             Tweet TEXT,
                                             Retweet_Count INT,
                                             Created_At TIMESTAMP WITH TIME ZONE,
                                             CONSTRAINT fk_user
                                                 FOREIGN KEY(User_Id)
                                                     REFERENCES TwitterUser(User_Id));''')
    # Create the cursor to execute SQL commands
    cur = conn.cursor()

    # Execute SQL commands
    for command in commands:
        # Execute the queries
        cur.execute(command)

    # Disconnect
    conn.commit()
    cur.close()


def crypto_tables(crypto):
    # We create the exact same process as the previous function, but this time for creating tables that will ONLY
    # contain tweets from a specific cryptocurrency
    command = "Create Table "+crypto+"     (Tweet_Id BIGINT PRIMARY KEY, \
                                            User_Id BIGINT, \
                                            Tweet TEXT, \
                                            Retweet_Count INT, \
                                            sentiment_score DECIMAL(10, 2), \
                                            outcome TEXT, \
                                            Created_At TIMESTAMP WITH TIME ZONE, \
                                            CONSTRAINT fk_user \
                                                FOREIGN KEY(User_Id) \
                                                    REFERENCES TwitterUser(User_Id));"

    cur = conn.cursor()
    # Execute the query
    cur.execute(command)

    # Disconnect
    conn.commit()
    cur.close()


"""
EXECUTION
"""
# Run main
if __name__ == "__main__":
    main()
