"""
-----------------------------------------------------------------------------------------------------------------------
PROGRAM #2
-----------------------------------------------------------------------------------------------------------------------
The objective of this code is to firstly obtain all the tweets from 2008 to 2021 of our crypto-influencers to then
store them in the DB we created in the first program of this folder.
"""

"""
IMPORT LIBRARIES
"""
# Import psycopg2 for creating and manipulating SQL databases
import psycopg2
# Import tweepy for accessing the Twitter API
import tweepy
# Import pandas for dataframe manipulation
import pandas as pd
# Import Login_access for logging in in our recently created DB and the Twitter API
from Login_access import host, port, user, password, bearer_token

"""
MAIN FUNCTION
"""
# Main function
def main():
    # Defining global variables
    global conn, cur, stored_influencers
    # Connection to database server
    conn = psycopg2.connect(host=host, database="TwitterDB", port=port, user=user, password=password)
    # Read the crypto-influencers' data, like their nametag in Twitter
    df_influencers = pd.read_csv('/Volumes/GoogleDrive-108620229832084644835/Mi unidad/CE&M/'
                                 '2nd year/2do semestre/TFM/TFM-coding/Twitter/influencers.csv')
    try:
        # Create the cursor to execute SQL commands
        cur = conn.cursor()
        # Check all influencers already in the data, in case there is one or more, the code will not retrieve date from
        # him/her
        check_query = "SELECT DISTINCT B.user_name " \
                      "FROM twittertweet A " \
                      "LEFT JOIN twitteruser B ON A.user_id=B.user_id"
        # Execute the query
        cur.execute(check_query)
        # Store already scraped influencers
        stored_influencers = cur.fetchall()

    # Print comment in case there is none
    except (Exception, psycopg2.Error) as error:
        print("No data in the database", error)

    # Disconnect
    if conn:
        cur.close()

    # For each influencer in our dataframe, if it's not already in our DB, retrieve all its tweets and store them
    # into the TwitterUser and the TwitterTweet tables
    for influencer in df_influencers.username.drop_duplicates():
        if (influencer.replace('@', ''),) not in stored_influencers:
            print(influencer)
            extract_tweets_and_store(influencer)
        else:
            pass

    # Disconnect
    conn.close()


"""
DEFINE OTHER FUNCTIONS
"""
# This will be a function inside another function (extract_tweets_and_store) and will append each retrieved tweet from
# the API into the TwitterUser and the TwitterTweet tables. In case the information is repeated, it will not append.
def dbConnect(user_id, user_name, tweet_id, tweet, retweet_count, created_at):
    # Create the cursor to execute SQL commands
    cur = conn.cursor()
    # insert user information into the TwitterUser table
    command = '''INSERT INTO TwitterUser (user_id, user_name) VALUES (%s,%s) ON CONFLICT
                 (User_Id) DO NOTHING;'''
    # Execute the query
    cur.execute(command, (user_id, user_name))
    # insert tweet information into the TwitterTweet table
    command = '''INSERT INTO TwitterTweet (tweet_id, user_id, tweet, retweet_count, created_at) 
                 VALUES (%s,%s,%s,%s,%s);'''
    # Execute the query
    cur.execute(command, (tweet_id, user_id, tweet, retweet_count, created_at))
    # Commit changes
    conn.commit()
    # Disconnect
    cur.close()

# this function will retrieve non-retweeted tweets from our influencers since the creation of Twitter. More precisely,
# it will take the tweet itself, as well as the user id, username, tweet id, # of retweets  and creation date.
def extract_tweets_and_store(influencer):
    # Twitter API search query
    # Will use it for searching non-retweeted tweets from our crypto-influencers
    query = 'from:{} -is:retweet'.format(influencer.replace('@', ''))

    # Set start results' date
    start_time = '2008-01-01T00:00:00Z'

    # Set last results' date
    end_time = '2021-12-31T00:00:00Z'

    n_results = 500

    # Connect to Twitter API client
    client = tweepy.Client(bearer_token=bearer_token, wait_on_rate_limit=True)

    # For every input from the API save the features in it with the dbConnect function
    for count, response in enumerate(tweepy.Paginator(client.search_all_tweets,
                                                      query=query,
                                                      user_fields=['username', 'public_metrics', 'description',
                                                                   'location'],
                                                      tweet_fields=['created_at', 'geo', 'public_metrics', 'text'],
                                                      expansions='author_id',
                                                      start_time=start_time,
                                                      end_time=end_time,
                                                      max_results=n_results)):

        for tweet in response.data:
            dbConnect(user_id=tweet.author_id,
                      user_name=response.includes['users'][0].username,
                      tweet_id=tweet.id,
                      tweet=tweet.text,
                      retweet_count=tweet.public_metrics['retweet_count'],
                      created_at=tweet.created_at)

        print("{} tweets stored in PostGreSQL for {}".format(n_results * (count + 1), influencer))


"""
EXECUTION
"""
# Run main
if __name__ == "__main__":
    main()
