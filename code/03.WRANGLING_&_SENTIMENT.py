"""
-----------------------------------------------------------------------------------------------------------------------
PROGRAM #3
-----------------------------------------------------------------------------------------------------------------------
The objective of this code is to firstly filter, among all of our tweet database, the ones related to a certain crypto.
After that, we wrangle the data and clean it so that the output is in a better format for the computer to read it.
Finally, we apply a sentiment score to each one of the tweets with a rule-base analysis (with some alien predefined
set of rules)
"""

"""
IMPORT LIBRARIES
"""
# Import psycopg2 for creating and manipulating SQL databases
import psycopg2
# Import re for regular expressions' wrangling
import re
# Import pandas for dataframe manipulation
import pandas as pd
# We use SentimentIntensityAnalyzer because it is a lexicon and rule-based sentiment analysis tool that is specifically
# for sentiments analysis in social media because it works well on special characters such as emojis and some
# combination of punctuations.
from nltk.sentiment.vader import SentimentIntensityAnalyzer

analyser = SentimentIntensityAnalyzer()
from nltk.corpus import stopwords

stopwords = stopwords.words('english')
from nltk.stem.snowball import SnowballStemmer

snowballstemmer = SnowballStemmer("english")
from nltk.tokenize import word_tokenize
# Import Login_access for logging in in our DB
from Login_access import host, port, user, password

"""
MAIN FUNCTION
"""
def main():
    global crypto, symbol, creation_date, sentiment_score, outcome
    cryptos = pd.read_csv('./crypto_names.txt', sep=',')
    for i, symbol in enumerate(cryptos.symbol):
        crypto = cryptos['name'][i]
        creation_date = cryptos['release_date'][i]
        # Connecting to the DB and applying certain requirements
        dbCondition(crypto, symbol, creation_date)
        # For each tweet meeting the requirements below in the DB...
        for tweet_info in cur.fetchall():
            # Extract tweet's ID and the tweets itself
            user_id, tweet_id, tweet, retweet_count, created_at = tweet_info
            # Perform a first stage text cleaning
            tweet_edit = semiclean_text(tweet, crypto, symbol)
            # Perform stemming of each word in each tweet (i.e., changing --> chang)
            tweet_edit = tokenization_and_stem(tweet_edit)
            # Applying a rule-based algorithm for guessing sentiments for each tweet
            score, outcome = sentiment_analysis(tweet_edit)
            # Save changes inside each crypto's table
            db_append(symbol, tweet_id, user_id, tweet, retweet_count, score, outcome, created_at)
        print(symbol)

    # Disconnect
    cur.close()
    conn.close()


"""
DEFINE OTHER FUNCTIONS
"""
def dbCondition(crypto, symbol, creation_date):
    global cur, conn
    conn = psycopg2.connect(host=host, database="TwitterDB", port=port, user=user, password=password)
    # Create the cursor to execute SQL commands
    cur = conn.cursor()
    # Filtering each tweet
    command = "SELECT user_id, tweet_id, tweet, retweet_count, created_at " \
              "FROM twittertweet WHERE (tweet LIKE '%" + symbol + "%' OR tweet LIKE '%" + crypto + "%') AND " \
              "created_at > '" + creation_date + "';"
    cur.execute(command)
    # Commit changes
    conn.commit()


# This function will be used for performing a series of actions that will clean our text before calculating
# a sentiment analysis to it
def semiclean_text(tweet, crypto, symbol):
    tweet = tweet.lower()  # Transforms text to lower caps
    tweet = re.sub('#{}'.format(crypto), '{}'.format(crypto), tweet)  # Substitutes crypto hashtag for crypto
    tweet = re.sub('@[\S]+', '', tweet)  # Removing mentions
    tweet = re.sub('#[A-Za-z0-9]+', '', tweet)  # Removing any other hashtag
    tweet = re.sub('https?:\/\/\S+', '', tweet)  # Removing links to webpages
    tweet = re.sub('\\n', '', tweet)  # Removing the new line character
    tweet = re.sub(r"(”|“|-|\+|`|#|,|;|\|)*", "", tweet)  # Removing special character
    tweet = re.sub(r"&amp", "", tweet)  # Removing spaces
    tweet = re.sub(r"[0-9]*", "", tweet)  # Removing numbers

    # Now, let's transform some slang acronyms into real sentences, so that the AI can understand them better
    tweet = re.sub('idk', "i don't know", tweet)
    tweet = re.sub('smh', "shaking my head", tweet)
    tweet = re.sub('ikr', "i know",
                   tweet)  # We purposely avoid writing "right" because in this context it does not have a positive meaning
    tweet = re.sub('immd', "it made my day", tweet)
    tweet = re.sub('snh', "sarcasm noted here", tweet)
    tweet = re.sub('ama', "ask me anything", tweet)
    tweet = re.sub('icymi', "in case you missed it", tweet)
    tweet = re.sub('dr', "double rainbow", tweet)
    tweet = re.sub('mfw', "my face when", tweet)
    tweet = re.sub('rofl', "rolling on floor laughing", tweet)
    tweet = re.sub('stfu', "shut the fuck up", tweet)
    tweet = re.sub('nvm', "never mind", tweet)
    tweet = re.sub('tbh', "to be honest", tweet)
    tweet = re.sub('btw', "by the way", tweet)
    tweet = re.sub('aka', "also known as", tweet)
    tweet = re.sub('asap', "as soon as possible", tweet)
    tweet = re.sub('np', "no problem", tweet)
    return tweet


def tokenization_and_stem(tweet):
    snowballstemmer_token_ls = []
    tokens = word_tokenize(tweet)
    for token in tokens:
        if token not in stopwords:
            snowballstemmer_token_ls.append(snowballstemmer.stem(token))
    return ' '.join(snowballstemmer_token_ls)


def sentiment_analysis(tweet):
    score = analyser.polarity_scores(tweet)
    if score['compound'] >= 0.05:
        return score['compound'], 'positive'
    if score['compound'] <= -0.05:
        return score['compound'], 'negative'
    else:
        return score['compound'], 'neutral'


# This will be a function that will append each processed tweet inside its own crypto table
def db_append(symbol, tweet_id, user_id, tweet, retweet_count, sentiment_score, outcome, created_at):
    # Create the cursor to execute SQL commands
    cur = conn.cursor()
    # insert tweet information into the TwitterTweet table
    command = "INSERT INTO " + symbol + "(tweet_id,user_id,tweet,retweet_count,sentiment_score,outcome,created_at)" \
                                        "VALUES (%s,%s,%s,%s,%s,%s,%s);"
    # Execute the query
    cur.execute(command, (tweet_id, user_id, tweet, retweet_count, sentiment_score, outcome, created_at))
    # Commit changes
    conn.commit()
    # Disconnect
    cur.close()


"""
EXECUTION
"""
if __name__ == "__main__":
    main()
