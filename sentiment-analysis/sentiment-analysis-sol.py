# %%
import nltk
from nltk.tokenize import WordPunctTokenizer, TreebankWordTokenizer
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import string
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# %%
# Find positive and negative reviews

# %%
wine = pd.read_csv(r"C:\Users\trucn\Documents\repositories\ds-mod-2-0924-code-solutions\sentiment-analysis\winemag-data.csv", index_col = 0)

# %%
wine

# %%
desc = wine['description']

# %%
analyzer = SentimentIntensityAnalyzer()

# %%
wine['description_sentiment'] = desc.apply(lambda x: analyzer.polarity_scores(x)['compound'])

# %%
wine['sentiment'] = wine['description_sentiment'].apply(lambda x: "positive" if x > 0.05 else "negative" if x < -0.05 else 'neutral')

# %%
wine

# %%
wine.loc[wine['sentiment'] == 'negative', 'description']

# %%
desc_pol = [TextBlob(d).sentiment for d in desc]

# %%
desc_pol[0][0]

# %%
polarity_scores = [desc_pol[p][0] for p in range(len(desc_pol))]

# %%
wine['description_polarity'] = pd.Series(polarity_scores)

# %%
wine['sentiment_2'] = wine['description_polarity'].apply(lambda x: "positive" if x > 0 else "negative" if x < 0 else 'neutral')

# %%
wine


