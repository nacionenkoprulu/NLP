import pandas as pd
# natural language processing: n-gram ranking
import re
import unicodedata
import nltk
from nltk.corpus import stopwords
# add appropriate words that will be ignored in the analysis
ADDITIONAL_STOPWORDS = ['covfefe']

import matplotlib.pyplot as plt



df = pd.read_csv("/Users/hayirliolsun/Desktop/Yüksek Lisans Dosyaları/N Gram Uygulama/tweets.csv")

df.head()



def basic_clean(text):
  """
  A simple function to clean up the data. All the words that
  are not designated as a stop word is then lemmatized after
  encoding and basic regex parsing are performed.
  """
  wnl = nltk.stem.WordNetLemmatizer()
  durakKelime = stopwords.words('english') + ADDITIONAL_STOPWORDS
  text = (unicodedata.normalize('NFKD', text)
    .encode('ascii', 'ignore')
    .decode('utf-8', 'ignore')
    .lower())
  words = re.sub(r'[^\w\s]', '', text).split()
  return [wnl.lemmatize(word) for word in words if word not in durakKelime]

words = basic_clean(''.join(str(df['text'].tolist())))

print(words[:20])

print((pd.Series(nltk.ngrams(words, 2)).value_counts())[:10])

print((pd.Series(nltk.ngrams(words, 3)).value_counts())[:10])



