import requests
import numpy as np
from bs4 import BeautifulSoup
import pandas as pd

user_name= input('Please Enter URL')
r = requests.get(user_name)



#r = requests.get("https://www.amazon.in/Apple-iPhone-13-256-GB/product-reviews/B09V4MXBSN/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews")
soup = BeautifulSoup(r.content,"html.parser")
contents = soup.prettify()
#print(contents)
#print(contents)
#info_box = soup.find(class_="a-row a-spacing-small review-data")
reviews = soup.find_all('div', {'data-hook': 'review'})
reviewlist = []
for item in reviews:
            review = {
            'Reviews': item.find('span', {'data-hook': 'review-body'}).text.strip(),
            }
            reviewlist.append(review)

#print(reviewlist)

a = pd.DataFrame(reviewlist)
a.to_csv('amazon_review.csv')
fb = pd.read_csv('amazon_review.csv')
fb.rename(columns = {'Unnamed: 0':'Id'}, inplace = True)
fb

import nltk
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ggplot')

from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm
from tqdm import tqdm
import nltk
nltk.downloader.download('vader_lexicon')

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()

res = {}
for i, row in tqdm(fb.iterrows(), total=len(fb)):
    text = row['Reviews']
    myid = row['Id']
    res[myid] = sia.polarity_scores(text)

vaders = pd.DataFrame(res).T
vaders = vaders.reset_index().rename(columns={'index': 'Id'})
vaders = vaders.merge(fb, how='left')

#m = vaders.filter(['compound'])
m = vaders.compound
#m=vaders.iloc[4]
from numpy.ma.core import negative
Result=[]
for x in m:
  if int(x >= 0):
      pos = "positive"
      Result.append(pos)
  else:
      neg = "negative"
      Result.append(neg)
      
vaders['Result'] = Result
print(vaders.head)
len_vaders = len(vaders)
print(f'Total no of Reviews : {len_vaders}')

from numpy.ma.core import negative
POSITIVEf = []
NEGATIVEf = []
for x in m:
    if (x >= 0):
      pos = "positive"
      POSITIVEf.append(pos)
    else:
      neg = "negative"
      NEGATIVEf.append(neg)


lenpos=len(POSITIVEf)
print(f'Total no of Positive Reviews : {lenpos}')


lennev=len(NEGATIVEf)
print(f'Total no of Negative Reviews : {lennev}')
