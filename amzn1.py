import requests
from bs4 import BeautifulSoup
import pandas as pd
user_name= input('Please Enter URL')
r = requests.get(user_name)
reviewlist = []

def get_soup(url):
    global soup
    #r = requests.get('http://localhost:8050/render.html', params={'url': url, 'wait': 2})
    soup = BeautifulSoup(r.text, 'html.parser')
    return soup


def get_reviews(soup):
    reviews = soup.find_all('div', {'data-hook': 'review'})
    try:
        for item in reviews:
            review = {
            'product': soup.title.text.replace('Amazon.co.uk:Customer reviews:', '').strip(),
            'title': item.find('a', {'data-hook': 'review-title'}).text.strip(),
            'rating':  float(item.find('i', {'data-hook': 'review-star-rating'}).text.replace('out of 5 stars', '').strip()),
            'Reviews': item.find('span', {'data-hook': 'review-body'}).text.strip(),
            }
            reviewlist.append(review)
    except:
        pass

for x in range(1,999):
    soup = get_soup(f'https://www.amazon.co.uk/product-reviews/B07WD58H6R/ref=cm_cr_arp_d_paging_btm_next_2?ie=UTF8&reviewerType=all_reviews&pageNumber={x}')
    #print(f'Getting page: {x}')
    get_reviews(soup)
    #print(len(reviewlist))
    if not soup.find('li', {'class': 'a-disabled a-last'}):
        pass
    else:
        break

#df = pd.DataFrame(reviewlist)
# df.to_excel('sony-headphones.xlsx', index=False)
# print('Fin.')

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
#data.columns = data.columns.str.strip()
#vaders.compound = vaders.compound.str.strip()
#data = data.rename(columns={'Number ': 'Number'})
vaders = vaders.rename(columns={'compound': 'compound'})
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
