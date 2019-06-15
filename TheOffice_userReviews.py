#!/usr/bin/env python
# coding: utf-8

# In[57]:


import requests
import pandas as pd
import urllib.request
from bs4 import BeautifulSoup
import nltk
from nltk import word_tokenize
import re
import base64
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('vader_lexicon')
import bokeh
import bokeh.plotting as bkh
import plotly.offline as pyo
import plotly.graph_objs as go
import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
from flask import send_from_directory
import dash_html_components as html

# from bokeh.models import HoverTool
bkh.output_notebook()

def preprocessing(sentence):
    
    sentence=word_tokenize(sentence)
    wnl = nltk.WordNetLemmatizer()
    sentence=[wnl.lemmatize(t) for t in sentence]
    sentence=' '.join(sentence)
    return sentence


# In[2]:


URL_critics=["https://www.metacritic.com/tv/the-office/season-1/critic-reviews",
      "https://www.metacritic.com/tv/the-office/season-2/critic-reviews",
     "https://www.metacritic.com/tv/the-office/season-3/critic-reviews",
     "https://www.metacritic.com/tv/the-office/season-4/critic-reviews",
     "https://www.metacritic.com/tv/the-office/season-5/critic-reviews",
     "https://www.metacritic.com/tv/the-office/season-6/critic-reviews",
     "https://www.metacritic.com/tv/the-office/season-7/critic-reviews",
     "https://www.metacritic.com/tv/the-office/season-8/critic-reviews",
     "https://www.metacritic.com/tv/the-office/season-9/critic-reviews" 
    ]

URL_users=["https://www.metacritic.com/tv/the-office/season-1/user-reviews",
      "https://www.metacritic.com/tv/the-office/season-2/user-reviews",
     "https://www.metacritic.com/tv/the-office/season-3/user-reviews",
     "https://www.metacritic.com/tv/the-office/season-4/user-reviews",
     "https://www.metacritic.com/tv/the-office/season-5/user-reviews",
     "https://www.metacritic.com/tv/the-office/season-6/user-reviews",
     "https://www.metacritic.com/tv/the-office/season-7/user-reviews",
     "https://www.metacritic.com/tv/the-office/season-8/user-reviews",
     "https://www.metacritic.com/tv/the-office/season-9/user-reviews" 
    ]

user_agent = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'
headers={'User-Agent':user_agent}

#Season wise data
Raw_data_critics=[]
Raw_data_users=[]

for url in URL_critics:
    print(url)
    request=urllib.request.Request(url,None,headers)
    response=urllib.request.urlopen(request)
    data=response.read()
    soup=BeautifulSoup(data,"html.parser")

    
    critics_comments=( soup.find_all('div',attrs={'class':'review_body'}))
    scores=(soup.find_all('div',attrs={'class':('metascore_w medium tvshow positive indiv','metascore_w medium tvshow mixed indiv','metascore_w medium tvshow negative indiv')}))
#     print(len(critics_comments[:len(scores)]),len(scores))
    Raw_data_critics.append(pd.DataFrame([ [ i.text for i in critics_comments[:len(scores)] ] ,  [j.text for j in  scores]  ] ).T)
    
for url in URL_users:
    
    request=urllib.request.Request(url,None,headers)
    response=urllib.request.urlopen(request)
    data=response.read()
    soup=BeautifulSoup(data,"html.parser")
    
    users_comments=( soup.find_all('div',attrs={'class':'review_body'}))
    scores=(soup.find_all('div',attrs={'class':("metascore_w user medium tvshow positive indiv",'metascore_w user medium tvshow positive indiv perfect','metascore_w user medium tvshow mixed indiv','metascore_w user medium tvshow negative indiv' )}))
#     print(len(users_comments[:len(scores)]),len(scores))
    Raw_data_users.append(pd.DataFrame( [ [ i.text for i in users_comments[:len(scores)]  ],[j.text for j in  scores] ]).T)
    


# In[4]:


sid = SentimentIntensityAnalyzer()

Season_scores_users=[]
Season_scores_critics=[]

for season_number in range(9):
    total_sum=0
    
    for i in Raw_data_users[season_number][0]:
        ss=sid.polarity_scores(i)
        total_sum=total_sum+ss['compound']
    
    Season_scores_users.append(total_sum*(1/Raw_data_users[season_number].shape[0]))
    
for season_number in range(9):
    total_sum=0
    
    for i in Raw_data_critics[season_number][0]:
        ss=sid.polarity_scores(i)
        total_sum=total_sum+ss['compound']
    
    Season_scores_critics.append(total_sum*(1/Raw_data_users[season_number].shape[0]))
       
    


# In[5]:


f = bkh.figure(title="User Reviews",width=600, height=400, tools='box_zoom,reset,save')


f.line([1,2,3,4,5,6,7,8,9], Season_scores_critics)

bkh.show(f)


# In[6]:


characters={
    
    "jim":['jim halpert','jim','halpert','john krasinski','john','krasinski'],
    "micheal": ['micheal','scott','steve carell','steve','carell'],
    "dwight": ["dwight schrute","schrute","dwight","rainn wilson","rainn","wilson"],
    "pam": ["pam","beesly","jenna fischer","jenna","fischer"],
    "andy": ["andy bernard","andy","bernard","ed helms",'ed','helms'],
    "angela": ["angela","martin","kinsey"],
    "kevin": ["kevin", "malone" , "brian","baumgartner"],
    "toby":["toby", "flenderson", "paul","lieberstein"],
    "stanley":["stanley", "james", "hudson", "leslie", "david", "baker"],
    "ryan": ['ryan','howard','novak'],
    "kelly": ["kelly","kapoor","mindy", "kaling"],
    "creed": ["creed","bratton"],
    "erin": ["erin","hannon","ellie","kemper"],
    "karen": ["karen","fillipelli", "rashida","jones"],
    "meredith":["meredith","kate","flannery"],
    "phyllis":["phyllis","vance","smith"],
    "oscar":["oscar"]    
}

character_polarity={
    
    "jim":[],
    "micheal": [],
    "dwight": [],
    "pam": [],
    "andy": [],
    "angela": [],
    "kevin": [],
    "toby":[],
    "stanley":[],
    "ryan": [],
    "kelly": [],
    "creed": [],
    "erin": [],
    "karen": [],
    "meredith":[],
    "phyllis":[],
    "oscar":[] 
    
}
   
for season_number in range(9):

    #similar structure to character_polarity dictory ( copying the dictionary instead of referencing it)
    character_polarity_seasonwise={
    
    "jim":[],
    "micheal": [],
    "dwight": [],
    "pam": [],
    "andy": [],
    "angela": [],
    "kevin": [],
    "toby":[],
    "stanley":[],
    "ryan": [],
    "kelly": [],
    "creed": [],
    "erin": [],
    "karen": [],
    "meredith":[],
    "phyllis":[],
    "oscar":[] 
    
    }
 
      
    #iterate through the comments
    for i in Raw_data_critics[season_number][0]:

        for actor in characters.keys():
            if( len(set(characters[actor]) & set(word_tokenize(i.lower())))>1):         
                ss=sid.polarity_scores(i)
                character_polarity_seasonwise[actor].append(ss['compound'])
                
    for i in Raw_data_users[season_number][0]:

        for actor in characters.keys():
            if( len(set(characters[actor]) & set(word_tokenize(i.lower())))>1):         
                ss=sid.polarity_scores(i)
                character_polarity_seasonwise[actor].append(ss['compound'])
                
    
    for a in characters.keys():

        if(len(character_polarity_seasonwise[a])>0):
            character_polarity[a].append(sum(character_polarity_seasonwise[a])/len(character_polarity_seasonwise[a]))
        else:
            character_polarity[a].append(0)


# In[ ]:


data=[]
app = dash.Dash()
seasons_Yaxis=[1,2,3,4,5,6,7,8]
# character_colors=['grey','red','blue','green','yellow','orange','rgb(158,202,225)','rgb(8,48,107)']
# color_flag=0
                  
for i in character_polarity:
    print(i,character_polarity[i])
    
    print(color_flag)
    trace=go.Scatter(
                        y =character_polarity[i],
                        x =seasons_Yaxis,
                        mode = 'lines',
                        name = i
                    )
                  
    data.append(trace)
    
graphi=dcc.Graph(id="hmm",
                    figure = {'data':data,
                    'layout':go.Layout(title="Character trends over the seasons",
                                        xaxis = {'title':'Season'})} )

#host resources    
resource_directory = '/home/srvopsresearch/anomaly_detection'
resources = ['the-office.jpg']
static_resource_route = '/static/'

@app.server.route('{}<resource>'.format(static_resource_route))
def serve_stylesheet(resource):
    if resource not in resources:
        raise Exception(
            '"{}" is excluded from the allowed static files'.format(
                resource
            )
        )
    return send_from_directory(resource_directory, resource)
  
ok=html.Img(
                                    src= 'http://vc2coma1365148n:8050/static/the-office.jpg',
                                    style = {
                                        'position':'relative',
                                        'heigth':'80px',
                                        'width':'80px'
                                    }
                                ) 

app.layout = html.Div([
    ok,graphi
])


app.run_server(host='0.0.0.0')


# In[ ]:




