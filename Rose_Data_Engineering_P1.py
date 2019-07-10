#!/usr/bin/env python
# coding: utf-8

# # Will you accept this model? Predicting love on the Bachelorette
# 
# ## Introduction
# Recently I started watching The Bachelorette with my girlfriend and found out that like fantasy football, she plays fantasy Bachelorette with her friends.  Part of the scoring system involves deciding who is going to receive a rose and who is going to be eliminated.  Partially to keep myself engaged and partially to help her win, I wondered if machine learning could be used to predict which rounds contestants were going to be eliminated?
# 
# In summary, I ended up not being able to get lower than a two round error when predicting elimination since the data collected didn't have enough correlation to the target variable.  We can still have a vague idea of if a contestant will be eliminated towards the beginning or end of the show, but there is certainly room for improvement which is exciting.  After looking around, I couldn't find a good dataset to use and ended up constructing one from scratch by scraping various webpages and sorting the information into a nice table.  
# 
# This project ended up being rather long, so I broke it up into three parts and created a table of contents. If you are interested in the data engineering read parts 1 and 2.  If you want to learn more about the machine learning read part 3.
# 
# ## Table of Contents
# 
# [Part 1](https://github.com/desdelgado/Predicting_Roses/blob/master/Rose_Data_Engineering_P1.ipynb) - Initial dataset from 538's website, target variable extraction, and feature engineering
# 
# [Part 2](https://github.com/desdelgado/Predicting_Roses/blob/master/Rose_Data_Engineering_P2.ipynb) - Does the bachelorette and a contestant have the same political leanings, hometown, and cultural background?
# 
# [Part 3](https://github.com/desdelgado/Predicting_Roses/blob/master/Predicting_Roses.ipynb) - Modeling of custom-built dataset and conclusion.

# # Part 1
# 
# ## Introduction
# 
# Like in all good projects, the first step was finding a good data set.  After some major googling, I found [this](https://github.com/fivethirtyeight/data/tree/master/bachelorette) dataset put together by 538. So, let's import it and explore what we have.  Before we do let's import some important libraries.

# In[1]:


import pandas as pd
import re
import numpy as np

#Scrape websites
from bs4 import BeautifulSoup 
import requests
import warnings


# I downloaded the csv off the github to my own computer.

# In[2]:


elim_data = pd.read_csv('Bachelorette_Data/Contestant.csv')


# ## Explore the Data
# 
# Great, now let's get a feel for what we are working with:

# In[3]:


print(elim_data.head(10))
print(elim_data.columns)


# It seems like we get the contestant name, which show and season they appeared on.  After reading a bit more on the website, it looks the 'elimination' columns tell us at which round or rose ceremony did people either get a "R" for a rose and stayed or some form of "E" for eliminated.  Additionally, the "DATES" columns tell us at each round what kind of date each contestant went on.  'D3' for example was a three-person date.  Finally, there seems to be a 'R1' which according to the website means that person got the 'First Impression' rose. Let's explore this data a little more.

# In[4]:


elim_data.info()


# Wow, a total of 921 entries but lots of these columns have less than that which means they're filled with null values. This makes sense, however, as once a contestant is eliminated it means that they no long have data there.  In a way, this is very nice for us as it provides a complete square data set we can work through while keeping in mind the null values. 
# 
# 
# ## Clean Up the Data
# 
# Now let's clean up some of this table.  One thing I also noticed is that the first row is almost a repeat of the columns let's get rid of that real quick.

# In[5]:


print(elim_data.iloc[0,:])
elim_data = elim_data.drop(elim_data.index[0])
print(elim_data.head())


# In that row it looked like 'ID' was under the 'CONTESTANT' column.  Let's check to make sure there are not more rows like that by using a .sum() function.

# In[6]:


elim_data[elim_data.CONTESTANT == 'ID'].sum()


# Ahh it looks that type of row repeats a bunch.  We can quickly get rid of those types of rows and reset the index.

# In[7]:


elim_data = pd.DataFrame(elim_data[elim_data.CONTESTANT != 'ID'])
elim_data = elim_data.reset_index()


# Let's then also check how complete of a data set this is by looking at which seasons of each show there are.

# In[8]:


Bachelorette_count = elim_data[elim_data['SHOW'] == 'Bachelorette']['SEASON'].unique()
Bachelor_count = elim_data[elim_data['SHOW'] == 'Bachelor']['SEASON'].unique()

print('Number of Bachelorette seasons = '  +str(Bachelorette_count))
print('Number of Bachlor seasons = '  +str(Bachelor_count))


# Okay so after some googling there seems to be 23 bachelor seasons and 14 seasons of the bachelorette that have aired.  So, we have most of the seasons covered. Let's then see how many winners there are. We can do this by looking for a 'W' in each column which according to the website denotes the winner.

# In[9]:


winner_count = 0 # Come back to this 
for col in elim_data.columns[4:]: 
    winner_count = winner_count + elim_data[col].str.count("W").sum()
print('Number of Winners: ' + str(int(winner_count)))


# I would have thought 34 (21 bachelor + 13 bachelorette) which means we're missing a season winner.  After some googling and consulting with my girlfriend apparently season 11 didn’t have a winner.  In a small context this is a good example of where talking to domain experts allows one to avoid unnecessary writing code to find out why you're missing some data.
# 
# So now that we know a little bit more about this data, it appears it doesn't give us the features we want to look at upfront.  Thus, let's do a bit of our own data engineering so we can tackle this question.  My first inking is to grab the contestants, and which show (bachelor/bachelorette) they showed up on.

# In[10]:


data_table = pd.DataFrame(elim_data[['CONTESTANT', 'SHOW']])


# ##  Create Target Variable
# 
# Next we want our target variable.  In this case it's going to be the round they were eliminated in.  My initial guess was to use some sort of nested loop and check where 'E' showed up and somehow count how many rounds it was in.  I, however, immediately felt this dread every first year CS student feels when they have to code some complicated loop for the first time.  Additionally, using some sort of nested loop on any large dataset also sends alarm bells off in my head as it has O(n^2) time complexity which is no buno.  While it doesn't matter here, let's make sure to practice good habits for when we are working with a real big dataset.  
# 
# After some thought, I remembered the .apply() function allows one to all at once apply a function to each row or column.  While this took a bit of time, we can write a function that will search the row and find if 'E' or some form of 'E' shows up.  Then we can grab the number in the column (i.e 2 for 'Elimination-2, etc) and report that number to a separate column for each contestant as 'Round Eliminated.'

# In[11]:


elim_data.columns = elim_data.columns.str.replace('ELIMINATION-', '') # Instead of finding the number we can just strip 'ELIMINATION-'


# In[12]:


def r_elim(ind):
    '''
        Returns the column name where any type of elimination was found.  Column name is the round
        they were eliminated.  Returns 0 for winners
    '''
    if 'E' in ind.values:
        return ind[ind == 'E'].index[0]
    elif 'ED' in ind.values:
        return ind[ind == 'ED'].index[0]
    elif 'EQ' in ind.values:
        return ind[ind == 'EQ'].index[0]
    elif 'EF' in ind.values:
        return ind[ind == 'EF'].index[0]
    elif 'EU' in ind.values:
        return ind[ind == 'EU'].index[0] 
    #Now for winners
    else:
        return 0


# In[13]:


data_table['Round_Eliminated'] = elim_data.apply(r_elim, axis = 1)
print(data_table.head())


# Dope, we can do some googling and find that Bryan was in fact the winner of season 13 and Peter lost in the finals.  Side note: I spent more time that I care to admit reading about the Rachel and Peter drama.
# 
# ## Feature Engineering
# 
# Now that we have our target variable, we can start thinking about constructing some features. Reading a bit more from this [article](https://fivethirtyeight.com/features/the-bachelorette/), it was noted that first impression roses play a strong indicator of people that will win or go far. Again, it's important to point out that doing some research on the topic can go a long way.  Let's use the same .apply() principle and create a categorical variable for if a contestant got a first impression rose or not. 

# In[14]:


def FI_rose(ind):
    if 'R1' in ind.values:
        return 1
    else:
        return 0

data_table['First_Impression_Rose'] = elim_data.apply(FI_rose, axis = 1)


# Furthermore, according to the same 538 article going on an early first date is a good indicator a person might do well.  To get such a variable, we would need to first figure out how many rounds each season has, figure out when a contestant went on a 'D1', and then see what percentage of the season is left. 
# 
# So, let's first get the number of rounds in each season.  We can pick this off by using a trick from before.  Wherever 'W' appears is also the last round.  So, we can again grab that round number but also the show and season and save that data to a list.

# In[15]:


max_episode_list = []

def max_episode(ind):
    if 'W' in ind.values:
       max_episode_list.append((ind.get(key = 'SHOW'), ind.get(key = 'SEASON'), int(ind[ind == 'W'].index[0])))

elim_data.apply(max_episode, axis = 1)

#add season 11 max episode
max_episode_list.append(('Bachelor', '11', 8))


# We can now merge this list into the "elim_data" dataframe so each contestant has the max episode in their row.  We can then print them to check if we got the right numbers.

# In[16]:


max_episode_df = pd.DataFrame(max_episode_list, columns = ['SHOW', 'SEASON', 'MAX_EPISODE'])
elim_data = pd.merge(elim_data, max_episode_df, on = ['SHOW', 'SEASON'], how = 'left')

print(elim_data.columns)
print(elim_data.MAX_EPISODE.head())


# Cool, now for simplicity sake let's just make a different data frame that is only dealing with the dating data.

# In[17]:


dating_data = pd.DataFrame(elim_data[['SHOW', 'SEASON', 'CONTESTANT']])
dating_data = pd.concat([dating_data, elim_data.loc[:,'DATES-1':]], axis = 1)


# We can use the same trick we used to find when a contestant was eliminated and find the round of the first 'D1'.  We can then divide that round number by the overall total rounds in each season which will give us a percentage. 

# In[18]:


#Again stripping the 'DATES-' so each column is just the round number 
dating_data.columns = dating_data.columns.str.replace('DATES-', '')

def Date1_first(ind):
    if 'D1' in ind.values:
        #find the week the first D1 happened
        week_D1 = int(ind[ind == 'D1'].index[0])
        max_episode_number = ind.get(key = 'MAX_EPISODE')
        #need to divide to normalize by number of episodes as some episodes have more seasons than others
        weeks_left_percentage = ((max_episode_number - week_D1)/max_episode_number)*100
    else:
        weeks_left_percentage = np.nan
        
    return weeks_left_percentage


# Now let's apply it to the data_table which again is our table we are constructing to eventually use in our models. 

# In[19]:


data_table['Percentage Left after D1'] = dating_data.apply(Date1_first, axis = 1)
print(data_table['Percentage Left after D1'].head())


# Let's save our current datasets to csv files so that they can be used in parts of this project going forward

# In[20]:


data_table.to_csv('Bachelorette_Data/data_table.csv')
elim_data.to_csv('Bachelorette_Data/elim_data.csv')


# ## Conclusion
# 
# At this point, we have most of the low hanging fruit in terms of features from this dataset.  We could maybe add something in terms of date composition or average pick order but for now let's start with the simple stuff.  This is a good place to break.  In [part_2](https://github.com/desdelgado/Predicting_Roses/blob/master/Rose_Data_Engineering_P2.ipynb) we'll only focus on the bachelorette data since that is the current season being watched/played.  Additionally, we will look to add more features relating to if the bachelorette and a contestant have the same hometown, political leanings, and cultural background.
