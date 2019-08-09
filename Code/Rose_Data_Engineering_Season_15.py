#!/usr/bin/env python
# coding: utf-8

# # Data Engineering for Season 15
# 
# This is just a quick script to load all the data from season 15.  Most of the code is copied from [part 2](https://github.com/desdelgado/Predicting_Roses/blob/master/Rose_Data_Engineering_P2.ipynb) and thus I won't narrate as much.   The exception being that my girlfriend and I were keeping track of certain statistics such as first impression rose and one on on dates as the season went on.  We kept a excel file and I used that to load in some of the features of each contestant. 

# In[1]:


import pandas as pd
import re

from bs4 import BeautifulSoup
import requests
import warnings


# In[2]:


seasons_wiki = ['https://en.wikipedia.org/wiki/The_Bachelorette_(season_15)']
wiki_df = pd.DataFrame()
missed_season_tracker = []
warnings.filterwarnings("ignore")
for season in seasons_wiki:

    try:
        URL= season
        response = requests.get(URL)
        soup = BeautifulSoup(response.text, 'lxml')
        
        #Need to add additional try statment because
        try:
            My_table = soup.find("table",{"class" :"wikitable sortable"})
        except:
            pass
        
        try:
            My_table = soup.find("table",{"class" :"wikitable"})
        except:
            pass
        
        contest = []
        
        for record in My_table.findAll('tr'):
                contest.append(record.text)
        
        #Convert list into dataframe
        
        contest_df = pd.DataFrame(contest)
        
        #Split the dataframe by the \n
        contest_df = contest_df.iloc[:,0].str.split('\n', expand = True)
        
        new_header = contest_df.iloc[0] #grab the first row for the header
        contest_df.columns = new_header
        
        contest_df = contest_df.iloc[1:]
        print(contest_df.columns)
        
        
        occup = contest_df[['Name', 'Hometown','Age', 'Outcome']]
        
        #need to get which season we are working with in order to construct the name to merge the tables with
        #instead of inputting a list use a regrex equations to pull the season number out of the wiki url
        season_number = int(re.findall('\d+', URL )[0])
        
        occup['SEASON'] = season_number
        
        
        #Getting the strings 
        occup['Name'] = occup['Name'].str.replace('\d+', '')
        occup.Name = occup.Name.str.strip('[]')
        occup.Name = occup.Name.str.strip('.')
        
        occup.Age = occup.Age.str.extract('(\d+)')
        
        #Have to check the varity of names make sure to talk about this 
        
        #If they have the nickname grab the middle one
        occup.loc[occup['Name'].str.split().str.len() == 3, 'First_name'] = occup['Name'].str.split().str[1]
        #If they have just two names grab the first
        occup.loc[occup['Name'].str.split().str.len() == 2, 'First_name'] = occup['Name'].str.split().str[0]
        #If they have just one name like in the ealier seasons
        occup.loc[occup['Name'].str.split().str.len() == 1, 'First_name'] = occup['Name'].str.split().str[0]
        #strip the parathesis
        occup.First_name = occup.First_name.str.strip(' "" ')
        #grab the last names
        
        #could use this concept to speed up HT match loop
        occup.loc[occup['Name'].str.split().str.len() == 3, 'Last_name'] = occup.Name.str.split().str[-1]
        occup.loc[occup['Name'].str.split().str.len() == 2, 'Last_name'] = occup.Name.str.split().str[-1]
        occup.loc[occup['Name'].str.split().str.len() == 1, 'Last_name'] = 'X'
        
        occup['Last_name'] = occup['Last_name'].astype(str).str[0]
                
        
        occup["Name"] = occup["First_name"].map(str) + '_' + occup["Last_name"]
                
        #Adds a 0 if the season is less than 9 so we can properly match stuff
        #print(occup.SEASON)
        if occup.SEASON.iloc[0] > 9:
            occup["Name"] = occup["SEASON"].map(str) + '_' + occup["Name"]           
        else:
            occup["Name"] = '0'+ occup["SEASON"].map(str) + '_' + occup["Name"]
        
        #strip any hidden spaces
        occup.Name = occup.Name.str.strip()
        
        occup.Name = occup.Name.str.upper()
        #Rename it to match the elim_data table
        occup.rename(columns={'Name':'CONTESTANT'}, inplace=True)
        wiki_df = pd.concat([wiki_df, occup], sort = True)
    except:
        print('Missed season: ' + season)
        
        missed_season_tracker.append(season)


# In[3]:


season_15 = wiki_df[['CONTESTANT','Hometown', 'Age', 'SEASON']]

#%% Make a table we will eventually use in our trained model
Validation_15 = pd.DataFrame(wiki_df['CONTESTANT'])


# In[4]:


new_england = ['Maine', 'Vermont', 'New Hampshire', 'Massachusetts', 'Rhode Island', 'Connecticut']
#Could put Maryland somewhere else
south = ['Alabama','Florida', 'Georgia', 'Kentucky', 'Louisiana', 'Mississippi',
         'North Carolina', 'South Carolina', 'West Virgina', 
         'Virgina', 'Maryland', 'Tennessee']
midatlatic = ['Pennsylvania', 'New Jersey', 'Delaware', 'New York']
upper_midwest = ['Ohio','Indiana', 'Illinois', 'Michigan','Wisconsin', 'Iowa', 'Minnesota','Nebraska',
                 'North Dakota', 'South Dakota', 'Nebraska']
lower_midwest = ['Kansas', 'Missouri']
northern_mountain = ['Montana', 'Idaho', 'Wyoming']
northwest = ['Washington', 'Oregon']
southwest = ['Arizona', 'New Mexico', 'Texas', 'Oklahoma', 'Arizona']
mountain = ['Colorado','Utah']
west = ['California', 'Nevada', 'Alaska', 'Hawaii']

regions = new_england + south + midatlatic + upper_midwest +lower_midwest + northern_mountain + northwest +southwest+ mountain +west


# In[5]:


season_15['Home State'] = season_15['Hometown'].str.split(",").str[1].str.strip()


# In[6]:


def findregion(ind):
    homestate = ind.get(key = 'Home State')
    if homestate in new_england:
        return 'New England'
    elif homestate in south:
        return 'South'
    elif homestate in midatlatic:
        return 'Midatlatic'    
    elif homestate in upper_midwest:
        return 'Upper midwest'
    elif homestate in lower_midwest:
        return 'Lower Midwest'
    elif homestate in northern_mountain:
        return 'Northern Mountain'    
    elif homestate in northwest:
        return 'Northwest'    
    elif homestate in southwest:
        return 'Southwest'    
    elif homestate in mountain:
        return 'Mountain'    
    elif homestate in west:
        return 'West'   
    #In case the contestant comes from outside the US
    else:
        return homestate
    
season_15['Home State'] = season_15['Home State'].str.strip()
season_15['Culture Region'] = season_15.apply(findregion, axis = 1)


# In[7]:


season_15['Culture Region'].isnull().any().sum()


# In[8]:


bachelorettesHT = pd.read_excel('Bachelorette_Data/Hometown_Bacherlorette.xlsx')

bachelorettesHT['Home State'] = bachelorettesHT['Hometown'].str.split(",").str[1].str.strip()
bachelorettesHT['Culture Region'] = bachelorettesHT.apply(findregion, axis = 1)


# In[9]:


season_15['Match Region'] = 0
season_15['Match City'] = 0

#%%
          
bachelorettesHT.index = bachelorettesHT['Bachelorette']
season_15.index = season_15['CONTESTANT']

for row in bachelorettesHT.index.tolist():
    for contest in season_15.index.tolist():
        if (bachelorettesHT.loc[row,'Season'] == season_15.loc[contest, 'SEASON']) and (bachelorettesHT.loc[row,'Culture Region'] == season_15.loc[contest, 'Culture Region']):
            season_15.loc[contest, 'Match Region'] = 1
            if bachelorettesHT.loc[row,'Hometown'] == season_15.loc[contest, 'Hometown']:
                season_15.loc[contest, 'Match City'] = 1
        


# In[10]:


Validation_15 = pd.merge(Validation_15, season_15[['CONTESTANT','Match Region', 'Match City']], on = 'CONTESTANT')      


# In[11]:


state_leanings = pd.read_csv('Bachelorette_Data/state_leanings.csv', index_col = 0)


# In[12]:


canada_pol = pd.read_csv('Bachelorette_Data/Canada_Wiki.csv', index_col = 0)


# In[14]:


def setCanPolitical(ind):
    lean = ind.get(key = 'Canada Leanings')
    if lean == 'Centre-right':
        return 2
    elif lean == 'Centre-left to left-wing':
        return -7
    elif lean == 'Centre to centre-right':
        return 5    
    elif lean == 'Centre to centre-left':
        return -5
    #non partisan
    else:
        return 0
 
canada_pol['Canada Leanings'] = canada_pol.apply(setCanPolitical, axis = 1)


# In[15]:


def FindPolLean(on_going_table, PVI_table, canada_table):
    '''
        In takes data table you are working with and the polictical table pulled from
        the internet and gives you a number that indicates their PVI - is Liberal + is conservative
        0 is either even or not in the USA
        On going table should have a 'Home State' column and PVI_Table should have a "State" table
        
        Canada table needs to have table labeled "Province/Territory"
    '''
    on_going_table = on_going_table.merge(PVI_table, how = 'left', left_on = 'Home State', right_on = 'State')
    on_going_table = on_going_table.replace('EVEN', 'N+0')
    on_going_table['PVI'] = on_going_table['PVI'].astype(str)
    #season_15['PVI'] = season_15['PVI'].str.strip('+')
    #Need to split the values based on political parties
    on_going_table['PVI'] = on_going_table['PVI'].str.split("+") 
    
    def setPolitical(ind):
        pair = ind.get(key = 'PVI')
        if pair[0] == 'R':
            return int(pair[1])
        elif pair[0] == 'D':
            point = int(pair[1])
            return point*-1
        else:
            return 0
    
    on_going_table['Poltical Spectrum'] = on_going_table.apply(setPolitical, axis = 1)   
    on_going_table = on_going_table.drop(['State', 'PVI'], axis = 1)
    
    on_going_table = on_going_table.merge(canada_table, how = 'left', left_on = 'Home State', right_on = 'Province/Territory')
    
    #add Canada's leanings    
    def addCanLean(ind):
        canada_regions = ['Alberta','British Columbia','Manitoba','New Brunswick',
                          'Newfoundland and Labrador','Nova Scotia','Ontario','Prince Edward Island',
                          'Quebec','Saskatchewan','Northwest Territories','Nunavut','Yukon']
        if ind.get(key = 'Home State') in canada_regions:
            return ind.get(key = 'Canada Leanings')
        else:
            return ind.get(key = 'Poltical Spectrum')
    
    on_going_table['PVI'] = on_going_table.apply(addCanLean, axis = 1)
    on_going_table = on_going_table.drop(['Poltical Spectrum', 'Canada Leanings', 'Province/Territory'], axis = 1)
    
    
    
    return on_going_table


# In[16]:


season_15 = FindPolLean(season_15,state_leanings, canada_pol)


# In[17]:


bachelorettesHT = FindPolLean(bachelorettesHT, state_leanings, canada_pol)


# In[18]:


bachelorette_pol_lean = pd.DataFrame({
    "Season": bachelorettesHT.Season,
    "B_PVI": bachelorettesHT.PVI,
    "B_Age": bachelorettesHT.Age})


# In[19]:


season_15 = season_15.merge(bachelorette_pol_lean, left_on = 'SEASON', right_on = 'Season')
season_15 = season_15.drop('Season', axis = 1)


# In[20]:


season_15['Political Difference'] = season_15.PVI - season_15.B_PVI
season_15['Age Difference'] = season_15.Age.astype(int) - season_15.B_Age


# In[21]:


Validation_15 = pd.merge(Validation_15, season_15[['CONTESTANT','Political Difference', 'Age Difference']], on = 'CONTESTANT', how = 'left')


# In[22]:


round_data = pd.read_excel('Bachelorette_Data/season_15_Elim.xlsx')

round_data = round_data.drop(['Round D1', 'GF_pick'], axis = 1)


# In[23]:


Validation_15 = pd.merge(Validation_15, round_data, on = 'CONTESTANT', how = 'left')


# In[24]:


Validation_15.to_csv('Bachelorette_Data/Validation_15.csv')

print(Validation_15.head())

