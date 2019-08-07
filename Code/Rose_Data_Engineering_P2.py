#!/usr/bin/env python
# coding: utf-8

# # Will you accept this model? Predicting love on the Bachelorette
# 
# # Part 2 
# 
# # Introduction
# 
# Going forward, I would like to only look at the bachelorette seasons since that is currently airing and being played fantasy wise. As noted, we have exhausted the low hanging fruit from the 538 dataset and need to look elsewhere.
# 
# When watching the most recent season, Hannah, the bachelorette, mentioned that one thing that makes one of her contestants, Jed, so attractive is that they come from the same southern background.  It's also been well documented that a good amount of people fall in people who are of similar backgrounds.  Thus, perhaps adding a categorical feature that tells us if the contestant is from the same cultural roots will help train our model.  To me, this question can be broken down into a few parts: same cultural region, same hometown, political leanings, and age.  I am sure there are other ways, but for now we'll focus on those four.
# 
# Before we go any further let's import the libraries and datasets from [part 1](https://github.com/desdelgado/Predicting_Roses/blob/master/Rose_Data_Engineering_P1.ipynb).

# In[26]:


import pandas as pd
import re
import numpy as np

#Scrape Websites
from bs4 import BeautifulSoup 
import requests
import warnings


# In[27]:


elim_data = pd.read_csv('Bachelorette_Data/elim_data.csv')
elim_data = elim_data.drop(['Unnamed: 0', 'index'], axis = 1)

data_table = pd.read_csv('Bachelorette_Data/data_table.csv')
data_table = data_table.drop('Unnamed: 0', axis = 1)


# ## Scrape and Organize Contestant Information from Wikipedia
# 
# Let's start with figuring out the cultural region.
# 
# To do this we first need to home town of every contestant and bachelorette.  On the contestant side, there’s luckily a [wikipedia](https://en.wikipedia.org/wiki/The_Bachelorette) page that has links to every season.  We can grab all those links and add it into a list for us to scrape.

# In[28]:


#Only grab bachelorette data.  This is our new table that we will be adding all the features to
bachelorette_predict = pd.DataFrame(data_table[data_table['SHOW'] == 'Bachelorette']) 


Bachelorette_seasons = ['https://en.wikipedia.org/wiki/The_Bachelorette_(season_1)', 'https://en.wikipedia.org/wiki/The_Bachelorette_(season_2)',
                    'https://en.wikipedia.org/wiki/The_Bachelorette_(season_3)', 'https://en.wikipedia.org/wiki/The_Bachelorette_(season_4)',
                    'https://en.wikipedia.org/wiki/The_Bachelorette_(season_5)', 'https://en.wikipedia.org/wiki/The_Bachelorette_(season_6)',
                    'https://en.wikipedia.org/wiki/The_Bachelorette_(season_7)', 'https://en.wikipedia.org/wiki/The_Bachelorette_(season_8)',
                    'https://en.wikipedia.org/wiki/The_Bachelorette_(season_9)', 'https://en.wikipedia.org/wiki/The_Bachelorette_(season_10)',
                    'https://en.wikipedia.org/wiki/The_Bachelorette_(season_11)', 'https://en.wikipedia.org/wiki/The_Bachelorette_(season_12)',
                    'https://en.wikipedia.org/wiki/The_Bachelorette_(season_13)']


# Each of these pages has a table with the contestant, age, occupation, hometown among other things.  Scraping this data turned out to be a wild ride because between finishing the project and writing it up, those wiki pages changed and instead of using pd.read_html, I needed to use beautifulsoup, etc. 
# 
# I'll quickly show how we can scrape the page and get the information into the dataframe.

# In[29]:


URL= 'https://en.wikipedia.org/wiki/The_Bachelorette_(season_13)'
response = requests.get(URL)
soup = BeautifulSoup(response.text, 'lxml')

My_table = soup.find("table",{"class" :"wikitable sortable"})

#Make a list to hold all the data we scrape from the HTML table
contest = []

#Search through the table class to find all the rows denoted by 'tr' in the HTML
for record in My_table.findAll('tr'):
        contest.append(record.text)

#Convert list into dataframe
contest_df = pd.DataFrame(contest)

#Turns out the column data is seperated by '\n's so we can use that to split it into a dataframe
contest_df = contest_df.iloc[:,0].str.split('\n', expand = True)

new_header = contest_df.iloc[0] #grab the first row for the header
contest_df.columns = new_header
contest_df = contest_df.iloc[1:]
print(contest_df.head())


# Cool, now the goal is to link the names to the names from the 538 table.  For example "Bryan Abasolo[5][6]" needs to be "13_BRYAN_A" and thus the hometown and age can be added to our overall table.  I choose this convention of "Season_Firstname_LastNameLetter" as it gives us unique names for each contestant and we don't have to worry about duplicate names.  
# 
# Below is a loop that takes in a wiki URL from the list above, finds the relevant information, sorts the data into the naming convention we want, and then saves it into an overall dataframe. For a more intricate explanation of the idea feel free to read the code comments.

# In[30]:


seasons_wiki = Bachelorette_seasons

wiki_df = pd.DataFrame() # Dataframe to keep everything recorded
missed_season_tracker = []

for season in seasons_wiki:
    '''
        This "try: except" method helped me figure out which seasons were giving me problems so I could go 
        back and adjust for those cases. 
    '''
    #Ignore depreciation warnings 
    warnings.filterwarnings("ignore")
    try:
        URL= season
        response = requests.get(URL)
        soup = BeautifulSoup(response.text, 'lxml')
        
        #Need to add additional try statment because the table name in the HTML for seasons 2 and 3 were different
        try:
            My_table = soup.find("table",{"class" :"wikitable sortable"})
        except:
            pass
        
        try:
            My_table = soup.find("table",{"class" :"wikitable"})
        except:
            pass
        
        #Make a list to hold all the data we scrape from the HTML table
        contest = []
        
        #Search through the table class to find all the rows denoted by 'tr' in the HTML
        for record in My_table.findAll('tr'):
                contest.append(record.text)
        
        #Convert list into dataframe
        contest_df = pd.DataFrame(contest)
        
        #Turns out the column data is seperated by '\n's so we can use that to split it into a dataframe
        contest_df = contest_df.iloc[:,0].str.split('\n', expand = True)
        
        new_header = contest_df.iloc[0] #grab the first row for the header
        contest_df.columns = new_header
        contest_df = contest_df.iloc[1:]
        
        #Only grab what we need from the wiki table 
        occup = contest_df[['Name', 'Hometown','Age']]
        
        #Need to get which season we are working with in order to construct the name to merge the tables with
        #Instead of inputting a list use a regrex equations to pull the season number out of the wiki url
        season_number = int(re.findall('\d+', URL )[0])
        
        occup['SEASON'] = season_number
        
        #Get each name and strip any links, periods, etc
        occup['Name'] = occup['Name'].str.replace('\d+', '')
        occup.Name = occup.Name.str.strip('[]')
        occup.Name = occup.Name.str.strip('.')
        
        #Get the age from the age column
        occup.Age = occup.Age.str.extract('(\d+)')
        
        #Since there are a varitiy of different length names, we need to make sure we are always grabbing the
        #right first name when using indexing.  We also have to account for cases where a "nickname" is recorded.
        
        #If they have the nickname grab the middle one
        occup.loc[occup['Name'].str.split().str.len() == 3, 'First_name'] = occup['Name'].str.split().str[1]
        #If they have just two names grab the first
        occup.loc[occup['Name'].str.split().str.len() == 2, 'First_name'] = occup['Name'].str.split().str[0]
        #If they have just one name like in the ealier seasons
        occup.loc[occup['Name'].str.split().str.len() == 1, 'First_name'] = occup['Name'].str.split().str[0]
        
        #strip the parathesis
        occup.First_name = occup.First_name.str.strip(' "" ')
        
        '''
            Same problem as above.  Varity of ways names were recorded.  Our goal here is just to get the first letter
            but in the first few seasons just first names were recorded.  We can infill an 'X' here so it matches our running
            table. 
        '''
        occup.loc[occup['Name'].str.split().str.len() == 3, 'Last_name'] = occup.Name.str.split().str[-1]
        occup.loc[occup['Name'].str.split().str.len() == 2, 'Last_name'] = occup.Name.str.split().str[-1]
        occup.loc[occup['Name'].str.split().str.len() == 1, 'Last_name'] = 'X'
        
        occup['Last_name'] = occup['Last_name'].astype(str).str[0]
                
        #Link everything together
        occup["Name"] = occup["First_name"].map(str) + '_' + occup["Last_name"]
                
        '''
            The first 9 seasons in our main table have a '0#_Name' convention compared to a '#_Name' we get here.
            Need to add that small change into here.
        '''
        if occup.SEASON.iloc[0] > 9:
            occup["Name"] = occup["SEASON"].map(str) + '_' + occup["Name"]           
        else:
            occup["Name"] = '0'+ occup["SEASON"].map(str) + '_' + occup["Name"]
        
        #strip any hidden spaces
        occup.Name = occup.Name.str.strip()
        occup.Name = occup.Name.str.upper()
        
        #Rename it to match the elim_data table
        occup.rename(columns={'Name':'CONTESTANT'}, inplace=True)
        
        #Add it to a running table
        wiki_df = pd.concat([wiki_df, occup], sort = True)
        
        '''    
            If anything above throws an error we can record which season was missed and print it out.  
        '''
    except:
        print('Missed season: ' + season)
        missed_season_tracker.append(season)


# In[31]:


print(wiki_df.head())


# Cool.  Now that we have the Wikipedia data in a standard format, we can link it to our original "elim_data" and form "running_table."  This table will be a staging area of sorts that we can use to further build our goal data set of "Bachelorette_predict."  This way we can mess around with the data and not worry about changing the original elim_data.
# 
# Now let's merge the two tables together on the contestant names. Once we do that let's do a quick check to make sure the tables are the same length, and nothing got lost in the merge.

# In[32]:


elim_data.CONTESTANT = elim_data.CONTESTANT.str.strip()

running_table = pd.merge(elim_data, wiki_df[['CONTESTANT','Hometown', 'Age']], on = 'CONTESTANT')

#Cast the season number and age number into a float so that we can compare 
running_table.SEASON = running_table.SEASON.astype(float)
running_table.Age = running_table.Age.astype(int)

print("running_table length is " + str(len(running_table)))
print("wiki_df length is " + str(len(wiki_df)))


# Dang, it seems like we lost ~30 contestants or 10% of our data.  We can either just carry on and drop the data, but I have a feeling that we can do a bit better.  We can make a list of all the contestants that aren't in the data set and see what's up.

# In[33]:


missing = set(wiki_df.CONTESTANT) ^ set(bachelorette_predict.CONTESTANT)

print(missing)


# Looking closely, it seems like some of them are only 1 or 2 letters off (i.e '03_KEVIN' vs '03_KEVIN_X').  We can write a quick function that will compare two strings and return true if they're only two letters different.

# In[34]:


def isEditDistanceTwo(s1, s2): 
  
    # Find lengths of given strings 
    m = len(s1) 
    n = len(s2) 
    # If difference between lengths is more than 2, 
    # then strings can't be at one distance 
    if abs(m - n) > 2: 
        return False 
    count = 0    # Count of isEditDistanceTwo 
    i = 0
    j = 0
    while i < m and j < n: 
        # If current characters dont match 
        if s1[i] != s2[j]: 
            if count == 2: 
                return False 
            # If length of one string is
            # more, then only possible edit 
            # is to remove a character 
            if m > n: 
                i+=1
            elif m < n: 
                j+=1
            else:    # If lengths of both strings is same 
                i+=1
                j+=1
            # Increment count of edits 
            count+=1
        else:    # if current characters match 
            i+=1
            j+=1
    # if last character is extra in any string 
    if i < m or j < n: 
        count+=1
    return count == 1


# We can then iterate through the two different tables and change the "wiki_df" contestant name to the actual name if it's only two letters different from a contestant entry on the table we're trying to merge on.  For example "03_KEVIN_X" on the wiki_df table would return true when compared to "03_KEVIN" on bachelorette_predict table  The wiki_df entry would then be changed to "03_KEVIN" and we can then easily match the Wikipedia data to that contestant.  For further details on how this is done, feel free to read the code comments.

# In[35]:


#Get the names that are in the wiki_df but not our target dataframe
wiki_missing = list(missing & set(wiki_df.CONTESTANT))

#Get the names that are in our target dataframe but not our wiki_df
predict_missing = list(missing & set(bachelorette_predict.CONTESTANT))

wiki_name = []
predict_name = []    

#Check if the missing names from the wiki_df are only one or two letters away from the target dataframe names
for wiki_element in wiki_missing:
    for predict_element in predict_missing:
        if isEditDistanceTwo(wiki_element,predict_element):
                #record the slightly different names in to two lists. 
                wiki_name.append(wiki_element)
                predict_name.append(predict_element)
                
# Convert to dictionary to avoid another double loop
convert_dict = dict(zip(wiki_name, predict_name))

#Find where the names in wiki_df are only one or two leters off and replace them with the standard name format version.
for counter in range(0, len(wiki_df)):
    contest = wiki_df.iloc[counter,1]
    if contest in convert_dict.keys():
        wiki_df.iloc[counter,1] = convert_dict.get(contest)


# Now let's try merging again and print out the table lengths to check.

# In[36]:


elim_data.CONTESTANT = elim_data.CONTESTANT.str.strip()

running_table = pd.merge(elim_data, wiki_df[['CONTESTANT','Hometown', 'Age']], on = 'CONTESTANT')

#Cast the season number and age number into a float so that we can compare 
running_table.SEASON = running_table.SEASON.astype(float)
running_table.Age = running_table.Age.astype(int)

print("running_table length is " + str(len(running_table)))
print("wiki_df length is " + str(len(wiki_df)))


# Nice, now we're only missing about 17 names which is about 5% of our data. 
# 
# 
# ## Find the Cultural Region of Each Contestant
# 
# Remember the overarching goal was to use this hometown data to see if the contestants are from the same cultural region and even the same hometown as the bachelorette.  This leads into the question of how do we split up the "cultural" regions of the United States?  Digging around I found [this article](https://www.businessinsider.com/regional-differences-united-states-2018-1) where journalist Colin Woodard broke the US down into 11 different cultural regions.  While this map would have been great to use, it was difficult to quickly find a list of the counties in each of these regions.  Instead I found a [britannica](https://www.britannica.com/place/United-States/The-newer-culture-areas) article that gave a nice breakdown.  We can then make some lists of which states below in each region.  I know there are a bunch of ways to slice this and some states are in multiple regions, however, this breakdown is a good first attempt.

# In[37]:


new_england = ['Maine', 'Vermont', 'New Hampshire', 'Massachusetts', 'Rhode Island', 'Connecticut'] 
south = ['Alabama','Florida', 'Georgia', 'Kentucky', 'Louisiana', 'Mississippi'
         'North Carolina', 'Oklahoma', 'Virginia', 'West Virgina', 
         'Virgina', 'Maryland']
midatlatic = ['Pennsylvania', 'New Jersey', 'Delaware', 'New York']
upper_midwest = ['Ohio','Indiana', 'Illinois', 'Michigan','Wisconsin', 'Iowa', 'Minnesota','Nebraska',
                 'North Dakota', 'South Dakota', 'Nebraska']
lower_midwest = ['Kansas', 'Missouri']
northern_mountain = ['Montana', 'Idaho', 'Wyoming']
northwest = ['Washington', 'Oregon']
southwest = ['Arizona', 'New Mexico', 'Texas', 'Oklahoma', 'Arizona']
mountain = ['Colorado','Utah']
west = ['California', 'Nevada', 'Alaska', 'Hawaii']


# We can then write a function that will return the name of the region the hometown of the contestants are from.

# In[38]:


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


# Let's then grab the home state from the hometown column and then run it through that function.  Finally, we'll quickly check if there's any null values denoting we missed a contestant.

# In[39]:


running_table['Home State'] = running_table['Hometown'].str.split(",").str[1].str.strip()
    
running_table['Home State'] = running_table['Home State'].str.strip()
running_table['Culture Region'] = running_table.apply(findregion, axis = 1)

#Check if there is any missing 
print('Number of missing contestants: ' +str(running_table['Culture Region'].isnull().any().sum()))


# Nice.  Now we need to get the bachelorette data to match. I went through the Wikipedia pages and put the hometown, season, and age of each bachelorette in an excel sheet.  I am sure given more time, there's a better computational way to do this but with only 13 entries sometimes you just gotta do it the hard way.

# In[40]:


bachelorettesHT = pd.read_excel('Bachelorette_Data/Hometown_Bacherlorette.xlsx')

bachelorettesHT['Home State'] = bachelorettesHT['Hometown'].str.split(",").str[1].str.strip()
bachelorettesHT['Culture Region'] = bachelorettesHT.apply(findregion, axis = 1)
print(bachelorettesHT.head(10))


# We can now go through and see if the "culture region" and "Hometown" columns match between the contestants and their respective bachelorettes.

# In[41]:


running_table['Match Region'] = 0
running_table['Match City'] = 0

# Create indexs to be able to use .loc rather than .iloc..makes the code more generalizable
bachelorettesHT.index = bachelorettesHT['Bachelorette']
running_table.index = running_table['CONTESTANT']
running_table.drop_duplicates('CONTESTANT', inplace = True)

for row in bachelorettesHT.index.tolist():
    # Get each bachelorette contestant
    for contest in running_table.index.tolist():
        if (bachelorettesHT.loc[row,'Season'] == running_table.loc[contest, 'SEASON']) and (bachelorettesHT.loc[row,'Culture Region'] == running_table.loc[contest, 'Culture Region']):
            # Check if the bachelorette and contestant are from the same cultural region
            running_table.loc[contest, 'Match Region'] = 1
            if bachelorettesHT.loc[row,'Hometown'] == running_table.loc[contest, 'Hometown']:
                # Check if the bachelorette and contestant are from the same city
                running_table.loc[contest, 'Match City'] = 1
            
#Add this data to the prediction table.
bachelorette_predict = pd.merge(bachelorette_predict, running_table[['CONTESTANT','Match Region', 'Match City']], on = 'CONTESTANT')      
  
#Print out to see what we got
print(running_table[['CONTESTANT','Match City', 'Match Region']].head(20))


# We can see that some of the contestant's are even from the same hometown!

# ## Find the Political Leanings of each Contestant
# 
# Next, let's get the political leanings of each contestant.  Like with the previous features there's several ways to do this.  At first glance, we could assign each state conservative or liberal based on who they voted for in 2016.  After digging around, however, I found this concept called the [Cook Partisan Voting Index](https://en.wikipedia.org/wiki/Cook_Partisan_Voting_Index) (PVI) where each state is assigned a political leaning and a number where that number is the difference between the country's average and the state average.  For example, Alabama is Republican +15 which means if the country voted say 40% republican Alabama voted 55% republican.  While we could try to go down to the district level, let's use the state’s first. 
# 
# On a different note, while writing this up the wiki article changed multiple times.  So, for the sake of reproducibility, I will write the scrape code into comments and load the data from a saved csv file.   

# In[42]:


'''
voting_df = pd.read_html('https://en.wikipedia.org/wiki/Cook_Partisan_Voting_Index') 

state_leanings = voting_df[6]
state_leanings.columns = state_leanings.iloc[0]
state_leanings = state_leanings.drop(state_leanings.index[0])
state_leanings = state_leanings.drop(['Party ofGovernor', 'Partyin Senate', 'Housebalance'], axis = 1)

state_leanings.to_csv('state_leanings.csv')
'''
state_leanings = pd.read_csv('Bachelorette_Data/state_leanings.csv', index_col = 0)
print(state_leanings.head())


# Before matching these numbers to the contestants, I did notice some of the contestants (and even a bachelorette!) are from Canada.  So, let's first grab the PVI for the different Canadian territories so we don't miss out on that data. We can go to Canada's Wikipedia page that has a table of the territory and each respective political leanings.  Like the PVI wiki page, this page changed multiple times even while writing this project up.  So again, for the sake of reproducibility, I'll write the scrape code in comments and have saved csv file from that output we can just load in.  

# In[43]:


'''
    canada_pol = pd.read_html('https://en.wikipedia.org/wiki/Provinces_and_territories_of_Canada')

    canada_pol = canada_pol[3]
    canada_pol.columns = canada_pol.iloc[0]
    canada_pol = canada_pol.drop(canada_pol.index[0])

    #%% Columns seemed to be fixed
    #canada_pol = canada_pol.iloc[:-2,[0,4]]
    #%%
    canada_pol.rename(columns={'Majority/Minority':'Canada Leanings'}, inplace=True)
    canada_pol['Canada Leanings'] = canada_pol['Canada Leanings'].str.replace('\d+', '')
    canada_pol['Canada Leanings'] = canada_pol['Canada Leanings'].str.strip('[]')

    #Consolidate and remove last two rows
    canada_pol = canada_pol[['Province/Territory', 'Canada Leanings']].iloc[:-2]

    canada_pol.to_csv('Canada_Wiki.csv')
'''
canada_pol = pd.read_csv('Bachelorette_Data/Canada_Wiki.csv', index_col = 0)
print(canada_pol)


# We can assign PVI numbers to each territory based on its leanings.  For example, we can denote "Centre-right" as 5.  Having some foresight, when we get around to assigning PVI numbers to contestants, let's make democrats negative numbers and republicans positive.  This way when we can take the difference in PVI numbers between contestants and bachelorettes, couples that lean in different ideologies will have a bigger difference.  Moreover, we can make this assumption cause in the 2016 election roughly the same number of people voted conservative and liberal.

# In[44]:


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
print(canada_pol)


# Awesome, now let's write a function that will intake contestant data and match their home state to a PVI number.  I also included matching the Canada PVI numbers we created.  Eventually we will want to look at the bachelor data (when that season rolls around) so writing a reusable function now will save us time down the road.  

# In[45]:


def FindPolLean(on_going_table, PVI_table, canada_table):
    '''
        Intakes data table you are working with and the polictical table pulled from
        the internet and gives you a number that indicates their PVI - is Liberal + is conservative
        0 is either even or not in the USA
        On going table should have a 'Home State' column and PVI_Table should have a "State" table
        
        Canada table needs to have table labeled "Province/Territory"
    '''
    
    #Merge an input table with the contestants with a list of PVI for each state
    on_going_table = on_going_table.merge(PVI_table, how = 'left', left_on = 'Home State', right_on = 'State')
    on_going_table = on_going_table.replace('EVEN', 'N+0')
    on_going_table['PVI'] = on_going_table['PVI'].astype(str)

    #Need to split the values based on political parties
    on_going_table['PVI'] = on_going_table['PVI'].str.split("+") 
    
    #Write function to put the PVI on the +- spectrum we talked about.
    def setPolitical(ind):
        pair = ind.get(key = 'PVI')
        if pair[0] == 'R':
            return int(pair[1])
        elif pair[0] == 'D':
            point = int(pair[1])
            return point*-1
        #If they're from Canada return 0 and we'll add that later
        else:
            return 0
    
    #Above the setPolitical function to give the PVI number on a spectrum for each contestant
    on_going_table['Poltical Spectrum'] = on_going_table.apply(setPolitical, axis = 1)   
    on_going_table = on_going_table.drop(['State', 'PVI'], axis = 1)
    
    on_going_table = on_going_table.merge(canada_table, how = 'left', left_on = 'Home State', right_on = 'Province/Territory')
    
    #add Canada's leanings    
    def addCanLean(ind):
        canada_regions = ['Alberta','British Columbia','Manitoba','New Brunswick',
                          'Newfoundland and Labrador','Nova Scotia','Ontario','Prince Edward Island',
                          'Quebec','Saskatchewan','Northwest Territories','Nunavut','Yukon']
        #Only look at Canadians 
        if ind.get(key = 'Home State') in canada_regions:
            return ind.get(key = 'Canada Leanings')
        else:
            return ind.get(key = 'Poltical Spectrum')
    
    on_going_table['PVI'] = on_going_table.apply(addCanLean, axis = 1)
    on_going_table = on_going_table.drop(['Poltical Spectrum', 'Canada Leanings', 'Province/Territory'], axis = 1)
    
    
    
    return on_going_table


# Now apply this function to our table.

# In[46]:


running_table = FindPolLean(running_table,state_leanings, canada_pol)
print(running_table.head())


# Awesome.  Let's check really quick that we got the right PVI for the Canadian contestants by looking for 'Alberta' which should have a PVI number of 2.

# In[47]:


running_table[running_table['Home State'] == 'Alberta'].iloc[:,-1]


# Let's apply the same function to the bachelorette table.

# In[48]:


bachelorettesHT = FindPolLean(bachelorettesHT, state_leanings, canada_pol)
print(bachelorettesHT.head())


# ## Compare Political Leanings to the Bachelorettes
# 
# Let's now compare the PVI and age of each bachelorette and their respective contestants.

# In[49]:


bachelorette_pol_lean = pd.DataFrame({
    "Season": bachelorettesHT.Season,
    "B_PVI": bachelorettesHT.PVI,
    "B_Age": bachelorettesHT.Age})

#Now need to see how much difference between bachorlette
#Cast the running_table into a float so that we can merge them together
running_table = running_table.merge(bachelorette_pol_lean, left_on = 'SEASON', right_on = 'Season')
running_table = running_table.drop('Season', axis = 1)

#%% Find the difference
running_table['Political Difference'] = running_table.PVI - running_table.B_PVI
running_table['Age Difference'] = running_table.Age - running_table.B_Age

print(running_table.head())


# Now that we have the political difference and age difference, we can merge those columns into our final dataset that we will use to try to predict eliminations.  

# In[50]:


bachelorette_predict = pd.merge(bachelorette_predict, running_table[['CONTESTANT','Political Difference', 'Age Difference']], on = 'CONTESTANT', how = 'left')

#Save the data set to a csv
bachelorette_predict.to_csv('Bachelorette_Data/Bachelorette_Predict.csv')

print(bachelorette_predict.head())


# ## Conclusion
# 
# Awesome, in [part 3](https://github.com/desdelgado/Predicting_Roses/blob/master/Predicting_Roses.ipynb), we'll look at trying to apply machine learning techniques to this constructed data set.  Obviously, there is other data we could add in such as pick order, make the PVI numbers more refined, or look at physical height differences.  I think, however, an important part of any open-ended project like this is determining when to stop.  Thus, let's try to apply some techniques to get started and let that guide us to see if we need to come back and improve.     
