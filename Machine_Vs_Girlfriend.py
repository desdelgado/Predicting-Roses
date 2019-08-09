#!/usr/bin/env python
# coding: utf-8

# #  Will you accept this model? Predicting love on the Bachelorette
# 
# 
# # Part 4
# 
# ## Introduction
# 
# Finally, as a last step let's compare our model's performance vs. my girlfriend's picks using the most recent season of the Bachelorette that started this whole project.  We'll again use the k-nearest neighbors model that we decided to use in [part 3](https://github.com/desdelgado/Predicting-Roses/blob/master/Predicting_Roses.ipynb) with the hyperparameters we found using a random and then grid search.  Let's dive in.

# In[1]:


# Load relevant libraries
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error as MSE


# ## Load in Data
# 
# Let's load in the contestant data from season named "validation_15".  I added the data engineering workbook [here](https://github.com/desdelgado/Predicting-Roses/blob/master/Rose_Data_Engineering_Season_15.ipynb) from season 15 which is very similar to the first two data engineering parts.  We'll also load in the training data we used in [part 3](https://github.com/desdelgado/Predicting-Roses/blob/master/Predicting_Roses.ipynb) to train our model.  

# In[2]:


validation_15 = pd.read_csv('Bachelorette_Data/Validation_15.csv').iloc[:, 1:]

training_data = pd.read_csv('Bachelorette_Data/Training_Data.csv', header = 0 ).iloc[:, 2:]
print(validation_15.head())


# Let's then create our target and features from each dataset.

# In[3]:


target_train = training_data['Round_Eliminated']

features_train = training_data.drop(['Round_Eliminated'], axis = 1)


# In[4]:


target_15 = validation_15['Round_Eliminated']

features_15 = validation_15[['First_Impression_Rose','Percentage Left after D1',
                             'Match Region','Match City','Political Difference', 'Age Difference']]


# ## Fit and Predict With Model
# 
# Now we can load in the best parameters and use them to instantiate the model.

# In[5]:


best_est = {'weights': 'uniform', 'p': 2, 'n_neighbors': 10, 'n_jobs': -1, 'leaf_size': 60, 'algorithm': 'kd_tree'}

KNN = KNeighborsRegressor(**best_est)


# We can now fit to our training data and predict on our season 15 data.

# In[6]:


KNN.fit(features_train, target_train)

predicted = KNN.predict(features_15)
# Round the numbers since were using discrete numbers
predicted = predicted.round()


# ## Compare Against Girlfriend's Picks
# 
# Let's load in my girlfriend's picks from the data table created from last season.

# In[7]:


GF_picks = pd.read_excel('Bachelorette_Data/season_15_Elim.xlsx')['GF_pick']


# Finally, we can look at the RMSE score of my girlfriends pick's and compare them to the model. 

# In[8]:


computer_RMSE = round((MSE(target_15,predicted))**0.5, 2)

GF_RMSE = round((MSE(target_15,GF_picks))**0.5,2)

print('My model got a RMSE score of ' + str(computer_RMSE))
print('My girlfriend got a RMSE score of ' + str(GF_RMSE))


# ## Conclusion 
# 
# Dang, it seems that my girlfriend can predict almost half a round better than us.  I feel that again this is due to some of features such as "Age Difference" or "Match City" in our model having low correlation and low mutual information. Either these features do not particularly matter to a bachelorette or possibly the setting being a mansion in LA or New York removes some of the comfort/advantage of a similar hometown/culture from the equation.  That being said, being able to predict with a RMSE score of ~2 rounds does allow us to at least get a sense of if the contestant will be eliminated in the first few rounds or more towards the end of the show.  For example, if one contestant had a predicted round of 2 and an RMSE of 2, then we could guess they have a higher chance of being eliminated before someone with a predicted round of 8 and RMSE of 2. 
# 
# Going forward, I would like to add other features to our table to try to improve the model. One idea that comes to mind is scraping twitter data about #bachernation's feelings about each contestant and do some sort of sentiment analysis.  The idea being that audience in the aggregate will have a good idea of who will make it far and will reflect those ideas via tweets.  Furthermore, I would like to explore trying to predict overall points in bachelorette fantasy.  Beyond contestants being eliminated, there are other ways for contestants to earn or loose points such as kisses, popping champagne, or crying.  This would lend me towards a more round by round approach using Bayesian statistics. The idea being that one would start with a distribution of possible points and with each episode that distribution would be updated.  Furthermore, this approach would mirror how audiences think about the game as their notions on each contestant are similarly updated round by round.  Though the modeling here didn't go as we planned, I still enjoyed constructing a dataset from scratch as well as learned a lot about the bachelorette. 
# 
# As always, this is a learning experience, so I welcome questions, comments, and suggestions for improvements. Email me at davidesmoleydelgado@gmail.com or @davidesdelgado on twitter.
# 
