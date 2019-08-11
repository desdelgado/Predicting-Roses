# Will you accept this model? Predicting love on the Bachelorette

## Introduction
Recently I started watching The Bachelorette with my girlfriend and found out that like fantasy football, she plays fantasy Bachelorette with her friends.  Part of the scoring system involves deciding who is going to receive a rose and who is going to be eliminated.  Partially to keep myself engaged and partially to help her win, I wondered if machine learning could be used to predict which rounds contestants were going to be eliminated?

In summary, when comparing the model's picks vs. my girlfriend's picks, she was able to have about half a round less error than the model.  Moreover, I ended up not being able to get lower than a two round error when predicting elimination since the data collected didn't have enough correlation to the target variable.  We can still have a vague idea of if a contestant will be eliminated towards the beginning or end of the show, but there is certainly room for improvement which is exciting.  After looking around, I couldn't find a good dataset to use and ended up constructing one from scratch by scraping various webpages and sorting the information into a nice table.  

This project ended up being rather long, so I broke it up into four parts and created a table of contents. If you are interested in the data engineering, read parts 1 and 2 and the season 15 data engineering section.  If you want to learn more about the machine learning, read part 3.  If you want to see how I compared the model to my girlfriend's picks, see part 4.  Finally, feel free to play with the code which is saved as .py files in the "Code" folder.

## Table of Contents

Rose_Data_Engineering_P1 - Initial dataset from 538's website, target variable extraction, and feature engineering

Rose_Data_Engineering_P2 - Does the bachelorette and a contestant have the same political leanings, hometown, and cultural background?

Predicting_Roses - Modeling of custom-built dataset and conclusion.

Model_Vs_Girlfriend - Comparing the best model we could come up with using our custom built dataset to my girlfriend's picks.
