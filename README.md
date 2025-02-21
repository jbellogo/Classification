# Classification


In this kaggle contest, we will use a dataset to predict answers to the question "On a scale of 1 to 10, how rich or poor do you think you are right now? 1 means very poor, and 10 means very rich.". There are 10 levels responses for this question, which are from 1 to 10.

Each observation represents a survey response of one person. The combination of the first three columns of each dataset, "psu", "hh", and "idcode" will indicate a unique person, where "psu" stands for primary sampling unit and "hh" stands for household. Your goal is to predict the correct response for this question based on the training dataset.

A more detailed description for each column could be found in the given file "codebook.xlsx".


##  Data
Files
module_Education_train_set.csv - the training set for education information of each respondents
module_HouseholdInfo_train_set.csv - the training set for basic personal information of each respondents
module_Education_test_set.csv - the test set for education information of each respondents
module_HouseholdInfo_test_set.csv - the test set for basic personal information of each respondents
module_SubjectivePoverty_train_set.csv - the training set of target (first three columns are pre-combined as "psu_hh_idcode")
sample_submission.csv - a sample submission file in the correct format (first three columns are pre-combined as "psu_hh_idcode")
codebook.xlsx - detailed descriptions of each column