This repository has everything needed to reproduce the model we explain in our report.
The Data and Notebook directories have the CSVs from the Invisible Institute's dropbox 
and the notebooks we used to test and fine-tune the model.

What will be of interest to graders is the pipeline directory. The Make_By_Officer_DF.py and 
Feat_Engineering.py contain the code used to filter for the correct years, collate the different datasets, sum up counts per officer, etc. The Run_Model.py and PreProcessing.py are for actually running the models and perfoming steps such as normalizing, encoding, running a grid search, and plotting precision-recall curves.

We hope that this code is clean and well-organized enough that it would fairly 
easy for us or someone else to use it to continue building features, testing new 
time periods, and trying different models.

The dataframe that is passed to the Run_Model code is produced by a function that takes in
two arguments: t1 and t2, which are tuples where the values are the first and last years in a time period.
t1 is the observation period and t2 is the test period.
To set the time period to only one year, simply put the same year for both (e.g., t2 = (2016, 2016)).
So it's quite easy to try different study periods, which we did at the beginning 
to find a time period that presented enough data to work with.

Changing the target column and the way features are engineered should be fairly easy as different groups of features (count of complaints, demographics of victims, TRR data etc.) are engineering in different functions that can be changed without affecting the overall function of the pipeline.