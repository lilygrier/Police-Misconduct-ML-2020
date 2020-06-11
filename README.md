This repository has everything needed to reproduce the model we explain in our report.
The Data and Notebook directories have the CSVs from the Invisible Institute's dropbox 
and the notebooks we used to test and fine-tune the model.

What will be of interest to graders is the pipeline directory. The Make_By_Officer_DF.py and 
Feat_Engineering.py contain the code used to filter for the correct years, collate the different datasets, sum up counts per officer, etc. The Run_Model.py and PreProcessing.py are for actually running the models and performing steps such as normalizing, encoding, running a grid search, and plotting precision-recall curves.

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

To create a model, first one calls make_df from Make_By_Officer_DF with their chosen study period.
To look at observations from 2012-2014 to make predictions about officer behavior in 2015, 
one would call:
additional_cont_feat, final_df = make_df((2012, 2014), (2015, 2015)).
This returns most of the continuous features as a list as well as the engineered features.

Then one calls functions from Run_Model with any chosen feature combination and "severe_complaints" as the target.
try_four_models() does grid search across all parameters for a Logistic Regression, Naive Bayes,
Decision Tree, and Random Forest while single_model() allows the user to specify which type of model they would like to run. 
To use all the features we included in our model, you have to perform a bit of manual feature manipulation between calling make_df and running models, as shown below.

The functions are used as follows:

To run all four model types:

additional_cont_feat, final_df = Make_By_Officer_DF.make_df((2012, 2014), (2015, 2015))<br/>
cont_feat = ["birth_year", "start_date_timestamp", "male"]<br/>
cont_feat.extend(additional_cont_feat)<br/>
cat_feat = ["race"]<br/>
target_col = "severe_complaint"<br/>
many_models = Run_Model.try_four_models(final_df, target_col, cont_feat, cat_feat, "balanced_accuracy")<br/>
many_models is a list containing the best estimator of each model type

To run a single model, you would do:

additional_cont_feat, final_df = Make_By_Officer_DF.make_df((2012, 2014), (2015, 2015))<br/>
cont_feat = ["birth_year", "start_date_timestamp", "male"]<br/>
cont_feat.extend(additional_cont_feat)<br/>
cat_feat = ["race"]<br/>
rf, rf_feature_importances = Run_Model.single_model(final_df, "RandomForest", 
                                                      target_col, cont_feat, cat_feat, "balanced_accuracy")<br/>
log is the best log model object and log_feature_importances is a dataframe of feature importances associated with the model<br/>
To try a model type other than LogisticRegression, you would simply enter the name of the model
type into the function.