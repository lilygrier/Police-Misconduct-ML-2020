"""
This .py file has code to:
Split into train/test
Impute NA with median
Normalize
One hot encode
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split


def tt_split(df, rs):
    '''
    Returns train/test split based on typical 80-20 split.
    Inputs:
        df: a Pandas dataframe
        rs (int): number for random state
    Returns:
        one dataframe with training data, one with testing data
    '''

    return train_test_split(df, test_size=.2, random_state=rs)


def na_to_median(train, test, cont_feat):
    '''
    Takes a dataframe with one or more continuous features specified and
    replaces na with median value for those features.
    Most recent change is to use median of train for both train and test
    Inputs:
        train: dataframe of training data
        test: dataframe of testing data
        cont_feat (list): list of continuous features
    Returns:
        nothing, makes changes to df in place
    '''
    for f in cont_feat:
        train_median = train[f].median()
        train[f].fillna(train_median, inplace=True)
        test[f].fillna(train_median, inplace=True)


def normalize(df, feat_to_norm, my_scaler=None):
    '''
    Takes a dataframe with one or more continuous features specified and
    adds column that is that feature normalized.
    Inputs:
        df: a Pandas dataframe
        feat_to_norm (list): list of names of continuous features to be normalized
        my_scaler: a scaler object
            if my_scaler is none then fit and return a new StandardScaler object
    Returns:
        list of scaler objects to normalize train data, 
        list of labels for normalized columns
    '''

    if not my_scaler:
        my_scaler = StandardScaler()
        my_scaler.fit(df[feat_to_norm])
    feat_norm = my_scaler.transform(df[feat_to_norm])
    norm_col = []
    for i in range(len(feat_to_norm)):
        norm =  feat_to_norm[i] + "_norm"
        df.loc[:, norm] = feat_norm[:,i].copy()
        norm_col.append(norm)
    return  my_scaler, norm_col

def one_hot(df, cat_feat, OH_encoder = None):
    '''
    One-hot encodes categorical features
    Inputs:
        df: a Pandas dataframe
        cat_feat: list of the names of categorical features
        OH_encoder: a OneHotEncoder() object,
            if OH_encoder is None, creates an object
    Returns:
        OH_encoder: a OH_encoder object
        a dataframe with one-hot encoded categories
    '''
    df.loc[:,cat_feat] = df[cat_feat].fillna("None").astype("str").copy()
    if not OH_encoder:
        OH_encoder = OneHotEncoder(handle_unknown = "ignore")
        OH_encoder.fit(df[cat_feat].values)
    oh_encoded = OH_encoder.transform(df[cat_feat].values).toarray()
    oh_encoded = pd.DataFrame(oh_encoded, columns=OH_encoder.get_feature_names())
    return OH_encoder, pd.concat([df.reset_index(drop = True),
                         oh_encoded.reset_index(drop=True)], axis= 1)

def limit_for_fit(df, target_col, cont_feat = [], OHE_feat = []):
    '''
    Take target attribute and processed features to be passed to .fit
    Inputs:
        df: a Pandas dataframe
        target_col: the name of the target column
        cont_feat: list of names of continuous features
        OHE_feat: list of names of one-hot encoded features
    Returns:
        a dataframe with features relevant for model
    '''
    final_col = [target_col]+ cont_feat +OHE_feat

    return df[final_col], final_col

def feat_target_split(df, target_col):
    '''
    Divides data into features and targets.
    Inputs:
        df: a dataframe
        target_col: the name of the target column
    Returns:
        one dataframe with features, one dataframe with target
    '''
    Y = df[target_col].values.ravel()
    X_col = list(df.columns)
    X_col.remove(target_col)
    X = df[X_col].values
    if len(X_col) == 1:
        X = X.reshape(-1,1)
    return X, Y
    