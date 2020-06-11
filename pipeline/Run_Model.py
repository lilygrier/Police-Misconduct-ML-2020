'''
Takes in a cleaned dataframe, list of variabless to include.
Builds, runs, and evaluates models.
'''
import PreProcessing as PreProcessing
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from graphviz import Source
import sklearn

models_dict = {
    'LogisticRegression': LogisticRegression(max_iter = 1000),
    'GaussianNB': GaussianNB(),
    'DecisionTree': DecisionTreeClassifier(random_state=3),
    "RandomForest": RandomForestClassifier(class_weight="balanced", random_state =4)
}

big_params_dict = {
    'LogisticRegression': [{'penalty': ("l2", "none"), 'C': [.01,.1,1,10,100], 'random_state': (3,)}],
    'GaussianNB': [{'priors': (None,)}],
    "DecisionTree": [{"criterion":("gini", "entropy"),
                     "max_depth":[3, 7, 9, 13],
                     "min_samples_split":[2,5,10]}],
    "RandomForest": [{"criterion":("gini", "entropy"),
                     "max_depth":[3, 5, 7, 9],
                     "min_samples_split":[2, 5, 7],
                     "n_estimators": [100, 1000, 5000]}]
}
big_how_score = ["recall", "precision", "balanced_accuracy"]
log_how_score = ["recall", "precision", "balanced_accuracy"]

def build_model(train_X, train_Y, refit, model_type):
    '''
    Builds and fits a gridsearchCV object
    Inputs:
        train_X: dataframe with training features
        train_Y: dataframe with training target
        refit (str or False): how model shoud be refit
        model_type (str): type of model to run
    Returns:
        grid: a GridSearchCV object
    '''
    estimator = models_dict[model_type]
    params = big_params_dict[model_type]
    grid = GridSearchCV(estimator=estimator,
                        param_grid=params,
                        scoring=log_how_score,
                        cv=5,
                        refit=refit)
    grid.fit(train_X, train_Y)

    return grid

def single_model(df, model_type, target_col, cont_feat, cat_feat, refit):
    '''
    Runs a grid search of a single type of model.
    Inputs:
        df: a Pandas dataframe
        model_type (str): the type of model to be run
        target_col (str): the name of the target column
        cont_feat (list): list of continuous features
        cat_feat (list): list of categorical features
        refit (str or False): how the best model should be refit
            For decision tree refit can be one
    Returns:
        best_model: model object of the best model
        dataframe of feature importances
    '''
    train, test = PreProcessing.tt_split(df[[target_col] + cont_feat+cat_feat], 30)
    normalize_cont = True
    if model_type == "RandomForest" or model_type == "DecisionTree":
        normalize_cont = False
    train_X, train_Y, test_X, test_Y, labels = pre_processing(target_col, train, 
                                                test, cont_feat, cat_feat, normalize_cont)
    grid = build_model(train_X, train_Y, refit, model_type)
    best_model = eval_model(grid, test_X, test_Y, model_type)
    fixed_val_threshold(best_model, test_X, test_Y)
    feature_headers = list(labels)
    feature_headers.remove(target_col)

    return best_model, pd.DataFrame(index=feature_headers, 
            data=best_model.feature_importances_).sort_values(by=0, ascending=False)

def eval_single_model(best_model, test_X, test_Y):
    '''
    Evaluates a single model. Prints metrics and plots precision recall curve
    Inputs:
        best_model: model to be evaluated
        test_X: dataframe of test features
        test_Y: dataframe of test target
    Returns:
        best_model: model object of the best model
    '''
    pred_Y = best_model.predict(test_X)
    print(sklearn.metrics.classification_report(test_Y, pred_Y, output_dict=True)["True"])
    print("balanced_accuracy:", sklearn.metrics.balanced_accuracy_score(test_Y, pred_Y))
    pred_Y_probs = best_model.predict_proba(test_X)[:,(best_model.classes_ ==True)]
    sklearn.metrics.precision_recall_curve(test_Y, pred_Y_probs)
    print(sklearn.metrics.plot_precision_recall_curve(best_model, test_X, test_Y))

    return best_model

def try_four_models(df, target_col, cont_feat, cat_feat, refit):
    '''
    Processes data and runs grid search of logistic regression, naive bayes, 
    decision tree, and random forest models.
    Inputs:
        df: a Pandas dataframe
        target_col (str): the name of the target column
        cont_feat (list): list of continuous features
        cat_feat (list): list of categorical features
        refit (str or False): specifies if/how the model should be refit
    Returns:
        the best estimator from the grid search
    '''
    train, test = PreProcessing.tt_split(df[[target_col] + cont_feat+cat_feat], 30)
    train_X, train_Y, test_X, test_Y, labels = pre_processing(target_col, train, test, cont_feat, cat_feat)

    return big_grid_search(train_X, train_Y, test_X, test_Y, refit=refit)

def big_grid_search(train_X, train_Y, test_X, test_Y, refit="recall"):
    '''
    Performs grid scearch of specified models
    Inputs:
        train_X: dataframe of training features
        train_Y: dataframe of training target
        test_X: dataframe of test features
        test_Y: dataframe of test target
        refit (str or False): specifies if/how the model should be refit
    Returns:
         list of the best-performing model objects of each type
    '''
    out_list = []
    for k in models_dict.keys():
        print()
        print("got to ", k)
        estimator = models_dict[k]
        params = big_params_dict[k]
        print("parameter dictionary is", params)
        print()
        grid = GridSearchCV(estimator=estimator,
                            param_grid=params,
                            scoring=log_how_score,
                            cv=5,
                            refit=refit)
        grid.fit(train_X, train_Y)
        eval_model(grid, test_X, test_Y, k)
        out_list.append(grid.best_estimator_)

    return out_list

def eval_model(grid_search, test_X, test_Y, model_type):
    '''
    Finds and evaluates best estimator from a grid search.
    Inputs:
        grid_search: a GridSearchCV object
        test_X: dataframe with test features
        test_Y: dataframe with test target
        model_type (str): type of model to evaluate
    Returns:
        best_model: model object of best model
    '''
    best_model = grid_search.best_estimator_
    pred_Y = best_model.predict(test_X)
    print("best ", model_type, " metrics:")
    print(sklearn.metrics.classification_report(test_Y, pred_Y, output_dict=True)["True"])
    print("balanced_accuracy:", sklearn.metrics.balanced_accuracy_score(test_Y, pred_Y))
    pred_Y_probs = best_model.predict_proba(test_X)[:,(best_model.classes_ ==True)]
    sklearn.metrics.precision_recall_curve(test_Y, pred_Y_probs)
    print(sklearn.metrics.plot_precision_recall_curve(best_model, test_X, test_Y))

    return best_model

def visualize_tree(tree, labels):
    '''
    Creates a visualization of a decision tree.
    Inputs:
        tree: a decision tree object
        labels: a list of decision tree features
    Returns:
        a visualization of the decision tree
    '''
    tree_string_1 = sklearn.tree.export_graphviz(tree, feature_names=labels)
    tree_vis_1 = Source(tree_string_1)

    return tree_vis_1

def fixed_val_threshold(best_model, test_X, test_Y, custom_cutoff = None):
    '''
    Finds and prints evaluation metrics for classifying at different thresholds
    Inputs:
        best_model: model to evaluate
        test_X: dataframe with test features
        test_Y: dataframe with test target
        custom_cutoff: list of cutoffs values to use
    Returns:
        None
    '''
    fixed_val_threshold_metrics = pd.DataFrame(columns = ["Cutoff", "precision", 
                                                "recall", "f1-score","support"])
    pred_Y_prob_True = best_model.predict_proba(test_X)[:,1]
    if not custom_cutoff:
        cutoffs = [0, .00001, .01, .05, .1, .2, .3, .4, .5, .6, .7, .8, .9, .95]
    else:
        cutoffs = custom_cutoff
    for c in cutoffs:
        p_y = (pred_Y_prob_True > c)
        d = sklearn.metrics.classification_report(test_Y, p_y, output_dict = True)["True"]
        d["balanced_accuracy"] = sklearn.metrics.balanced_accuracy_score(test_Y, p_y)
        d["Cutoff"] = c
        fixed_val_threshold_metrics = fixed_val_threshold_metrics.append(d, ignore_index = True)
    fixed_val_threshold_metrics = fixed_val_threshold_metrics[["Cutoff", "precision", "recall", "balanced_accuracy"]]
    fixed_val_threshold_metrics = fixed_val_threshold_metrics.set_index("Cutoff")
    display("fixed val threshold metrics: ", fixed_val_threshold_metrics)

    return None

def pre_processing(target_col, train, test, cont_feat, cat_feat, normalize_cont = True):
    '''
    Imputes missing values, normalizes continuous values, 
    one-hot encodes categorical values, splits up features and targets.
    Inputs:
        target_col (str): the name of the target column
        train: dataframe with training data
        test: dataframe with testing data
        cont_feat (list): list of continuous features to include
        cat_feat (list): list of categorical features to include
        normalize_cont (bool): whether or not continuous features
            should be normalized
    Returns:
        train_X: dataframe of training features
        train_Y: dataframe of training target
        test_X: dataframe of test features
        test_Y: dataframe of test target
        labels: list of all included columns for later use
    '''
    # impute na to median
    PreProcessing.na_to_median(train, test, cont_feat)
    # normalize
    if normalize_cont:
        my_scaler, cont_feat_final = PreProcessing.normalize(train, cont_feat)
        PreProcessing.normalize(test, cont_feat, my_scaler)
    else:
        cont_feat_final = cont_feat
    # one-hot encode
    OHE, train = PreProcessing.one_hot(train, cat_feat)
    _, test = PreProcessing.one_hot(test, cat_feat, OHE)
    cat_feat_final = list(OHE.get_feature_names())
    train_limited, labels = PreProcessing.limit_for_fit(train, target_col, cont_feat_final, cat_feat_final)
    test_limited, labels = PreProcessing.limit_for_fit(test, target_col, cont_feat_final, cat_feat_final)
    train_X, train_Y = PreProcessing.feat_target_split(train_limited, target_col)
    test_X, test_Y = PreProcessing.feat_target_split(test_limited, target_col)

    return train_X, train_Y, test_X, test_Y, labels
