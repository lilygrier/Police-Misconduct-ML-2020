import pandas as pd
import os
import sys
sys.path.insert(0, os.path.abspath('../Pipeline'))
import Feat_Engineering as feat_engineer_helpers
from datetime import datetime

# files as global vars
officer_profiles = pd.read_csv("../data/final-profiles.csv.gz", compression="gzip")
TRR_main = pd.read_csv("../data/TRR-main_2004-2016_2016-09.csv.gz", compression="gzip")
TRR_officers = pd.read_csv("../data/TRR-officers_2004-2016_2016-09.csv.gz", compression="gzip")
complaints_accused = pd.read_csv("../data/complaints-accused.csv.gz", compression="gzip")
complaints = pd.read_csv("../data/complaints-complaints.csv.gz", compression="gzip")
complaints_victims = pd.read_csv("../data/complaints-victims.csv.gz", compression="gzip")
salary_ranks = pd.read_csv("../data/salary-ranks_2002-2017_2017-09.csv.gz", compression="gzip")

na_to_zero = ["suspension_length", "settlement", "count_sustained", "trr_total",
                              "trr_firearms_total", "trr_taser_total"]

def make_df(t1, t2):
    '''
    Wrapper function that takes in desired date ranges and makes the dataframe.
    Inputs:
        t1: a tuple with start and end years for t1 period
        t2: a tuple with start and end years for t2 period
        cat_feat: list of categorical features to include
        target col: string defining target column
    returns:
        dataframe ready to send to build model
        bin_names: will need this for model preprocessing in run_model.py
    '''
    complaints_t1, complaints_t2 = get_relevant_complaints(t1, t2)
    relevant_UID = list(complaints_t1["UID"].unique()) + list(complaints_t2["UID"].unique())
    by_officer_df = officer_profiles[officer_profiles["UID"].isin(relevant_UID)]
    by_officer_df = add_settlements_data(by_officer_df, t1)
    by_officer_df = add_trr(by_officer_df, t1)
    # by_officer_df = add_trr(add_settlements_data(by_officer_df, t1), t1)
    # add complaint data:
    bin_names, by_officer_df = add_complaint_data(complaints_t1, by_officer_df)
    # add salary data:
    by_officer_df = add_salary_data(by_officer_df, t1)
    # remove resigned officers:
    by_officer_df["resignation_date"] = pd.to_datetime(by_officer_df["resignation_date"])
    by_officer_df = by_officer_df[(by_officer_df["resignation_date"].dt.year > t2[0]) | (by_officer_df['resignation_date'].isnull())]
    # merge target with final
    target_df = feat_engineer_helpers.prep_y(complaints_t2)
    final_df = by_officer_df.merge(target_df, on="UID", how="left")
    # some final cleaning steps
    final_df = pare_df(final_df, bin_names)
    final_df["cleaned_rank"].fillna(value="Unknown", inplace=True)
    return (bin_names, final_df)

def pare_df(df, bin_names):
    '''
    Fills in NA values resulting from joining and removes unnecessary columns.
    Sasha's Note: made na_to_zero a global variable so that I could add racial_col to it in add_complaint_data
    '''

    target_cols = ["target_use_of_force", "target_drug", "target_racial", "target_sustained",
                   "target_nonviolent", "target_other"]
    na_to_zero.extend(bin_names)
    other_vars_to_include = ["start_date_timestamp", "cleaned_rank", "birth_year",
                             "current_unit", "average_salary", "salary_change", "race", "gender"]
    df[target_cols] = df[target_cols].fillna(value=False)
    df["start_date"] = pd.to_datetime(df["start_date"])
    df["start_date_timestamp"] = df["start_date"].apply(lambda x: datetime.timestamp(x) if 
                                                                  (pd.isnull(x) == False) else None)
    df[na_to_zero] = df[na_to_zero].fillna(value=0)
    df["cleaned_rank"] = df["cleaned_rank"].fillna(value="Unknown")
    vars_to_include = target_cols + na_to_zero + other_vars_to_include
    
    return df[vars_to_include]

def get_relevant_complaints(t1, t2):
    complaints_t1, complaints_t2 = feat_engineer_helpers.relevant_complaints(complaints, complaints_accused, t1, t2)
    return (complaints_t1, complaints_t2)

def add_settlements_data(officer_profiles, t1):
    '''
    Clean settlements data, attach it to officer profiles.
    Returns a dataframe with officer profiles and t1 settlement data.
    '''
    settlements = pd.read_csv("../data/settlements_1952-2016_2017-01.csv.gz", compression="gzip")
    settlements["incident_date"] = pd.to_datetime(settlements["incident_date"])
    settlements["settlement"] = settlements["settlement"].str.replace("$", "")
    settlements["settlement"] = settlements["settlement"].str.replace(",", "")
    settlements["settlement"] = settlements["settlement"].astype(int)
    settlements_relevant = settlements[((settlements["incident_date"].dt.year>=t1[0]) &
                                    (settlements["incident_date"].dt.year<=t1[1]))]
    settlements_t1_by_officer = pd.DataFrame(settlements_relevant.groupby("UID").agg({"settlement":"sum"})).reset_index()
    by_officer_df = officer_profiles.merge(settlements_t1_by_officer[["UID", "settlement"]], how="left", on ="UID")
    by_officer_df["settlement"].fillna(0, inplace=True)
    return by_officer_df

def add_trr(by_officer_df, t1):
    '''
    Cleans and adds TRR data to by-officer dataframe.
    '''

    TRR_main["trr_date"] = pd.to_datetime(TRR_main["trr_date"])
    TRR_main_t1 = TRR_main[(TRR_main["trr_date"].dt.year >= t1[0]) & 
                       (TRR_main["trr_date"].dt.year <= t1[1])]
    
    TRR_main_t1 = pd.merge(TRR_main_t1, TRR_officers[["trr_id", "UID"]], how = "left", 
                       on = "trr_id")
    trr_by_officer = TRR_main_t1.groupby("UID").agg({"trr_id":"count", "firearm_used":"sum", "taser":"sum"})\
                 .rename(columns = {'trr_id':'trr_total','firearm_used':'trr_firearms_total', 'taser':"trr_taser_total"})
    by_officer_df = pd.merge(by_officer_df, trr_by_officer, how = "left", on = "UID" )
    return by_officer_df


def add_complaint_data(complaints_t1, by_officer_df):
    '''
    Cleans and bins complaints and adds it to the df.
    Returns dataframe and list of bin names for later use.
    '''
    feat_engineer_helpers.complaint_bins(complaints_t1)
    complaint_bins_by_UID = complaints_t1.groupby(["UID", "complaints_binned"]).size().unstack().fillna(0)
    bin_names = list(complaint_bins_by_UID.columns)
    by_officer_df = by_officer_df.merge(complaint_bins_by_UID, how = "left", on = "UID")
    # add victim race info
    by_officer_racial_breakdown, racial_cols = feat_engineer_helpers.add_victim_race(by_officer_df,
                                                                        complaints_t1, complaints_victims)
    by_officer_df = by_officer_df.merge(by_officer_racial_breakdown,
                                        how="left", left_on="UID", right_index=True)
    na_to_zero.extend(racial_cols)
    # add discipline info
    feat_engineer_helpers.add_suspension_length(complaints_t1)
    complaints_t1["count_sustained"] = \
    (complaints_t1["final_outcome"].str.contains("no action taken") == False) & \
    (complaints_t1["final_outcome"].str.contains("unknown") == False) & \
    (complaints_t1["final_outcome"].isna() == False)
    disciplines_t1_by_UID = complaints_t1.groupby("UID").agg({"count_sustained":"sum", "suspension_length":"sum"})
    assert ((disciplines_t1_by_UID["suspension_length"]>0) & (disciplines_t1_by_UID["count_sustained"] ==0)).sum() ==0
    by_officer_df = by_officer_df.merge(disciplines_t1_by_UID, how = "left", on="UID")
    return bin_names, by_officer_df

def add_salary_data(by_officer_df, t1):
    salary_ranks_t1_T = feat_engineer_helpers.add_salary_data(salary_ranks, t1)
    by_officer_df = by_officer_df.merge(salary_ranks_t1_T[["average_salary", "salary_change"]], how='left', on='UID')
    return by_officer_df



