'''
This code creates a by-officer dataframe ready for use in models.
'''
import pandas as pd
import os
import sys
sys.path.insert(0, os.path.abspath('../Pipeline'))
import Feat_Engineering as feat_engineer_helpers
from datetime import datetime

def make_df(t1, t2):
    '''
    Wrapper function that takes in desired date ranges and makes the dataframe.
    Inputs:
        t1: a tuple with start and end years for t1 period
        t2: a tuple with start and end years for t2 period
    returns:
        dataframe ready to send to build model
        bin_names: will need this for model preprocessing in run_model.py
    '''
    officer_profiles = pd.read_csv("../data/final-profiles.csv.gz", compression="gzip")

    na_to_zero = []
    cont_feat_col = []
    complaints_t1, complaints_t2 = get_relevant_complaints(t1, t2)
    relevant_UID = list(complaints_t1["UID"].unique()) + list(complaints_t2["UID"].unique())
    by_officer_df = officer_profiles[officer_profiles["UID"].isin(relevant_UID)]
    by_officer_df["male"] = (by_officer_df["gender"] =="MALE")
    settlement_col, by_officer_df = add_settlements_data(by_officer_df, t1)
    na_to_zero.extend(settlement_col)
    trr_bins, by_officer_df = add_trr(by_officer_df, t1)
    na_to_zero.extend(trr_bins)
    complaint_cols, by_officer_df = add_complaint_data(complaints_t1, by_officer_df)
    na_to_zero.extend(complaint_cols)
    victim_demo_cols, by_officer_df = add_victim_demo(complaints_t1, by_officer_df)
    cont_feat_col.extend(victim_demo_cols)
    officer_filed_complaints_col, by_officer_df = get_officer_filed_complaints(complaints_t1, by_officer_df)
    na_to_zero.extend(officer_filed_complaints_col)
    salary_col, by_officer_df = add_salary_data(by_officer_df, t1)
    cont_feat_col.extend(salary_col)
    cont_feat_col.extend(na_to_zero)
    # remove resigned officers:
    by_officer_df["resignation_date"] = pd.to_datetime(by_officer_df["resignation_date"])
    by_officer_df = by_officer_df[(by_officer_df["resignation_date"].dt.year > t2[0]) | (by_officer_df['resignation_date'].isnull())]
    # merge target with final
    target_df = feat_engineer_helpers.prep_y(complaints_t2)
    final_df = by_officer_df.merge(target_df, on="UID", how="left")
    # some final cleaning steps
    final_df = pare_df(final_df, na_to_zero, cont_feat_col)
    final_df["cleaned_rank"].fillna(value="Unknown", inplace=True)
    
    return cont_feat_col, final_df

def pare_df(df, na_to_zero, cont_feat_col):
    '''
    Fills in NA values resulting from joining and removes unnecessary columns.
    Inputs:
        df: a Pandas dataframe
        na_to_zero: list of cols for which NA should be imputed to zero
        cont_feat_col: list of continuous feature columns
    Returns:
        a Pandas dataframe pared down
    '''
    target_col = "severe_complaint"

    other_vars_to_include = ["UID","start_date_timestamp", "cleaned_rank", "birth_year",
                             "current_unit", "race", "male"]
    df[target_col] = df[target_col].fillna(value=False)
    df["start_date"] = pd.to_datetime(df["start_date"])
    df["start_date_timestamp"] = df["start_date"].apply(lambda x: datetime.timestamp(x) if 
                                                                  (pd.isnull(x) == False) else None)
    df[na_to_zero] = df[na_to_zero].fillna(value=0)
    df["cleaned_rank"] = df["cleaned_rank"].fillna(value="Unknown")
    vars_to_include = [target_col] + cont_feat_col + other_vars_to_include
    
    return df[vars_to_include]

def get_relevant_complaints(t1, t2):
    '''
    Generates dataframes of relevant complaints from files.
    Inputs:
        t1 (tuple): start and end years of observation period
        t2 (tuple): start and end years of target period
    Returns:
        complaints_t1: dataframe of complaints for observation period
        complaints_t2: dataframe of complaints for target period
    '''
    complaints_accused = pd.read_csv("../data/complaints-accused.csv.gz", compression="gzip")
    complaints = pd.read_csv("../data/complaints-complaints.csv.gz", compression="gzip")
    complaints_t1, complaints_t2 = feat_engineer_helpers.relevant_complaints(complaints, complaints_accused, t1, t2)

    return (complaints_t1, complaints_t2)

def get_officer_filed_complaints(complaints_t1, final_df):
    '''
    Gets per-officer count of complaints filed by other officers.
    Inputs:
        complaints_t1: dataframe of complaints from t1
        final_df: dataframe with other complaint data
    Returns:
        name of column in list form for later use
        merged_final: dataframe with officer_filed_complaints column added
    '''
    officer_filed_complaints = pd.read_csv("../data/officer-filed-complaints__2017-09.csv.gz", compression="gzip")
    merged_df = complaints_t1[["cr_id","UID"]].merge(officer_filed_complaints, on="cr_id", how="right")
    merged_fc_df = merged_df.groupby('UID').count().reset_index(). sort_values(['cr_id'], ascending = False)
    merged_fc_df.rename(columns={"cr_id":"officer_filed_complaints"}, inplace=True)
    merged_final = merged_fc_df.merge(final_df, on = "UID", how = "right")

    return ["officer_filed_complaints"], merged_final

def add_settlements_data(officer_profiles, t1):
    '''
    Clean settlements data, attach it to officer profiles.
    Inputs:
        officer_profiles: dataframe with officer information
        t1 (tuple): start and end years of observation period
    Returns:
        the name of the new column in list form for later use
        by_officer_df: a dataframe with officer profiles and t1 settlement data.

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

    return ["settlement"],by_officer_df

def add_trr(by_officer_df, t1):
    '''
    Cleans and adds TRR data to by-officer dataframe.
    Inputs:
        by_officer_df: dataframe of officer-level data
        t1 (tuple): start and end years of observation period
    Returns:
        trr_bins: list of new TRR-related columns
        by_officer_df: dataframe of officer-level data updated to
            include TRR data
    '''
    TRR_main = pd.read_csv("../data/TRR-main_2004-2016_2016-09.csv.gz", 
                            compression="gzip")
    TRR_officers = pd.read_csv("../data/TRR-officers_2004-2016_2016-09.csv.gz", 
                                compression="gzip")
    TRR_action_responses = pd.read_csv("../data/TRR-actions-responses_2004-2016_2016-09.csv.gz", 
                                        compression="gzip")

    TRR_main["trr_date"] = pd.to_datetime(TRR_main["trr_date"])
    TRR_main_t1 = TRR_main[(TRR_main["trr_date"].dt.year >= t1[0]) & 
                       (TRR_main["trr_date"].dt.year <= t1[1])]
    
    TRR_main_t1 = pd.merge(TRR_main_t1, TRR_officers[["trr_id", "UID"]], how = "left", 
                       on = "trr_id")
    trr_by_officer = TRR_main_t1.groupby("UID").agg({"trr_id":"count"})\
                     .rename(columns = {'trr_id':'trr_total'})
    by_officer_df = pd.merge(by_officer_df, trr_by_officer, how="left", on="UID")
    trr_bins = ["trr_total"]

    TRR_action_t1 = TRR_main_t1[["UID","trr_id", "trr_date"]].merge(TRR_action_responses, how = "left", on = "trr_id")
    TRR_action_t1_member = TRR_action_t1[TRR_action_t1["person"] == "Member Action"]
    TRR_action_t1_member["resistance_bin"] = TRR_action_t1_member["resistance_type"].map(lambda x: "Passive" if
                                                                                                   "Passive" in x
                                                                                                    else "Active")
    TRR_action_t1_member["force_bin"] = TRR_action_t1_member["force_type"]
    TRR_action_t1_member["force_bin"] = TRR_action_t1_member["force_bin"].map(lambda x: "Physical Force" if
                                                                                              "Force" in x
                                                                                              or "Impact" in x
                                                                                              else x)

    TRR_action_t1_member["force_bin"] = TRR_action_t1_member["force_bin"].map(lambda x: "Non-Lethal Weapon" if
                                                                                         ("Taser" in x
                                                                                          and "Display" not in x)
                                                                                          or "Chemical" in x
                                                                                          else x)
    TRR_action_t1_member["force_bin"] = TRR_action_t1_member["force_bin"].map(lambda x: "Firearm" if
                                                                                                "Firearm" in x
                                                                                                else x)

    force_bins = ["Physical Force", "Non-Lethal Weapon", "Firearm"]
    TRR_action_t1_member["force_bin"][TRR_action_t1_member["force_bin"].isin(force_bins) == False] = "Other"


    TRR_action_t1_member["force_type"].replace("Chemical (Authorized)", 
                                                "Chemical", inplace=True)
    TRR_action_t1_member["force_type"].replace(["Verbal Commands", "Member Presence"], 
                                                "Other", inplace=True)
    TRR_action_t1_member["force_resistance_feat"] = TRR_action_t1_member["resistance_bin"] + " - " + \
                                                    TRR_action_t1_member["force_bin"]
    TRR_actions_by_officer = TRR_action_t1_member.groupby(["UID", "force_resistance_feat"]).size().unstack().fillna(0)
    TRR_action_cols = TRR_actions_by_officer.columns
    TRR_actions_by_officer.reset_index(inplace=True)
    by_officer_df = by_officer_df.merge(TRR_actions_by_officer, how="left", on="UID")
    trr_bins.extend(list(TRR_action_cols.values))

    return trr_bins, by_officer_df

def add_complaint_data(complaints_t1, by_officer_df):
    '''
    Cleans and bins complaints and adds them to the dataframe.
    Inputs:
        complaints_t1: Pandas dataframe of complaints from observation period
        by_officer_df: Pandas dataframe of officer-level data
    Returns:
        new_cols: list of bin names for later use
        by_officer_df: officer-level dataframe updated to 
            reflect each officer's complaint history
    '''
    new_cols = []
    feat_engineer_helpers.complaint_bins(complaints_t1)
    complaint_bins_by_UID = complaints_t1.groupby(["UID", "complaints_binned"]).size().unstack().fillna(0)
    new_cols.extend(list(complaint_bins_by_UID.columns))
    by_officer_df = by_officer_df.merge(complaint_bins_by_UID, how = "left", on = "UID")
    feat_engineer_helpers.add_suspension_length(complaints_t1)
    complaints_t1["count_sustained"] = \
    (complaints_t1["final_outcome"].str.lower().str.contains("no action taken") == False) & \
    (complaints_t1["final_outcome"].str.lower().str.contains("unknown") == False) & \
    (complaints_t1["final_outcome"].isna() == False)
    disciplines_t1_by_UID = complaints_t1.groupby("UID").agg({"count_sustained":"sum", "suspension_length":"sum"})
    assert ((disciplines_t1_by_UID["suspension_length"]>0) & \
            (disciplines_t1_by_UID["count_sustained"] ==0)).sum() ==0
    by_officer_df = by_officer_df.merge(disciplines_t1_by_UID, how = "left", on="UID")
    new_cols.extend(list(disciplines_t1_by_UID.columns))
    return new_cols, by_officer_df

def add_victim_demo(complaints_t1, by_officer_df):
    '''
    Adds information about demographics of victims.
    Inputs:
        complaints_t1: dataframe of complaints
        by_officer_df: Pandas dataframe of officer-level data
    Returns:
        new_cols: list of newly added columns for later use
        by_officer_df: officer-level dataframe updated to include breakdown of each
            officer's victims
    '''
    new_cols = []
    complaints_victims = pd.read_csv("../data/complaints-victims.csv.gz", compression="gzip")
    by_officer_racial_breakdown, victim_race_cols = feat_engineer_helpers.add_victim_cat(complaints_t1,
                                                                                          complaints_victims, "race")
    by_officer_df = by_officer_df.merge(by_officer_racial_breakdown,
                                                          how="left", left_on="UID", right_index=True)
    new_cols.extend(victim_race_cols)
    by_officer_gender_breakdown, victim_gender_cols = feat_engineer_helpers.add_victim_cat(complaints_t1,
                                                                                            complaints_victims,
                                                                                            "gender")
    by_officer_df = by_officer_df.merge(by_officer_gender_breakdown,
                                        how="left", left_on="UID", right_index=True)
    new_cols.extend(victim_gender_cols)

    return new_cols, by_officer_df

def add_salary_data(by_officer_df, t1):
    '''
    Reads salary data file and creates dataframe
    Inputs:
        by_officer_df: Pandas dataframe with officer-level information
        t1 (tuple): start and end years of observation period
    Returns
        salary_cols: list of added salary columns for later use
        by_officer_df: officer-level dataframe updated to include
            salary data
    '''
    salary_ranks = pd.read_csv("../data/salary-ranks_2002-2017_2017-09.csv.gz", compression="gzip")
    salary_ranks_t1_T = feat_engineer_helpers.add_salary_data(salary_ranks, t1)
    salary_cols = ["average_salary", "salary_change"]
    by_officer_df = by_officer_df.merge(salary_ranks_t1_T[salary_cols], how='left', on='UID')

    return salary_cols, by_officer_df
