'''
Engineers features of complaint bins, victim demographics, 
suspension length, and salary.
'''

import pandas as pd

def relevant_complaints(complaints, complaints_accused, t1, t2):
    '''
    Finds all complaints within the desired time period
    Inputs:
        complaints: a dataframe of complaints
        complaints_accused: second complaints dataframe from CPDP
        t1 (tuple): start and end years of observation period
        t2 (tuple): start and end years of target period
    Returns:
        complaints_t1_full: dataframe of complaints in observation period
        complaints_t2_full: dataframe of complaints in target period
    '''
    complaints["incident_date"] = pd.to_datetime(complaints["incident_date"])
    complaints_t1 = complaints[(complaints["incident_date"].dt.year >= t1[0]) &
                               (complaints["incident_date"].dt.year <= t1[1])]
    complaints_t2 =  complaints[(complaints["incident_date"].dt.year >= t2[0]) &
                               (complaints["incident_date"].dt.year <= t2[1])]

    complaints_t1_full = pd.merge(complaints_t1[["cr_id", "incident_date"]], complaints_accused,
                                  how="left", on="cr_id")
    complaints_t2_full = pd.merge(complaints_t2[["cr_id", "incident_date"]], complaints_accused,
                                  how="left", on = "cr_id")

    return complaints_t1_full, complaints_t2_full

def complaint_bins(complaints):
    '''
    Sorts complaints into bins based on their categories
    Inputs:
        complaints: a Pandas dataframe of complaints
    Returns:
        None, adds complaints_binned column to complaints dataframe
    '''

    complaints["complaint_category"].fillna("None", inplace=True)

    complaints["complaints_binned"] = complaints["complaint_category"]
    #Fold violence with weapon down into one bin
    complaints["complaints_binned"] = complaints["complaints_binned"].map(lambda x: "Use of Force - Weapon" if
                                                                            "TASER" in x or "FIREARM" in x else x)
    #Fold non-weapon violence down into one bin
    complaints["complaints_binned"] = complaints\
                                       ["complaints_binned"].map(lambda x:
                                                                  "Use of Force - No Weapon"
                                                                  if "KICKED" in x
                                                                  or "CHOKED" in x
                                                                  or "GRAB" in x
                                                                  or "STOMP" in x
                                                                  or "ASSAULT" in x
                                                                  or "PUNCH" in x
                                                                  or "TAKE DOWN" in x
                                                                  or "UNNECESSARY PHYSICAL CONTACT" in x
                                                                  or ("EXCESSIVE FORCE" in x and
                                                                      "FIREARM" not in x and
                                                                      "TASER" not in x)
                                                                  else x)

    complaints["complaints_binned"] = complaints["complaints_binned"].map(lambda x: "Non-Violent Civilian Contact" if
                                                                            "ILLEGAL ARREST" in x or
                                                                            "DETENTION" in x or "ALTERCATION" in x or
                                                                            "WARRANT" in x else x)
    #Fold racial complaints down into one bin
    complaints["complaints_binned"] = complaints["complaints_binned"].map(lambda x:
                                                                            "Racial/Ethnic" if "RACIAL" in x else x)


    #Fold Drug use down into one bin
    complaints["complaints_binned"] = complaints["complaints_binned"].map(lambda x: "Officer Drug Use/Possession"
                                                                                      if "DRUG" in x or
                                                                                      "INTOXIC" in x or "D.U.I" in x
                                                                                      or "ALCOHOL" in x
                                                                                      else x)

    bins = ["Use of Force - Weapon", "Use of Force - No Weapon", "Racial/Ethnic", 
            "Officer Drug Use/Possession", "Non-Violent Civilian Contact"]
    complaints["complaints_binned"][complaints["complaints_binned"].isin(bins) == False] = "Other/Unknown"

    return None

def add_salary_data(salary_ranks, t1):
    '''
    Creates dataframe of officer average salaries and changes in salary
    during observation period
    Inputs:
        salary_ranks: a dataframe with salary data
        t1 (tuple): start and end years of the observation period
    Returns:
        salary_ranks_t1_T: a dataframe with per-officer salary data for 
        the observation period
    '''
    salary_ranks['UID'] = salary_ranks['UID'].astype(int)
    salary_ranks['year'] = salary_ranks['year'].astype(int)
    salary_ranks_t1 = salary_ranks[(salary_ranks['year'] >= t1[0]) & (salary_ranks['year'] <= t1[1]) ]
    salary_ranks_t1_T = salary_ranks_t1.pivot(index = "UID", columns ='year', values = 'salary')
    salary_ranks_t1_T['average_salary'] = salary_ranks_t1_T.mean(axis=1, skipna=True)
    salary_ranks_t1_T_change = salary_ranks_t1_T.fillna(0)
    salary_ranks_t1_T_change['salary_change'] = salary_ranks_t1_T_change[t1[1]] - salary_ranks_t1_T_change[t1[0]]
    salary_ranks_t1_T['salary_change'] = salary_ranks_t1_T_change['salary_change']

    return salary_ranks_t1_T

def add_suspension_length(complaints):
    '''
    Adds length of suspension to complaints dataframe. If no suspension, inputs 0.
    Inputs:
        complaints: Pandas dataframe of complaints
    Returns:
        none, adds suspension_length column to complaints
    '''
    final_outcome = complaints["final_outcome"].fillna("")
    complaints["suspension_length"] = final_outcome.map(lambda x: int(x.split()[0])
                                                                                if x is not None and
                                                                                "Day Suspension" in x
                                                                                else 0)

    return None

def prep_y(complaints_t2):
    '''
    Prepares target variables.
    Inputs:
        complaints_t2: Pandas dataframe of complaints for target period
    Returns:
        target_df: dataframe of categorized complaints in target period
    '''
    complaints_t2["final_outcome"] = complaints_t2["final_outcome"].str.lower()
    complaints_t2["complaint_sustained"] = \
        (complaints_t2["final_outcome"].str.contains("no action taken") == False) & \
        (complaints_t2["final_outcome"].str.contains("unknown") == False) & \
        (complaints_t2["final_outcome"].isna() == False)
    complaints_t2["target_sustained"] = complaints_t2.groupby("UID")["complaint_sustained"].transform("sum") > 0
    complaint_bins(complaints_t2) # make bins
    complaints_t2["complaints_binned"] = complaints_t2["complaints_binned"].str.lower()
    complaints_t2["comp_use_of_force"] = complaints_t2["complaints_binned"].str.contains("use of force")
    complaints_t2["target_use_of_force"] = complaints_t2.groupby("UID")["comp_use_of_force"].transform("sum") > 0
    complaints_t2["comp_racial"] = complaints_t2["complaints_binned"].str.contains("racial")
    complaints_t2["target_racial"] = complaints_t2.groupby("UID")["comp_racial"].transform("sum") > 0
    complaints_t2["comp_drug"] = complaints_t2["complaints_binned"].str.contains("drug")
    complaints_t2["target_drug"] = complaints_t2.groupby("UID")["comp_drug"].transform("sum") > 0
    complaints_t2["comp_nonviolent"] = complaints_t2["complaints_binned"].str.contains("non-violent")
    complaints_t2["target_nonviolent"] = complaints_t2.groupby("UID")["comp_nonviolent"].transform("sum") > 0
    desired_targets = ["target_use_of_force", "target_drug", "target_racial",
                        "target_sustained", "target_nonviolent"]
    make_target_col(complaints_t2, desired_targets, "severe_complaint")
    target_df = complaints_t2[["UID", "severe_complaint"]].drop_duplicates()

    return target_df

def add_victim_cat(complaints_t1, complaints_victims, demographic):
    '''
    Adds demographic category for victim.
    Inputs:
        complaints_t1: Pandas dataframe of complaints during observation period
        complaints_victims: Pandas dataframe of who filed the complaints
        demographic (str): demographic category to be added
    Returns:
        by_officer_demo_breakdown: by-officer dataframe with percent of 
            complaints by demographic
        victim_demo_cols: list of labels of new victim demographic columns
    '''
    complaints_w_victim_data = complaints_t1.merge(complaints_victims, how="left", on="cr_id")
    by_officer_groupby = pd.DataFrame(complaints_w_victim_data.groupby(["UID", demographic]).size())
    by_officer_demo_breakdown = pd.pivot_table(by_officer_groupby, values=0, index=['UID'], 
                                                 columns=[demographic]).reset_index()
    by_officer_demo_breakdown.fillna(0, inplace=True)
    by_officer_demo_breakdown.set_index("UID", inplace=True)
    by_officer_demo_breakdown = by_officer_demo_breakdown.div(by_officer_demo_breakdown.sum(axis=1), axis=0)
    by_officer_demo_breakdown.columns = ["Pcnt Complaints Against " + x for x in by_officer_demo_breakdown.columns]
    victim_demo_cols = list(by_officer_demo_breakdown.columns)
    if demographic == "gender": # Fixes multicollinearity issues of male/female
        victim_demo_cols.remove("Pcnt Complaints Against FEMALE")

    return by_officer_demo_breakdown, victim_demo_cols

def make_target_col(df, desired_targets, col_name):
    '''
    Creates a target column in the target_df that will be true if at least one 
    of the desired targets is true.
    Inputs:
        df: a data frame with target period complaints
        desired_targets: list of target columns to include
        col_name: name of the new target column
    Returns:
        None: adds target column to the dataframe
    '''
    df[col_name] = df[desired_targets].any(axis='columns')
    return None

