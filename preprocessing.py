
import pandas as pd


def minimal_preprocessing():
    ed = pd.read_csv('./data/train_data/ed_train_raw.csv') # education DataFrame
    hh = pd.read_csv('./data/train_data/hh_train_raw.csv') # household DataFrame
    poverty = pd.read_csv('./data/train_data/poverty_train_raw.csv') # poverty/labels


    def preprocess_df(df, suffix:str):

        # merge first three columns into psu_hh_idcode identifier
        uids = df['psu'].astype(str) + "_"  + df['hh'].astype(str) + "_" + df['idcode'].astype(str) 
        
        # delete the three columns
        df = df.drop(columns=['psu', 'hh', 'idcode'])

        ## Capitalize all Q's in column name prefixes. Add ED or HH prefix to identify variate group
        df.columns = [suffix + "_" + col.capitalize() for col in df.columns]

        # Insert uid as first column, lowercase, no prefix.
        df.insert(0, 'uid', uids)

        return df

    ed = preprocess_df(ed, 'ED')
    hh = preprocess_df(hh, 'HH')

    # Convert subjective poverty score from one-hot to categorical [1-10] outcome
    for i in range(1,11):
        col = 'subjective_poverty_'+ str(i)
        poverty.loc[poverty[col]==1, 'poverty_score'] = i

    poverty['uid'] = poverty['psu_hh_idcode']
    y = poverty[['uid', 'poverty_score']]

    # Filter labeled data
    ed = ed[ed['uid'].isin(poverty['uid'])]
    hh = hh[hh['uid'].isin(poverty['uid'])]

    # ensure rows match between ed, hhand y
    X_raw = pd.merge(ed, hh, on='uid', how='inner')
    Xy = pd.merge(X_raw, y, on='uid', how='left')
    assert(Xy['poverty_score'].isna().sum() == 0)
    y = Xy['poverty_score']
    X_raw = Xy.drop(columns=['poverty_score', 'HH_Hhid'])


    return X_raw, y



def drop_nans_threshold(X, threshold=0.7):
    '''
    Returns X with columns with more than threshold*100% of NaN values dropped
    '''
    threshold = len(X) * threshold
    X = X.dropna(axis=1, thresh=len(X)-threshold)
    return X


def rename_columns(X):
    renames = {
        'ED_Q01': 'read', 
        'ED_Q02': 'write', 
        'ED_Q03': 'attended_school', 
        'ED_Q04': 'highest_school_lvl', 
        'ED_Q05': 'highest_school_lvl_grade', 
        'ED_Q06': 'highest_diploma',
        'ED_Q07': 'preschool', 
        'ED_Q08': 'now_enrolled', 
        'ED_Q11': 'now_not_enroll_reason', 
        'ED_Q14': 'past_enrolled', 
        'ED_Q17': 'past_not_enroll_reason', 
        'ED_Q18': 'finish_school_age', 
        'ED_Q19': 'less_than_19',
        'HH_Q02': 'sex', 
        'HH_Q03': 'family_role', 
        'HH_Q04': 'DOB', 
        'HH_Q05y': 'age_yrs', 
        'HH_Q05m': 'age_months', 
        'HH_Q06': 'marital_status',
        'HH_Q07': 'lives_with_partner', 
        'HH_Q08': 'partner_id_code', 
        'HH_Q09': 'time_away', 
        'HH_Q10': 'present_in_past_year', 
        'HH_Q11': 'lives_with_mother', 
        'HH_Q13': 'mother_education', 
        'HH_Q14': 'mother_alive',
        'HH_Q15': 'mother_death_age', 
        'HH_Q16': 'age_mother',
        'HH_Q17': 'lives_with_father', 
        'HH_Q19': 'father_education', 
        'HH_Q20': 'father_alive', 
        'HH_Q21': 'father_death_age', 
        }
    return X.rename(columns=renames)