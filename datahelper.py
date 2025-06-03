import numpy as np

def fetch_opsim_cols(
    visits,
    col_list=['paraAngle', 'altitude', 'rotSkyPos'],
    db_path='/Users/clairealice/Documents/share/sim_baseline/baseline_v3.3_10yrs.db'):
    """This is pretty redundant with above, but want list of visit-level information"""
    import sqlite3
    import pandas as pd
    con = sqlite3.connect(db_path)

    query = \
        f"""SELECT observationId, {", ".join(c for c in col_list)}
            FROM observations
            WHERE observationId in ({", ".join([str(v) for v in visits])})
            """

    # Read these columns in from the "Summary" table, convert into a Pandas df
    db = pd.read_sql_query(query, con)

    return db

def get_visit_info(cat, col_list=['paraAngle', 'altitude', 'rotSkyPos']):

    db = fetch_opsim_cols(np.unique(cat['visit']), col_list)

    for col in col_list:
        vals = np.zeros(len(cat))
        for visitId in np.unique(cat['visit']):
            vals[np.where(cat['visit']==visitId)[0]] = db[col][db['observationId']==visitId]
        cat[col] = vals

    for col in col_list:
        if col not in cat.columns:
            raise ValueError(f"Column {col} not found in the database or catalog.")