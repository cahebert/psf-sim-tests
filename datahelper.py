import numpy as np
import fitsio
import pandas as pd
import ngmix

def load_catalog(catpath, reserved, fpcoords=False, trim=False):
    catalog = fitsio.read(catpath)
    catalog = catalog[~catalog['flagged']]

    if trim:
        catalog = catalog[(catalog['dec'] > -44) & (catalog['dec'] < -36)]
        catalog = catalog[(catalog['ra'] > 66.5) & (catalog['ra'] < 73.5)]

    cat = pd.DataFrame(catalog)

    if 'rotSkyPos' not in cat.columns:
        get_visit_info(cat)

    if reserved:
        cat = cat[cat['reserved']]

    cat['dtt'] = (cat['T'] - cat['psf_T']) / cat['psf_T']
    cat['dg1'] = (cat['g1'] - cat['psf_g1'])
    cat['dg2'] = (cat['g2'] - cat['psf_g2'])

    cat['dtt4'] = (cat['T4'] - cat['psf_T4']) / cat['psf_T4']
    cat['de4_1'] = (cat['e4_1'] - cat['psf_e4_1'])
    cat['de4_2'] = (cat['e4_2'] - cat['psf_e4_2'])

    if fpcoords:
        ## positive rotSkyPos moves sky CW relative to camera, so camera moves CCW relative to sky
        cat['dg1_pix'], cat['dg2_pix'] = ngmix.shape.rotate_shape(
            cat['dg1'],
            cat['dg2'],
            -cat['rotSkyPos'] * np.pi/180
        )
        cat['de4_1_pix'], cat['de4_2_pix'] = ngmix.shape.rotate_shape(
            cat['de4_1'],
            cat['de4_2'],
            -cat['rotSkyPos'] * np.pi/180
        )

        cat['g1_pix'], cat['g2_pix'] = ngmix.shape.rotate_shape(
            cat['g1'],
            cat['g2'],
            -cat['rotSkyPos'] * np.pi/180
        )
        cat['e4_1_pix'], cat['e4_2_pix'] = ngmix.shape.rotate_shape(
            cat['e4_1'],
            cat['e4_2'],
            -cat['rotSkyPos'] * np.pi/180
        )

    return cat

def fetch_opsim_cols(
    visits,
    col_list=['paraAngle', 'altitude', 'rotSkyPos'],
    db_path='/Users/clairealice/Documents/share/sim_baseline/baseline_v3.3_10yrs.db'):
    """This is pretty redundant with above, but want list of visit-level information"""
    import sqlite3
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