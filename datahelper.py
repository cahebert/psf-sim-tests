import numpy as np
import fitsio
import pandas as pd

def rotate_to_fp(e1, e2, skyrot):
    import ngmix
    e1_p, e2_p = ngmix.shape.rotate_shape(e1, e2, -skyrot*np.pi/180)
    return -e1_p, -e2_p

def load_catalog(catpath, reserved, fpcoords=False, trim=False):
    fits = fitsio.FITS(catpath)

    parameters = [
    'ra','dec','detector','x','y',
    'T','g1','g2',
    'T4','e4_1','e4_2',
    'psf_T','psf_g1','psf_g2',
    'psf_T4','psf_e4_1','psf_e4_2',
    'visit','flagged','reserved','visit_T'
    ]
    if trim: parameters += ['ra','dec']
    catalog = fits[1][parameters][:]
    catalog = catalog[~catalog['flagged']]

    if trim:
        catalog = catalog[(catalog['dec'] > -44) & (catalog['dec'] < -36)]
        catalog = catalog[(catalog['ra'] > 66.5) & (catalog['ra'] < 73.5)]

    if reserved:
        catalog = catalog[catalog['reserved']]

    cat = pd.DataFrame(catalog, dtype='f8')
    for col in ['flagged', 'reserved', 'detector','visit']:
        cat[col] = cat[col].astype('i8')

    cat['dtt'] = (cat['T'] - cat['psf_T']) / cat['psf_T']
    cat['dg1'] = (cat['g1'] - cat['psf_g1'])
    cat['dg2'] = (cat['g2'] - cat['psf_g2'])

    cat['dtt4'] = (cat['T4'] - cat['psf_T4']) / cat['psf_T4']
    cat['de4_1'] = (cat['e4_1'] - cat['psf_e4_1'])
    cat['de4_2'] = (cat['e4_2'] - cat['psf_e4_2'])

    if fpcoords:
        if 'rotSkyPos' not in cat.columns:
            get_visit_info(cat, col_list=['rotSkyPos'])

        for p in ['g','dg','e4_','de4_']:
            cat[p+'1_pix'], cat[p+'2_pix'] = rotate_to_fp(
                cat[p+'1'],
                cat[p+'2'],
                cat['rotSkyPos']
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