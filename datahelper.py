import numpy as np
import fitsio
import pandas as pd

CORNERS=[188, 168, 123, 27, 0, 20, 65, 161]

def rotate_to_fp(e1, e2, skyrot):
    import ngmix
    e1_p, e2_p = ngmix.shape.rotate_shape(e1, e2, (90-skyrot)*np.pi/180)
    return e1_p, e2_p

def load_rho_samples(filename, n):
    import json
    with open(filename, 'rb') as f:
        rhos = json.load(f)

    rho_summary = {}
    error_norm = 1508/n  # divide by the number of independent samples

    for k, rho in rhos.items():
        if n==1508:
            rho_summary[k] = {
                'xip_var' : np.array(rho['varxip']).flatten(),
                'xim_var' : np.array(rho['varxim']).flatten(),
                'xip'     : np.array(rho['xip']).flatten(),
                'xim'     : np.array(rho['xim']).flatten(),
                'meanr'   : np.array(rho['meanr']).flatten()
            }
        else:
            rho_summary[k] = {
                'xip_var' : np.var(rho['xip'], axis=0).flatten() / error_norm,
                'xim_var' : np.var(rho['xim'], axis=0).flatten() / error_norm,
                'xip'     : np.mean(np.array(rho['xip']), axis=0).flatten(),
                'xim'     : np.mean(np.array(rho['xim']), axis=0).flatten(),
                'meanr'   : np.array(rho['meanr']).flatten()
            }

    return rho_summary

def load_catalog(catpath, reserved, fpcoords=False, trim=False, cutcorners=True, other_cols=[]):
    fits = fitsio.FITS(catpath)

    parameters = [
    'ra','dec','detector','x','y',
    'T','e2_1','e2_2',
    'T4','e4_1','e4_2',
    'psf_T','psf_e2_1','psf_e2_2',
    'psf_T4','psf_e4_1','psf_e4_2',
    'visit','flagged','reserved',
    ]
    parameters += other_cols

    if trim: parameters += ['ra','dec']
    catalog = fits[1][parameters][:]
    catalog = catalog[~catalog['flagged']]

    if trim:
        catalog = catalog[(catalog['dec'] > -44) & (catalog['dec'] < -36)]
        catalog = catalog[(catalog['ra'] > 66.5) & (catalog['ra'] < 73.5)]

    if reserved:
        catalog = catalog[catalog['reserved']]

    if cutcorners:
        catalog = catalog[~np.isin(catalog['detector'].astype('int32'), CORNERS)]

    cat = pd.DataFrame(catalog, dtype='f8')
    for col in ['flagged', 'reserved', 'detector','visit']:
        cat[col] = cat[col].astype('i8')

    cat['dtt'] = (cat['T'] - cat['psf_T']) / cat['psf_T']
    cat['de2_1'] = (cat['e2_1'] - cat['psf_e2_1'])
    cat['de2_2'] = (cat['e2_2'] - cat['psf_e2_2'])

    cat['dtt4'] = (cat['T4'] - cat['psf_T4']) / cat['psf_T4']
    cat['de4_1'] = (cat['e4_1'] - cat['psf_e4_1'])
    cat['de4_2'] = (cat['e4_2'] - cat['psf_e4_2'])

    if fpcoords:
        if 'rotSkyPos' not in cat.columns:
            get_visit_info(cat)

        for p in ['e2_','de2_','e4_','de4_']:
            cat[p+'1_pix'], cat[p+'2_pix'] = rotate_to_fp(
                cat[p+'1'],
                cat[p+'2'],
                cat['rotSkyPos']
                )

    return cat

def get_visit_info(cat, col_list=['paraAngle', 'altitude', 'rotSkyPos', 'seeingFwhm500']):
    opsimparams = pd.read_json('../data/opsim-params.json')

    for col in col_list:
        vals = np.zeros(len(cat))
        for visitId in opsimparams['observationId'].values:
            vals[np.where(cat['visit']==visitId)[0]] = opsimparams[col][opsimparams['observationId']==visitId]
        cat[col] = vals

    for col in col_list:
        if col not in cat.columns:
            raise ValueError(f"Column {col} not found in the database or catalog.")

# def fetch_opsim_cols(
#     visits,
#     col_list=['paraAngle', 'altitude', 'rotSkyPos'],
#     db_path='/Users/clairealice/Documents/share/sim_baseline/baseline_v3.3_10yrs.db'):
#     """This is pretty redundant with above, but want list of visit-level information"""
#     import sqlite3
#     con = sqlite3.connect(db_path)

#     query = \
#         f"""SELECT observationId, {", ".join(c for c in col_list)}
#             FROM observations
#             WHERE observationId in ({", ".join([str(v) for v in visits])})
#             """

#     # Read these columns in from the "Summary" table, convert into a Pandas df
#     db = pd.read_sql_query(query, con)

#     return db

# def get_visit_info(cat, col_list=['paraAngle', 'altitude', 'rotSkyPos']):

#     db = fetch_opsim_cols(np.unique(cat['visit']), col_list)

#     for col in col_list:
#         vals = np.zeros(len(cat))
#         for visitId in np.unique(cat['visit']):
#             vals[np.where(cat['visit']==visitId)[0]] = db[col][db['observationId']==visitId]
#         cat[col] = vals

#     for col in col_list:
#         if col not in cat.columns:
#             raise ValueError(f"Column {col} not found in the database or catalog.")