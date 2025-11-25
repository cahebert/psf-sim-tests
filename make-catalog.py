import ngmix
import fitsio
import pathlib
import numpy as np
import pandas as pd

def get_desairmass_visits(db_path = '/astro/u/esheldon/oh2/rubin_sim_data/opsim-databases/baseline_v3.3_10yrs.db'):
    import sqlite3

    con = sqlite3.connect(db_path)
    query = \
    """select
            observationId
        from
            observations
        where
            target =''
            and filter='g'
            and airmass between 1 and 1.4
       """

    # Read these columns in from the "Summary" table, convert into a Pandas df
    db = pd.read_sql_query(query, con)
    return db

def get_fourth_moms(M11, M13, M31, M22):
    """Return spin-0 and spin-2 combinations of the fourth moments."""
    T4 = M22 / M11
    e4_1 = M31 / M11**2
    e4_2 = M13 / M11**2
    return T4, e4_1, e4_2

def get_sources(cat_list):
    dfs = []

    for f in cat_list:
        try:
            df = pd.DataFrame(fitsio.read(f))
        except FileNotFoundError:
            print(f"File {f} not found!")
        if np.sum(df['reserved']) != 0:
            df['raw_stars_in_det'] = np.sum(df['star_select'])

            # only save actual stars
            df = df.loc[df['star_select']==True]

            # is detector in corner of focal plane (ie vignetted) 
            det = f.split("-")[-1].strip("det").strip(".fits")
            df['detector'] = np.ones(len(df), dtype=int) * int(det)

            # save visit number
            try:
                visit = int(f.split("/")[-2]) 
            except ValueError:
                print(f)
            df['visit'] = np.ones(len(df), dtype=int) * visit
            
            df['flagged'] = (df['am_flags'] != 0) | (df['am_psf_flags'] != 0) 

            # save average seeing
            # df['visit_T'] = np.ones(len(df), dtype=int) * np.mean(df['am_T'])

            dfs.append(df)

    cat = pd.concat(dfs)

    cat.rename(columns={
        'am_e1':'e2_1',
        'am_e2':'e2_2',
        'am_psf_e1':'psf_e2_1',
        'am_psf_e2':'psf_e2_2',
        'am_T':'T',
        'am_psf_T':'psf_T'}, inplace=True)
    
    # get fourth moments
    raw_T4, raw_e4_1, raw_e4_2 = get_fourth_moms(cat['M11'],
                                                 cat['M13'],
                                                 cat['M31'],
                                                 cat['M22'])
    # subtract second order mom components from fourth order
    # this designed to give 0 fourth moments for Gaussian profile
    cat['T4'] = raw_T4 - cat['T']
    cat['e4_1'] = raw_e4_1 - 3*cat['e2_1']
    cat['e4_2'] = raw_e4_2 - 3*cat['e2_2']
    
    raw_psf_T4, raw_psf_e4_1, raw_psf_e4_2 = get_fourth_moms(cat['psf_M11'],
                                                             cat['psf_M13'], 
                                                             cat['psf_M31'], 
                                                             cat['psf_M22'])
    cat['psf_T4'] = raw_psf_T4 - cat['psf_T']
    cat['psf_e4_1'] = raw_psf_e4_1 - 3*cat['psf_e2_1']
    cat['psf_e4_2'] = raw_psf_e4_2 - 3*cat['psf_e2_2']
    
    # only keep necessary columns to reduce file size
    cat = cat[[
        'detector',
        'reserved',
        'star_select',
        'flagged',
        'visit',
        'raw_stars_in_det',
        'am_s2n',
        'ra', 
        'dec', 
        'x',
        'y',
        'e2_1',
        'e2_2',
        'am_e1_err',
        'am_e2_err',
        'T',
        'psf_e2_1',
        'psf_e2_2',
        'psf_T',
        'e4_1',
        'e4_2',
        'T4',
        'psf_e4_1',
        'psf_e4_2',
        'psf_T4',
    ]]

    return cat

def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--flist', nargs="*", default=[])
    parser.add_argument('--catdir', default='/gpfs02/astro/workarea/chebert/piff-catalogs/')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    print("Running python file...")
    for flistpath in args.flist:
        flist = np.loadtxt('./flists/' + flistpath, dtype=str)
        print("Loaded file list")
        simref = flist[0].split('/')[-3].strip('run-')
        band = simref.split('-')[0]

        savecat = pathlib.Path(args.catdir + 'cat-' + simref  + '.fits')

        if savecat.is_file():
            print("catalog already exists!")
        else:
            cat = get_sources(cat_list=flist)
        
            fitsio.write(savecat, cat.to_records(index=False))
