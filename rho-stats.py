import numpy as np
import ngmix
import json

CORNERS=[188, 168, 123, 27, 0, 20, 65, 161]
    
def catalog_select(catalog, args):
    """Run various selections on the catalog."""
    # get rid of flagged entries first
    try:
        catalog = catalog[~catalog['flagged']]
    except ValueError:
        print('did not find flag column')
        
    reserved=False if args.justtrain else True,
    training=args.justtrain
    if reserved:
        catalog = catalog[catalog['reserved']]
    elif training:
        catalog = catalog[(catalog['star_select']) & (~catalog['reserved'])]
    
    if args.desairmass:
        desvisits = get_desairmass_visits()
        keep = np.array([i for i in range(len(catalog)) if catalog['visit'][i] in desvisits['observationId']])
        catalog = catalog[keep]
        
    if args.nstarcut is not None:
        catalog = catalog[catalog['stars_in_visit']>args.nstarcut]
        
    if args.cornercut:
        catalog = catalog[~np.isin(catalog['detector'].astype('int32'), CORNERS)]

    if args.edgecut:
        dec_min, dec_max = -44, -36
        ra_min, ra_max = 66.6, 73.5
        
        select = (catalog['ra'] > ra_min) & (catalog['ra'] < ra_max)
        select = select & (catalog['dec'] > dec_min) & (catalog['dec'] < dec_max)
        
        catalog = catalog[select]
    
    return catalog

def get_catalog_samples(catalog, rng, args):
    unique_visits = np.unique(catalog['visit'])
    if args.Nsamples == 1:
        # if we are not bootstrapping, then we need to select Nvisits and return catalog ids
        if (args.Nvisits is not None) and (len(unique_visits)>args.Nvisits):
            sample = rng.choice(unique_visits, replace=False, size=args.Nvisits)
            # iterate over sample instead of catalog, prob faster
            keep = np.concatenate(
                [np.arange(len(catalog))[catalog['visit']==v] for v in sample]
                )
        else:
            # no sampling and we just take the whole catalog
            keep = True
    elif args.Nsamples > 1:
        # bootstrapping. Set whether to replace or not depending on Nvisits
        if (args.Nvisits is not None) and (len(unique_visits)>args.Nvisits):
            replace = False
            n_to_sample = args.Nvisits
        elif (args.Nvisits is None) or (len(unique_visits)==args.Nvisits):
            replace = True
            n_to_sample = len(unique_visits)
            
        sample = rng.choice(unique_visits, replace=replace, size=n_to_sample)
        keep = np.concatenate(
            [np.arange(len(catalog))[catalog['visit']==v] for v in sample]
        )
    return catalog[keep]


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

def get_psf_correlations(
    catalog,
    min_sep=0.6,
    max_sep=200,
    nbins=40,
    sep_units='arcmin',
    cov=False,
    npatch=250,
    ):
    import treecorr

    ## notation convention: 
    # g = reduced shear, 
    # e = distortion
    # number next to g/e gives moment order, eg e2 is second order distortion
    # underscore gives component of spin-2 
    
    # second order reduced shear residual \delta g2
    dg2_1 = catalog['e2_1'] - catalog['psf_e2_1']
    dg2_2 = catalog['e2_2'] - catalog['psf_e2_2']
    # fourth order distortion residual \delta e4
    de4_1 = catalog['e4_1'] - catalog['psf_e4_1']
    de4_2 = catalog['e4_2'] - catalog['psf_e4_2']
    
    ## size residuals -- T_* in denominator matching Yamamoto et al. 2025
    dTT2 = (catalog['T'] - catalog['psf_T']) / catalog['T']
    dTT4 = (catalog['T4'] - catalog['psf_T4']) / catalog['T4']
    # wij = ei * dTj/Tj
    # to match Schutt et al., Yamamoto et al. 2025 we use e_psf rather than e_*
    w22_1 = catalog['psf_e2_1'] * dTT2
    w22_2 = catalog['psf_e2_2'] * dTT2
    w24_1 = catalog['psf_e2_1'] * dTT4
    w24_2 = catalog['psf_e2_2'] * dTT4
    w42_1 = catalog['psf_e4_1'] * dTT2
    w42_2 = catalog['psf_e4_2'] * dTT2
    w44_1 = catalog['psf_e4_1'] * dTT4
    w44_2 = catalog['psf_e4_2'] * dTT4

    # to match Schutt et al., Yamamoto et al. 2025 we use e_psf rather than e_*
    moments = [(catalog['psf_e2_1'], catalog['psf_e2_2'], 'e2'), 
               (catalog['psf_e4_1'], catalog['psf_e4_2'], 'e4'),
               (dg2_1, dg2_2, 'de2'), 
               (de4_1, de4_2, 'de4'), 
               (w22_1, w22_2, 'w22'), 
               (w24_1, w24_2, 'w24'), 
               (w42_1, w42_2, 'w42'), 
               (w44_1, w44_2, 'w44'),
              ]

    correlations = {}
    
    ggconfig = {'min_sep'    : min_sep,
                'max_sep'    : max_sep,
                'nbins'      : nbins,
                'sep_units'  : sep_units,
                'var_method' : 'shot' if cov is None else cov,
               }
    
    catconfig = {'ra'        : catalog['ra'],
                 'dec'       : catalog['dec'],
                 'ra_units'  : 'degree', 
                 'dec_units' : 'degree',
                }
    if cov:
        catconfig['npatch'] = npatch
    
    # All the combinations for stats
    for i in range(len(moments)):
        for j in range(i, len(moments)):
            corr = treecorr.GGCorrelation(**ggconfig)

            if i!=0 and cov:
                catconfig['patch_centers'] = x_i.patch_centers

            x_i = treecorr.Catalog(g1=moments[i][0], g2=moments[i][1], **catconfig)

            if i==j:
                corr.process(x_i)
            else:
                if i==0 and cov: # only condition in which this has not already been set
                    catconfig['patch_centers'] = x_i.patch_centers
                x_j = treecorr.Catalog(g1=moments[j][0], g2=moments[j][1], **catconfig)
                corr.process(x_i, x_j)     
                
            correlations[moments[i][2] + moments[j][2]] = corr

    return correlations           


def test_significance(stats):
    chi2_list = {}
    for k, rho in stats.items():
        n = rho.nbins
        cov_inv = np.linalg.inv(rho.cov[:n,:n])

        chi2 = rho.xip.reshape((1,-1)) @ cov_inv @ rho.xip
        chi2_list[k] = chi2
        
    return chi2_list
    
def get_info_string(args):
    info = f'{"-trainonly" if args.justtrain else ""}' + \
           f'{"-nstarcut"+str(args.nstarcut) if args.nstarcut is not None else ""}' + \
           f'{"-cornercut" if args.cornercut else ""}' + \
           f'{"-edgecut" if args.edgecut else ""}' + \
           f'{"-desairmass" if args.desairmass else ""}' + \
           f'{"-summary" if args.summary else ""}' + \
           f'{f"-{args.Nsamples}x" if args.Nsamples>1 else "-cov"}'
    return info


def get_rho_summary(rstats, summary=None):
    if summary is None:
        summary = {k:{'xip':[],'xim':[],'varxip':[],'varxim':[],'meanr':[]} for k in rstats.keys()}
    for k in rstats.keys():
        summary[k]['xip'].append(list(rstats[k].xip))
        summary[k]['xim'].append(list(rstats[k].xim))
        summary[k]['varxip'].append(list(rstats[k].varxip))
        summary[k]['varxim'].append(list(rstats[k].varxim))
        summary[k]['meanr'] = list(rstats[k].meanr)
    return summary
    

def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--catname', type=str, required=True)
    parser.add_argument('--summary', default=False, action='store_true')
    parser.add_argument('--dxi', default=False, action='store_true')
    parser.add_argument('--cov', default=False, action='store_true')
    parser.add_argument('--index', type=int, default=None)
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--Nvisits', type=int, default=None)
    parser.add_argument('--npatch', type=int, default=200)
    parser.add_argument('--Nsamples', type=int, default=1)
    parser.add_argument('--justtrain', default=False, action='store_true')
    parser.add_argument('--nstarcut', type=float, default=None)
    parser.add_argument('--cornercut', default=False, action='store_true')
    parser.add_argument('--edgecut', default=False, action='store_true')
    parser.add_argument('--desairmass', default=False, action='store_true')
    parser.add_argument('--catdir', default='/gpfs02/astro/workarea/chebert/piff-catalogs/')
    parser.add_argument('--rhodir', default='/gpfs02/astro/workarea/chebert/rho-outputs/')
    return parser.parse_args()

if __name__ == '__main__':
    import pathlib
    import fitsio
    
    args = get_args()
    if args.index is not None:
        rng = np.random.default_rng(seed=args.seed + args.index)
    else:
        rng = np.random.default_rng(seed=args.seed)
    
    info_string = args.catname.strip('cat-') + get_info_string(args)
    saverho = args.rhodir + 'rho-' + info_string
    
    print(f"Running rho stats with info string: {info_string} and requested {args.Nvisits}")
    
    # load catalog, with selections in args
    catfile = pathlib.Path(args.catdir + args.catname + '.fits')
    if catfile.is_file():
        cat = fitsio.read(catfile)
    else:
        raise FileNotFoundError(f'Catalog not found at: {catfile}!')
    
    catalog = catalog_select(cat, args)
    
    # set some treecorr settings
    corr_args = {'sep_units': 'arcmin'}
    if args.summary:
        corr_args['nbins'] = 1
        corr_args['min_sep'] = 0.5
        corr_args['max_sep'] = 50
    else:
        corr_args['nbins'] = 25
        corr_args['min_sep'] = 0.5
        corr_args['max_sep'] = 50
    
    if args.Nsamples > 1:
        # we do no jackknife sampling, so don't set cov 
        corr_args['cov'] = None
        
        rho_summary = None
        # run stats on subset of catalog for each sample
        for i in range(args.Nsamples):
            cat = get_catalog_samples(catalog, rng, args)
            
            print(f'Running sample {i+1} of {args.Nsamples}')
            nvisits = len(np.unique(cat['visit']))
            print('Number of visits: ', nvisits)
        
            rstats = get_psf_correlations(cat, **corr_args)
            # add summary of results to dict
            rho_summary = get_rho_summary(rstats, summary=rho_summary)
    else:
        # this will just subsample if Nvisits is set, otherwise will return whole catalog
        cat = get_catalog_samples(catalog, rng, args)
        
        # no bootstrapping so sample_indices is either single array or True
        nvisits = len(np.unique(cat['visit']))
        print(f'Number of visits: {nvisits}')
        
        if args.cov:
            # set covariance to jackknife
            corr_args['cov'] = 'jackknife'
            corr_args['npatch'] = args.npatch
        else:
            # set covariance to None
            corr_args['cov'] = None
    
        rstats = get_psf_correlations(catalog, **corr_args)
        rho_summary = get_rho_summary(rstats, summary=None)

    # change the file name to include Nvisits
    saverho += str(nvisits)
    if args.index is not None:
        # keep track of bootstrap samples when doing parallel runs
        saverho += f'-{args.index}'

    print(f"Saving rho result summary to {saverho}.json")
    with open(saverho + '.json', 'w', encoding='utf-8') as f:
        json.dump(rho_summary, f, ensure_ascii=False, indent=4)

    if args.cov:
        import pickle
        with open(saverho + '.pkl', 'wb') as f:
            pickle.dump(rstats, f)
