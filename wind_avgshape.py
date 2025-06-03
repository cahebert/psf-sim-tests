import numpy as np
import fitsio
import pandas as pd
import datahelper
from colors import color_scheme as colors

import matplotlib.pyplot as plt
plt.style.use('./paper.mplstyle')

def get_lsstsim_winds(cat):
    import psfws
    ws = psfws.ParameterGenerator()
    wind_dirs = []

    # in principle just need one row per visit, so:
    tmp = cat[['visit', 'altitude', 'azimuth']].drop_duplicates()
    for v in tmp.index:
        row = tmp.loc[v]
        winds = []
        for _ in range(100):
            pt = ws.draw_datapoint()
            winds.append(ws.get_parameters(pt, skycoord=True,
                                        alt=row['altitude'],
                                        az=row['azimuth'])['phi'][0])
        wind_dirs.append(np.mean(winds))
    return np.array(wind_dirs)

def load_lsstsim(catpath):
    """
    Load the LSSTsim catalog into a DataFrame and get array of wind directions
    """
    fits = fitsio.FITS(catpath)
    catalog = fits[1][['g1','g2','flagged','visit']][:]
    # can include the training stars here too bc it's not a model question.
    catalog = pd.DataFrame(catalog[~catalog['flagged']], dtype='f8')

    datahelper.get_visit_info(catalog, col_list=['azimuth', 'altitude'])
    winds = get_lsstsim_winds(catalog)
    winds = change_wind_coordinate(winds)

    return catalog[['g1','g2']].rename(columns={'g1':'e1','g2':'e2'}), winds

def get_ctio_direction(ctio_wind):
    import psfws
    ctio_wind['wind_dir'] = np.zeros(len(ctio_wind))
    for i in ctio_wind.index:
        obs_nez, sky_nez = psfws.utils.get_both_nez(
            90-ctio_wind.at[i, 'zenith_distance'],
            ctio_wind.at[i, 'azimuth'],
            lat=-30.169, lon=70.806
        )
        v = ctio_wind.at[i, 'windspd'] * np.cos(np.radians(ctio_wind.at[i, 'winddir']))
        u = ctio_wind.at[i, 'windspd'] * np.sin(np.radians(ctio_wind.at[i, 'winddir']))

        obs_wind = v * obs_nez[0] + u * obs_nez[1]
        sky_wind = np.dot(sky_nez, obs_wind)

        ctio_wind.at[i, 'wind_dir'] = psfws.utils.to_direction(sky_wind[0], sky_wind[1])
    return ctio_wind[['expnum', 'wind_dir']]

def load_desy3(catpath, windpath1, windpath2):
    ctio_wind_1 = pd.read_csv(windpath1)
    ctio_wind_2 = pd.read_csv(windpath2)
    # combine the two wind datasets
    ctio_wind = pd.concat([ctio_wind_1, ctio_wind_2]).reset_index()

    fits = fitsio.FITS(catpath)
    catalog = fits[1][['obs_e1','obs_e2','exp']][:]
    catalog = pd.DataFrame(catalog, dtype='f8')

    # keep only the visits that are in both catalogs
    union_exp = np.intersect1d(catalog['exp'].unique(), ctio_wind['expnum'].unique())
    catalog = catalog[catalog['exp'].isin(union_exp)]
    ctio_wind = ctio_wind[ctio_wind['expnum'].isin(union_exp)]

    # get the projected wind directions
    ctio_wind = get_ctio_direction(ctio_wind)
    ctio_wind = change_wind_coordinate(ctio_wind['wind_dir'])

    if ctio_wind['expnum'].iloc[0] not in catalog['exp']:
        raise ValueError("DESY3 and CTIO wind catalogs do not match on expnum.")

    return catalog.rename(columns={'obs_e1':'e1','obs_e2':'e2'}), ctio_wind

def load_psfws(catpath):
    import json
    with open(catpath, 'r') as f:
        psfwssim = json.load(f)
    psfws_wind = psfwssim.pop('wind dir')
    psfws_wind = change_wind_coordinate(psfws_wind)

    psfwssim = pd.DataFrame(psfwssim).rename(columns={'g1': 'e1', 'g2': 'e2'})
    return psfwssim, psfws_wind

def change_wind_coordinate(wind_dirs):
    """Convert wind directions from E of N to degrees from +x axis."""
    # wind_dirs are in degrees E of N, so subtract 90 and flip sign
    return -(np.array(wind_dirs) % 180 - 90)

def beta_angle(e1, e2):
    """Calculate the beta angle from e1 and e2."""
    return 0.5 * np.arctan2(e2, e1) * 180 / np.pi


# Load the catalogs and wind data
desy3_cat, ctio_wind = load_desy3(
    catpath='/Users/clairealice/Documents/shear/des_psf_y3a1-v29.fits',
    windpath1='/Users/clairealice/Downloads/ctio_sispi_wind_250000-275000.csv',
    windpath2='/Users/clairealice/Downloads/ctio_sispi_wind_350000-375000.csv',
)
lsst_cat, lsst_wind = load_lsstsim(
    catpath='~/Documents/shear/testing-piff/data/cat-i-full-radec-03-piff-02.fits'
)
psfws_cat, psfwssim_wind = load_psfws(
    catpath='/Users/clairealice/Downloads/2025_psfws_sim_summary.json'
)

f, a = plt.subplots(3,1,figsize=(3.35, 5.5), sharex=True)
bins = np.linspace(-90,90,19) # degrees from +x axis

labels = [
    r'atm-only sim',
    r'LSST sim',
    r'DESY3'
]

for ax, cat, wind, label in zip(
    a,
    [psfws_cat, lsst_cat, desy3_cat],
    [psfwssim_wind, lsst_wind, ctio_wind],
    labels,
    ):
    # calculate the mean angle for the beta angle
    mean_angle = beta_angle(np.mean(cat['e1']), np.mean(cat['e2']))

    ax.hist(
        beta_angle(cat['e1'], cat['e2']),
        bins=bins, alpha=0.1, color=colors.g,label=r'$\beta(e_1, e_2)$'
    )
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-1,1))
    ax.axvline(
        mean_angle,
        lw=2.5, color=colors.g, ls='--', alpha=0.5,
        label=r'$\beta(\langle e_1\rangle, \langle e_2\rangle)$'
    )
    aw = ax.twinx()
    aw.hist(wind, bins=bins, histtype='step', lw=2, color=colors.y, label=r'$\theta_{wind}$')

    aw.set_ylabel(r'$N_{visit}$', color=colors.y)
    ax.set_ylabel(r'$N_{PSF}$', color=colors.g)

    ax.set_title(label, pad=0.15)

a[0].set_xticks([-90, -60, -30, 0, 30, 60, 90], [-90, -60, -30, 0, 30, 60, 90]);
a[0].set_xlim(-90,90)

a[2].set_xlabel('degrees from RA axis')

# make legend
handles, labels = ax.get_legend_handles_labels()
wind_h, wind_l = aw.get_legend_handles_labels()

a[0].legend(
    handles=handles + wind_h,
    labels=labels + wind_l,
    loc='center right', borderaxespad=0.1
)

plt.savefig('../figures/wind-shapes-psfws-lsstsim-desy3.jpg', dpi=300)

plt.show()
