import matplotlib.pyplot as plt
import numpy as np
import fitsio
import pandas as pd
import skyproj
plt.style.use('./paper.mplstyle')  # change this to a paper specific one

def load_catalog(catpath, residual):
    catalog = fitsio.read(catpath)
    catalog = catalog[~catalog['flagged']]

    cat = pd.DataFrame(catalog)

    if residual:
        cat['dtt'] = (cat['T'] - cat['psf_T']) / cat['psf_T']
        cat['dg1'] = (cat['g1'] - cat['psf_g1'])
        cat['dg2'] = (cat['g2'] - cat['psf_g2'])

        cat['dtt4'] = (cat['T4'] - cat['psf_T4']) / cat['psf_T4']
        cat['de4_1'] = (cat['e4_1'] - cat['psf_e4_1'])
        cat['de4_2'] = (cat['e4_2'] - cat['psf_e4_2'])

    return cat

def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--residual', action='store_true', default=False)
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    cat = load_catalog(
        '~/Documents/shear/testing-piff/data/cat-i-full-radec-03-piff-02.fits',
        args.residual)

    f, a = plt.subplots(3,2, figsize=(3.35,5.25))

    if args.residual:
        parameters = ['dtt', 'dtt4', 'dg1', 'dg2', 'de4_1', 'de4_2']
        labels = [
            r'$\delta T^{(2)} / T^{(2)}$',
            r'$\delta T^{(4)} / T^{(4)}$',
            r'$\delta g_1^{(2)}$',
            r'$\delta g_2^{(2)}$',
            r'$\delta e_1^{(4)}$',
            r'$\delta e_2^{(4)}$',
        ]
        vmaxs = [.0025, .02, .0002, .0002, .0002, .0002]
        ticks = [.002, .015, .00015, .00015, .00015, .00015]
    else:
        parameters = ['T', 'T4', 'g1', 'g2', 'e4_1', 'e4_2']
        labels = [
            r'$T^{(2)}$',
            r'$T^{(4)}$',
            r'$g_1^{(2)}$',
            r'$g_2^{(2)}$',
            r'$e_1^{(4)}$',
            r'$e_2^{(4)}$',
        ]
        vmaxs = [None, None, 5e-3, 5e-3, 1.5e-3, 1.5e-3]
        ticks = [None, None, 4e-3, 4e-3, 1e-3, 1e-3]
    for ax, param, label, vmax, tick in zip(
        a.flatten(),
        parameters,
        labels,
        vmaxs,
        ticks,
        ):

        if 'T' in param:
            sp = skyproj.McBrydeSkyproj(ax=ax, n_grid_lat=3)
            sp.draw_hpxbin(
                cat['ra'], cat['dec'], C=cat[param],
                zoom=True, xsize=600, cmap='magma',
                )

            cb = sp.draw_colorbar(
                label=label, location='top', pad=0.05,
                fontsize=9,
                )
        else:
            sp = skyproj.McBrydeSkyproj(ax=ax, n_grid_lat=3)
            sp.draw_hpxbin(
                cat['ra'], cat['dec'], C=cat[param],
                zoom=True, xsize=600, cmap='RdBu_r',
                vmin=-vmax, vmax=vmax,
                )

            cb = sp.draw_colorbar(
                label=label, location='top', pad=0.05,
                ticks=[-tick, 0, tick],
                fontsize=9,
                )
        sp.ax._ticklabels_visibility['top']=False

        if 't' not in param and 'T' not in param:
            cb.ax.ticklabel_format(style='sci', scilimits=(0,0), axis='both')
        if 'e4' not in param:
            sp.ax.set_xlabel('')
            sp.ax._ticklabels_visibility['bottom']=False
        else:
            sp.ax.set_xlabel('Right Ascension', fontsize=10)
        if param in ['dtt4', 'dg2', 'de4_2'] or param in ['T4', 'g2', 'e4_2']:
            sp.ax.set_ylabel('')
            sp.ax._ticklabels_visibility['left']=False
        else:
            sp.ax.set_ylabel('Declination', fontsize=10)

    # a[-1,1].axis('off')

    plt.subplots_adjust(top=0.95, bottom=0.05, right=0.95, left=0.2, wspace=0.1, hspace=0.1)
    plt.savefig(f'../figures/{"residual" if args.residual else "param"}-sky-maps.jpg', dpi=300)
    plt.show()
