import numpy as np
import datahelper
import plothelper
from colors import color_scheme as colors
import matplotlib.pyplot as plt
plt.style.use('./paper.mplstyle')

def plot_focalplane(
    cat,
    parameters,
    labels,
    vmaxs,
    cbarticks,
    whisker_properties,
    savepath,
    cat2=None
    ):
    a = make_mosaic_axes()

    for axkey, param, label, vmax, tick in zip(
        ['t_l', 't_r', 'e_l', 'e_r'],
        parameters,
        labels,
        vmaxs,
        cbarticks,
        ):
        if 'g' in param or 'e' in param:
            if cat2 is not None and 'r' in axkey:
                plotcat = cat2
            else:
                plotcat = cat
            # whisker plot
            plothelper.plot_whisker(
                a[axkey],
                plotcat,
                param,
                LSSTCAM,
                **whisker_properties
            )
        else:
            if cat2 is not None and 'r' in axkey:
                plotcat = cat2
            else:
                plotcat = cat

            # size plot
            plotcat['z'] = plotcat[param]

            if cat['reserved'].mean() == 1:
                bins = 151
            else:
                bins = 61

            plothelper.plot_fpbin_param(
                a[axkey],
                plotcat,
                LSSTCAM,
                vmin=None if 'd' not in param else -vmax,
                vmax=vmax,
                axcbar=a['cbar_'+axkey[-1]],
                numBins=bins,
                cbarlabel=label,
                cbartick=tick,
            )

    a['e_l'].text(
        0.1, 0.875, labels[2],
        horizontalalignment='center', verticalalignment='bottom',
        transform=a['e_l'].transAxes, fontsize=9
    )
    a['e_r'].text(
        0.1, 0.875, labels[3],
        horizontalalignment='center', verticalalignment='bottom',
        transform=a['e_r'].transAxes, fontsize=9
    )

    plt.subplots_adjust(left=0.025, right=0.975, bottom=0.025, top=0.9)

    plt.savefig(savepath, dpi=300)
    plt.show()

def plot_residual_skycoord(a, cat, savedir):
    import skyproj
    parameters = ['dtt', 'dtt4', 'dg1', 'dg2', 'de4_1', 'de4_2']
    labels = [
        r'$\delta T^{(2)} / T^{(2)}$',
        r'$\delta T^{(4)} / T^{(4)}$',
        r'$\delta g_1^{(2)}$',
        r'$\delta g_2^{(2)}$',
        r'$\delta e_1^{(4)}$',
        r'$\delta e_2^{(4)}$',
    ]
    vmaxs = [.002, .02, .00045, .00045, .00045, .00045]
    ticks = [.0015, .015, .0004, .0004, .0004, .0004]
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
                zoom=True, xsize=200, cmap=colors.cmap_s,
                nside=256,
                )

            cb = sp.draw_colorbar(
                label=label, location='top', pad=0.05,
                fontsize=9,
                )
        else:
            sp = skyproj.McBrydeSkyproj(ax=ax, n_grid_lat=3)
            sp.draw_hpxbin(
                cat['ra'], cat['dec'], C=cat[param],
                zoom=True, xsize=600, cmap=colors.cmap_d,
                vmin=-vmax, vmax=vmax,
                nside=256,
                )

            cb = sp.draw_colorbar(
                label=label, location='top', pad=0.05,
                ticks=[-tick, 0, tick],
                fontsize=9,
                )
        sp.ax._ticklabels_visibility['top']=False

        sp.draw_polygon([75, 65, 65, 75], [-35, -35, -45, -45], edgecolor='w', lw=1.)

        # if 't' not in param and 'T' not in param:
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

    plt.subplots_adjust(top=0.95, bottom=0.075, right=0.95, left=0.2, wspace=0.1, hspace=0.1)
    plt.savefig(savedir + 'sky-maps-residual' + '.jpg', dpi=300)
    plt.show()

def make_mosaic_axes():
    mosaic = [
        ['cbar_l', 'cbar_r'],
        ['t_l', 't_r'],
        ['e_l', 'e_r']
    ]
    f, a = plt.subplot_mosaic(
        mosaic,
        height_ratios=[0.03, 1, 1],
        figsize=(3.35, 3.75),
        gridspec_kw={'hspace': 0.02, 'wspace': 0.03}
    )

    [ax.set_yticks([]) for k, ax in a.items() if 'cbar' in k]
    [ax.set_yticklabels([]) for k, ax in a.items() if 'cbar' not in k]
    [ax.set_xticklabels([]) for k, ax in a.items() if 'cbar' not in k]
    [ax.set_aspect('equal') for k, ax in a.items() if 'cbar' not in k]

    return a

if __name__ == '__main__':
    savedir = '../figures/'

    from lsst.obs.lsst import LsstCam
    LSSTCAM = LsstCam.getCamera()

    cat = datahelper.load_catalog(
        '~/Documents/shear/testing-piff/data/cat-i-full-radec-03-piff-02.fits',
        reserved=False,
        fpcoords=True,
        trim=False,
    )

    # if printing statistics, format to close to tex table
    means = cat[['dtt', 'dtt4', 'dg1', 'dg2', 'de4_1', 'de4_2', 'reserved']].groupby('reserved').mean()
    stds = cat[['dtt', 'dtt4', 'dg1', 'dg2', 'de4_1', 'de4_2', 'reserved']].groupby('reserved').std()
    for col in ['dtt', 'dtt4', 'dg1', 'dg2', 'de4_1', 'de4_2']:
        print(f'& {means[col].iloc[0]:.2g} & {stds[col].iloc[0]:.2g} & {means[col].iloc[1]:.2g} & {stds[col].iloc[1]:.2g} \\ ')

    subcat = cat[['visit','visit_T']]
    grouped = subcat.groupby('visit').mean().sort_values('visit_T').reset_index()
    lowseeing = grouped.at[40, 'visit']
    highseeing = grouped.at[1450, 'visit']

    # PSF focal plane map for example exposures
    parameters = ['T', 'T', 'g', 'g']
    labels = [
        r'$T^{(2)}$ $($arcsec$^2)$',
        r'$T^{(2)}$ $($arcsec$^2)$',
        r'$g^{(2)}$',
        r'$g^{(2)}$',
    ]
    vmaxs = [None, None, 0, 0]
    ticks = [None, None, 0, 0]
    whisker_dict = {
        'scaling': 0.4,
        'keysize': 0.01,
        'fontsize': 5,
        'keyposition': (225, 250)
    }
    plot_focalplane(
        cat=cat.loc[cat.index[cat['visit']==lowseeing]],
        parameters=parameters,
        labels=labels,
        vmaxs=vmaxs,
        cbarticks=ticks,
        whisker_properties=whisker_dict,
        savepath=savedir+'focalplane-example.jpg',
        cat2=cat.loc[cat.index[cat['visit']==highseeing]],
    )

    ## residual version of the above
    parameters = ['dtt', 'dtt', 'dg', 'dg']
    labels = [
        r'$\delta T^{(2)} / T^{(2)}$',
        r'$\delta T^{(2)} / T^{(2)}$',
        r'$\delta g^{(2)}$',
        r'$\delta g^{(2)}$',
    ]
    vmaxs = [0.025, 0.025, 0, 0]
    whisker_dict['scaling'] = 0.09
    whisker_dict['keysize'] = 0.002
    plot_focalplane(
        cat=cat.loc[cat.index[cat['visit']==lowseeing]],
        parameters=parameters,
        labels=labels,
        vmaxs=vmaxs,
        cbarticks=ticks,
        whisker_properties=whisker_dict,
        savepath=savedir+'focalplane-example-residual.jpg',
        cat2=cat.loc[cat.index[cat['visit']==highseeing]],
    )
    # now use only reserve stars
    cat = cat[cat['reserved']==1]

    ## focal plane map of all residuals
    parameters = ['dtt', 'dtt4', 'dg', 'de4_']
    labels = [
        r'$\delta T^{(2)} / T^{(2)}$',
        r'$\delta T^{(4)} / T^{(4)}$',
        r'$\delta g^{(2)}$',
        r'$\delta e^{(4)}$',
    ]
    vmaxs = [0.002, 0.022, 0, 0]
    ticks = [0.0015, 0.015, 0, 0]
    whisker_dict = {
        'scaling': 0.01,
        'keysize': 0.001,
        'fontsize': 5,
        'keyposition': (225, 250)
    }
    plot_focalplane(
        cat=cat,
        parameters=parameters,
        labels=labels,
        vmaxs=vmaxs,
        cbarticks=ticks,
        whisker_properties=whisker_dict,
        savepath=savedir+'focalplane-residuals.jpg',
    )

    f, a = plt.subplots(3,2, figsize=(3.35,5.2))
    plot_residual_skycoord(a, cat, savedir)
