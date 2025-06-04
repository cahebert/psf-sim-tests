import numpy as np
import datahelper
import plothelper
from colors import color_scheme as colors
import matplotlib.pyplot as plt
plt.style.use('./paper.mplstyle')

def plot_seeing_examples(a, cat, savedir, residual=False):
    subcat = cat[['visit','visit_T']]
    grouped = subcat.groupby('visit').mean().sort_values('visit_T').reset_index()
    lowseeing = grouped.at[40, 'visit']
    highseeing = grouped.at[1450, 'visit']

    for ax, seeing in zip(a, [lowseeing, highseeing]):
        plothelper.plot_whisker(
            ax,
            cat.loc[cat.index[cat['visit']==seeing]],
            'dg' if residual else 'g',
            LSSTCAM,
            scaling=0.2 if residual else 0.5,
            keysize=0.01,
            fontsize=5,
            keyposition=(220,240)
        )
    [ax.set_yticklabels([]) for ax in a]
    [ax.set_xticklabels([]) for ax in a]

    plt.subplots_adjust(left=0.025, right=0.975, top=0.95, bottom=0.075)
    plt.savefig(savedir + 'focalplane-examples' + f'{"-residual" if residual else ""}' + '.jpg', dpi=300)
    plt.show()
    return a

def plot_residual_fp(a, cat, savedir):
    parameters = ['dtt', 'dtt4', 'dg', 'de4_']
    labels = [
        r'$\delta T^{(2)} / T^{(2)}$',
        r'$\delta T^{(4)} / T^{(4)}$',
        r'$\delta e^{(2)}$',
        r'$\delta e^{(4)}$',
    ]
    vmaxs = [0.002, 0.022, 0, 0]

    for axkey, param, label, vmax, tick in zip(
        ['t2', 't4', 'e2', 'e4'],
    parameters,
        labels,
        vmaxs,
        [.0015, .015, 0, 0]
        ):
        if 'g' in param or 'e' in param:
            # whisker plot
            plothelper.plot_whisker(
                a[axkey],
                cat,
                param,
                LSSTCAM,
                scaling=.01,
                keysize=0.001,
                fontsize=5,
                keyposition=(225,245)
            )
        else:
            # size plot
            cat['z'] = cat[param]

            plothelper.plot_fpbin_param(
                a[axkey],
                cat,
                LSSTCAM,
                vmin=0 if 'T' in param else -vmax,
                vmax=vmax,
                axcbar=a['cbar'+axkey[-1]],
                numBins=151,
                cbarlabel=label,
                cbartick=tick,
            )

    a['e2'].text(
        0.1, .875, r'$\delta e^{(2)}$',
        horizontalalignment='center', verticalalignment='bottom',
        transform=a['e2'].transAxes, fontsize=9
    )
    a['e4'].text(
        0.1, 0.875, r'$\delta e^{(4)}$',
        horizontalalignment='center', verticalalignment='bottom',
        transform=a['e4'].transAxes, fontsize=9
    )

    plt.subplots_adjust(left=0.025, right=0.975)

    plt.savefig(savedir + 'focalplane-residuals' + '.jpg', dpi=300)
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

    # a[-1,1].axis('off')

    plt.subplots_adjust(top=0.95, bottom=0.075, right=0.95, left=0.2, wspace=0.1, hspace=0.1)
    plt.savefig(savedir + 'sky-maps-residual' + '.jpg', dpi=300)
    plt.show()


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

    f, a = plt.subplots(
        1,2, figsize=(3.35, 2),
        sharex=True, sharey=True,
        gridspec_kw={'wspace': 0.01}
    )
    plot_seeing_examples(a, cat, savedir, residual=False)

    f, a = plt.subplots(
        1,2, figsize=(3.35, 2),
        sharex=True, sharey=True,
        gridspec_kw={'wspace': 0.02}
    )
    plot_seeing_examples(a, cat, savedir, residual=True)

    # now use only reserve stars
    cat = cat[cat['reserved']]

    # more complicated plot mosaic for this
    mosaic = [
        ['cbar2', 'cbar4'],
        ['t2', 't4'],
        ['e2', 'e4']
    ]
    f, a = plt.subplot_mosaic(
        mosaic,
        height_ratios=[0.05, 1, 1],
        figsize=(3.35, 4.25),
        gridspec_kw={'hspace': 0.01, 'wspace': 0.02}
    )

    [ax.set_yticks([]) for k, ax in a.items() if 'cbar' in k]
    [ax.set_yticklabels([]) for k, ax in a.items() if 'cbar' not in k]
    [ax.set_xticklabels([]) for k, ax in a.items() if 'cbar' not in k]
    [ax.set_aspect('equal') for k, ax in a.items() if 'cbar' not in k]

    plot_residual_fp(a, cat, savedir)


    # f, a = plt.subplots(3,2, figsize=(3.35,5.2))
    # plot_residual_skycoord(a, cat, savedir)
