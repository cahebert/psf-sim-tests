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
            scaling=0.5,
            keysize=0.01
        )
    a[0].set_ylabel('y (mm)', labelpad=-5)
    a[1].set_ylabel('y (mm)', labelpad=-5)
    a[1].set_xlabel('x (mm)')

    plt.subplots_adjust(left=0.15,right=0.95)
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
    vmaxs = [0.0025, 0.025, 0, 0]

    for ax, param, label, vmax in zip(
        a.flatten(),
        parameters,
        labels,
        vmaxs,
        ):
        if 'g' in param or 'e' in param:
            # whisker plot
            plothelper.plot_whisker(
                ax,
                cat,
                param,
                LSSTCAM,
                scaling=.01,
                keysize=0.0002,
                fontsize=5
            )
        else:
            # size plot
            cat['z'] = cat[param]

            plothelper.plot_fpbin_param(
                ax,
                cat,
                LSSTCAM,
                vmin=0 if 'T' in param else -vmax,
                vmax=vmax,
                numBins=151,
                cbarlabel=label
            )

    for ax in a.flatten():
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    # for ax in a[:,0]:
    #     ax.set_ylabel('y (mm)', labelpad=-7)
    # for ax in a[-1,:]:
    #     ax.set_xlabel('x (mm)')

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

    f, a = plt.subplots(2,1, figsize=(3.35, 5.8), sharex=True, sharey=True)
    plot_seeing_examples(a, cat, savedir, residual=False)

    f, a = plt.subplots(2,1, figsize=(3.35, 5.8), sharex=True, sharey=True)
    plot_seeing_examples(a, cat, savedir, residual=True)

    # now use only reserve stars
    cat = cat[cat['reserved']]
    f, a = plt.subplots(2,2, figsize=(3.35, 4.1), sharex=True, sharey=True)
    plot_residual_fp(a, cat, savedir)

    f, a = plt.subplots(3,2, figsize=(3.35,5.2))
    plot_residual_skycoord(a, cat, savedir)
