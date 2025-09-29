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
        if 'e' in param:
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
                axcbar=a['cbar_'+axkey],
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
                bins = 12
            else:
                bins = 5

            plothelper.plot_fpbin_param(
                a[axkey],
                plotcat,
                LSSTCAM,
                vmin=None if 'd' not in param else -vmax,
                vmax=vmax,
                axcbar=a['cbar_'+axkey],
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

    plt.subplots_adjust(left=0.025, right=0.975, bottom=0.1, top=0.9)

    plt.savefig(savepath, dpi=300)
    plt.show()

def plot_residual_skycoord(a, cat, savedir):
    import skyproj
    parameters = ['dtt', 'dtt4', 'de2_1', 'de2_2', 'de4_1', 'de4_2']
    labels = [
        r'$\delta T^{(2)} / T^{(2)}$',
        r'$\delta T^{(4)} / T^{(4)}$',
        r'$\delta e_1^{(2)}$',
        r'$\delta e_2^{(2)}$',
        r'$\delta e_1^{(4)}$',
        r'$\delta e_2^{(4)}$',
    ]
    vmaxs = [.002, .02, .00055, .00055, .00055, .00055]
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

        sp.draw_polygon([75, 65, 65, 75], [-35, -35, -45, -45], edgecolor='w', lw=1.25, zorder=10)

        # if 't' not in param and 'T' not in param:
        cb.ax.ticklabel_format(style='sci', scilimits=(0,0), axis='both')
        if 'e4' not in param:
            sp.ax.set_xlabel('')
            sp.ax._ticklabels_visibility['bottom']=False
        else:
            sp.ax.set_xlabel('Right Ascension', fontsize=10)
        if param in ['dtt4', 'de2_2', 'de4_2'] or param in ['T4', 'e2_2', 'e4_2']:
            sp.ax.set_ylabel('')
            sp.ax._ticklabels_visibility['left']=False
        else:
            sp.ax.set_ylabel('Declination', fontsize=10)

    plt.subplots_adjust(top=0.95, bottom=0.075, right=0.95, left=0.2, wspace=0.1, hspace=0.1)
    plt.savefig(savedir + 'sky-maps-residual' + '.jpg', dpi=300)
    plt.show()

def make_mosaic_axes():
    mosaic = [
        ['cbar_t_l', 'cbar_t_r'],
        ['t_l', 't_r'],
        ['e_l', 'e_r'],
        ['cbar_e_l', 'cbar_e_r'],
    ]
    f, a = plt.subplot_mosaic(
        mosaic,
        height_ratios=[0.03, 1, 1, 0.03],
        figsize=(3.35, 4.15),
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
        '~/Documents/git/testing-piff-paper/data/cat-i-full-radec-04-piff-01.fits',
        reserved=False,
        fpcoords=True,
        trim=False,
        other_cols=['am_s2n']
    )

    ## slightly fewer bins for single exps, way smaller whiskers for ind exp residuals

    # # if printing statistics, format to close to tex table
    # means = cat[['dtt', 'dtt4', 'dg1', 'dg2', 'de4_1', 'de4_2', 'reserved']].groupby('reserved').mean()
    # stds = cat[['dtt', 'dtt4', 'dg1', 'dg2', 'de4_1', 'de4_2', 'reserved']].groupby('reserved').std()
    # for col in ['dtt', 'dtt4', 'dg1', 'dg2', 'de4_1', 'de4_2']:
    #     print(f'& {means[col].loc[0]:.3g} & {stds[col].loc[0]:.3g} & {means[col].loc[1]:.3g} & {stds[col].loc[1]:.3g} \\ ')

    subcat = cat[['visit','T']]
    grouped = subcat.groupby('visit').mean().sort_values('T').reset_index()
    lowseeing = grouped.at[30, 'visit']
    highseeing_list = grouped['visit'].loc[[1248, 1252, 1242, 1246, 1261]].values
    # 1242, 1246, 1248, 1252, 1261
    # PSF focal plane map for LOW seeing example exposure
    parameters = ['T', 'T4', 'e2', 'e4']
    labels = [
        r'$T^{(2)}$ $($arcsec$^2)$',
        r'$T^{(4)}$ $($arcsec$^2)$',
        r'$e^{(2)}$',
        r'$e^{(4)}$',
    ]
    vmaxs = [None, None, 0, 0]
    ticks = [None, None, 0, 0]
    whisker_dict = {
        'scaling': 0.5,
        'keysize': 0.01,
        'fontsize': 5,
        'keyposition': (230,250)
    }
    plot_focalplane(
        cat=cat.loc[cat.index[cat['visit']==lowseeing]],
        parameters=parameters,
        labels=labels,
        vmaxs=vmaxs,
        cbarticks=ticks,
        whisker_properties=whisker_dict,
        savepath=savedir+'focalplane-example-low.jpg',
    )
    # PSF focal plane map for HIGH seeing example exposure
    for i, v in enumerate(highseeing_list):
        plot_focalplane(
            cat=cat.loc[cat.index[cat['visit']==v]],
            parameters=parameters,
            labels=labels,
            vmaxs=vmaxs,
            cbarticks=ticks,
            whisker_properties=whisker_dict,
            savepath=savedir+f'focalplane-example-high-{i}.jpg',
        )

    ## residual version of the above
    parameters = ['dtt', 'dtt4', 'de2', 'de4']
    labels = [
        r'$\delta T^{(2)} / T^{(2)}$',
        r'$\delta T^{(4)} / T^{(4)}$',
        r'$\delta e^{(2)}$',
        r'$\delta e^{(4)}$',
    ]
    vmaxs = [0.0125, 0.125, 0, 0]
    whisker_dict['scaling'] = 0.25
    whisker_dict['keysize'] = 0.005
    plot_focalplane(
        cat=cat.loc[cat.index[cat['visit']==lowseeing]],
        parameters=parameters,
        labels=labels,
        vmaxs=vmaxs,
        cbarticks=ticks,
        whisker_properties=whisker_dict,
        savepath=savedir+'focalplane-example-low-residual.jpg',
    )
    for i, v in enumerate(highseeing_list):
        plot_focalplane(
            cat=cat.loc[cat.index[cat['visit']==v]],
            parameters=parameters,
            labels=labels,
            vmaxs=vmaxs,
            cbarticks=ticks,
            whisker_properties=whisker_dict,
            savepath=savedir+f'focalplane-example-high-residual-{i}.jpg',
        )
    # now use only reserve stars
    cat = cat[cat['reserved']==1]

    # focal plane map of all residuals
    parameters = ['dtt', 'dtt4', 'de2', 'de4']
    labels = [
        r'$\delta T^{(2)} / T^{(2)}$',
        r'$\delta T^{(4)} / T^{(4)}$',
        r'$\delta e^{(2)}$',
        r'$\delta e^{(4)}$',
    ]
    vmaxs = [0.0015, 0.015 , 0, 0]
    ticks = [0.001, 0.01, 0, 0]
    whisker_dict = {
        'scaling': 0.0175,
        'keysize': 0.001,
        'fontsize': 5,
        'keyposition': (230,250)
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

    # f, a = plt.subplots(3,2, figsize=(3.35,5.2))
    # plot_residual_skycoord(a, cat, savedir)
