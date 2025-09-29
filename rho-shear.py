import matplotlib.pyplot as plt
import numpy as np
import json

import datahelper

from colors import color_scheme as colors
plt.style.use('./paper.mplstyle')

markers = ['s','D','o']

rho_labels = {'e2e2'   : r'$\langle e^{(2)} e^{(2)} \rangle$',
              'e2e4'   : r'$\langle e^{(2)} e^{(4)} \rangle$',
              'e4e4'   : r'$\langle e^{(4)} e^{(4)} \rangle$',
              'e2de2'  : r'$\langle e^{(2)} \delta e^{(2)} \rangle$',
              'e2de4'  : r'$\langle e^{(2)} \delta e^{(4)} \rangle$',
              'e4de2'  : r'$\langle e^{(4)} \delta e^{(2)} \rangle$',
              'e4de4'  : r'$\langle e^{(4)} \delta e^{(4)} \rangle$',
              'e2w22'  : r'$\langle e^{(2)} w_{22} \rangle$',
              'e2w24'  : r'$\langle e^{(2)} w_{24} \rangle$',
              'e2w42'  : r'$\langle e^{(2)} w_{42} \rangle$',
              'e2w44'  : r'$\langle e^{(2)} w_{44} \rangle$',
              'e4w22'  : r'$\langle e^{(4)} w_{22} \rangle$',
              'e4w24'  : r'$\langle e^{(4)} w_{24} \rangle$',
              'e4w42'  : r'$\langle e^{(4)} w_{42} \rangle$',
              'e4w44'  : r'$\langle e^{(4)} w_{44} \rangle$',
              'de2de2' : r'$\langle \delta e^{(2)}\delta e^{(2)} \rangle$',
              'de2de4' : r'$\langle \delta e^{(2)}\delta e^{(4)} \rangle$',
              'de4de4' : r'$\langle \delta e^{(4)}\delta e^{(4)} \rangle$',
              'de2w22' : r'$\langle \delta e^{(2)} w_{22} \rangle$',
              'de2w24' : r'$\langle \delta e^{(2)} w_{24} \rangle$',
              'de2w42' : r'$\langle \delta e^{(2)} w_{42} \rangle$',
              'de2w44' : r'$\langle \delta e^{(2)} w_{44} \rangle$',
              'de4w22' : r'$\langle \delta e^{(4)} w_{22} \rangle$',
              'de4w24' : r'$\langle \delta e^{(4)} w_{24} \rangle$',
              'de4w42' : r'$\langle \delta e^{(4)} w_{42} \rangle$',
              'de4w44' : r'$\langle \delta e^{(4)} w_{44} \rangle$',
              'w22w22' : r'$\langle w_{22} w_{22} \rangle$',
              'w22w24' : r'$\langle w_{22} w_{24} \rangle$',
              'w22w42' : r'$\langle w_{22} w_{42} \rangle$',
              'w22w44' : r'$\langle w_{22} w_{44} \rangle$',
              'w24w24' : r'$\langle w_{24} w_{24} \rangle$',
              'w24w42' : r'$\langle w_{24} w_{42} \rangle$',
              'w24w44' : r'$\langle w_{44} w_{24} \rangle$',
              'w42w42' : r'$\langle w_{42} w_{42} \rangle$',
              'w42w44' : r'$\langle w_{44} w_{42} \rangle$',
              'w44w44' : r'$\langle w_{44} w_{44} \rangle$',
               }

def plot_rho(ax, rho, label, marker, params, c=1):
    line2dparams = {k:p for k,p in params.items() if 'cap' not in k}
    ax.plot(rho['meanr'], rho['xip']*c, '-', **line2dparams)
    ax.plot(rho['meanr'], -rho['xip']*c, ':', **line2dparams)
    ax.errorbar(
        rho['meanr'][rho['xip']*c>0],
        rho['xip'][rho['xip']*c>0]*c,
        yerr=np.sqrt(rho['xip_var'][rho['xip']*c>0])*abs(c),
        fmt=marker, label=label, **params
    )
    ax.errorbar(
        rho['meanr'][rho['xip']*c<0],
        -rho['xip'][rho['xip']*c<0]*c,
        yerr=np.sqrt(rho['xip_var'][rho['xip']*c<0])*abs(c),
        fmt=marker, mfc='none', **params
    )
    return ax

def fit_amplitude_scaling(rhoy1, rhoy10):
    from scipy import odr
    y10_from_y1 = odr.Model(lambda p, x: p[0] * x)

    y1_stats = np.array([v['xip'] for k,v in rhoy1.items()])
    y10_stats = np.array([v['xip'] for k,v in rhoy10.items()])
    y1_errs = np.array([np.sqrt(v['xip_var']) for k,v in rhoy1.items()])
    y10_errs = np.array([np.sqrt(v['xip_var']) for k,v in rhoy10.items()])

    data = odr.RealData(x=abs(y10_stats), y=abs(y1_stats), sx=y10_errs, sy=y1_errs)
    myodr = odr.ODR(data, y10_from_y1, beta0=[1.0])
    myoutput = myodr.run()
    return myoutput.beta[0], myoutput.sd_beta[0]

def plot_rho_examples(rhohomy10, rhohomy1, a, color_list):
    for ax, rhokey in zip(a, ['de2de2', 'de2de4', 'de2w22']):
        ax.grid(which='major', color='lightgrey', alpha=0.7, lw=0.5, zorder=0)
        for rho, color, marker, label in zip(
            [rhohomy10[rhokey], rhohomy1[rhokey]],
            color_list,
            markers,
            ['Y10', 'Y1']
        ):
            plot_rho(
                ax, rho, label, marker, {'color':color, 'alpha':1, 'mew':0.75, 'zorder':3}
                )
            ax.set_ylabel(rho_labels[rhokey]+r'$(\theta)$')
    a[0].legend(ncols=2, handletextpad=0.05, columnspacing=1, edgecolor='lightgrey', borderaxespad=0.3)

    [ax.set_yscale('log', nonpositive='clip') for ax in a.flatten()]

    a[1].set_xscale('log')
    a[1].set_xlim(.6, 200)
    a[-1].set_xlabel(r'$\theta$ (arcmin)')
    a[1].set_ylim(bottom=5e-11)
    plt.subplots_adjust(hspace=0)
    plt.savefig('../figures/rho-examples-hom-i-full-radec-04-piff-01-cornercut.jpg', dpi=300)
    # plt.show()

def plot_rho_summary(
    rhohomy10,
    rhohomy1,
    a,
    labels,
    color_list,
    title,
    coefficients=False,
    ):
    bad = ['nvisits']
    alpha2_terms = ['e2e2', 'e2e4', 'e4e4']
    alpha_terms = ['e2de2', 'e2de4', 'e4de4', 'e4dg2', 'e4w22', 'e4w24', 'e4w42', 'e4w44']

    if coefficients:
        with open('../data/coeffs-i-full-radec-04-piff-01.json', 'r') as f:
            coefficients = json.load(f)
        coeffy10 = coefficients['y10']['regular']
        coeffy1 = coefficients['y1']['regular']
        # if we apply coefficients on everything, all terms are included in first pass
        noalpha_terms = [k for k in rhohomy10.keys() if k not in bad]
    else:
        coeffy10 = {k:1 for k in rhohomy10.keys()}
        coeffy1 = {k:1 for k in rhohomy1.keys()}
        noalpha_terms = [k for k in rhohomy10.keys() if k not in alpha_terms+alpha2_terms+bad]

    sorted = {}
    xlabels = []

    for rhohom, transparancy, c, coeffs, fmt in zip(
        [rhohomy10, rhohomy1],
        [1, 0.75],
        color_list,
        [coeffy10, coeffy1],
        ['o', 'D']
    ):
        a.grid(which='major', axis='y', color='lightgrey', alpha=0.7, lw=0.5, zorder=0)
        n_start = 0
        params = {'color': c, 'alpha':transparancy, 'capsize': 1.5, 'capthick': 0.5, 'zorder':3}

        for alpha_factor, terms in zip(
            [0,1,2] if not coefficients else [0],
            [noalpha_terms, alpha_terms, alpha2_terms]
        ):
            means = np.array(
                [rhohom[rho]['xip'] * coeffs[rho] for rho in terms]
            ).flatten()
            sigma = np.array(
                [np.sqrt(rhohom[rho]['xip_var']) * abs(coeffs[rho]) for rho in terms]
            ).flatten()

            if transparancy==1:
                # means we're doing y10; sort by these values
                sorted[alpha_factor] = np.argsort(abs(means))[::-1]
                xlabels += [rho_labels[k] for k in np.array(terms)[sorted[alpha_factor]]]
            pltpts = np.arange(len(terms))+n_start

            pos = means[sorted[alpha_factor]] > 0
            a.errorbar(
                pltpts[pos],
                means[sorted[alpha_factor]][pos],
                yerr=sigma[sorted[alpha_factor]][pos],
                fmt=fmt,
                **params,
            )
            a.errorbar(
                pltpts[~pos],
                -means[sorted[alpha_factor]][~pos],
                yerr=sigma[sorted[alpha_factor]][~pos],
                fmt=fmt,
                mfc='none',
                mew=0.75,
                **params
            )

            if alpha_factor>0:
                a.errorbar(
                    pltpts[pos],
                    means[sorted[alpha_factor]][pos]*(0.025**alpha_factor),
                    yerr=sigma[sorted[alpha_factor]][pos]*(0.025**alpha_factor),
                    fmt='s',
                    **params,
                )
                a.errorbar(
                    pltpts[~pos],
                    -means[sorted[alpha_factor]][~pos]*(0.025**alpha_factor),
                    yerr=sigma[sorted[alpha_factor]][~pos]*(0.025**alpha_factor),
                    fmt= 's',
                    mfc='none',
                    mew=0.75,
                    **params,
                )

                a.fill_betweenx([0,1], n_start-0.5, len(rhohom.keys())-0.5, alpha=0.05, color='k')

            n_start += len(terms)

    a.set_xticks(
        np.arange(len(rhohomy1.keys())),
        labels=xlabels,
        rotation=65,
        ha="right",
        rotation_mode='anchor'
    )

    a.set_yscale('log', nonpositive='clip')

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], ls='', marker='D', alpha=0.9, color=color_list[1], label=labels[1], markerfacecolor=color_list[1]),
        Line2D([0], [0], ls='', marker='o', color=color_list[0], label=labels[0], markerfacecolor=color_list[0]),
        ]

    if coefficients:
        a.legend(handles=handles, loc='upper right', frameon=True, edgecolor='lightgrey')
        a.set_ylim(bottom=1e-13, top=1e-7)
        a.set_ylabel(r"$c_ic_j\langle P_iP_j\rangle_{0.5'-50'}$")
    else:
        a.legend(handles=handles, loc='upper left', frameon=True, edgecolor='lightgrey')
        a.set_ylim(bottom=1e-13, top=1e-5)
        a.set_ylabel(r"$\langle P_iP_j\rangle_{0.5'-50'}$")

    a.set_xlim(-0.5, 35.5)
    plt.savefig('../figures/' + title + '.jpg', dpi=300)
    plt.show()

def plot_all_rho_c(rhoy10, rhoy1, a):
    with open('../data/coeffs-i-full-radec-04-piff-01-empirical.json', 'r') as f:
        coefficients = json.load(f)
    coeffy10 = coefficients['y10']
    coeffy1 = coefficients['y1']
    parameters = ['e2','e4','de2','de4','w22','w24','w42','w44']
    maxs = []
    rs = []
    for pi in parameters:
        for pj in parameters:
            r = pi + pj
            if r in rhoy10.keys():
                maxs.append(np.max(abs(coeffy10[r] * rhoy10[r]['xip'])))
                rs.append(r)
    sorted_ind = np.argsort(maxs)[::-1]

    for ax, ri in zip(a.flatten(), sorted_ind):
        r = rs[ri]
        ax.grid(which='major', color='lightgrey', alpha=0.7, lw=0.5, zorder=0)
        for rho, color, marker, alpha, label, c in zip(
            [rhoy10[r], rhoy1[r]],
            [colors.g, colors.y],
            markers, [1, 0.75],
            ['Y10', 'Y1'], [coeffy10[r], coeffy1[r]]
        ):
            # plot_rho(
            #     ax, rho, label, marker, 
            #     params={'color':color, 'alpha':alpha, 'zorder':3, 'ms':3, 'mew': 0.65,  'capsize': 1.5, 'capthick': 0.5})
            # ax.text(
            #     0.95, .925, rho_labels[r], transform=ax.transAxes,
            #     verticalalignment='top', horizontalalignment='right'
            #     )
            plot_rho_band(
                ax, rho, label, c=c,
                params={'marker':marker, 'color':color, 'alpha':alpha,
                        'zorder':3, 'lw':0, 'ms':2, 'mew': 0.5})
            ax.text(
                0.95, 0.925, rho_labels[r], transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='right'
            )

    [ax.set_yscale('log', nonpositive='clip') for ax in a.flatten()]
    [ax.set_xlabel(r'$\theta$ (arcmin)') for ax in a[-1]]
    a[-1,-1].legend(
        edgecolor='lightgrey', borderaxespad=0.1, loc='center left', reverse=True)

    a[0,0].set_xscale('log')
    a[0,0].set_xlim(.6, 50)
    a[0,0].set_ylim(top=2e-6, bottom=2e-13)
    a[0,0].set_yticks([1e-12, 1e-10, 1e-8, 1e-6])
    a[1,0].set_ylabel(r"$c_ic_j\langle P_iP_j\rangle (\theta)$")

    plt.savefig('../figures/rho-c-all-i-full-radec-04-piff-01-cornercut-empirical.jpg', dpi=300)
    plt.show()
    return a

def plot_rho_band(ax, rho, label, params, c=1):
    line2dparams = {k:p for k,p in params.items() if 'cap' not in k}
    ax.plot(rho['meanr'], rho['xip']*c, **line2dparams, label=label)
    ax.plot(rho['meanr'], -rho['xip']*c, mfc='none', **line2dparams)

    ax.fill_between(
        rho['meanr'],
        abs(rho['xip']*c) - np.sqrt(rho['xip_var'])*abs(c),
        abs(rho['xip']*c) + np.sqrt(rho['xip_var'])*abs(c),
        alpha=0.2, color=params['color'], zorder=2
    )
    return ax

def plot_all_rho(rhoy10, rhoy1, a,
                 pi_list=None, pj_list=None, title='rho-all-i-full-radec-04-piff-01'):
    if pi_list is None:
        pi_list = ['de2','de4','w22','w44','w24','w42']
    if pj_list is None:
        pj_list = ['de2','de4','w22','w44','w24','w42']

    a = a.T
    for i, pi in enumerate(pi_list):
        for j, pj in enumerate(pj_list):
            r = pi + pj
            if r == 'w24w44': r = 'w44w24'
            elif r == 'w42w44': r = 'w44w42'
            elif r == 'w44w24': r = 'w24w44'
            elif r == 'w44w42': r = 'w42w44'
            if len(pi_list)>len(pj_list):
                r = pj + pi
            ax = a[i,j]
            if r in rhoy10.keys():
                ax.grid(which='major', color='lightgrey', alpha=0.7, lw=0.5, zorder=0)
                for rho, color, marker, alpha, label in zip(
                    [rhoy10[r], rhoy1[r]],
                    [colors.g, colors.y],
                    markers, [1, 0.9],
                    ['Y10', 'Y1']
                ):
                    plot_rho_band(
                        ax, rho, label,
                        params={'marker':marker, 'color':color, 'alpha':alpha,
                                'zorder':3, 'lw':0, 'ms':2, 'mew': 0.5})
                    tx, ty = (0.95, 0.25) if r in ['e2e2', 'e4e4', 'e2e4'] else (0.95, .925)
                    ax.text(
                        tx, ty, rho_labels[r], transform=ax.transAxes,
                        verticalalignment='top', horizontalalignment='right'
                    )
            else:
                ax.axis('off')

    [ax.set_yscale('log', nonpositive='clip') for ax in a.flatten()]
    nx = round(len(pi_list)/2)
    ny = round(len(pj_list)/2)
    a[nx,-1].set_xlabel(r'$\theta$ (arcmin)',x=0)
    a[0,ny].set_ylabel(r"$\langle P_iP_j\rangle (\theta)$", y=1)
    if len(pi_list)>len(pj_list):
        a[1,0].legend(
            reverse=True,handletextpad=0.,
            loc='center right', bbox_to_anchor=(0, 0.5))

    else:
        a[0,0].legend(
            reverse=True,handletextpad=0.,
            loc='center left', bbox_to_anchor=(0.85, 0.5))

    a[0,0].set_xscale('log')
    a[0,0].set_xlim(.5, 50)

    a[0,0].set_ylim(bottom=2e-13)
    a[0,0].set_yticks([1e-12, 1e-10, 1e-8, 1e-6], labels=[r'$10^{-12}$', r'$10^{-10}$', r'$10^{-8}$', r'$10^{-6}$'])

    plt.savefig(f'../figures/{title}.jpg', dpi=300)
    plt.show()
    return a

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', type=str, default='../data/')
    args = parser.parse_args()

    # rho(theta)
    rhohomy10 = datahelper.load_rho_samples(
        args.datapath+'/patch35-rho-i-full-radec-04-piff-01-cornercut-cov1508.json',
        n=1508
    )
    rhohomy1 = datahelper.load_rho_samples(
        args.datapath+'/rho-i-full-radec-04-piff-01-cornercut-50x150.json',
        n=150
    )

    # full set
    # f, a = plt.subplots(
    #     3,7, figsize=(7.5, 3.5),
    #     sharex=True, sharey=True,
    #     gridspec_kw={'hspace': 0, 'wspace': 0}
    #     )
    # plot_all_rho_c(
    #     rhohomy10,
    #     rhohomy1,
    #     a,
    # )
    f, a = plt.subplots(
        6,6, figsize=(7.5, 5),
        sharex=True, sharey=True,
        gridspec_kw={'hspace': 0, 'wspace': 0}
        )
    plot_all_rho(
        rhohomy10,
        rhohomy1,
        a,
        title='rho-all-i-full-radec-04-piff-01-cornercut'
    )
    f, a = plt.subplots(
        2,8, figsize=(7.5, 2.5),
        sharex=True, sharey=True,
        gridspec_kw={'hspace': 0, 'wspace': 0}
        )
    plot_all_rho(
        rhohomy10,
        rhohomy1,
        a,
        pj_list=['e4','e2'],
        pi_list=['e2','e4','de2','de4','w22','w24','w42','w44'],
        title='rho-ge-i-full-radec-04-piff-01-cornercut'
    )

    # # examples
    # f, a = plt.subplots(3,1, figsize=(3.35,4.5), sharex=True, sharey=True)
    # plot_rho_examples(
    #     rhohomy10,
    #     rhohomy1,
    #     a,
    #     color_list=[colors.g, colors.p, colors.y],
    # )

    # summary
    # rhohomy10 = datahelper.load_rho_samples(
    #     args.datapath+'/rho-i-full-radec-04-piff-01-cornercut-summary-cov1508.json',
    #     n=1508
    # )

    # rhohomy1 = datahelper.load_rho_samples(
    #     args.datapath+'/rho-i-full-radec-04-piff-01-cornercut-summary-50x150.json',
    #     n=150
    # )

    # f, a = plt.subplots(1,1, figsize=(7.5, 3))
    # plot_rho_summary(
    #     rhohomy10,
    #     rhohomy1,
    #     a,
    #     labels=['Y10 $i$', 'Y1 $i$'],
    #     color_list=[colors.g, colors.p, colors.y],
    #     title='rho-summary-i-full-radec-04-piff-01-cornercut',
    #     coefficients=False,
    # )

    # f, a = plt.subplots(1,1, figsize=(7.5, 3))
    # plot_rho_summary(
    #     rhohomy10,
    #     rhohomy1,
    #     a,
    #     labels=['Y10 $i$', 'Y1 $i$'],
    #     color_list=[colors.g, colors.p, colors.y],
    #     title='rho-c-summary-i-full-radec-04-piff-01-cornercut-regular',
    #     coefficients=True,
    # )

    # # big vs small psfs
    # with open(args.datapath+'/rho-i-full-radec-03-piff-02-fwhmcutleq0.8-summary-cov567.json', 'rb') as f:
    #     rhohomsmall = json.load(f)
    # with open(args.datapath+'/rho-i-full-radec-03-piff-02-fwhmcutgg0.8-summary-cov567.json', 'rb') as f:
    #     rhohombig = json.load(f)

    # f, a = plt.subplots(1,1, figsize=(7.5, 3))
    # plot_rho_summary(
    #     rhohombig,
    #     rhohomsmall,
    #     a,
    #     labels=['fwhm$>0.8$', 'fwhm$\leq0.8$'],
    #     color_list=[colors.g, colors.p],
    #     title='rho-summary-hom-i-full-radec-03-piff-02-jk-fwhmcut'
    # )
