import matplotlib.pyplot as plt
import numpy as np
import json
from colors import color_scheme as colors
plt.style.use('./paper.mplstyle')

markers = ['o','D','s']

rho_labels = {'g2g2'   : r'$\langle e_2 e_2 \rangle$',
              'g2e4'   : r'$\langle e_2 e_4 \rangle$',
              'e4e4'   : r'$\langle e_4 e_4 \rangle$',
              'g2dg2'  : r'$\langle e_2 \delta e_2 \rangle$',
              'g2de4'  : r'$\langle e_2 \delta e_4 \rangle$',
              'e4dg2'  : r'$\langle e_4 \delta e_2 \rangle$',
              'e4de4'  : r'$\langle e_4 \delta e_4 \rangle$',
              'g2w22'  : r'$\langle e_2 w_{22} \rangle$',
              'g2w24'  : r'$\langle e_2 w_{24} \rangle$',
              'g2w42'  : r'$\langle e_2 w_{42} \rangle$',
              'g2w44'  : r'$\langle e_2 w_{44} \rangle$',
              'e4w22'  : r'$\langle e_4 w_{22} \rangle$',
              'e4w24'  : r'$\langle e_4 w_{24} \rangle$',
              'e4w42'  : r'$\langle e_4 w_{42} \rangle$',
              'e4w44'  : r'$\langle e_4 w_{44} \rangle$',
              'dg2dg2' : r'$\langle \delta e_2\delta e_2 \rangle$',
              'dg2de4' : r'$\langle \delta e_2\delta e_4 \rangle$',
              'de4de4' : r'$\langle \delta e_4\delta e_4 \rangle$',
              'dg2w22' : r'$\langle \delta e_2 w_{22} \rangle$',
              'dg2w24' : r'$\langle \delta e_2 w_{24} \rangle$',
              'dg2w42' : r'$\langle \delta e_2 w_{42} \rangle$',
              'dg2w44' : r'$\langle \delta e_2 w_{44} \rangle$',
              'de4w22' : r'$\langle \delta e_4 w_{22} \rangle$',
              'de4w24' : r'$\langle \delta e_4 w_{24} \rangle$',
              'de4w42' : r'$\langle \delta e_4 w_{42} \rangle$',
              'de4w44' : r'$\langle \delta e_4 w_{44} \rangle$',
              'w22w22' : r'$\langle w_{22} w_{22} \rangle$',
              'w22w24' : r'$\langle w_{22} w_{24} \rangle$',
              'w22w42' : r'$\langle w_{22} w_{42} \rangle$',
              'w22w44' : r'$\langle w_{22} w_{44} \rangle$',
              'w24w24' : r'$\langle w_{24} w_{24} \rangle$',
              'w24w42' : r'$\langle w_{24} w_{42} \rangle$',
              'w24w44' : r'$\langle w_{24} w_{44} \rangle$',
              'w42w42' : r'$\langle w_{42} w_{42} \rangle$',
              'w42w44' : r'$\langle w_{42} w_{44} \rangle$',
              'w44w44' : r'$\langle w_{44} w_{44} \rangle$',
               }

def plot_rho(ax, rho, label, marker, params):
    ax.plot(rho['meanr'], rho['xip'], '-', **params)
    ax.plot(rho['meanr'], -rho['xip'], ':', **params)
    ax.errorbar(
        rho['meanr'][rho['xip']>0],
        rho['xip'][rho['xip']>0],
        yerr=np.sqrt(rho['xip_var'][rho['xip']>0]),
        fmt=marker, label=label, **params
    )
    ax.errorbar(
        rho['meanr'][rho['xip']<0],
        -rho['xip'][rho['xip']<0],
        yerr=np.sqrt(rho['xip_var'][rho['xip']<0]),
        fmt=marker, mfc='none', mew=0.75, **params
    )
    return ax

def mean_rho(rho, lb=5, ub=100):
    points = (rho['meanr']>lb)&(rho['meanr']<ub)
    result = np.mean(rho['xip'][points])
    return result

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
    for ax, rhokey in zip(a, ['dg2dg2', 'g2dg2', 'g2w22']):
        ax.grid(which='major', color='lightgrey', alpha=0.7, lw=0.5, zorder=0)
        for rho, color, marker, label in zip(
            [rhohomy10[rhokey], rhohomy1[rhokey]],
            color_list,
            markers,
            ['Y10', 'Y1']
        ):
            plot_rho(ax, rho, label, marker, {'color':color, 'alpha':1, 'zorder':3})
            ax.set_ylabel(rho_labels[rhokey]+r'$(\theta)$')
    a[0].legend(ncols=2, handletextpad=0.05, columnspacing=1, edgecolor='lightgrey', borderaxespad=0.3)

    [ax.set_yscale('log', nonpositive='clip') for ax in a.flatten()]

    a[1].set_xscale('log')
    a[1].set_xlim(.6, 200)
    a[-1].set_xlabel(r'$\theta$ (arcmin)')
    a[1].set_ylim(bottom=5e-11)
    plt.subplots_adjust(hspace=0)
    plt.savefig('../figures/rho-examples-hom-i-full-radec-03-piff-02.jpg', dpi=300)
    # plt.show()

def plot_rho_summary(
    rhohomy10,
    rhohomy1,
    a,
    labels,
    color_list,
    title,
    coefficients=False,
    significant=None
    ):
    bad = ['nvisits']
    alpha2_terms = ['g2g2', 'g2e4', 'e4e4']
    alpha_terms = ['g2dg2', 'g2de4', 'e4dg2', 'e4de4', 'g2w22', 'g2w24', 'g2w42', 'g2w44', 'e4w22', 'e4w24', 'e4w42', 'e4w44']

    if coefficients:
        with open('../data/coeffs-i-full-radec-03-piff-02.json', 'r') as f:
            coefficients = json.load(f)
        coeffy10 = coefficients['y10']
        coeffy1 = coefficients['y1']
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

    if significant is not None:
        xpos = []
        for s in significant:
            xpos += [i for i,x in enumerate(xlabels) if rho_labels[s]==x]

        [plt.setp(a.get_xticklabels()[i], color=colors.y) for i in xpos]

    a.set_yscale('log', nonpositive='clip')

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], ls='', marker='D', alpha=0.9, color=color_list[1], label=labels[1], markerfacecolor=color_list[1]),
        Line2D([0], [0], ls='', marker='o', color=color_list[0], label=labels[0], markerfacecolor=color_list[0]),
        ]

    if coefficients:
        a.legend(handles=handles, loc='upper right', frameon=True, edgecolor='lightgrey')
        a.set_ylim(bottom=1e-13, top=1e-7)
        a.set_ylabel(r"$c_i\langle \rho(\theta)\rangle_{0.5'-50'}$")
    else:
        a.legend(handles=handles, loc='upper left', frameon=True, edgecolor='lightgrey')
        a.set_ylim(bottom=1e-13, top=1e-5)
        a.set_ylabel(r"$\langle \rho(\theta)\rangle_{0.5'-50'}$")

    # [a.axhline(10**x, color='lightgrey', alpha=0.5, zorder=1) for x in [-6, -8, -10, -12]]
    a.set_xlim(-0.5, 35.5)
    plt.savefig('../figures/' + title + '.jpg', dpi=300)
    plt.show()


def load_rho_samples(filename, n):
    with open(filename, 'rb') as f:
        rhos = json.load(f)

    rho_summary = {}
    error_norm = 1506/n  # divide by the number of independent samples

    for k, rho in rhos.items():
        if n==1506:
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

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', type=str, default='../data/')
    args = parser.parse_args()

    # # examples
    rhohomy10 = load_rho_samples(
        args.datapath+'/rho-i-full-radec-03-piff-02-cov1506.json',
        n=1506
    )
    rhohomy1 = load_rho_samples(
        args.datapath+'/rho-i-full-radec-03-piff-02-50x150.json',
        n=150
    )

    f, a = plt.subplots(3,1, figsize=(3.35,4.5), sharex=True, sharey=True)
    plot_rho_examples(
        rhohomy10,
        rhohomy1,
        a,
        color_list=[colors.g, colors.p, colors.y],
    )

    # summary

    # y10 vs y1
    rhohomy10 = load_rho_samples(
        args.datapath+'/rho-i-full-radec-03-piff-02-summary-cov1506.json',
        n=1506
    )

    rhohomy1 = load_rho_samples(
        args.datapath+'/rho-i-full-radec-03-piff-02-summary-50x150.json',
        n=150
    )

    # significant = []
    # for k, v in rhohomy1.items():
    #     pm_sig_y1 = v['xip'] - v['xip_var'], v['xip'] + v['xip_var']
    #     pm_sig_y10 = rhohomy10[k]['xip'] - rhohomy10[k]['xip_var'], rhohomy10[k]['xip'] + rhohomy10[k]['xip_var']

    #     if pm_sig_y1[0] > pm_sig_y10[1] or pm_sig_y1[1] < pm_sig_y10[0]:
    #         significant.append(k)
    # print("significant terms: ", significant)

    # y1toy10, y1toy10_sigma = fit_amplitude_scaling(rhohomy10, rhohomy1)
    # print("y1 to y10 scaling: ", y1toy10, "+/-", y1toy10_sigma)

    f, a = plt.subplots(1,1, figsize=(7.5, 3))
    plot_rho_summary(
        rhohomy10,
        rhohomy1,
        a,
        labels=['Y10 $i$', 'Y1 $i$'],
        color_list=[colors.g, colors.p, colors.y],
        title='rho-summary-i-full-radec-03-piff-02',
        coefficients=False,
    )

    f, a = plt.subplots(1,1, figsize=(7.5, 3))
    plot_rho_summary(
        rhohomy10,
        rhohomy1,
        a,
        labels=['Y10 $i$', 'Y1 $i$'],
        color_list=[colors.g, colors.p, colors.y],
        title='rho-c-summary-i-full-radec-03-piff-02',
        coefficients=True,
    )

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
