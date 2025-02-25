import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

plt.style.use('./paper.mplstyle')  # change this to a paper specific one
# plt.rcParams['text.usetex'] = False

colors = ['#D60270','#9B4F96','#0038A8']
markers = ['o','s','^']

rho_labels = {'g2g2'   : r'$\langle g_2 g_2 \rangle$',
              'g2e4'   : r'$\langle g_2 e_4 \rangle$',
              'e4e4'   : r'$\langle e_4 e_4 \rangle$',
              'g2dg2'  : r'$\langle g_2 \delta g_2 \rangle$',
              'g2de4'  : r'$\langle g_2 \delta e_4 \rangle$',
              'g2w22'  : r'$\langle g_2 w_{22} \rangle$',
              'g2w24'  : r'$\langle g_2 w_{24} \rangle$',
              'g2w42'  : r'$\langle g_2 w_{42} \rangle$',
              'g2w44'  : r'$\langle g_2 w_{44} \rangle$',
              'e4dg2'  : r'$\langle e_4 \delta g_2 \rangle$',
              'e4de4'  : r'$\langle e_4 \delta e_4 \rangle$',
              'e4w22'  : r'$\langle e_4 w_{22} \rangle$',
              'e4w24'  : r'$\langle e_4 w_{24} \rangle$',
              'e4w42'  : r'$\langle e_4 w_{42} \rangle$',
              'e4w44'  : r'$\langle e_4 w_{44} \rangle$',
              'dg2dg2' : r'$\langle \delta g_2\delta g_2 \rangle$',
              'dg2de4' : r'$\langle \delta g_2\delta e_4 \rangle$',
              'dg2w22' : r'$\langle \delta g_2 w_{22} \rangle$',
              'dg2w24' : r'$\langle \delta g_2 w_{24} \rangle$',
              'dg2w42' : r'$\langle \delta g_2 w_{42} \rangle$',
              'dg2w44' : r'$\langle \delta g_2 w_{44} \rangle$',
              'de4dg2' : r'$\langle \delta e_4\delta g_2 \rangle$',
              'de4de4' : r'$\langle \delta e_4\delta e_4 \rangle$',
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
              'w44w44' : r'$\langle w_{42} w_{44} \rangle$',
               }

def plot_rho(ax, rho, label, marker, params):
    ax.plot(rho.meanr, rho.xip, '-', **params)
    ax.plot(rho.meanr, -rho.xip, ':', **params)
    ax.errorbar(rho.meanr[rho.xip>0], rho.xip[rho.xip>0], yerr=np.sqrt(rho.varxip[rho.xip>0]), fmt=marker, label=label, **params)
    ax.errorbar(rho.meanr[rho.xip<0], -rho.xip[rho.xip<0], yerr=np.sqrt(rho.varxip[rho.xip<0]), fmt=marker, mfc='none', mew=0.75, **params)
    return ax

def mean_rho(rho, lb=5, ub=100):
    points = (rho.meanr>lb)&(rho.meanr<ub)
    result = np.mean(rho.xip[points])
    return result

def plot_rho_examples(rhohomy10, rhohomy1, a):
    for rhohom, alpha in zip([rhohomy10, rhohomy1], [1, 0.25]):
        for key, color, marker in zip(['dg2dg2', 'dg2de4', 'de4de4'], colors, markers):
            plot_rho(a[0], rhohom[key], rho_labels[key], marker, {'color':color, 'alpha':alpha})

        for key, color, marker in zip(['g2w24', 'dg2w24', 'w22w24'], colors, markers):
            plot_rho(a[1], rhohom[key], rho_labels[key], marker, {'color':color, 'alpha':alpha})

        if alpha == 1:
            [ax.legend(ncols=3, handletextpad=0.05, columnspacing=1, edgecolor='lightgrey', borderaxespad=0.2) for ax in a];

    a[0].text(0.85, 0.8, 'Y10', alpha=1, transform=a[0].transAxes, fontsize=10, va='top')
    a[0].text(0.85, 0.7, 'Y1', alpha=0.5, transform=a[0].transAxes, fontsize=10, va='top')

    [ax.set_yscale('log', nonpositive='clip') for ax in a.flatten()]

    a[1].set_xscale('log')
    a[1].set_xlim(.6, 200)
    a[1].set_xlabel(r'$\theta$ (arcmin)')# for ax in a]
    [ax.set_ylabel(r'$\xi_+(\theta)$') for ax in a]
    a[1].set_ylim(bottom=1e-10)
    # [ax.tick_params(axis='both', which='both', direction='out') for ax in a]
    # plt.subplots_adjust(top=0.92, bottom=0.85, right=0.925, left=0.08, hspace=0.001)
    plt.subplots_adjust(hspace=0)
    plt.savefig('../figures/rho-examples-hom-i-full-radec-03-piff-02-jk.jpg', dpi=300)
    plt.show()

def plot_rho_summary(rhohomy10, rhohomy1, a):
    bad = ['nvisits']
    alpha2_terms = ['g2g2', 'g2e4', 'e4e4']
    alpha_terms = ['g2dg2', 'g2de4', 'g2w22', 'g2w24', 'g2w42', 'g2w44', 'e4dg2', 'e4de4', 'e4w22', 'e4w24', 'e4w42', 'e4w44']
    noalpha_terms = [k for k in rhohomy10.keys() if k not in alpha_terms+alpha2_terms+bad]
    sorted = {}
    xlabels = []
    for rhohom, transparancy in zip([rhohomy10, rhohomy1], [1, 0.5]):
        n_start = 0
        params = {'color': 'k', 'alpha':transparancy}
        for alpha_factor, terms in zip([0,1,2], [noalpha_terms, alpha_terms, alpha2_terms]):
            means = np.array([mean_rho(rhohom[rho]) for rho in terms])
            if transparancy==1:
                sorted[alpha_factor] = np.argsort(abs(means))[::-1]
                xlabels += [rho_labels[k] for k in np.array(terms)[sorted[alpha_factor]]]
            pltpts = np.arange(len(terms))+n_start
            a.plot(pltpts, means[sorted[alpha_factor]], 'o', **params)
            a.plot(pltpts, -means[sorted[alpha_factor]], 'o', mfc='none', mew=0.75, **params)

            if alpha_factor>0:
                a.plot(pltpts, means[sorted[alpha_factor]]*(0.025**alpha_factor), '^', **params)
                a.plot(pltpts, -means[sorted[alpha_factor]]*(0.025**alpha_factor), '^', mfc='none', mew=0.75, **params)

                a.fill_betweenx([0,1], n_start-0.5, len(rhohom.keys())-1.5, alpha=0.05, color='k')

            n_start += len(terms)

    a.set_xticks(np.arange(len(rhohom.keys())-1), xlabels, rotation=66)

    a.set_yscale('log', nonpositive='clip')

    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], ls='', marker='o', color='k', label='Y10', markerfacecolor='k'),
            Line2D([0], [0], ls='', marker='o', alpha=0.5, color='k', label='Y1', markerfacecolor='k')]
    a.legend(handles=handles, loc='upper left', frameon=True, edgecolor='lightgrey')

    a.set_ylabel(r"$\langle \rho(\theta)\rangle_{5'-100'}$")

    [a.axhline(10**x, color='lightgrey', alpha=0.5, zorder=1) for x in [-6, -8, -10, -12]]

    a.set_ylim(bottom=1e-13, top=2e-5)
    a.set_xlim(-0.5, 35.5)
    plt.savefig('../figures/rho-summary-hom-i-full-radec-03-piff-02-jk.jpg', dpi=300)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', type=str, required=True)
    args = parser.parse_args()

    with open(args.datapath+'/rho-i-full-radec-03-piff-02-patch25-hom1506.pkl', 'rb') as f:
        rhohomy10 = pickle.load(f)
    with open(args.datapath+'/rho-i-full-radec-03-piff-02-patch25-hom150.pkl', 'rb') as f:
        rhohomy1 = pickle.load(f)

    f, a = plt.subplots(2,1, figsize=(3.35,4.25), sharex=True, sharey=True)
    plot_rho_examples(rhohomy10, rhohomy1, a)

    f, a= plt.subplots(1,1, figsize=(7.5, 3))
    plot_rho_summary(rhohomy10, rhohomy1, a)
