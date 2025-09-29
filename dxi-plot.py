import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
from colors import color_scheme as colors
plt.style.use('./paper.mplstyle')
showsig = True

def plot_dxi_band(ax, xip, theta, sig, label, params, c=1, alpha_fill=0.2):
    line2dparams = {k:p for k,p in params.items() if 'cap' not in k}
    ax.plot(theta, xip*c, **line2dparams, label=label)
    ax.plot(theta, -xip*c, mfc='none', **line2dparams)

    ax.fill_between(
        theta, abs(xip*c) - sig * abs(c), abs(xip*c) + sig * abs(c),
        alpha=alpha_fill, color=params['color'], zorder=2,
    )
    return ax

f, a = plt.subplots(
    2, 1, figsize=(3.35, 4.25),
    sharex=True, sharey=True,
    # gridspec_kw={'hspace':0.4},
    )

grey = '#dde3eb'

# range_str='regular'
frac = 0.35
clist = [colors.g, colors.p, colors.p]

type = 'all'
for y10, ax in zip([False, True], a):
    for fname, color, marker, label, alpha in zip(
        ['empirical', 'alphae-2'], clist, ['o', 'v'], [r'$\alpha_i=0$', r'$\alpha_2=0.01, \alpha_4=-0.01$', 'test'], [0.2, 0.2]):
        with open(f'../data/dxi-i-full-radec-04-piff-01-cornercut-{fname}.json', 'r') as f:
            data = json.load(f)

        if y10:
            subset = data['y10'][type]
            xip = np.array(subset['xip']) * frac
            theta = np.array(subset['r'])
            sig = np.sqrt(subset['xip_var']) * frac
        else:
            subset = data['y1'][type]
            xip = np.mean(subset['xip'], axis=0) * frac
            theta = np.array(subset['r'])
            sig = np.sqrt(subset['xip_var']) * frac

        params = dict(
            marker=marker,
            color=color,
            ms=4,
            zorder=4,
            lw=0,
            mew=0.75,
        )
        plot_dxi_band(
            ax, xip, theta, sig,
            label=label, params=params,
            alpha_fill=alpha)

    if showsig:
        covfile = f'../data/thps_cov_test_matrix_V2_{"Y10" if y10 else "Y1"}.dat'
        constraint = pd.read_csv(covfile, delimiter='\t', header=None).to_numpy()
        covmin = np.sqrt(np.diag(constraint)[:25])  # first half is xi+
        theta_sig = np.copy(theta)
        theta_sig[0] = 0.5
        theta_sig[-1] = 50
        for fraction, ls, lab in zip([1, 0.3],['-', '--'], [r'$\sigma_{\xi_+}$', r'$0.3\sigma_{\xi_+}$']):
            sigma = covmin * fraction
            ax.plot(theta, sigma, ls=ls, lw=2, color=grey, zorder=2, label=lab)
            ax.fill_between(theta_sig, 0, sigma, color=grey, alpha=0.15, zorder=1)
        if not y10:
            ylimu = covmin[0] * 2
            ylimb = sigma[-1] / 100

a[1].text(0.7125, 0.8, r'Y10 ($riz$)', fontsize=10, transform=a[1].transAxes)
a[0].text(0.7125, 0.8, r'Y1 ($riz$)', fontsize=10, transform=a[0].transAxes)

h1, l1 = a[1].get_legend_handles_labels()
legend_order = [2,3,0,1]
a[1].legend(
    handles=[h1[i] for i in legend_order],
    labels=[l1[i] for i in legend_order],
    loc='lower left', 
    borderaxespad=0.1, ncol=2, handletextpad=0.2, columnspacing=1,
    bbox_to_anchor=(0,1))

a[1].set_xscale('log')
a[-1].set_xlabel(r'$\theta$ (arcmin)')
a[1].set_xlim(np.min(theta)-.05, np.max(theta)+5)

for ax in a:
    ax.set_yscale('log', nonpositive='clip')
    ax.set_ylabel(r'$\delta\xi_+^{\rm PSF}(\theta)$')
    ax.set_ylim(bottom=ylimb, top=ylimu)

plt.subplots_adjust(hspace=0)

plt.savefig(f'../figures/dxi-riz-full-radec-04-piff-01-cornercut-alpha!0.jpg', dpi=300)
plt.show()

# def bootstrap_ratio(data, type='all', nboot=50):
#     ratios = []
#     for i in range(nboot):
#         idx = np.random.randint(0, 50, 50)
#         ratio = abs(np.array(data['y10'][type]['xip']) / np.mean([data['y1'][type]['xip'][j] for j in idx], axis=0))
#         ratios.append(ratio)
#     return np.std(ratios, axis=0)

# f2, ax = plt.subplots(1, 1, figsize=(3.35, 2), sharex=True)

# for type, c, m, label in zip(
#     ['all'],#, 'w_0', 'w_12'],
#     [colors.g, colors.p, colors.y],
#     ['o','D', 's'],
#     [r'all terms', r'$\delta e^{(i)}\delta e^{(j)}$', r'remaining']):
#     std = bootstrap_ratio(data, type='all', nboot=1000)
#     ratio = abs(np.array(data['y10'][type]['xip']) / np.mean(data['y1'][type]['xip'], axis=0))
#     # ax.errorbar(
#     #     data['y10'][type]['r'], ratio, yerr=std,
#     #     color=c, zorder=1, fmt='o')
#     ax.plot(
#         data['y10'][type]['r'], ratio, 'o',
#         color=c, marker=m, lw=1, label=label)
#     ax.fill_between(
#         data['y10'][type]['r'], ratio-std, ratio+std,
#         color=c, alpha=0.2)

# ax.fill_between([.5, 50], 0.2, 0.05, color=colors.g, alpha=0.1)
# # ax.set_yscale('log', nonpositive='clip')
# ax.set_xscale('log')
# ax.set_ylim(-0.5,2)
# # ax.set_ylim(2e-1, 2e2)
# ax.set_ylabel(r'$\delta\xi_+^{\rm Y10} / \delta\xi_+^{\rm Y1}$')
# ax.set_xlabel(r'$\theta$ (arcmin)')
# ax.set_xlim(.5, 50)
# # ax.legend(loc='upper left', borderaxespad=0.5)

# # plt.savefig(f'../figures/dxi-ratio-full-radec-04-piff-01-cornercut-emp-boot.jpg', dpi=300)
# plt.show()