import matplotlib.pyplot as plt
import numpy as np
# import pickle
import json
# import treecorr
plt.style.use('./paper.mplstyle')  # change this to a paper specific one
showsig = True

# def get_dxi(data, coefficients, namemap_all, namemap_2, trr=None, ref='des'):
#     """
#     Combine the given PSF residual correlations and coefficients into dxi+.
#     Use option parameter trr to adjust for the ratio of the tratio between two surveys.
#     """
#     # fourth order
#     dxi_4 = np.zeros_like(data['g2g2'].xip)
#     dxi_4_var = np.zeros_like(data['g2g2'].xip)
#     dxi_2 = np.zeros_like(data['g2g2'].xip)
#     dxi_2_var = np.zeros_like(data['g2g2'].xip)

#     for v1 in ['g2','dg2', 'w22', 'e4', 'de4', 'w24', 'w42', 'w44']:
#         for v2 in ['g2','dg2', 'w22', 'e4', 'de4', 'w24', 'w42', 'w44']:
#             if v1+v2 in data.keys() or v2+v1 in data.keys():
#                 key = v1+v2 if v1+v2 in data.keys() else v2+v1

#                 c1 = coefficients[ref][namemap_all[v1]]
#                 c2 = coefficients[ref][namemap_all[v2]]

#                 if trr is not None:  # adjust for tratio except for alpha parameter
#                     if 'alpha' not in namemap_all[v1]:
#                         c1 *= trr
#                     if 'alpha' not in namemap_all[v2]:
#                         c2 *= trr

#                 dxi_4 += c1 * c2 * data[key].xip
#                 dxi_4_var += (c1 * c2)**2 * data[key].varxip

#                 # and if doing second order terms only:
#                 if '4' not in key:
#                     c1 = coefficients[ref][namemap_2[v1]]
#                     c2 = coefficients[ref][namemap_2[v2]]

#                     if trr is not None:  # adjust for tratio except for alpha parameter
#                         if 'alpha' not in namemap_2[v1]:
#                             c1 *= trr
#                         if 'alpha' not in namemap_2[v2]:
#                             c2 *= trr
#                     dxi_2 += c1 * c2 * data[key].xip
#                     dxi_2_var += (c1 * c2)**2 * data[key].varxip

#     dxi_2_var = np.sqrt(dxi_2_var)
#     dxi_4_var = np.sqrt(dxi_4_var)

#     return dxi_2, dxi_2_var, dxi_4, dxi_4_var

constraint = np.loadtxt('/Users/clairealice/Documents/git/psf_req/srd_y1_tightest.txt')
# with open('../data/contamination-coeffs.js', 'r') as f:
#     coefficients, namemap, namemap_2, tratios = json.load(f)

with open('../data/dxi-4th-jk.js', 'r') as f:
    data = json.load(f)

f, a = plt.subplots(2, 1, figsize=(3.35, 3.75), sharex=True)

for y10, ax in zip([False, True], a):
    # load the sim rho stats
    # if y10:
    #     # label = r'$\delta \xi_+^{Y10}$'
    #     with open('../data/rho-i-full-radec-03-piff-02-cov-hom1506.pkl', 'rb') as f:
    #         rhodata = pickle.load(f)
    # else:
    #     # label = r'$\delta \xi_+^{Y1}$'
    #     with open('../data/rho-i-full-radec-03-piff-02-cov-hom150.pkl', 'rb') as f:
    #         rhodata = pickle.load(f)

    for survey, color, label, marker in zip(['desy6', 'hscy3'],
                                            ['#0038A8', '#D60270'],
                                            # ['#EC7357', '#5775c3'],
                                            ['DES', 'HSC'],
                                            ['o', '^']):
        # tratioratio = tratios['lssty10'] / tratios[survey]

        # dxi_2_ref, _, dxi_4_ref, _ = get_dxi(rhodata,
        #                                     coefficients,
        #                                     namemap,
        #                                     namemap_2,
        #                                     None,
        #                                     ref=survey)

        # if y10:
        #     print(f"Fraction of {'Y10' if y10 else 'Y1'} dxi using raw {label} coefficients:")
        # for Trr in [0.9*tratioratio, 1.1*tratioratio, tratioratio]:
        #     dxi_2, dxi_2_var, dxi_4, dxi_4_var = get_dxi(rhodata,
        #                                                 coefficients,
        #                                                 namemap,
        #                                                 namemap_2,
        #                                                 Trr,
        #                                                 ref=survey)
        #     if y10:
        #         print(dxi_2[0]/dxi_2_ref[0], dxi_4[0]/dxi_4_ref[0])

        if label == 'DES':
            label = 'DES coefficients' #r'$\delta \xi_+^{\rm DES}$'
        else:
            label = 'HSC coefficients' #r'$\delta \xi_+^{\rm HSC}$'

        subset = data[survey]['y10' if y10 else 'y1']
        theta = np.array(subset['r'])
        xip = np.array(subset['xip'])

        # second order
        # ax.plot(theta, dxi_2, '-', color=color, lw=1.25, zorder=4, alpha=0.2)
        # ax.plot(theta, -dxi_2, ':', color=color, lw=1.25, zorder=4, alpha=0.2)
        # ax.errorbar(theta[dxi_2>0], dxi_2[dxi_2>0], yerr=dxi_2_var[dxi_2>0],
        #             fmt=marker, color=color, ms=4, zorder=4, label=label, alpha=0.2)
        # ax.errorbar(theta[dxi_2<0], -dxi_2[dxi_2<0], yerr=dxi_2_var[dxi_2<0],
        #             fmt=marker, mfc='none', mew=0.75, color=color, ms=4, zorder=4, alpha=0.2)

        # fourth order
        ax.plot(theta, xip, '-', color=color, lw=1.25, zorder=4)
        ax.plot(theta, -xip, ':', color=color, lw=1.25, zorder=4)
        ax.errorbar(theta[xip>0], xip[xip>0], yerr=np.array(subset['sig'])[xip>0],
                    fmt=marker, color=color, ms=4, zorder=4, label=label)
        ax.errorbar(theta[xip<0], -xip[xip<0], yerr=np.array(subset['sig'])[xip<0],
                    fmt=marker, mfc='none', mew=0.75, color=color, ms=4, zorder=4)
        # label = r'$\delta \xi_+^{(4)}$'
        # theta = rhofull['g2g2'].meanr
        # plt.plot(theta, dxi_4, '-', color='orangered')
        # plt.plot(theta, -dxi_4, ':', color='orangered')
        # plt.errorbar(theta[dxi_4>0], dxi_4[dxi_4>0], yerr=dxi_4_var[dxi_4>0], fmt='o', color='orangered', label=label)
        # plt.errorbar(theta[dxi_4<0], -dxi_4[dxi_4<0], yerr=dxi_4_var[dxi_4<0], fmt='o', mfc='none', mew=0.75, color='orangered')

    if showsig:
        grey = '#dde3eb'
        ## divide by sqrt(10) to get Y10, srqt(5) to adjust for sample size from redshift bins
        ## multiply by fraction to get PSF budget from total uncertainty
        covmin = constraint[1] #/ np.sqrt(5)

        if y10:
            covmin /= np.sqrt(10)

        for fraction, ls in zip([0.1, 0.2, 0.3], ['-', '--', ':']):
            sigma = covmin * fraction
            ax.plot(constraint[0], sigma, ls=ls, lw=2, color=grey, zorder=2)
            ax.fill_between(constraint[0], 0, sigma, color=grey, alpha=0.3, zorder=1)

    ax.text(0.125, 0.875, f'Y{"10" if y10 else "1"}', fontsize=10, transform=ax.transAxes)

from matplotlib.lines import Line2D
# from matplotlib.container import ErrorbarContainer
# from matplotlib.collections import LineCollection
# labels = [r'DES coefficients', r'HSC coefficients']
# lines = [Line2D([0], [0], color=c, ls='', marker=marker, label=lab)
#             for marker, c, lab in zip(['^', 'o'], ['#D60270', '#0038A8'], labels)]
# barline = LineCollection(np.empty((2,2,2)))
# err = [ErrorbarContainer((line, [line], [barline]), has_yerr=True) for line in lines]
# handles = [Line2D([0], [0], color=c, ls='', marker=marker, label=lab)
#             for marker, c, lab in zip(['^', 'o'], ['#D60270', '#0038A8'], labels)]
# a[0].legend(handles=err, loc='upper right', borderaxespad=0.75)

sig_labels = [r'$0.1\sigma_+^{\rm min}$',
              r'$0.2\sigma_+^{\rm min}$',
              r'$0.3\sigma_+^{\rm min}$']
handles = [Line2D([0], [0], color='#dde3eb', lw=2, ls=ls, label=lab)
            for ls, lab in zip(['-', '--', ':'], sig_labels)]
a[1].legend(handles=handles, loc='upper right', borderaxespad=0.75)
a[0].legend(borderaxespad=0.6, handletextpad=0.1, title=r'LSST-sim $\rho$ stats and:')

a[1].set_xscale('log')
a[1].set_xlabel(r'$\theta$ (arcmin)')
a[1].set_xlim(.6, 200)

for ax in a:
    ax.set_yscale('log', nonpositive='clip')
    ax.set_ylabel(r'$\delta\xi_+^{\rm PSF}(\theta)$')
    ax.set_ylim(bottom=5e-10, top=2e-5)
    # ax.tick_params(axis='both', which='both', direction='out', top=0, right=0)

# plt.subplots_adjust(hspace=0.01, top=0.92, right=0.925, bottom=0.85, left=0.08)
plt.subplots_adjust(hspace=0)

plt.savefig('../figures/dxi-y1y10-i-full-radec-03-piff-02-jk.jpg', dpi=300)
# plt.show()