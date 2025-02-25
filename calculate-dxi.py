import numpy as np
import json
import treecorr
import pickle

def get_dxi(corrs, keys, coefficients, namemap, trr=None, ref='desy6'):
    """
    Combine the given PSF residual correlations and coefficients into dxi+.
    Use option parameter trr to adjust for the ratio of the tratio between two surveys.
    """
    # fourth order
    dxi_4 = np.zeros_like(corrs[0].xip)
    # dxi_2 = np.zeros_like(corrs[0].xip)

    for v1 in ['g2','dg2', 'w22', 'e4', 'de4', 'w24', 'w42', 'w44']:
        for v2 in ['g2','dg2', 'w22', 'e4', 'de4', 'w24', 'w42', 'w44']:
            if v1+v2 in keys.keys() or v2+v1 in keys.keys():
                key = v1+v2 if v1+v2 in keys.keys() else v2+v1

                c1 = coefficients[ref][namemap[v1]]
                c2 = coefficients[ref][namemap[v2]]

                if trr is not None:  # adjust for tratio except for alpha parameter
                    if 'alpha' not in namemap[v1]:
                        c1 *= trr
                    if 'alpha' not in namemap[v2]:
                        c2 *= trr

                dxi_4 += c1 * c2 * corrs[keys[key]].xip

                # and if doing second order terms only:
                # if '4' not in key:
                #     c1 = coefficients[ref][namemap_2[v1]]
                #     c2 = coefficients[ref][namemap_2[v2]]

                #     if trr is not None:  # adjust for tratio except for alpha parameter
                #         if 'alpha' not in namemap_2[v1]:
                #             c1 *= trr
                #         if 'alpha' not in namemap_2[v2]:
                #             c2 *= trr
                #     dxi_2 += c1 * c2 * corrs[keys[key]].xip

    return dxi_4

with open('../data/contamination-coeffs.js', 'r') as f:
    coefficients, namemap, namemap_2, tratios = json.load(f)

results = {'desy6':{'y10':{},'y1':{}}, 'hscy3':{'y10':{},'y1':{}}}

for survey in ['desy6', 'hscy3']:
    tratioratio = tratios['lssty10'] / tratios[survey]
    for y10 in [True, False]:
        # load the sim rho stats
        if y10:
            # label = r'$\delta \xi_+^{Y10}$'
            with open('../data/rho-i-full-radec-03-piff-02-patch25-hom1506.pkl', 'rb') as f:
                rhodata = pickle.load(f)
        else:
            # label = r'$\delta \xi_+^{Y1}$'
            with open('../data/rho-i-full-radec-03-piff-02-patch25-hom150.pkl', 'rb') as f:
                rhodata = pickle.load(f)

        corrs = [v for k,v in rhodata.items() if 'n' not in k]#list(rhodata.values())
        keys = {k:i for i,k in enumerate(rhodata.keys()) if 'n' not in k}

        func = lambda data: get_dxi(
            data,
            keys,
            coefficients,
            namemap,
            tratioratio,
            ref=survey)

        cov = treecorr.estimate_multi_cov(corrs, 'jackknife', func=func)

        results[survey]['y10' if y10 else 'y1']['sig'] = list(np.sqrt(np.diag(cov)))
        results[survey]['y10' if y10 else 'y1']['xip'] = list(func(corrs))
        results[survey]['y10' if y10 else 'y1']['r'] = list(corrs[0].meanr)

with open('../data/dxi-4th-jk.js', 'w') as f:
    json.dump(results, f)