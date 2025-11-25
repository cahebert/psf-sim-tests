import numpy as np
import json
import treecorr
import pickle

def get_coeff_combos(keys, coefficients, namemap, trr=None):
    """
    Combine the given PSF coefficients into dxi+/-.
    Use option parameter trr to adjust for the ratio of the tratio between two surveys.
    """
    parameters = ['e2','de2', 'w22', 'e4', 'de4', 'w24', 'w42', 'w44']
    coefficient_combos = {}

    for v1 in parameters:
        for v2 in parameters:
            if v1+v2 in keys or v2+v1 in keys:
                key = v1+v2 if v1+v2 in keys else v2+v1

                c1 = coefficients[namemap[v1]]
                c2 = coefficients[namemap[v2]]

                if trr is not None:  # adjust for tratio except for alpha parameter
                    if 'alpha' not in namemap[v1]:
                        c1 *= trr
                    if 'alpha' not in namemap[v2]:
                        c2 *= trr

                coefficient_combos[key] = c1 * c2

                # need to add a factor of 2 for the cross terms!!
                if v1 != v2:
                    coefficient_combos[key] *= 2

    return coefficient_combos

def run_dxi_estimate(rstats, coeffs, y10=None):
    error_norm = 1508/150  # divide variance by the number of independent samples

    results = {'all': {}, 'w_0': {}, 'w_12': {}}
    keys_all = {k:i for i,k in enumerate(rstats.keys())}
    keys_0 = {k:i for i,k in enumerate(rstats.keys()) if k in ['de2de2','de4de4','de2de4']}
    keys_tmp = {k:i for i,k in enumerate(rstats.keys()) if k not in ['de2de2','de4de4','de2de4']}
    corrs = [v for k,v in rstats.items()]

    for keys, tmp in zip(
        [keys_all],#, keys_0, keys_tmp],
        ['all', 'w_0', 'w_12']
        ):
        if y10:
            def get_dxi(data, xi='xip'):
                out = np.zeros_like(data[0].xip)
                for k in keys.keys():
                    if xi=='xip':
                        out += data[keys[k]].xip * coeffs[k]
                    elif xi=='xim':
                        out += data[keys[k]].xim * coeffs[k]
                return out

            func_xip = lambda data: get_dxi(data,xi='xip')
            func_xim = lambda data: get_dxi(data,xi='xim')

            cov_xip = treecorr.estimate_multi_cov(corrs, 'jackknife', func=func_xip, cross_patch_weight='match')
            cov_xim = treecorr.estimate_multi_cov(corrs, 'jackknife', func=func_xim, cross_patch_weight='match')

            results[tmp]['xip_var'] = list(np.diag(cov_xip))
            results[tmp]['xip'] = list(func_xip(corrs))
            results[tmp]['xim_var'] = list(np.diag(cov_xim))
            results[tmp]['xim'] = list(func_xim(corrs))
            results[tmp]['r'] = list(corrs[0].meanr)

        else:
            def get_dxi(data, xi='xip'):
                out = np.zeros_like(data[0][xi])
                for k in keys.keys():
                    out += np.array(data[keys[k]][xi]) * coeffs[k]
                return out
            dxip = get_dxi(corrs, xi='xip')
            dxim = get_dxi(corrs, xi='xim')

            results[tmp]['xip_var'] = list(np.var(dxip, axis=0) / error_norm)
            results[tmp]['xim_var'] = list(np.var(dxim, axis=0) / error_norm)
            results[tmp]['xip'] = list(list(d) for d in dxip)
            results[tmp]['xim'] = list(list(d) for d in dxim)
            results[tmp]['r'] = list(corrs[0]['meanr'])

    return results

namemap = {
    "e2"  : "alpha2",
    "de2" : "beta2",
    "w22" : "eta22",
    "e4"  : "alpha4",
    "de4" : "beta4",
    "w44" : "eta44",
    "w24" : "eta24",
    "w42" : "eta42"
}
results = {'y10': {}, 'y1': {}}
coeff_combo = {'y10': {}, 'y1': {}}

# with open('../data/contamination-coeffs.js', 'r') as f:
#     coefficients, namemap, tratios = json.load(f)
with open('../data/contamination-coeffs-empirical.json', 'r') as f:
    coefficients = json.load(f)

alpha_mag = 1e-2 / 0.35
coefficients['lssty1_mcal']['alpha2'] = alpha_mag
coefficients['lssty1_mcal']['alpha4'] = -alpha_mag
coefficients['lssty10_mcal']['alpha2'] = alpha_mag
coefficients['lssty10_mcal']['alpha4'] = -alpha_mag

for y10 in [False, True]:
    # load the sim rho stats
    if y10:
        with open(f'../data/patch35-rho-i-full-radec-04-piff-01-cornercut-cov1508.pkl', 'rb') as f:
            rhodata = pickle.load(f)
        ckey = 'lssty10_mcal'

    else:
        with open(f'../data/rho-i-full-radec-04-piff-01-cornercut-50x150.json', 'r') as f:
            rhodata = json.load(f)
        ckey = 'lssty1_mcal'

    ## for coeff_type in ['wide', 'regular']:
    ##     coeffs = get_coeff_combos(rhodata.keys(), coefficients['desy6'][coeff_type], namemap, trr=None)
    ##     coeff_combo['y10' if y10 else 'y1'][coeff_type] = coeffs
    ##     results['y10' if y10 else 'y1'][coeff_type] = run_dxi_estimate(rhodata, coeffs, y10)

    coeffs = get_coeff_combos(rhodata.keys(), coefficients[ckey], namemap, trr=None)
    coeff_combo['y10' if y10 else 'y1'] = coeffs
    results['y10' if y10 else 'y1'] = run_dxi_estimate(rhodata, coeffs, y10)

with open('../data/dxi-i-full-radec-04-piff-01-cornercut-alphapme-2.json', 'w') as f:
    json.dump(results, f)

# with open('../data/coeffs-i-full-radec-04-piff-01-empirical.json', 'w') as f:
#     json.dump(coeff_combo, f)
