import numpy as np
import galsim
from numpy import mgrid, sum
import scipy.linalg as alg
import ngmix
import scipy

#### most of this code is from TQ Zhang ####

class shapeletXmoment:
    def __init__(self, psf,n, bmax = 10, pixel_scale = 1.0):
        self.n = n
        self.bmax = bmax
        self.pixel_scale = pixel_scale
        self.base_psf = psf
        self.base_psf_image = psf.drawImage(scale = pixel_scale)
        self.base_psf_result = galsim.hsm.FindAdaptiveMom(self.base_psf_image)
        self.base_shapelet = galsim.Shapelet.fit(self.base_psf_result.moments_sigma, bmax, self.base_psf_image, normalization = 'sb')
        self.base_bvec = self.base_shapelet.bvec

    def moment_measure(self, image, p, q):
        n = p+q
        if n<2:
            print( "Does not support moment measure less than second order.")
            return 0
        elif n==2:
            return self.get_second_moment(image,p,q)
        else:
            return self.higher_weighted_moment(image,p,q)

    def get_second_moment(self, image, p ,q):
        image_results = galsim.hsm.FindAdaptiveMom(image)
        if p==2:
            return image_results.observed_shape.e1
        elif q==2:
            return image_results.observed_shape.e2
        else:
            return image_results.moments_sigma

    def higher_weighted_moment(self,gsimage,p,q):
        image = gsimage.array
        y, x = mgrid[:image.shape[0],:image.shape[1]]+1

        psfresults = galsim.hsm.FindAdaptiveMom(galsim.Image(image, scale = self.pixel_scale))
        M = np.zeros((2,2))
        e1 = psfresults.observed_shape.e1
        e2 = psfresults.observed_shape.e2
        sigma4 = psfresults.moments_sigma**4
        c = (1+e1)/(1-e1)
        M[1][1] = np.sqrt(sigma4/(c-0.25*e2**2*(1+c)**2))
        M[0][0] = c*M[1][1]
        M[0][1] = 0.5*e2*(M[1][1]+M[0][0])
        M[1][0] = M[0][1]

        pos = np.array([x-psfresults.moments_centroid.x, y-psfresults.moments_centroid.y])
        pos = np.swapaxes(pos,0,1)
        pos = np.swapaxes(pos,1,2)

        inv_M = np.linalg.inv(M)
        sqrt_inv_M = alg.sqrtm(inv_M)
        std_pos = np.zeros(pos.shape)
        weight = np.zeros(pos.shape[0:2])
        for i in range(pos.shape[0]):
            for j in range(pos.shape[1]):
                this_pos = pos[i][j]
                this_standard_pos = np.matmul(sqrt_inv_M, this_pos)
                std_pos[i][j] = this_standard_pos
                weight[i][j] = np.exp(-0.5* this_standard_pos.dot(this_standard_pos))

        std_x, std_y = std_pos[:,:,0],std_pos[:,:,1]

        return sum(std_x**p*std_y**q*weight*image)/sum(image*weight)

    def modify_pq(self, m, c, delta = 0.0001):
        n = self.n
        mu = self.get_mu(n)
        pq_list = self.get_pq_full(n)
        shapelet_list = self.pq2shapelet(pq_list)

        ori_moments = self.get_all_moments(self.base_psf_image, pq_list)

        A = np.zeros(shape =(mu, mu))

        #i is the mode index
        #j is the moment index
        #measure d_moment_j / d_mode_i

        for i in range(mu):
            mode_index = shapelet_list[i]
            pert_bvec = self.base_bvec.copy()
            pert_bvec[mode_index]+=delta
            ith_pert = galsim.Shapelet(self.base_psf_result.moments_sigma, self.bmax, pert_bvec)
            pert_moment = self.get_all_moments(ith_pert.drawImage(scale = self.pixel_scale,method = 'no_pixel'), pq_list)
            for j in range(mu):
                A[i][j] = (pert_moment[j] - ori_moments[j])/delta
        self.A = A

        dm = np.zeros(mu)
        dm += m*ori_moments + c
        ds = np.linalg.solve(A.T,dm)

        true_mod_bvec = self.base_bvec.copy()
        for i in range(mu):
            true_mod_bvec[shapelet_list[i]]+=ds[i]

        self.true_mod = galsim.Shapelet(self.base_psf_result.moments_sigma,  self.bmax, true_mod_bvec)
        return self.true_mod

    def step_modify_pq(self,current_moments,current_dm, current_mod_bvec ,current_psf,mu,shapelet_list,delta, pq_list):
        A = np.zeros(shape =(mu, mu))
        for i in range(mu):
            mode_index = shapelet_list[i]

            pert_bvec = current_mod_bvec.copy()
            pert_bvec[mode_index]+=delta
            ith_pert = galsim.Shapelet(self.base_psf_result.moments_sigma, self.bmax, pert_bvec)
            pert_moment = self.get_all_moments(ith_pert.drawImage(scale = self.pixel_scale), pq_list)
            for j in range(mu):
                A[i][j] = (pert_moment[j] - current_moments[j])/delta

        ds = np.linalg.solve(A.T,current_dm)

        for i in range(mu):
            current_mod_bvec[shapelet_list[i]]+=ds[i]
        return current_mod_bvec

    def iterative_modify_pq(self,m,c,delta = 0.0001, threshold = 1e-6):
        iterative_n = 10

        n = self.n
        mu = self.get_mu(n)
        pq_list = self.get_pq_full(n)
        shapelet_list = self.pq2shapelet(pq_list)
        base_shapelet_image = self.base_shapelet.drawImage(scale = self.pixel_scale)
        original_moment = self.get_all_moments(base_shapelet_image, pq_list)
        current_moment = self.get_all_moments(base_shapelet_image, pq_list)
        current_dm = np.zeros(mu)
        current_dm += m*current_moment + c

        destiny_moment = current_moment + current_dm

        current_mod_bvec = self.base_bvec.copy()
        current_psf = galsim.Shapelet(self.base_psf_result.moments_sigma,  self.bmax, current_mod_bvec)

        while (np.max(np.abs(current_dm)) > threshold):

            current_mod_bvec = self.step_modify_pq(current_moment,current_dm, current_mod_bvec ,current_psf,mu,shapelet_list,delta,pq_list)
            current_psf = galsim.Shapelet(self.base_psf_result.moments_sigma,  self.bmax, current_mod_bvec)
            current_moment = self.get_all_moments(current_psf.drawImage(scale = self.pixel_scale), pq_list)

            current_dm = destiny_moment - current_moment

        return current_psf

    def get_all_moments(self, image, pq_list):
        results_list = []
        for tup in pq_list:
            results_list.append(self.moment_measure(image, tup[0], tup[1]))

        return np.array(results_list)

    def pq2mode(self,p,q):
        if p<=q:
            return (p+q)*(p+q+1)//2 + 2*min(p,q)
        else:
            return (p+q)*(p+q+1)//2 + 2*min(p,q)+1

    def pq2shapelet(self,pq_list):
        shapelet_index = []
        for tup in pq_list:
            shapelet_index.append(self.pq2mode(tup[0], tup[1]))
        return shapelet_index

    def get_mu(self, n):
        mu = 0
        for i in range(2,n+1):
            mu+=i+1
        return mu

    def get_pq_full(self,nmax):
        pq_list = []
        for n in range(2, nmax+1):
            p = 0
            q = n
            pq_list.append((p,q))

            while p<n:
                p+=1
                q-=1
                pq_list.append((p,q))
        return pq_list

    def get_pq_except(self,nmax,p,q):
        pq_full = self.get_pq_full(nmax)
        pq_except = []
        for tup in pq_full:
            if tup != (p,q):
                pq_except.append(tup)

        return pq_except


class HOMExShapeletPair:
    def __init__(
        self,
        gal_type, gal_sigma,
        e1,e2,
        g1,g2,
        psf_type, psf_sigma,
        gal_flux=1.e2,
        pixel_scale=1.0,
        sersicn=-1,
        psf_sersicn=-1,
        subtract_intersection=True,
        is_self_defined_PSF=False,
        self_defined_PSF=None,
        self_define_PSF_model=None,
        metacal_method='estimateShear',
        bpd_params = None,
        psf_gsobj = None):

        #Define basic variables
        self.pixel_scale = pixel_scale
        self.subtract_intersection = subtract_intersection
        self.is_self_defined_PSF = is_self_defined_PSF
        self.metacal_method = metacal_method

        #Define galaxy
        self.gal_type = gal_type
        self.gal_sigma = gal_sigma
        self.gal_flux=gal_flux
        self.e1 = e1
        self.e2 = e2
        self.g1 = g1
        self.g2 = g2
        self.cosmic_shear = galsim.Shear(g1 = g1, g2 = g2)
        self.g = np.array([g1,g2])
        self.e = np.array([e1,e2])
        self.sersicn=sersicn
        self.e_truth = self.e
        self.bpd_params = bpd_params

        if gal_type == 'gaussian':
            gaussian_profile = galsim.Gaussian(sigma = gal_sigma)
            self.gal_light = gaussian_profile.withFlux(self.gal_flux)
            self.gal_light = self.gal_light.shear(e1=e1, e2=e2)
        elif gal_type == 'sersic':
            sersic_profile = galsim.Sersic(sersicn, half_light_radius = self.gal_sigma)
            self.gal_light = sersic_profile.withFlux(self.gal_flux)
            self.gal_light = self.gal_light.shear(e1=e1, e2=e2)
        elif gal_type == 'bpd':
            bulge = galsim.Sersic(4, half_light_radius = self.bpd_params[0])
            bulge = bulge.shear(e1 = bpd_params[1], e2 = bpd_params[2])
            disk = galsim.Sersic(1, half_light_radius = self.bpd_params[3])
            disk = disk.shear(e1 = bpd_params[4], e2 = bpd_params[5])
            bulge_to_total = bpd_params[6]
            self.gal_light = bulge_to_total*bulge + (1-bulge_to_total)*disk

        self.gal_rotate_light = self.gal_light.rotate(90 * galsim.degrees)
        self.gal_light = self.gal_light.shear(g1 = g1, g2 = g2)
        self.gal_rotate_light = self.gal_rotate_light.shear(g1 = g1, g2 = g2)

        if not is_self_defined_PSF:
            self.psf_type = psf_type
            self.psf_sigma = psf_sigma
            self.psf_model_sigma = psf_sigma

            if psf_type == 'gaussian':
                self.psf_base = galsim.Gaussian(flux = 1.0, sigma = self.psf_sigma)
            elif psf_type == 'kolmogorov':
                self.psf_base  = galsim.Kolmogorov(flux = 1.0, half_light_radius = 0.5)
                self.psf_base = self.toSize(self.psf_base, self.psf_sigma,weighted = True)
            elif psf_type == 'opticalPSF':
                self.psf_base = galsim.OpticalPSF(1.0,flux = 1.0)
                self.psf_base = self.toSize(self.psf_base, self.psf_sigma)
            elif psf_type == 'sersic':
                self.psf_base = galsim.Sersic(psf_sersicn, half_light_radius = 1.0)
                self.psf_base = self.toSize(self.psf_base, self.psf_sigma,weighted = True)
            elif psf_type == 'gsobj':
                self.psf_base = psf_gsobj

        else:
            self.psf_type = "self_define"
            truth_image = self_defined_PSF
            truth_psf = galsim.InterpolatedImage(truth_image,scale = pixel_scale)
            truth_measure = galsim.hsm.FindAdaptiveMom(truth_image)
            truth_sigma = truth_measure.moments_sigma
            self.psf_light = truth_psf

    def setup_shapelet_psf(self, m, c, n, bmax = 10):
        self.n = n
        self.sxm = shapeletXmoment(self.psf_base,n,pixel_scale = self.pixel_scale)
        self.psf_light = self.sxm.base_shapelet
        self.psf_model_light = self.sxm.iterative_modify_pq(m, c)
        self.dm = m*self.sxm.get_all_moments(self.sxm.base_psf_image, self.sxm.get_pq_full(n))+c

    def speed_setup_shapelet_psf(self,m,c,n, psf_light, psf_model_light, dm):
        self.n = n
        self.sxm = shapeletXmoment(self.psf_base,n,pixel_scale = self.pixel_scale)
        self.psf_light = psf_light
        self.psf_model_light = psf_model_light
        self.dm = dm

    def perc_bias(self,metacal = True):
        base_ori_r, base_ori_e = self.measure(metacal = metacal,rot = False, base = True)
        mod_ori_r, mod_ori_e = self.measure(metacal = metacal,rot = False, base = False)
        base_rot_r, base_rot_e = self.measure(metacal = metacal,rot = True, base = True)
        mod_rot_r, mod_rot_e = self.measure(metacal = metacal,rot = True, base = False)

        R_base = np.mean(np.array([base_ori_r,base_rot_r]),axis = 0).reshape(2,2)
        base_shape = np.mean(np.array([base_ori_e,base_rot_e]),axis = 0)
        g_base = np.matmul(np.linalg.inv(R_base),base_shape)

        R_mod = np.mean(np.array([mod_ori_r,mod_rot_r]),axis = 0).reshape(2,2)
        mod_shape = np.mean(np.array([mod_ori_e,mod_rot_e]),axis = 0)
        g_mod = np.matmul(np.linalg.inv(R_mod),mod_shape)
        #print (g_mod[0] - g_base[0])/self.g1
        self.abs_bias = (g_mod - g_base)
        return (g_mod - g_base)/self.g

    def measure(self,metacal=True,rot = False, base = False):
        if base:
            image_epsf = self.psf_light.drawImage(scale=self.pixel_scale)
        else:
            image_epsf = self.psf_model_light.drawImage(scale=self.pixel_scale)

        if rot:
            galaxy = self.gal_rotate_light
        else:
            galaxy = self.gal_light

        final = galsim.Convolve([galaxy,self.psf_light])
        image = final.drawImage(scale = self.pixel_scale)
        if metacal == False:
            results = galsim.hsm.EstimateShear(image,image_epsf)
            shape = galsim.Shear(e1 = results.corrected_e1, e2 = results.corrected_e2)
            return np.array([[1.0,0,0,1.0]]),np.array([shape.g1,shape.g2])
        else:
            results = self.perform_metacal(image,image_epsf)
            return results["R"].reshape((-1)), results["noshear"]

    def perform_metacal(self,image,image_epsf):
        metacal = metacal_shear_measure(image,image_epsf)
        metacal.measure_shear(self.metacal_method)
        results = metacal.get_results()
        return results

    def findAdaptiveSersic(self,sigma,n):
        good_half_light_re = bisect(Sersic_sigma,sigma/3,sigma*5,args=(n,self.pixel_scale,sigma))
        return galsim.Sersic(n=n,half_light_radius=good_half_light_re)

    def findAdaptiveKolmogorov(self,sigma):
        good_half_light_re = bisect(Kolmogorov_sigma,max(self.psf_sigma/5,self.pixel_scale),self.psf_sigma*5,args = (self.pixel_scale,sigma))
        return galsim.Kolmogorov(half_light_radius = good_half_light_re)

    def findAdaptiveOpticalPSF(self,sigma):
        good_fwhm = bisect(OpticalPSF_sigma,max(self.psf_sigma/3,self.pixel_scale),self.psf_sigma*5,args = (self.pixel_scale,sigma))
        return galsim.OpticalPSF(good_fwhm)

    def toSize(self, profile, sigma , weighted = True, tol = 1e-4):
        if weighted:
            apply_pixel =  max(self.pixel_scale, sigma/10)
            true_sigma = galsim.hsm.FindAdaptiveMom(profile.drawImage(scale =apply_pixel,method = 'no_pixel')).moments_sigma*apply_pixel
        else:
            image = profile.drawImage(scale = self.pixel_scale, method = 'no_pixel')
            true_sigma = image.calculateMomentRadius()

        ratio = sigma/true_sigma
        new_profile = profile.expand(ratio)

        while abs(true_sigma - sigma)>tol:
            ratio = sigma/true_sigma
            new_profile = new_profile.expand(ratio)

            if weighted:
                apply_pixel =  max(self.pixel_scale, sigma/10)
                true_sigma = galsim.hsm.FindAdaptiveMom(new_profile.drawImage(scale =apply_pixel,method = 'no_pixel'),hsmparams=galsim.hsm.HSMParams(max_mom2_iter = 2000)).moments_sigma*apply_pixel
            else:
                #true_sigma = profile.calculateMomentRadius(scale = self.pixel_scale, rtype='trace')
                image = new_profile.drawImage(scale = self.pixel_scale, method = 'no_pixel')
                true_sigma = image.calculateMomentRadius()
        return new_profile

    def real_gal_sigma(self):
        image = self.gal_light.drawImage(scale = self.pixel_scale,method = 'no_pixel')
        return galsim.hsm.FindAdaptiveMom(image).moments_sigma*self.pixel_scale

    def get_actual_dm(self):
        m_truth = self.sxm.get_all_moments(self.psf_light.drawImage(scale=self.pixel_scale), self.sxm.get_pq_full(self.n))
        m_model = self.sxm.get_all_moments(self.psf_model_light.drawImage(scale=self.pixel_scale), self.sxm.get_pq_full(self.n))
        return m_model - m_truth

    def get_gal_trace(self):
        return 2*self.gal_light.calculateMomentRadius(scale = self.pixel_scale, rtype='trace')**2

    def get_psf_trace(self):
        return 2*self.psf_light.calculateMomentRadius(scale = self.pixel_scale, rtype='trace')**2

    def get_results(self,metacal = True):
        results = dict()

        results['shear_bias'] = self.perc_bias(metacal = metacal)
        results['abs_bias'] = self.abs_bias
        results["gal_type"] = self.gal_type
        results["psf_type"] = self.psf_type
        results["gal_sigma"] = self.gal_sigma
        results["psf_sigma"] = self.psf_sigma
        results["e1"] = self.e1
        results["e2"] = self.e2
        results["e"] = self.e
        results["sersicn"] = self.sersicn
        results["gal_hlr"] = self.gal_light.calculateHLR()
        results["psf_hlr"] = self.psf_base.calculateHLR()
        results["psf_model_sigma"] = self.psf_model_sigma
        results['g'] = self.g
        results["dm"] = self.dm
        results["actual_dm"] = self.get_actual_dm()
        results["gal_trace"] = self.get_gal_trace()
        results["psf_trace"] = self.get_psf_trace()

        return results

# Metacalibration implemented in this code is based on ngmix=1.3.8. The code breaks if you install the >2.0.0 versions.

class metacal_shear_measure:
    def __init__(self, final_image, psf_image):
        self.final_image = final_image
        self.final_image_array = final_image.array
        self.psf_image = psf_image
        self.psf_image_array = psf_image.array
        return None

    def measure_shear(self, method):
        self.results = {}
        if method == "estimateShear":
            shear = self.measure_shear_estimateShear()
        elif method == "ngmix":
            shear = self.measure_shear_ngmix()
        elif method == "admomBootstrap":
            shear = self.measure_shear_admombootstrap()
        self.results["g_cal"] = shear
        return 0

    def measure_shear_estimateShear(self):
        obs_results = galsim.hsm.EstimateShear(self.final_image, self.psf_image)

        psf_obs = ngmix.Observation(self.psf_image_array)
        obs = ngmix.Observation(self.final_image_array, psf=psf_obs)

        obdic = ngmix.metacal.get_all_metacal(obs, fixnoise=False)

        g_obs = galsim.Shear(e1=obs_results.corrected_e1, e2=obs_results.corrected_e2)

        self.results["g"] = g_obs

        mcal_results = {}

        for key in obdic:

            mobs = obdic[key]
            mpsf_array = mobs.get_psf().image
            mimage_array = mobs.image

            this_image = galsim.Image(mimage_array)
            this_image_epsf = galsim.Image(mpsf_array)

            res = galsim.hsm.EstimateShear(this_image, this_image_epsf)

            res_shear = galsim.Shear(e1=res.corrected_e1, e2=res.corrected_e2)
            this_res = {"g1": res_shear.g1, "g2": res_shear.g2}
            # print key,this_res
            mcal_results[key] = this_res

        # calculate response R11. The shear by default
        # is 0.01, so dgamma=0.02

        g = np.array([mcal_results["noshear"]["g1"], mcal_results["noshear"]["g2"]])

        R11 = (mcal_results["1p"]["g1"] - mcal_results["1m"]["g1"]) / (0.02)
        R22 = (mcal_results["2p"]["g2"] - mcal_results["2m"]["g2"]) / (0.02)
        R12 = (mcal_results["1p"]["g2"] - mcal_results["1m"]["g2"]) / (0.02)
        R21 = (mcal_results["2p"]["g1"] - mcal_results["2m"]["g1"]) / (0.02)

        R = np.array([[R11, R12], [R21, R22]])
        self.results["R"] = R
        self.results["noshear"] = g
        Rinv = np.linalg.inv(R)

        # print R11,R22

        g_truth = np.matmul(Rinv, g)
        return galsim.Shear(g1=g_truth[0], g2=g_truth[1])

    def make_guess(self, array):

        eps = 0.01
        # shape = galsim.hsm.FindAdaptiveMom(galsim.Image(array))
        pars = np.zeros(6)
        pars[0] = 0
        pars[1] = 0
        pars[2] = 0
        pars[3] = 0
        pars[4] = 100
        pars[5] = 1
        return pars

    def get_results(self):

        return self.results

def do_tests_speed(tests,test_m, test_c, n, debug=False):
    testsresult=[]
    for i in range(len(tests)):
        test = HOMExShapeletPair(*tests[i][:-1],**tests[i][-1])
        if i!=0:
            test.speed_setup_shapelet_psf(test_m[i],test_c[i],n,psf_light, psf_model_light, dm)

            if debug:
                model_img = psf_model_light.drawImage(nx=200, ny=200, scale=0.1)
                psf_img = psf_light.drawImage(nx=200, ny=200, scale=0.1)
                diff = psf_img.array - model_img.array

                # model_e4_1 = test.sxm.moment_measure(model_img, 4, 0)
                # model_e4_1 -= test.sxm.moment_measure(model_img, 0, 4)
                # psf_e4_1 = test.sxm.moment_measure(psf_img, 4, 0)
                # psf_e4_1 -= test.sxm.moment_measure(psf_img, 0, 4)
                # print(model_e4_1, psf_e4_1, model_e4_1-psf_e4_1)

                # model_e4_2 = 2*test.sxm.moment_measure(model_img, 1, 3)
                # model_e4_2 += 2*test.sxm.moment_measure(model_img, 3, 1)
                # psf_e4_2 = 2*test.sxm.moment_measure(psf_img, 1, 3)
                # psf_e4_2 += 2*test.sxm.moment_measure(psf_img, 3, 1)
                # print(model_e4_2, psf_e4_2, model_e4_2-psf_e4_2)

                # model_T4 = test.sxm.moment_measure(model_img, 4, 0) + test.sxm.moment_measure(model_img, 0, 4)
                # model_T4 += 2*test.sxm.moment_measure(model_img, 2, 2)
                # psf_T4 = test.sxm.moment_measure(psf_img, 4, 0) + test.sxm.moment_measure(psf_img, 0, 4)
                # psf_T4 += 2*test.sxm.moment_measure(psf_img, 2, 2)
                # print(model_T4, psf_T4, model_T4 - psf_T4)

                # if abs(model_e4_1 - psf_e4_1) > 1e-4:
                #     print(f"Warning: e4_1 difference is {model_e4_1 - psf_e4_1} at test {i}")
                # if abs(model_e4_2 - psf_e4_2) > 1e-4:
                #     print(f"Warning: e4_2 difference is {model_e4_2 - psf_e4_2} at test {i}")
                # psf_shape_mag = 0.005
                # if abs(model_e4_1 - psf_shape_mag) > 1e-4:
                #     print(f"Warning: model e4_1 not equal to input! {model_e4_1 - psf_shape_mag} at test {i}")
        else:
            test.setup_shapelet_psf(test_m[i],test_c[i],n)
            psf_light = test.psf_light
            psf_model_light = test.psf_model_light
            dm = test.dm
        results = test.get_results(metacal = True)
        testsresult.append(results)
    return testsresult, diff if debug else None

def e2(e1,e):
    return np.sqrt(e**2 - e1**2)


#### below functions are my implementations ####
n_shapelet = 4
if n_shapelet == 4:
    n_moments = 12
elif n_shapelet == 6:
    n_moments = 25

def add_e4_1(c_list, delta):
    c_list[7] -= delta / 2
    c_list[11] += delta / 2

def add_e4_2(c_list, delta):
    c_list[8] += delta / 4
    c_list[10] += delta / 4

def add_e2_1(c_list, delta):
    c_list[2] += delta

def add_e2_2(c_list, delta):
    c_list[0] += delta

def add_T2(c_list, delta, psf_sigma):
    # dsigma = dT / (4 * sigma)
    delta_sigma = delta / (4 * psf_sigma)
    c_list[1] += delta_sigma

def add_T4(m_list, delta, psf_sigma, rho4=2):
    # t4 = m11 * rho4, and m11 = T/2 = sigma**2
    delta_rho4 = delta / (psf_sigma**2)
    # inject drho/rho into m
    m_list[7] += delta_rho4 / rho4
    m_list[9] += delta_rho4 / rho4
    m_list[11] += delta_rho4 / rho4

def run_eta_tests(e_order, t_order, delta_t=0.001):
    psf_shape_mag = 0.005
    psf_sigma = 1.5
    if e_order == 2:
        ## set up PSFs with non-zero g2 components ##
        psf_real = galsim.Gaussian(sigma=psf_sigma).shear(e1=psf_shape_mag, e2=0.0)
        psf_complex = galsim.Gaussian(sigma=psf_sigma).shear(e1=0.0, e2=psf_shape_mag)

    elif e_order == 4:
        config_psf = ["gaussian", 1.5, 0.0, 0.0, 1e-8, 1e-8, "gaussian", 1.5, {'subtract_intersection':True}]
        ## set up PSFs with non-zero e4 components ##
        test_real = HOMExShapeletPair(*config_psf[:-1], **config_psf[-1])
        test_complex = HOMExShapeletPair(*config_psf[:-1], **config_psf[-1])

        m_psf = np.zeros(shape=(n_moments))
        c_psf = np.zeros(shape=(n_moments))
        add_e4_1(c_psf, psf_shape_mag)

        test_real.setup_shapelet_psf(m_psf, c_psf, n_shapelet)
        psf_real = test_real.psf_model_light
        psf_r_img = psf_real.drawImage(scale=0.1, nx=200, ny=200)
        e4_1 = -test_real.sxm.moment_measure(psf_r_img, q=4, p=0)
        e4_1 += test_real.sxm.moment_measure(psf_r_img, q=0, p=4)

        m_psf = np.zeros(shape=(n_moments))
        c_psf = np.zeros(shape=(n_moments))
        add_e4_2(c_psf, psf_shape_mag)

        test_complex.setup_shapelet_psf(m_psf, c_psf, n_shapelet)
        psf_complex = test_complex.psf_model_light
        psf_c_img = psf_complex.drawImage(scale=0.1, nx=200, ny=200)
        e4_2 = 2*test_complex.sxm.moment_measure(psf_c_img, q=1, p=3)
        e4_2 += 2*test_complex.sxm.moment_measure(psf_c_img, q=3, p=1)

        if abs(e4_2 - psf_shape_mag) > 5e-3:
            print('e4 mag not equal to input, something is wrong!')
            print(f"in: {psf_shape_mag}, out: {e4_2}")
        if abs(e4_1 - psf_shape_mag) > 5e-3:
            print('e4 mag not equal to input, something is wrong!')
            print(f"in: {psf_shape_mag}, out: {e4_1}")
        if abs(e4_1 - e4_2)/e4_2 > 1e-3:
            print('e4_1 and e4_2 are not equal, something is wrong!')
            print(f"e4_1: {e4_1}, e4_2: {e4_2}")
        # use the actual measured shape for the dc calculation to get norm correct.
        psf_shape_mag = np.mean([e4_1, e4_2])
    # for size residual
    N = 40
    m_T = np.zeros(shape=(N,n_moments))
    c_T = np.zeros(shape=(N,n_moments))
    if t_order == 2:
        for size_index in range(N):
            add_T2(c_T[size_index], delta_t, psf_sigma)
    elif t_order == 4:
        for size_index in range(N):
            add_T4(m_T[size_index], delta_t, psf_sigma, rho4=2)

    config_eta_real = [(
        "gaussian", sgal,
        0.0, 0.0, 1e-8, 1e-8,
        "gsobj", psf_sigma,
        {'subtract_intersection':True, 'psf_gsobj':psf_real}
        ) for sgal in sigma_gal]

    config_eta_complex = [(
        "gaussian", sgal,
        0.0, 0.0, 1e-8, 1e-8,
        "gsobj", psf_sigma,
        {'subtract_intersection':True, 'psf_gsobj':psf_complex}
        ) for sgal in sigma_gal]

    results_eta_real, diff_real = do_tests_speed(config_eta_real, m_T, c_T, n_shapelet, debug=False)
    results_eta_complex, diff_complex = do_tests_speed(config_eta_complex, m_T, c_T, n_shapelet, debug=False)

    # import matplotlib.pyplot as plt
    # f,a = plt.subplots(1,3, figsize=(8,2.5), sharex=True, sharey=True)

    # m=a[0].imshow(diff_real, vmin=-np.max(diff_real), vmax=np.max(diff_real),
    #               cmap='RdBu', origin='lower')
    # a[0].set_title(rf'Re($e${e_order}), $T${t_order}')
    # plt.colorbar(m, ax=a[0])
    # a[0].grid(alpha=0.25, color='grey')

    # m=a[1].imshow(diff_complex, vmin=-np.max(diff_complex), vmax=np.max(diff_complex),
    #               cmap='RdBu', origin='lower')
    # a[1].set_title(rf'Im($e${e_order}), $T${t_order}')
    # plt.colorbar(m, ax=a[1])
    # a[1].grid(alpha=0.25, color='grey')

    # m=a[2].imshow(diff_real-diff_complex,
    #               vmin=-np.max(diff_real-diff_complex),
    #               vmax=np.max(diff_real-diff_complex),
    #               cmap='RdBu', origin='lower')
    # a[2].set_title(r'$\Delta$')
    # plt.colorbar(m, ax=a[2])
    # a[2].grid(alpha=0.25, color='grey')
    # plt.tight_layout()
    # # plt.savefig('/Users/clairealice/Desktop/e{}_T{}_realvcomplex.png'.format(e_order, t_order), dpi=200)
    # plt.show()

    # f,a = plt.subplots(2,2, figsize=(6,6), sharex=True, sharey=True)

    # from scipy.ndimage import rotate
    # m=a[0,0].imshow(diff_real, #vmin=-np.max(diff_real), vmax=np.max(diff_real),
    #               cmap='RdBu', origin='lower')
    # a[0,0].set_title(r'$\Delta$PSF$_{e1}$')
    # plt.colorbar(m, ax=a[0,0])
    # a[0,0].grid(alpha=0.25, color='grey')
    # m=a[0,1].imshow(rotate(diff_real, angle=45, reshape=False), #vmin=-np.max(diff_real), vmax=np.max(diff_real),
    #               cmap='RdBu', origin='lower')
    # a[0,1].set_title(r'rot($\Delta$PSF$_{e1}$)')
    # plt.colorbar(m, ax=a[0,1])
    # a[0,1].grid(alpha=0.25, color='grey')

    # m=a[1,0].imshow(diff_complex, #vmin=-np.max(diff_complex), vmax=np.max(diff_complex),
    #               cmap='RdBu', origin='lower')
    # a[1,0].set_title(r'$\Delta$PSF$_{e2}$')
    # plt.colorbar(m, ax=a[1,0])
    # a[1,0].grid(alpha=0.25, color='grey')
    # m=a[1,1].imshow(rotate(diff_complex, angle=45, reshape=False), #vmin=-np.max(diff_complex), vmax=np.max(diff_complex),
    #               cmap='RdBu', origin='lower')
    # a[1,1].set_title(r'rot($\Delta$PSF$_{e2}$)')
    # plt.colorbar(m, ax=a[1,1])
    # a[1,1].grid(alpha=0.25, color='grey')
    # plt.tight_layout()
    # plt.savefig('/Users/clairealice/Desktop/e{}_T{}_rot.png'.format(e_order, t_order), dpi=200)
    # plt.show()

    # get size ratio and eta from the results
    size_ratio = np.array([t['psf_sigma']/t['gal_sigma'] for t in results_eta_real])**2

    if t_order == 2:
        t_psf_mag = ngmix.moments.fwhm_to_T(ngmix.moments.sigma_to_fwhm(1.5))
    elif t_order == 4:
        t_psf_mag = 2 * psf_sigma**2  # kurtosis = 2 for gaussian, and T4 = m11 * rho4
    dT_over_T = delta_t / t_psf_mag

    # minus sign bc my residual is obs - model
    eta_real = -np.array([t["abs_bias"][0] for t in results_eta_real]) / (dT_over_T * psf_shape_mag)
    eta_complex = -np.array([t["abs_bias"][1] for t in results_eta_complex]) / (dT_over_T * psf_shape_mag)

    return size_ratio, eta_real, eta_complex, diff_real

def run_beta_tests(e_order, delta_e=0.001):
    psf_sigma = 1.5

    N = 40
    m_e = np.zeros(shape=(N,n_moments))
    c_e1 = np.zeros(shape=(N,n_moments))
    c_e2 = np.zeros(shape=(N,n_moments))
    if e_order == 2:
        for size_index in range(N):
            add_e2_1(c_e1[size_index], delta_e)
            add_e2_2(c_e2[size_index], delta_e)
    elif e_order == 4:
        for size_index in range(N):
            add_e4_1(c_e1[size_index], delta_e)
            add_e4_2(c_e2[size_index], delta_e)

    config_beta_real = [(
        "gaussian", sgal,
        0.0, 0.0, 1e-8, 1e-8,
        "gaussian", psf_sigma,
        {'subtract_intersection':True}
        ) for sgal in sigma_gal]

    config_beta_complex = [(
        "gaussian", sgal,
        0.0, 0.0, 1e-8, 1e-8,
        "gaussian", psf_sigma,
        {'subtract_intersection':True}
        ) for sgal in sigma_gal]

    results_beta_real, diff_real = do_tests_speed(config_beta_real, m_e, c_e1, n_shapelet, debug=False)
    results_beta_complex, diff_complex = do_tests_speed(config_beta_complex, m_e, c_e2, n_shapelet, debug=False)

    # import matplotlib.pyplot as plt
    # f,a = plt.subplots(1,3, figsize=(8,2.5), sharex=True, sharey=True)

    # m=a[0].imshow(diff_real, vmin=-np.max(diff_real), vmax=np.max(diff_real),
    #               cmap='RdBu', origin='lower')
    # a[0].set_title(rf'Re($e${e_order})')
    # plt.colorbar(m, ax=a[0])
    # a[0].grid(alpha=0.25, color='grey')

    # m=a[1].imshow(diff_complex, vmin=-np.max(diff_complex), vmax=np.max(diff_complex),
    #               cmap='RdBu', origin='lower')
    # a[1].set_title(rf'Im($e${e_order})')
    # plt.colorbar(m, ax=a[1])
    # a[1].grid(alpha=0.25, color='grey')

    # m=a[2].imshow(diff_real-diff_complex,
    #               vmin=-np.max(diff_real-diff_complex),
    #               vmax=np.max(diff_real-diff_complex),
    #               cmap='RdBu', origin='lower')
    # a[2].set_title(r'$\Delta$')
    # plt.colorbar(m, ax=a[2])
    # a[2].grid(alpha=0.25, color='grey')
    # plt.tight_layout()
    # plt.savefig('/Users/clairealice/Desktop/e{}_realvcomplex.png'.format(e_order), dpi=200)
    # plt.show()

    # get size ratio and eta from the results
    size_ratio = np.array([t['psf_sigma']/t['gal_sigma'] for t in results_beta_real])**2

    # minus sign to keep sign convenction of residual the same as DES vs HSC
    beta_real = -np.array([t["abs_bias"][0] for t in results_beta_real]) / delta_e
    beta_complex = -np.array([t["abs_bias"][1] for t in results_beta_complex]) / delta_e

    return size_ratio, beta_real, beta_complex, diff_real


if __name__ == '__main__':
    import scipy

    size_ratio = np.sqrt(np.linspace(0.01, 3, 30))
    # size_ratio = np.sqrt(np.ones(2))
    sigma_gal = 1.5 / size_ratio

    results = {}
    # images = np.zeros((6,200,200))
    # order = {'2':0, '4':1, '22':2, '44':3,'24':4, '42':5 }
    for e_order in [2,4]:
        print(f"Running beta{e_order} tests...")
        size_ratio, beta_r, beta_c, diff_img_r = run_beta_tests(e_order, delta_e=0.001)
        results['beta' + str(e_order) + '_r'] = list(beta_r)
        results['beta' + str(e_order) + '_c'] = list(beta_c)
        # images[order[str(e_order)]] = diff_img_r
        for t_order in [2,4]:
            print(f"Running eta{e_order}{t_order} tests...")
            _, eta_r, eta_c, diff_img_r = run_eta_tests(e_order, t_order, delta_t=0.0005)
            results['eta' + str(e_order) + str(t_order) + '_r'] = list(eta_r)
            results['eta' + str(e_order) + str(t_order) + '_c'] = list(eta_c)
            # images[order[str(e_order)+str(t_order)]] = diff_img_r
    results['size_ratio'] = list(size_ratio)


    # np.save('test_images.npy', images)
    import json
    with open('moment_response_tratio_results_mcal.json', 'w') as f:
        json.dump(results, f)
