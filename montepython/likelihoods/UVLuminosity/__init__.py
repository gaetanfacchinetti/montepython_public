
from montepython.likelihood_class import Likelihood

import io_mp
import os

import pickle 
import numpy as np
import nnero

import classy

from copy import deepcopy


class UVLuminosity(Likelihood):

    def __init__(self, path, data, command_line):

        Likelihood.__init__(self, path, data, command_line)

        print("Initializing UV liminosity likelihood")

        self.need_cosmo_arguments(data, {'output': 'mPk'})
        
        self.uv_cosmo = classy.Class()
        self.run_uv_cosmo  = data.options.get('uv_cosmo', True)
        self.uv_cosmo_init = False


        # read the UV luminosity function data
        with open(os.path.join(self.data_directory, "UVData.pkl"), 'rb') as file:
            
            data_uv = pickle.load(file)

            self.z_uv_exp = data_uv['z_uv']
            self.m_uv_exp = data_uv['m_uv']

            self.phi_uv_exp = data_uv['p_uv']

            self.sigma_phi_uv_down = data_uv['sigma_p_uv_down']
            self.sigma_phi_uv_up   = data_uv['sigma_p_uv_up']



    def _get_k_max(self, data) -> (None | float):
        
    
        # if these specific cosmological parameters are used, 
        # it speeds up the computation by allowing to define a smart k_max
        if 'omega_b' not in data.cosmo_arguments:
            return self.kmax
        
        # check that some required parameters are defined
        if 'alpha_star' not in data.astro_arguments:
            raise io_mp.LikelihoodError("alpha_star is a required input parameter")    
        
        if 't_star' not in data.astro_arguments:
            raise io_mp.LikelihoodError("t_star is a required input parameter")
         
        if 'log10_f_star10' not in data.astro_arguments:
            raise io_mp.LikelihoodError("log10_f_star10 is a required input parameter") 
        

        # get the current values in the chain
        alpha_star = data.astro_arguments['alpha_star']
        t_star     = data.astro_arguments['t_star']
        f_star10   = 10**(data.astro_arguments['log10_f_star10'])

        omega_b = data.cosmo_arguments['omega_b']
        omega_m = 0.3 * (0.7**2)
        h       = 0.7 

        min_mh = np.inf

        for j in [1, 2, 3]:

            # loop on the redshift bins
            for iz, z, in enumerate(self.z_uv_exp[j]):

                hz = 100 * nnero.cosmology.h_factor_no_rad(z, omega_b, omega_m - omega_b, h)[0, 0] * nnero.CONVERSIONS.km_to_mpc # approximation of the hubble factor
                mh, _ = nnero.astrophysics.m_halo(hz, self.m_uv_exp[j][iz], alpha_star, t_star, f_star10, omega_b, omega_m)

                mh = mh[0, 0]

                # set the min of mh
                if np.min(mh) < min_mh:
                    min_mh = np.min(mh)
                    
        rhom0  = omega_m * nnero.CST_MSOL_MPC.rho_c_over_h2        
        k_max = 1.3 * self.c * (3*min_mh/(4*np.pi)/rhom0)**(-1/3)

        # one should (almost) never need self.kmax if large enough
        # set here as a security to do not make CLASS take to much
        # time and crash
        return np.min([k_max / h, self.kmax])
            


    def compute_uv_cosmo(self, data, k_max = None):

        # The UV luminosity functions are derived for a fixed cosmology
        uv_arguments = deepcopy(data.cosmo_arguments)

        
        # For the UV luminosity function we must ensure that omega_dm = 0.3, h = 0.7
        # therefore we remove the omega_dm part inside
        uv_arguments['h'] = 0.7
        uv_arguments['omega_m'] = 0.3 * (0.7**2)
        uv_arguments['output'] = 'mPk'

        if 'lensing' in uv_arguments:
            del uv_arguments['lensing']

        if 'l_max_scalars' in uv_arguments:
            del uv_arguments['l_max_scalars']

        if 'omega_cdm' in uv_arguments:
            del uv_arguments['omega_cdm']

        if 'non linear' in uv_arguments:
            del uv_arguments['non linear']
        

        # we can specify a custom value of k_max when we run the cosmology
        if k_max is not None:
            uv_arguments['P_k_max_h/Mpc'] = k_max
        else:
            if 'P_k_max_h/Mpc' in uv_arguments:
                uv_arguments['P_k_max_h/Mpc'] = np.max([uv_arguments['P_k_max_h/Mpc'], self._get_k_max(data)])
            else:
                uv_arguments['P_k_max_h/Mpc'] = self._get_k_max(data)

        # if the uv arguments are different then we recompute, else we do not
        self.uv_cosmo.set(uv_arguments)        

        # This makes the code much longer 
        # but at least we are consistent
        # with the data
        try:
            self.uv_cosmo.compute()
        except Exception as e :
            print("A problem appears in the execution of CLASS for the following parameters :", uv_arguments)
            raise e
        



    # start of the actual likelihood computation function
    def loglkl(self, cosmo, data):


        if 'alpha_star' not in data.mcmc_parameters:
            raise io_mp.LikelihoodError("alpha_star is a required input parameter")    
        
        if 't_star' not in data.mcmc_parameters:
            raise io_mp.LikelihoodError("t_star is a required input parameter")
         
        if 'log10_f_star10' not in data.mcmc_parameters:
            raise io_mp.LikelihoodError("log10_f_star10 is a required input parameter") 
         
        if 'log10_m_turn' not in data.mcmc_parameters:
            raise io_mp.LikelihoodError("log10_m_turn is a required input parameter") 

        # get the current values in the chain
        alpha_star = data.mcmc_parameters['alpha_star']['current']*data.mcmc_parameters['alpha_star']['scale']
        t_star     = data.mcmc_parameters['t_star']['current']*data.mcmc_parameters['t_star']['scale']
        f_star10   = 10**(data.mcmc_parameters['log10_f_star10']['current']*data.mcmc_parameters['log10_f_star10']['scale'])
        m_turn     = 10**(data.mcmc_parameters['log10_m_turn']['current']*data.mcmc_parameters['log10_m_turn']['scale'])
     

        # update the cosmology if needed
        if self.run_uv_cosmo: 
            self.compute_uv_cosmo(data)

        if not self.run_uv_cosmo and not self.uv_cosmo_init:
            self.compute_uv_cosmo(data, self.astro_only_kmax)
            self.uv_cosmo_init = True
        

        # get values from CLASS
        h       = 0.7
        omega_m = 0.3 * (0.7**2)
        omega_b = self.uv_cosmo.omega_b()


        k     = np.logspace(-5, np.log10(self.uv_cosmo.pars['P_k_max_h/Mpc'] * h), 50000)
        pk    = np.array([self.uv_cosmo.pk_lin(_k, 0) for _k in k])

        log_lkl = 0

        # loop on the datasets
        # we do not include Bouwens et al 2015 (10.1088/0004-637X/803/1/34)
        # stored at index 0, therefore we start the loop at 1
        for j in [1, 2, 3]:

            # loop on the redshift bins
            for iz, z, in enumerate(self.z_uv_exp[j]):

                hz = self.uv_cosmo.Hubble(z) * 1e-3 * nnero.CST_EV_M_S_K.c_light * nnero.CONVERSIONS.km_to_mpc
                mh, mask_mh = nnero.astrophysics.m_halo(hz, self.m_uv_exp[j][iz], alpha_star, t_star, f_star10, omega_b, omega_m)
                
                try:
                    # predict the UV luminosity function on the range of magnitude m_uv at that redshift bin
                    # in the future, could add sheth_a, sheth_q, sheth_p and c as nuisance parameters
                    phi_uv_pred_z = nnero.phi_uv(z, hz, self.m_uv_exp[j][iz], k, pk, alpha_star, t_star, f_star10, m_turn, omega_b, omega_m, h, 
                                                 self.sheth_a, self.sheth_q, self.sheth_p, window = self.window, c = self.c, mh = mh, mask = mask_mh)[0,0]
                
                except nnero.ShortPowerSpectrumRange:
                    # kill the log likelihood in that case by setting it to -infinity
                    #return -np.inf
                    raise ValueError('k_max must be too small')

                # get a sigma that is either the down or the up one depending 
                # if prediction is lower / higher than the observed value
                mask = (phi_uv_pred_z > self.phi_uv_exp[j][iz])
                sigma        = np.zeros_like(self.sigma_phi_uv_down[j][iz])
                sigma[~mask] = self.sigma_phi_uv_down[j][iz][~mask]
                sigma[mask]  = self.sigma_phi_uv_up[j][iz][mask]

                # update the log likelihood
                log_lkl = log_lkl + np.sum(np.log(np.sqrt(2.0/np.pi)/(self.sigma_phi_uv_up[j][iz] + self.sigma_phi_uv_down[j][iz])) - (phi_uv_pred_z - self.phi_uv_exp[j][iz])**2/(2*(sigma**2)))

        return log_lkl
