
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
        #self.need_cosmo_arguments(data, {'P_k_max_h/Mpc': self.kmax})

        # read the UV luminosity function data
        with open(os.path.join(self.data_directory, "UVData.pkl"), 'rb') as file:
            
            data_uv = pickle.load(file)

            self.z_uv_exp = data_uv['z_uv']
            self.m_uv_exp = data_uv['m_uv']

            self.phi_uv_exp = data_uv['p_uv']

            self.sigma_phi_uv_down = data_uv['sigma_p_uv_down']
            self.sigma_phi_uv_up   = data_uv['sigma_p_uv_up']



    def get_k_max(self, data) -> (None | float):
        
    
        # if these specific cosmological parameters are used, 
        # it speeds up the computation by allowing to define a smart k_max
        if 'omega_b' not in data.mcmc_parameters:
            return self.kmax
            
        #if 'omega_m' not in data.mcmc_parameters:
        #    return self.kmax

        #if 'h' not in data.mcmc_parameters:
        #    return self.kmax
    

        # check that some required parameters are defined
        if 'alpha_star' not in data.mcmc_parameters:
            raise io_mp.LikelihoodError("alpha_star is a required input parameter")    
        
        if 't_star' not in data.mcmc_parameters:
            raise io_mp.LikelihoodError("t_star is a required input parameter")
         
        if 'log10_f_star10' not in data.mcmc_parameters:
            raise io_mp.LikelihoodError("log10_f_star10 is a required input parameter") 
        

        # get the current values in the chain
        alpha_star = data.mcmc_parameters['alpha_star']['current']*data.mcmc_parameters['alpha_star']['scale']
        t_star     = data.mcmc_parameters['t_star']['current']*data.mcmc_parameters['t_star']['scale']
        f_star10   = 10**(data.mcmc_parameters['log10_f_star10']['current']*data.mcmc_parameters['log10_f_star10']['scale'])

        omega_b = data.mcmc_parameters['omega_b']['current']*data.mcmc_parameters['omega_b']['scale']
        omega_m = 0.3 * (0.7**2) #data.mcmc_parameters['omega_m']['current']*data.mcmc_parameters['omega_m']['scale']
        h       = 0.7 #data.mcmc_parameters['h']['current']*data.mcmc_parameters['h']['scale']

        min_mh = np.inf

        for j in [1, 2, 3]:

            # loop on the redshift bins
            for iz, z, in enumerate(self.z_uv_exp[j]):

                hz = 100 * nnero.cosmology.h_factor_no_rad(z, omega_b, omega_m - omega_b, h)[0, 0] * nnero.CONVERSIONS.km_to_mpc # approximation of the hubble factor
                mh, _ = nnero.astrophysics.m_halo(hz, self.m_uv_exp[j][iz], alpha_star, t_star, f_star10, omega_b, omega_m)[0, 0]

                # set the min of mh
                if np.min(mh) < min_mh:
                    min_mh = np.min(mh)
                    
        rhom0  = omega_m * nnero.CST_MSOL_MPC.rho_c_over_h2        
        k_max = 1.3 * self.c * (3*min_mh/(4*np.pi)/rhom0)**(-1/3)

        #print("We entrer here and return ", np.min([k_max / h, self.kmax]))

        # one should (almost) never need self.kmax if large enough
        # set here as a security to do not make CLASS take to much
        # time and crash
        return np.min([k_max / h, self.kmax])
            
            

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
     

        # The UV luminosity functions are derived for a fixed cosmology
        uv_cosmo = classy.Class()
        uv_arguments = deepcopy(data.cosmo_arguments)
        
        # For the UV luminosity function we must ensure that omega_dm = 0.3, h = 0.7
        # therefore we remove the omega_dm part inside
        uv_arguments['h'] = 0.7
        uv_arguments['omega_m'] = 0.3 * (0.7**2)
        del uv_arguments['omega_cdm']

        #print("UV arguments CLASS:", uv_arguments)
        #print("Original arguments:", data.cosmo_arguments)
        uv_cosmo.set(uv_arguments)

        # This makes the code much longer 
        # but at least we are consistent
        # with the data
        try:
            uv_cosmo.compute()
        except Exception as e :
            print("A problem appears in the execution of CLASS for the following parameters :", uv_arguments)
            raise e

        # get values from CLASS
        h       = 0.7
        omega_m = 0.3 * (0.7**2)
        omega_b = uv_cosmo.omega_b()

        #print("omega_m, h = ", omega_m, h, omega_b)

        k     = np.logspace(-5, np.log10(uv_cosmo.pars['P_k_max_h/Mpc'] * h), 50000)
        pk    = np.array([uv_cosmo.pk_lin(_k, 0) for _k in k])

        log_lkl = 0

        # loop on the datasets
        # we do not include Bouwens et al 2015 (10.1088/0004-637X/803/1/34)
        # stored at index 0, therefore we start the loop at 1
        for j in [1, 2, 3]:

            # loop on the redshift bins
            for iz, z, in enumerate(self.z_uv_exp[j]):

                hz = uv_cosmo.Hubble(z) * 1e-3 * nnero.CST_EV_M_S_K.c_light * nnero.CONVERSIONS.km_to_mpc
                mh, mask = nnero.astrophysics.m_halo(hz, self.m_uv_exp[j][iz], alpha_star, t_star, f_star10, omega_b, omega_m)
                
                try:
                    # predict the UV luminosity function on the range of magnitude m_uv at that redshift bin
                    # in the future, could add sheth_a, sheth_q, sheth_p and c as nuisance parameters
                    phi_uv_pred_z = nnero.phi_uv(z, hz, self.m_uv_exp[j][iz], k, pk, alpha_star, t_star, f_star10, m_turn, omega_b, omega_m, h, 
                                                 self.sheth_a, self.sheth_q, self.sheth_p, window = self.window, c = self.c, mh = mh, mask = mask)[0,0]
                
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
