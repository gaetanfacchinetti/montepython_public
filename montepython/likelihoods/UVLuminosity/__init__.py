
from montepython.likelihood_class import Likelihood

import io_mp
import os

import pickle 
import numpy as np
import nnero


class UVLuminosity(Likelihood):

    def __init__(self, path, data, command_line):

        Likelihood.__init__(self, path, data, command_line)

        print("Initializing UV liminosity likelihood")

        self.need_cosmo_arguments(data, {'output': 'mPk'})
        self.need_cosmo_arguments(data, {'P_k_max_h/Mpc': self.kmax})


    # Start of the actual likelihood computation function
    def loglkl(self, cosmo, data):

        # read the UV luminosity function data
        with open(os.path.join(self.data_directory, "UVData.pkl"), 'rb') as file:
            
            data_uv = pickle.load(file)

            z_uv_exp = data_uv['z_uv']
            m_uv_exp = data_uv['m_uv']

            phi_uv_exp = data_uv['p_uv']

            sigma_phi_uv_down = data_uv['sigma_p_uv_down']
            sigma_phi_uv_up   = data_uv['sigma_p_uv_up']


        if 'ALPHA_STAR' not in data.mcmc_parameters:
            raise io_mp.LikelihoodError("ALPHA_STAR is a required input parameter")    
        
        if 't_STAR' not in data.mcmc_parameters:
            raise io_mp.LikelihoodError("t_STAR is a required input parameter")
         
        if 'LOG10_f_STAR10' not in data.mcmc_parameters:
            raise io_mp.LikelihoodError("LOG10_f_STAR10 is a required input parameter") 
         
        if 'LOG10_M_TURN' not in data.mcmc_parameters:
            raise io_mp.LikelihoodError("LOG10_M_TURN is a required input parameter") 



        alpha_star = data.mcmc_parameters['ALPHA_STAR']['current']*data.mcmc_parameters['ALPHA_STAR']['scale']
        t_star     = data.mcmc_parameters['t_STAR']['current']*data.mcmc_parameters['t_STAR']['scale']
        f_star10   = 10**(data.mcmc_parameters['LOG10_f_STAR10']['current']*data.mcmc_parameters['LOG10_f_STAR10']['scale'])
        m_turn     = 10**(data.mcmc_parameters['LOG10_M_TURN']['current']*data.mcmc_parameters['LOG10_M_TURN']['scale'])
        omega_b    = data.mcmc_parameters['omega_b']['current']*data.mcmc_parameters['omega_b']['scale']
        omega_c    = data.mcmc_parameters['omega_cdm']['current']*data.mcmc_parameters['omega_cdm']['scale']
        
        h = cosmo.h()

        k     = np.logspace(-5, np.log10(cosmo.pars['P_k_max_h/Mpc'] * h), 50000)
        pk    = np.array([cosmo.pk_lin(_k, 0) for _k in k])

        log_lkl = 0

        # loop on the datasets
        for j in [1, 2, 3]:

            # loop on the redshift bins
            for iz, z, in enumerate(z_uv_exp[j]):
                
                try:
                    # predict the UV luminosity function on the range of magnitude m_uv at that redshift bin
                    phi_uv_pred_z = nnero.phi_uv(z, m_uv_exp[j][iz], k, pk, alpha_star, t_star, f_star10, m_turn, omega_b, omega_c, h)[0,0]
                
                except nnero.ShortPowerSpectrumRange:
                    # kill the log likelihood in that case by setting it to -infinity
                    return -np.inf

                # get a sigma that is either the down or the up one depending 
                # if prediction is lower / higher than the experimental value
                mask = (phi_uv_pred_z > phi_uv_exp[j][iz])
                sigma = sigma_phi_uv_down[j][iz]
                sigma[mask] = sigma_phi_uv_up[j][iz][mask]

                # update the log likelihood
                log_lkl = log_lkl + np.sum(np.log(np.sqrt(2.0/np.pi)/(sigma_phi_uv_up[j][iz] + sigma_phi_uv_down[j][iz])) - (phi_uv_pred_z - phi_uv_exp[j][iz])**2/(2*(sigma**2)))

        return log_lkl
