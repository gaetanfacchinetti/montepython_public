
from montepython.likelihood_class import Likelihood

import nnero

import numpy as np
from scipy import special, interpolate


class Reionization(Likelihood):

    def __init__(self, path, data, command_line):

        Likelihood.__init__(self, path, data, command_line)

        print("Initializing Reionization likelihood")

        # defining McGreer data
        # https://doi.org/10.1093/mnras/stu2449
        self.z_reio     = np.array([5.6, 5.9])
        self.x_reio     = np.array([0.96, 0.94])
        self.std_x_reio = np.array([0.05, 0.05])

        # we can imagine adding other constraints here without any problem
            

    # start of the actual likelihood computation function
    def loglkl(self, cosmo, data):
 
        thermo = cosmo.get_thermodynamics()
        xe     = interpolate.interp1d(thermo['z'], thermo['x_e'])(self.z_reio)

        # convert xe = ne/nH to xHII ~ ne / n_b
        # ATTENTION, need to fix YHe value to that in NNERO
        if nnero.constants.CST_NO_DIM.YHe != cosmo.get_current_derived_parameters(['YHe'])['YHe']:
            raise ValueError("If Reionization used, need to fix the value of YHe to that defined in NNERO")
        
        YHe = nnero.constants.CST_NO_DIM.YHe
        mHe = nnero.constants.CST_EV_M_S_K.mass_helium
        mH  = nnero.constants.CST_EV_M_S_K.mass_hydrogen
        fHe = YHe/(1.0-YHe)*mH/mHe

        xHII = xe / (1.0+fHe)


        # initialise the result
        res = np.zeros(self.z_reio.shape)

        # compute the truncated gaussian for the reionization data
        norm_reio = -np.log(1.0 - self.x_reio + np.sqrt(np.pi/2.0)*self.std_x_reio*special.erf(self.x_reio/(np.sqrt(2))/self.std_x_reio))
        res = res + norm_reio
        
        mask = xHII < self.x_reio
        res[mask] = res[mask] - 0.5 * (xHII[mask] - self.x_reio[mask])**2/(self.std_x_reio[mask]**2)

        return np.sum(res)

