"""
Fake star class to create singles and binaries for false positives analysis
"""
import numpy as np

from scipy.ndimage.interpolation import shift as sshift
from photutils.datasets import make_noise_image,apply_poisson_noise

class Fake_Star:
    def __init__(self, psf,flux,shift=None,Sky=0,eSky=0,PNoise=True):
        '''
        Initialize the fake star object


        Parameters
        ----------
        psf : numpy array
            PSF image.
        flux : float
            flux of the simualte star.
        shift : list, optional
            y,x shif for the simulated star to inject. The default is None.
        Sky : float, optional
            value of the Sky. The default is 0.
        eSky : float, optional
            uncertanties of the Sky. The default is 0.

        Returns
        -------
        None.

        '''
        self.flux=flux
        self.psf=psf
        self.Sky=Sky
        self.eSky=eSky
        self.star=(psf*flux).astype(np.float64)
        if shift!=None:self.star=sshift(self.star,shift,order=1,mode='constant')
        if Sky!=0:
            bkg_data=make_noise_image(self.star.shape, distribution='gaussian', mean=self.Sky ,stddev=self.eSky)
            self.star+=bkg_data
        self.star[self.star<0]=0
        if PNoise: self.star = apply_poisson_noise(self.star)


    def combine(self,companion,shift):
        """
        Combine the fake star with onother one to create a binary
        Parameters
        ----------
        companion: another fake_star
        sigmaclip: coordinate position in the tile where to place the companion
        Output
        ------
        None
        """
        companion_shift=sshift(companion,shift,order=1,mode='constant')
        self.binary=self.star.copy()+companion_shift
        self.binary = apply_poisson_noise(self.binary)
        
