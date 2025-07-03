import os, copy, sys,yaml,shutil,corner,emcee,textwrap,pickle
from straklip import config
from straklip.tiles import Tile
from straklip.stralog import getLogger
from copy import deepcopy
from glob import glob
import pyklip.fakes as fakes
import matplotlib.pyplot as plt
import pyklip.klip as klip
import pyklip.parallelized as parallelized
from scipy.interpolate import interp1d
import matplotlib.pylab as plt
import astropy.io.fits as pyfits
from pyklip.kpp.utils.mathfunc import *
from pyklip.instruments.Instrument import GenericData
import pyklip.rdi as rdi
import pyklip.fmlib.fmpsf as fmpsf
import pyklip.fm as fm
import pyklip.fitpsf as fitpsf
from io import StringIO
import scipy.ndimage.interpolation as sinterp
from scipy.ndimage import fourier_shift
from scipy.ndimage import shift
from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u
from mpl_toolkits.axes_grid1 import make_axes_locatable
from photutils import psf as pupsf


def star2epsf(stamp,weights=None, center=[20, 20]):
    """Get the stamp and uncertainties from a Star object and provide it to EPSFStar"""
    epsf = pupsf.EPSFStar(
        stamp,
        weights=weights,
        cutout_center=center
    )
    return epsf

def print_readme_to_file(output_path):
    """
    Prints the content of a README file to another file.

    Args:
        readme_path (str): The path to the README file.
        output_path (str): The path to the output file.
    """

    long_string = """\
    The extracted_candidate folder contains the output from StraKLIP analysis pipeline for this target.
    The 'test4PSFextraction' contains the output for all the MCMC fit performed on the companion to test the different pixel
    phases captured by the survey PSF library. For each of star in the library, an MCMC fit is run on the residuals adopting
    the model of the star deriving from pyKLIP as PSF for the fit.
    A chi2=sum(residual**2) is evaluated and the PSF providing the lower chi2 is selected as best match and its fit is copied
    in the extracted_candidate folder.
    The inj_candidates folder contains all the tiles with the fake injected companions to calibrate the contrast curves
    The masked_candidates folder contains tiles with the companions masked in the data and the residuals
    
    At the end of the analysis, in the 'extracted_candidate' there should be the following files:
    - {filter}_extracted.yaml: In this file are recoded all the relevant information obtained by the analysis pipeline
        regarding the companion and the primary (e.g.; contrast, separation, PA, etc.).
    - {filter}_{cand/ref}_corner.png: corner plot for the MCMC fit of the candidate companion or target star.
    - {filter}_{cand/ref}_residuals.png: diagnostic residuals plots for the MCMC fit of the candidate companion or target star.
    - {filter}_ref_traces.png: diagnostic traces plots for the MCMC fit of target star.
    - {filter}-{raw/cal/throuphut}.png: raw/calibrated contrast line and throughput.
    - {filter}-{raw/cal}_contrast_curves.pkl: pickled dictionary containing the contrast curves.
    """

    # Remove indentation
    dedented_string = textwrap.dedent(long_string).strip()

    # Wrap the text to 80 characters
    # wrapped_string = textwrap.fill(dedented_string, width=80)

    with open(output_path+"README.txt", "w") as f:
        f.write(dedented_string)


def get_MODEL_from_data(psf, centers, d=3):
    psf[psf < 0] = 0
    PSF = psf[centers[0] - d:centers[0] + d + 1, centers[1] - d:centers[1] + d + 1].copy()
    PSF = np.tile(PSF, (1, 1))
    return (PSF)


def setup_DATASET(DF, unq_id, mvs_id, filter, pipe_cfg):  # , remove_candidate=False):
    getLogger(__name__).info(f'Setting up Dataset for observation')
    filename = DF.path2out + f'/mvs_tiles/{filter}/tile_ID{mvs_id}.fits'
    hdulist = pyfits.open(filename)
    data = hdulist['SCI'].data
    data[data < 0] = 0
    centers = [int((data.shape[1] - 1) / 2), int((data.shape[0] - 1) / 2)]

    if pipe_cfg.mktiles['la_cr_remove'] or pipe_cfg.mktiles['cr_remove']:
        residuals = np.array([hdulist[f'CRCLEAN_KMODE{nb}'].data for nb in pipe_cfg.psfsubtraction['kmodes']])
    else:
        residuals = np.array([hdulist[f'KMODE{nb}'].data for nb in pipe_cfg.psfsubtraction['kmodes']])
    dataset = GenericData([data], [centers], filenames=[filename])

    dataset.fullframe_fitsname = [f"{pipe_cfg.paths['data']}/{i}_drc.fits" for i in DF.mvs_targets_df.loc[DF.mvs_targets_df.mvs_ids==mvs_id, f'fits_{filter}'].values]
    # Load WCS from original data
    hdulist = fits.open(dataset.fullframe_fitsname[0])
    dataset.wcs = [WCS(hdulist[1].header)]
    dataset.fullframe_ra_prim = DF.unq_targets_df.loc[DF.unq_targets_df.unq_ids == unq_id].ra.values[0]
    dataset.fullframe_dec_prim = DF.unq_targets_df.loc[DF.unq_targets_df.unq_ids == unq_id].dec.values[0]
    dataset.fullframe_x_prim = DF.mvs_targets_df.loc[DF.mvs_targets_df.mvs_ids==mvs_id, f'x_{filter}'].values-1
    dataset.fullframe_y_prim = DF.mvs_targets_df.loc[DF.mvs_targets_df.mvs_ids==mvs_id, f'y_{filter}'].values-1

    return (dataset, residuals)

def generate_psflib(DF, mvs_id, dataset, filter, d=3, KL=1, dir='./', min_corr=None, badfiles=None,epsf=False):
    getLogger(__name__).info(f'Generating PSF library')
    data = dataset.input[0]
    centers = dataset._centers[0]
    psf_list = [data]
    epsf_list = []
    weight_list = []
    psf_names = [DF.path2out + f'/mvs_tiles/{filter}/tile_ID{mvs_id}.fits']
    models_list = []
    target_cell = DF.mvs_targets_df.loc[(DF.mvs_targets_df.mvs_ids == mvs_id),f'cell_{filter}'].values[0]
    for psfid in DF.mvs_targets_df.loc[(DF.mvs_targets_df[f'cell_{filter}'] == target_cell) & (DF.mvs_targets_df[f'flag_{filter}'] == 'good_psf') & (DF.mvs_targets_df.mvs_ids != mvs_id)].mvs_ids.unique():
        hdul = fits.open(DF.path2out + f'/mvs_tiles/{filter}/tile_ID{psfid}.fits')
        data = hdul['SCI'].data
        data[data < 0] = 0
        if not epsf:
            try:
                model = hdul[f'MODEL{KL}'].data
                model[model < 0] = 0
                model_peak = np.nanmax(model)
                models_list.append(model / model_peak)
            except:
                getLogger(__name__).warning(f'MODEL{KL} missing in ID {psfid}. Skipping')
        else:
            edata = hdul['ERR'].data
            epsf_list.append(data)
            weight = 1/edata
            weight[~np.isfinite(weight)] = 0
            weight_list.append(weight)

        psf_list.append(data)
        psf_names.append(DF.path2out + f'/mvs_tiles/{filter}/tile_ID{psfid}.fits')
        hdul.close()

    if epsf:
        list_of_references=[]
        for psf, weight in zip(epsf_list,weight_list):
            list_of_references.append(star2epsf(psf, weight, center=centers))

        epsf = AnalysisTools.build_epsf(AnalysisTools, list_of_references, shape=psf.shape[0], oversampling=3)
        y, x = np.mgrid[:psf.shape[0], :psf.shape[1]]
        model = epsf.evaluate(x, y, 1, centers[0],centers[1])
        models_list.append(model)

    PSF = get_MODEL_from_data(np.median(models_list, axis=0), centers, d)

    # make the PSF library
    # we need to compute the correlation matrix of all images vs each other since we haven't computed it before
    psflib = rdi.PSFLibrary(np.array(psf_list), centers, np.array(psf_names), compute_correlation=True)

    # save the correlation matrix to disk so that we also don't need to recomptue this ever again
    # In the future we can just pass in the correlation matrix into the PSFLibrary object rather than having it compute it
    psflib.save_correlation(dir, overwrite=True)

    # now we need to prepare the PSF library to reduce this dataset
    # what this does is tell the PSF library to not use files from this star in the PSF library for RDI
    psflib.prepare_library(dataset, badfiles=badfiles)
    psflib.isgoodpsf = psflib.isgoodpsf[psflib.correlation[0][1:] > min_corr]

    return (psflib, PSF)

class Candidate:
    def __init__(self, input, chi2=[], fma=None, mag=None, emag=None, dmag=None, con=None, econ=None, mass=None,
                 emass=None,teff=None,av=None,R=None,ext='cand'):
        self.input = input
        self.chi2 = chi2
        self.mag = mag
        self.emag = emag
        self.dmag = dmag
        self.con = con
        self.econ = econ
        self.fma = fma
        self.mass = mass
        self.emass = emass
        self.teff = teff
        self.R = R
        self.ext = ext

class Primary:
    def __init__(self, mag=None, emag=None, age=None, dist=None, accr=None, av=None, logSPacc=None, ccd=None, ext='prim'):
        self.mag = mag
        self.emag = emag
        self.age = age
        self.dist = dist
        self.accr = accr
        self.av = av
        self.ccd = ccd
        self.logSPacc = logSPacc
        self.ext = ext

class MCMCfit:
    def __init__(self,initial_guess=[],best_fit_params=[],limits=[],filename='',burnin=None,thin=None,nsteps=500,ndim=3,nwalkers=50,debug=False,flux=1,contrast=1,oversampling=1, center=[], dxy=[],epsf=None,size=41,visual_binary=False):
        self.best_fit_params=best_fit_params
        self.filename=filename
        self.burnin=burnin
        self.thin=thin
        self.nsteps=nsteps
        self.initial_guess=initial_guess
        self.nwalkers=nwalkers
        self.ndim=ndim
        self.limits=limits
        self.debug=debug
        self.flux=flux
        self.contrast=contrast
        self.oversampling = oversampling
        self.dxy = dxy
        self.center = center
        self.epsf=epsf
        self.size=size
        self.visual_binary = visual_binary

    def build_model_from_psf(self, params, psf1, psf2, x_guess=0, y_guess=0,):
        '''
        Routine to inject a companion PSF around a single PSF, with an counter clockwise sangle and separation. 0 degree is up.
        :param params:
        :param psf:
        :param binarity:
        :param shifted:
        :return:
        '''
        if self.visual_binary:
            x1, y1, x2, y2, flux, contrast = params

            if psf1 is None or psf2 is None:
                y, x = np.mgrid[:self.size, :self.size]
                model1 = self.epsf.evaluate(x, y, 1, self.center[0] + x_guess - x1, self.center[1] + y1 - y_guess)
                # shifted_model1 = model1/np.max(model1) * flux
                shifted_model1 = model1 * flux
                model2 = self.epsf.evaluate(x, y, 1, self.center[0] + x_guess - x2, self.center[1] + y2 - y_guess)
                # shifted_model2 = model2/np.max(model2) * flux * contrast
                shifted_model2 = model2 * flux * contrast
            else:
                model1 = psf1 * flux
                shifted_model1 = shift(model1, shift=[y1 - y_guess, x_guess - x1], mode='constant', cval=0.0)
                model2 = psf2 * flux * contrast
                shifted_model2 = shift(model2, shift=[y2 - y_guess, x_guess - x2], mode='constant', cval=0.0)

            return shifted_model1 + shifted_model2

        else:

            x1, y1, flux = params
            if psf1 is None:
                y, x = np.mgrid[:self.size, :self.size]
                model1 = self.epsf.evaluate(x, y, flux, self.center[0] + x_guess - x1, self.center[1] + y1 - y_guess)
                shifted_model1 = model1
            else:
                model1 = psf1  * flux
                shifted_model1 = shift(model1, shift=[y1 - y_guess, x_guess - x1], mode='constant', cval=0.0)
            return shifted_model1


    def log_likelihood(self,params, star_image, psf1, psf2, centers, show_plots=False, vmin=None, vmax=None, vminres=None,
                       vmaxres=None, path2fitsfile=None):
        """Log-likelihood function for MCMC.

        Args:
            params (list): [x_shift, y_shift, flux]
            star_image (2D array): Observed image of the star.
            psf (2D array): Normalized PSF (flux sum is 1).

        Returns:
            float: Log-likelihood value.
        """
        model = self.build_model_from_psf(params, psf1, psf2)
        # # Scale the PSF by the flux value

        # Compute the residual between the star and the shifted, scaled PSF
        residual = star_image - model

        # Assuming Gaussian errors, the log-likelihood is proportional to the chi-squared
        log_likelihood = -0.5 * np.nansum(residual ** 2)
        # log_likelihood = -0.5 * np.nansum(residual ** 2/(np.sqrt(abs(residual))**2))

        if show_plots:
            if vmin is None:
                vmin = np.nanmin(star_image)
            if vmax is None:
                vmax = np.nanmax(star_image)
            if vminres is None:
                vminres = np.nanmin(residual)
            if vmaxres is None:
                vmaxres = np.nanmax(residual)
            if self.visual_binary:
                fig, ax = plt.subplots(1, 3, figsize=(21, 7))
                im0 = ax[0].imshow(star_image, origin='lower', vmin=vmin, vmax=vmax)
                ax[0].set_title('Data')
                ax[0].plot([self.center[0]+params[0],self.center[0]-params[2]], [self.center[0]+params[1], self.center[0]+params[3]],'r', marker='x')

                divider0 = make_axes_locatable(ax[0])
                cax0 = divider0.append_axes('right', size='5%', pad=0.05)
                fig.colorbar(im0, cax=cax0, orientation='vertical')

                im1 = ax[1].imshow(model, origin='lower', vmin=vmin, vmax=vmax)
                ax[1].set_title('Model')
                divider1 = make_axes_locatable(ax[1])
                cax1 = divider1.append_axes('right', size='5%', pad=0.05)
                fig.colorbar(im1, cax=cax1, orientation='vertical')

                im2 = ax[2].imshow(star_image - model, origin='lower', vmin=vminres, vmax=vmaxres)
                ax[2].set_title('Residual')
                divider2 = make_axes_locatable(ax[2])
                cax2 = divider2.append_axes('right', size='5%', pad=0.05)
                fig.colorbar(im2, cax=cax2, orientation='vertical')
            else:
                fig, ax = plt.subplots(1, 1, figsize=(7, 7))
                im0 = ax.imshow(star_image, origin='lower', vmin=vmin, vmax=vmax)
                ax.set_title('Data')
                ax.plot([self.center[0]+params[0],self.center[0]-self.dxy[0]], [self.center[0]+params[1], self.center[0]+self.dxy[1]],'r', marker='x')

            if path2fitsfile is not None:
                plt.savefig(path2fitsfile, bbox_inches='tight')
                plt.close()
            else:
                plt.show()

        return log_likelihood

    def log_prior(self,params, limits):
        """Log-prior function for MCMC.

        Args:
            params (list): [x_shift, y_shift, flux]
            limits (list): limits for [x_shift, y_shift]

        Returns:
            float: Log-prior value (log(1) for uniform priors, or -inf if out of bounds).
        """
        if self.visual_binary:
            x1, y1, x2, y2, flux, contrast = params

            # Define uniform priors
            if (limits[0][0] < x1 < limits[0][1]) and (limits[0][0] < y1 < limits[0][1]) and (limits[1][0] < x2 < limits[1][1]) and (limits[1][0] < y2 < limits[1][1]) and (limits[2][0] < flux < limits[2][1]) and (limits[3][0] < contrast < limits[3][1]):
                return 0.0
            else:
                return -np.inf
        else:
            x1, y1, flux = params

            # Define uniform priors
            if (limits[0][0] < x1 < limits[0][1]) and (limits[0][0] < y1 < limits[0][1])  and (limits[1][0] < flux < limits[1][1]):
                return 0.0
            else:
                return -np.inf

    def log_posterior(self,params, star_image, psf1, psf2, limits, centers, show_plots=False, vmin=None, vmax=None,
                      vminres=None, vmaxres=None, path2fitsfile=None):
        """Log-posterior function for MCMC.

        Args:
            params (list): [x_shift, y_shift, flux]
            star_image (2D array): Observed image of the star.
            psf (2D array): Normalized PSF (flux sum is 1).
            limits (list): limits for [x_shift, y_shift]

        Returns:
            float: Log-posterior value (log-likelihood + log-prior).
        """

        lp = self.log_prior(params, limits)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(params, star_image, psf1, psf2, centers, show_plots=show_plots, vmin=vmin,
                                   vmax=vmax, vminres=vminres, vmaxres=vmaxres, path2fitsfile=path2fitsfile)

    def run(self,data,psf1,psf2):
        # psf_masked = psf.copy()
        # Initialize the MCMC sampler
        # Add a small random offset to the initial guess to initialize walkers
        pos = self.initial_guess + 1e-4 * np.random.randn(self.nwalkers, self.ndim)

        moves = [(emcee.moves.DEMove(), 0.7), (emcee.moves.DESnookerMove(), 0.3), ]
        # Create the MCMC sampler object
        sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self.log_posterior, moves=moves,
                                        args=(data, psf1, psf2, self.limits, self.center))

        # Run the MCMC sampler for a number of steps
        sampler.run_mcmc(pos, self.nsteps, progress=True)

        # Extract the samples and compute the best-fit parameters
        samples = sampler.get_chain(flat=True)
        self.best_fit_params = []

        self.tau = sampler.get_autocorr_time(tol=0)
        if self.burnin is None:
            self.burnin = int(self.nsteps * 0.60)
        else:
            self.burnin = self.burnin
        if self.thin is None:
            self.thin = int(0.5 * np.nanmin(self.tau))
        else:
            self.thin = self.thin

        flat_samples = sampler.get_chain(discard=self.burnin, thin=self.thin, flat=True)
        pranges = []
        for i in range(flat_samples.shape[1]):
            pranges.append((np.nanmin(flat_samples[:, i][np.isfinite(flat_samples[:, i])]),
                            np.nanmax(flat_samples[:, i][np.isfinite(flat_samples[:, i])])))

        if self.visual_binary:
            labels = ['x1', 'y1', 'x2', 'y2', 'flux', 'contrast']
        else:
            labels = ['x1', 'y1', 'flux']

        samples = sampler.get_chain()  # Shape: (n_steps, n_walkers, n_dim)
        n_walkers = samples.shape[1]
        fig, ax = plt.subplots(len(labels), 1, figsize=(20, 20), sharex=True)
        for elno in range(len(labels)):
            for i in range(n_walkers):
                ax[elno].plot(samples[:, i, elno], alpha=0.5)
                ax[elno].axvline(self.burnin, color='k', linestyle='--')
            ax[elno].set_ylabel(f"{labels[elno]}")
        ax[elno].set_xlabel("Step number")
        plt.savefig(self.filename + '_traces.png')

        for i in range(self.ndim):
            mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
            q = np.diff(mcmc)
            self.best_fit_params.append([mcmc[1],q[0], q[1]])

        fig = corner.corner(flat_samples, labels=labels, quantiles=[0.16, 0.5, 0.84], show_titles=True,
                            prange=pranges,
                            title_kwargs={"fontsize": 12}, title_fmt=".5f")
        plt.savefig(self.filename + '_corners.png')
        plt.close()

        if self.debug:
            self.log_posterior([i[0] for i in self.best_fit_params], data, psf1, psf2, self.limits, self.center, show_plots=True,
                               path2fitsfile=self.filename + '_log_posterior_residuals.png')

        pass


class AnalysisTools():
    def __init__(self,DF=None,dataset=None,resdataset=None,maskeddataset=None, KLdetect=None, obspsflib = None, obsPSF = None,
                 pixelscale=None, xcomp=None, ycomp=None, guess_contrast=None, numbasis = None, extract_candidate = None,
                 mask_candidate = None, contrast_curves = None, cal_contrast_curves = None, fwhm=None,
                 subtract_companion=False):
        self.DF=DF
        self.obsdataset = dataset
        self.obspsflib = obspsflib
        self.obsPSF = obsPSF
        self.res = resdataset
        self.masked = maskeddataset
        self.KLdetect = KLdetect
        self.pixelscale=pixelscale
        self.xcomp = xcomp
        self.ycomp = ycomp
        self.guess_contrast = guess_contrast
        self.numbasis = numbasis
        self.extract_candidate = extract_candidate
        self.mask_candidate = mask_candidate
        self.contrast_curves = contrast_curves
        self.cal_contrast_curves = cal_contrast_curves
        self.fwhm = fwhm
        self.subtract_companion=subtract_companion

    def build_epsf(self,list_of_references, shape=41, oversampling=3):

        epsfs = pupsf.EPSFStars(list_of_references)
        epsf_builder = pupsf.EPSFBuilder(
            oversampling=oversampling,
            maxiters=10,
            shape=shape * oversampling,
            progress_bar=False
        )
        epsf, fitted_stars = epsf_builder(epsfs)
        return epsf

    def extract_subarray(self, data, center_x, center_y, size=3, flat_and_skip_center=False):
        """
        Extract a subarray from a 2D array based on the center coordinates and subarray size.

        Parameters:
        - data: 2D numpy array, the larger array.
        - center_x: The x-coordinate (row) of the center of the subarray.
        - center_y: The y-coordinate (column) of the center of the subarray.
        - size: The size of the subarray (e.g., 3 for a 3x3 subarray).

        Returns:
        - subarray: The extracted subarray.
        """
        print(int(round(center_x)),int(round(center_y)))
        half_size = int(size // 2)
        print(int(round(center_y)) - half_size,int(round(center_y)) + half_size + 1,
                   int(round(center_x)) - half_size,int(round(center_x)) + half_size + 1)
        subarray = data[int(round(center_y)) - half_size:int(round(center_y)) + half_size + 1,
                   int(round(center_x)) - half_size:int(round(center_x)) + half_size + 1]

        if flat_and_skip_center:
            # Flatten the subarray and remove the central value
            subarray_flat = subarray.flatten()
            center_index = len(subarray_flat) // 2
            subarray = np.delete(subarray_flat, center_index)

        return subarray

    def loss_function(self,
                      params,
                      psf,
                      flux,
                      target_array):
        '''
        Loss function for the minimization process in fit_for_extended_sources.
        '''
        x, y, contrast = params
        shifted_psf = np.fft.ifftn(fourier_shift(np.fft.fftn(psf.copy()*flux*contrast), [y,x])).real
        mse = np.nanmean((target_array - shifted_psf) ** 2)
        return mse

    def estimate_target_candidate_position(self,
                                 filter,
                                 target,
                                 guess_flux,
                                 guess_contrast,
                                 center,
                                 dxy,
                                 outputdir='./',
                                 bounds=None,
                                 initial_params=None,
                                 psf1=None,
                                 psf2=None,
                                 epsf=False,
                                 method='Powell'):
        '''
        Fit for extended sources with a 2D gaussian kernel.

        Parameters
        ----------
        target: 2-D array.
            The target tile containing the companion to be fitted.
        bounds: list of 2-D arrays.
            list of 3 elements, containing the min, max values for the sigma_x, sigma_y and theta_degrees parameters
            for the 2-D gaussian kernel. If None, use default bounds = [(0.01, 20),(0.01, 20),(-180, 180)]
        initial_params: list of floats
            list of initial guesses for the sigma_x, sigma_y and theta_degrees parameters
            for the 2-D gaussian kernel. If None, use default initial_params = [0.1, 0.1, 0]
        method: str
            Minimization method. Default is 'Powell'.

        Returns
        -------
        result: array
            Array containing the fitted  parameters from the minimization process

        '''
        oversampling=None
        if psf1 is None or psf2 is None or epsf:
            list_of_references = []
            for i in self.obspsflib.master_filenames[1:]:
                with fits.open(i) as hdul:
                    if epsf:
                        list_of_references.append(star2epsf(hdul[1].data,center=center))
                    else:
                        list_of_references.append(hdul[1].data)

            if epsf:
                epsf = self.build_epsf(list_of_references, shape=target.shape[0] , oversampling=3)

            else:
                epsf = None
                if psf1 is None:
                    psf1 = np.nanmedian(list_of_references,axis=0)/np.nanmax(np.nanmedian(list_of_references,axis=0))
                if psf2 is None:
                    psf2 = np.nanmedian(list_of_references,axis=0)/np.nanmax(np.nanmedian(list_of_references,axis=0))

        if initial_params is None:
            if not self.subtract_companion:
                initial_params = np.array([0, 0, dxy[0],dxy[1], guess_flux,guess_contrast])
            else:
                initial_params = np.array([0, 0, guess_flux])

        if bounds is None:
            if not self.subtract_companion:
                bounds = np.array([(-1, 1),(np.nanmin(dxy)-1, np.nanmax(dxy)+1), (guess_flux*1e-1,guess_flux*1e1),(0,1)])
            else:
                bounds = np.array([(-1, 1),(guess_flux*1e-1,guess_flux*1e1)])

        self.mcmcfit = MCMCfit(initial_guess=initial_params,limits=bounds, visual_binary = not self.subtract_companion,
                        filename=outputdir + f"/extracted_candidate/{filter}_ref_extracted",
                        burnin=1700,thin=None,nsteps=2000,nwalkers=100,debug=False,ndim=len(initial_params),flux=guess_flux,
                        contrast=guess_contrast,oversampling=oversampling, dxy=dxy, center=center, epsf=epsf,size=target.shape[0])

        data = target.copy()
        if self.subtract_companion:
            model = self.mcmcfit.build_model_from_psf([dxy[0],dxy[1], 1], psf1, psf2)
            target -= model/np.max(model) * guess_flux * guess_contrast
        self.mcmcfit.run(target,psf1,psf2)
        self.mcmcfit.log_posterior([i[0] for i in self.mcmcfit.best_fit_params], data, psf1, psf2, self.mcmcfit.limits, self.mcmcfit.center,
                           show_plots=True,
                           path2fitsfile=self.mcmcfit.filename + '_log_posterior_residuals.png')
        pass

    #Evaluate the separation and theta angle (from +y axis toward +x) between candidate and reference, having the candidate dx and dym where dx is positive to the left, dy is positive upwards
    def evaluate_separation_and_theta(self):
        """
        Evaluate the separation and theta angle (from +x axis toward +y) between candidate and reference.
        """
        # Calculate separation in pixels
        self.candidate.separation_pixels = np.sqrt((self.candidate.dx) ** 2 + (self.candidate.dy) ** 2)

        # Convert separation to arcseconds
        self.candidate.separation_arcsec = self.candidate.separation_pixels * self.pixelscale

        self.candidate.theta_angle = np.degrees(np.arctan2(self.candidate.dy, self.candidate.dx)) % 360

        #### TO DO: this is not the canonical PA since the wcs is wrong for these images so it's asuming noth is always up.
        #### To fix the WCS saved in the fits file, and then change how we evaluate the PA angle.
        ### For now it works, because in our reference frame we assume Noth is up, eas in left (as the default wcs),
        #### but it's not the right approach and can lead to a wrong final estimate of the PA angle.
        self.candidate.pa_angle = float(self.candidate.theta_angle-90)

        #### TO DO: add errors to sep, theta and PA

        getLogger(__name__).info(f"Separation: {self.candidate.separation_pixels:.4f} pixels")
        getLogger(__name__).info(f"Separation: {self.candidate.separation_arcsec:.4f} arcsec")
        getLogger(__name__).info(f"Theta Angle: {self.candidate.theta_angle:.2f} degrees")
        getLogger(__name__).info(f"PA Angle: {self.candidate.pa_angle:.2f} degrees")
        self.check_sep_theta_to_dx_dy()

    def check_sep_theta_to_dx_dy(self):
        """
        Convert separation and theta angle (from +y toward +x) to dx and dy.
        """
        # self.candidate.theta_rad = np.radians(self.candidate.theta_angle)

        dx = self.candidate.separation_pixels * np.cos(self.candidate.theta_angle* np.pi / 180.)
        dy = self.candidate.separation_pixels * np.sin(self.candidate.theta_angle* np.pi / 180.)
        getLogger(__name__).info(f"Checking we are able to retive the correct dx: {dx}, and dy: {dy} from separation: {self.candidate.separation_pixels} and theta angle: {self.candidate.theta_angle}")

    # def get_sep_and_posang(self, filter, cand_extracted, target, guess_flux, guess_contrast, outputdir='./', psf1=None, psf2=None, epsf=False):
    #     # Load WCS
    #     hdulist = fits.open(self.obsdataset.fullframe_fitsname[0])
    #     wcs = WCS(hdulist[1].header)
    #     # Convert pixel coordinates to world coordinates (RA, Dec)
    #     x, y = [self.obsdataset.fullframe_x_prim[0], self.obsdataset.fullframe_y_prim[0]]
    #     IDATA = Tile(data=hdulist[1].data, x=x, y=y, tile_base=self.DF.tilebase, delta=6, inst=self.DF.inst, Python_origin=False)
    #     IDATA.mk_tile(pad_data=True, legend=False, showplot=False, verbose=False, xy_m=True, xy_dmax=5,
    #                   title='OrigSCI', kill_plots=True, cbar=True)
    #
    #     deltax = IDATA.x_m - (IDATA.tile_base - 1) / 2
    #     deltay = IDATA.y_m - (IDATA.tile_base - 1) / 2
    #     self.x_ref, self.y_ref = [x + deltax, y + deltay]
    #
    #     self.estimate_target_candidate_position(filter,self.obsdataset.input[0],guess_flux, guess_contrast,self.obsdataset.centers[0], [cand_extracted['dx'],cand_extracted['dy']],outputdir=outputdir, psf1=psf1, psf2=psf2, epsf=epsf)
    #     self.x_ref_err, self.y_ref_err = [self.mcmcfit.best_fit_params[0][1], self.mcmcfit.best_fit_params[0][2]]
    #     self.x_err, self.y_err = [self.mcmcfit.best_fit_params[1][1], self.mcmcfit.best_fit_params[1][2]]
    #     self.x_ref, self.y_ref = [x + self.mcmcfit.best_fit_params[0][0], y + self.mcmcfit.best_fit_params[1][0]]
    #
    #     if self.mcmcfit.visual_binary:
    #         self.dx, self.dy = [self.mcmcfit.best_fit_params[2][0],self.mcmcfit.best_fit_params[3][0]]  # delta coordinate respect to the position of the target
    #     else:
    #         self.dx, self.dy = [cand_extracted['dx'], cand_extracted['dy']]
    #
    #     self.x, self.y = [self.x_ref - self.dx, self.y_ref + self.dy]
    #     self.ra_dec = wcs.pixel_to_world(self.x, self.y)
    #
    #     self.ra_dec_ref = wcs.pixel_to_world(self.x_ref, self.y_ref)
    #
    #     # Compute the angular separation in arcseconds
    #     self.separation_and_uncertanties()
    #     self.postion_angle_and_uncertanties()
    #
    #     getLogger(__name__).info(f"Separation: {self.separation_arcsec:.4f} arcsec")
    #     getLogger(__name__).info(f"Position Angle: {self.position_angle:.2f} degrees")
    #     passdef get_sep_and_posang(self, filter, cand_extracted, target, guess_flux, guess_contrast, outputdir='./', psf1=None, psf2=None, epsf=False):
    #     # Load WCS
    #     hdulist = fits.open(self.obsdataset.fullframe_fitsname[0])
    #     wcs = WCS(hdulist[1].header)
    #     # Convert pixel coordinates to world coordinates (RA, Dec)
    #     x, y = [self.obsdataset.fullframe_x_prim[0], self.obsdataset.fullframe_y_prim[0]]
    #     IDATA = Tile(data=hdulist[1].data, x=x, y=y, tile_base=self.DF.tilebase, delta=6, inst=self.DF.inst, Python_origin=False)
    #     IDATA.mk_tile(pad_data=True, legend=False, showplot=False, verbose=False, xy_m=True, xy_dmax=5,
    #                   title='OrigSCI', kill_plots=True, cbar=True)
    #
    #     deltax = IDATA.x_m - (IDATA.tile_base - 1) / 2
    #     deltay = IDATA.y_m - (IDATA.tile_base - 1) / 2
    #     self.x_ref, self.y_ref = [x + deltax, y + deltay]
    #
    #     self.estimate_target_candidate_position(filter,self.obsdataset.input[0],guess_flux, guess_contrast,self.obsdataset.centers[0], [cand_extracted['dx'],cand_extracted['dy']],outputdir=outputdir, psf1=psf1, psf2=psf2, epsf=epsf)
    #     self.x_ref_err, self.y_ref_err = [self.mcmcfit.best_fit_params[0][1], self.mcmcfit.best_fit_params[0][2]]
    #     self.x_err, self.y_err = [self.mcmcfit.best_fit_params[1][1], self.mcmcfit.best_fit_params[1][2]]
    #     self.x_ref, self.y_ref = [x + self.mcmcfit.best_fit_params[0][0], y + self.mcmcfit.best_fit_params[1][0]]
    #
    #     if self.mcmcfit.visual_binary:
    #         self.dx, self.dy = [self.mcmcfit.best_fit_params[2][0],self.mcmcfit.best_fit_params[3][0]]  # delta coordinate respect to the position of the target
    #     else:
    #         self.dx, self.dy = [cand_extracted['dx'], cand_extracted['dy']]
    #
    #     self.x, self.y = [self.x_ref - self.dx, self.y_ref + self.dy]
    #     self.ra_dec = wcs.pixel_to_world(self.x, self.y)
    #
    #     self.ra_dec_ref = wcs.pixel_to_world(self.x_ref, self.y_ref)
    #
    #     # Compute the angular separation in arcseconds
    #     self.separation_and_uncertanties()
    #     self.postion_angle_and_uncertanties()
    #
    #     getLogger(__name__).info(f"Separation: {self.separation_arcsec:.4f} arcsec")
    #     getLogger(__name__).info(f"Position Angle: {self.position_angle:.2f} degrees")
    #     pass

    def run_FMAstrometry(self,obj,sep,pa,filter,PSF,chaindir,boxsize,dr,fileprefix,outputdir,fitkernel,corr_len_guess,corr_len_range,xrange,yrange,frange,nwalkers,nburn,nsteps,nthreads,wkl):
        getLogger(__name__).info(f'Running forward modeling')
        # Reroute KLIP printing for our own progress bar
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = StringIO()
        sys.stderr = StringIO()

        # ####################################################################################################
        # setup FM guesses
        # initialize the FM Planet PSF class
        fm_class = fmpsf.FMPlanetPSF(inputs_shape=self.obsdataset.input.shape,
                                     input_wvs=[1],
                                     numbasis=self.numbasis,
                                     sep=sep,
                                     pa=pa,
                                     dflux=self.guess_flux,
                                     input_psfs=np.array([PSF]))

        # PSF subtraction parameters
        # run KLIP-FM
        fm.klip_dataset(self.obsdataset,
                        fm_class,
                        mode='RDI',
                        outputdir=outputdir,
                        fileprefix=fileprefix,
                        annuli=1,
                        subsections=1,
                        movement=1.,
                        numbasis=self.numbasis,
                        maxnumbasis=np.nanmax(self.numbasis),
                        aligned_center=self.obsdataset._centers[0],
                        psf_library=self.obspsflib,
                        mute_progression=True,
                        corr_smooth=0)

        # Open the FM dataset.
        with fits.open(outputdir + f"/{fileprefix}-fmpsf-KLmodes-all.fits") as hdul:
            fm_frame = hdul[0].data[wkl]
            fm_centx = hdul[0].header['PSFCENTX']
            fm_centy = hdul[0].header['PSFCENTY']
        with fits.open(outputdir + f"/{fileprefix}-klipped-KLmodes-all.fits") as hdul:
            data_frame = hdul[0].data[wkl]
            data_centx = hdul[0].header['PSFCENTX']
            data_centy = hdul[0].header['PSFCENTY']

        # Initialize pyKLIP FMAstrometry class.
        fma = fitpsf.FMAstrometry(guess_sep=sep,
                                  guess_pa=pa,
                                  fitboxsize=boxsize)
        fma.generate_fm_stamp(fm_image=fm_frame[0],
                              fm_center=[fm_centx, fm_centy],
                              padding=5)
        fma.generate_data_stamp(data=data_frame[0],
                                data_center=[data_centx, data_centy],
                                dr=dr,
                                exclusion_radius=5)

        corr_len_label = r'$l$'
        fma.set_kernel(fitkernel, [corr_len_guess], [corr_len_label])
        fma.set_bounds(xrange, yrange, frange, [corr_len_range])


        # Make sure that the noise map is invertible.
        noise_map_max = np.nanmax(fma.noise_map)
        fma.noise_map[np.isnan(fma.noise_map)] = noise_map_max
        fma.noise_map[fma.noise_map == 0.] = noise_map_max

        # Run the MCMC fit.
        chain_output = os.path.join(chaindir, f'{filter}-bka_chain.pkl')
        fma.fit_astrometry(nwalkers=nwalkers,
                           nburn=nburn,
                           nsteps=nsteps,
                           numthreads=nthreads,
                           chain_output=chain_output)

        # Restore printing
        sys.stdout = original_stdout
        sys.stderr = original_stderr

        obj.con = np.round(fma.fit_flux.bestfit * self.guess_contrast,3)
        obj.econ = np.round(fma.fit_flux.error_2sided * self.guess_contrast,3)
        getLogger(__name__).info(f' contrast: {obj.con}, error: {obj.econ}')
        obj.fma = fma

    def plot_traces(self, nburn=100, labels=[r"x", r"y", r"$\alpha$", "l"]):
        fig, ax = plt.subplots(len(labels), 1, figsize=(20, 20), sharex=True)
        samples = self.candidate.fma.sampler.get_chain()  # Shape: (n_steps, n_walkers, n_dim)
        n_walkers = samples.shape[1]
        for elno in range(len(labels)):
            for i in range(n_walkers):
                ax[elno].plot(samples[:, i, elno], alpha=0.5)
                # ax[elno].axvline(nburn, color='k', linestyle='--')
            ax[elno].set_ylabel(f"{labels[elno]}")
        ax[elno].set_xlabel("Step number")
        return fig

    def fit_astrometry(self, filter, elno, chaindir, test_outputdir, arc_sec, kwargs=None, obj_extracted={}):
        PSF = get_MODEL_from_data(self.obspsflib.master_library[elno] / np.nanmax(self.obspsflib.master_library[elno]),
                                  self.obsdataset._centers[0])

        if isinstance(kwargs,dict) and 'corr_len_guess' in kwargs.keys():
            corr_len_guess = kwargs['corr_len_guess']
        else:
            corr_len_guess = 3
        if isinstance(kwargs,dict) and 'corr_len_range' in kwargs.keys():
            corr_len_range = kwargs['corr_len_range']
        else:
            corr_len_range = 2
        if isinstance(kwargs,dict) and 'yrange' in kwargs.keys():
            yrange = kwargs['yrange']
        else:
            yrange = 2
        if isinstance(kwargs,dict) and 'xrange' in kwargs.keys():
            xrange = kwargs['xrange']
        else:
            xrange = 2
        if isinstance(kwargs,dict) and 'frange' in kwargs.keys():
            frange = kwargs['frange']
        else:
            frange = 1
        if isinstance(kwargs,dict) and 'nwalkers' in kwargs.keys():
            nwalkers = kwargs['nwalkers']
        else:
            nwalkers = 100
        if isinstance(kwargs,dict) and 'nburn' in kwargs.keys(): #this are the step to burn. It is supposed to be already converged when reach the endo of it.
            nburn = kwargs['nburn']
        else:
            nburn = 500
        if isinstance(kwargs,dict) and 'nsteps' in kwargs.keys(): #this are the step it supposded to run AFTER the burning period. It is supposed to be well converged here.
            nsteps = kwargs['nsteps']
        else:
            nsteps = 200
        if isinstance(kwargs, dict) and 'nthreads' in kwargs.keys():
            nthreads = kwargs['nthreads']
        else:
            nthreads = 4
        if isinstance(kwargs, dict) and 'boxsize' in kwargs.keys():
            boxsize = kwargs['boxsize']
        else:
            boxsize = 7
        if isinstance(kwargs, dict) and 'dr' in kwargs.keys():
            dr = kwargs['dr']
        else:
            dr = 5
        if isinstance(kwargs, dict) and 'fileprefix' in kwargs.keys():
            fileprefix = kwargs['fileprefix']
        else:
            fileprefix = f"{filter}_{self.candidate.ext}_{elno}"
        if isinstance(kwargs, dict) and 'fitkernel' in kwargs.keys():
            fitkernel = kwargs['fitkernel']
        else:
            fitkernel = 'diag'
        if isinstance(kwargs, dict) and 'labels' in kwargs.keys():
            labels = kwargs['labels']
        else:
            labels = [r"dx", r"dy", r"$\alpha$"]

        # theta_angle is a CCW angle stating from X+ toward Y+ axis. The angle requeaired by run_FMAstrometry is
        # a CCW angle from Y+ (North up) toward X- (East left) axis, so we need to subtract 90 degrees to theta_angle to econcile them.
        self.run_FMAstrometry(self.candidate, self.candidate.separation_pixels, self.candidate.pa_angle, filter, PSF, chaindir,
                              boxsize=boxsize, dr=dr,
                              fileprefix=fileprefix, outputdir=test_outputdir,
                              fitkernel=fitkernel, corr_len_guess=corr_len_guess, corr_len_range=corr_len_range,
                              xrange=xrange, yrange=yrange, frange=frange, nwalkers=nwalkers,
                              nburn=nburn, nsteps=nsteps, nthreads=nthreads,
                              wkl=np.where(self.KLdetect == np.array(self.numbasis))[0])

        self.candidate.fma.sampler.flatchain[:, 2] *= self.guess_contrast

        getLogger(__name__).info(f'Saving MCMC plots in: {test_outputdir}')

        # Plot the MCMC fit results.
        labels = np.append(labels, self.candidate.fma.covar_param_labels)
        fig = corner.corner(self.candidate.fma.sampler.flatchain, labels=labels, quantiles=[0.16, 0.5, 0.84],
                            show_titles=True,
                            title_fmt='.4f')
        path = os.path.join(test_outputdir, f'{filter}_{self.candidate.ext}_{elno}_corner.png')
        fig.savefig(path)
        plt.close(fig)

        fig = self.candidate.fma.best_fit_and_residuals()
        path = os.path.join(test_outputdir, f'{filter}_{self.candidate.ext}_{elno}_residuals.png')
        fig.savefig(path)
        plt.close(fig)

        fig = self.plot_traces(nburn=nburn, labels=labels)
        path = os.path.join(test_outputdir, f'{filter}_{self.candidate.ext}_{elno}_traces.png')
        fig.savefig(path)
        plt.close(fig)

        dx = self.candidate.fma.fit_x.bestfit - self.candidate.fma.data_stamp_x_center
        dy = self.candidate.fma.fit_y.bestfit - self.candidate.fma.data_stamp_y_center

        fm_bestfit = self.candidate.fma.fit_flux.bestfit * sinterp.shift(self.candidate.fma.fm_stamp, [dy, dx])
        if self.candidate.fma.padding > 0:
            fm_bestfit = fm_bestfit[self.candidate.fma.padding:-self.candidate.fma.padding, self.candidate.fma.padding:-self.candidate.fma.padding]

        # make residual map
        residual_map = self.candidate.fma.data_stamp - fm_bestfit
        self.candidate.chi2.append(np.nansum(residual_map) ** 2)
        obj_extracted['PSFref'] = str(self.obspsflib.master_filenames[elno])
        obj_extracted['chi2'] = float(np.nansum(residual_map) ** 2)
        obj_extracted['con'] = float(self.candidate.con)
        obj_extracted['econ'] = self.candidate.econ.tolist()
        obj_extracted['dx'] = float(self.candidate.fma.fit_x.bestfit-self.obsdataset._centers[0][0])
        obj_extracted['dy'] = float(self.candidate.fma.fit_y.bestfit-self.obsdataset._centers[0][1])
        obj_extracted['x'] = float(self.candidate.fma.fit_x.bestfit)+1
        obj_extracted['y'] = float(self.candidate.fma.fit_y.bestfit)+1
        obj_extracted['x_err'] = float(self.candidate.fma.fit_x.error)
        obj_extracted['y_err'] = float(self.candidate.fma.fit_y.error)
        obj_extracted['sep_arcsec'] = float(self.candidate.separation_arcsec)
        obj_extracted['sep'] = float(self.candidate.separation_pixels)
        obj_extracted['theta'] = float(self.candidate.theta_angle)
        obj_extracted['PA'] = float(self.candidate.pa_angle)

        return obj_extracted

    def candidate_extraction(self, filter, residuals, outputdir,overwrite=True,path2iso_interp=None, arc_sec=False,kwargs=None):
        getLogger(__name__).info(f'Extracting candidate from: {self.obsdataset.filenames}')
        test_outputdir = f'{outputdir}/extracted_candidate/testPSF4extraction/'
        os.makedirs(outputdir, exist_ok=True)
        os.makedirs(test_outputdir, exist_ok=True)
        chaindir = f'{outputdir}/extracted_candidate/chains'
        os.makedirs(chaindir, exist_ok=True)
        self.candidate.input=self.obsdataset.input[0]
        self.candidate.chi2 = []
        self.guess_flux = np.nanmax(self.obsdataset.input[0]) * self.guess_contrast

        self.candidate.dx = self.xcomp - self.obsdataset._centers[0][0]
        self.candidate.dy = self.ycomp - self.obsdataset._centers[0][1]
        getLogger(__name__).info(f'Getting preliminary sep and theta angle from rough estimate of dx and dy: {self.candidate.dx}, {self.candidate.dy}')

        path2yalms=glob(test_outputdir + f"/{filter}_cand_*_extracted.yaml")
        if len(path2yalms) == 0 or overwrite:
            path2extracted_list = []
            self.evaluate_separation_and_theta()
            for elno in range(1,len(self.obspsflib.master_library)):
                cand_extracted = self.fit_astrometry(filter,  elno, chaindir, test_outputdir, arc_sec,kwargs=kwargs)
                path2extracted_list.append(f'{test_outputdir}/{filter}_cand_{elno}_extracted.yaml')
                getLogger(__name__).info(f'Saving candidate dictionary to file: {test_outputdir}/{filter}_cand_{elno}_extracted.yaml')
                with open(path2extracted_list[-1], "w") as f:
                    yaml.dump(cand_extracted, f, sort_keys=False)
                pass

            q_cand=np.where(self.candidate.chi2 == np.nanmin(self.candidate.chi2))[0][0]
            path2loadyalm = path2extracted_list[q_cand]

        else:
            chi2_list=[]
            for file in path2yalms:
                with open(file, "r") as f:
                    cand_extracted = yaml.safe_load(f)
                    chi2_list.append(float(cand_extracted['chi2']))
            chi2_list=np.array(chi2_list)
            q_cand = np.where(chi2_list == np.nanmin(chi2_list))[0][0]

            path2loadyalm = path2yalms[q_cand]

        with open(path2loadyalm, "r") as f:
            cand_extracted = yaml.safe_load(f)

        self.candidate.chi2=cand_extracted['chi2']
        self.candidate.con=cand_extracted['con']
        self.candidate.econ=cand_extracted['econ']
        self.candidate.dx=cand_extracted['dx']
        self.candidate.dy=cand_extracted['dy']
        self.candidate.dmag=-2.5 * np.log10(cand_extracted['con'])
        self.candidate.mag=self.primary.mag -2.5 * np.log10(cand_extracted['con'])

        cand_extracted['mag'] = float(self.candidate.mag)
        cand_extracted['dmag'] = float(self.candidate.dmag)

        self.evaluate_separation_and_theta()

        guess_flux = np.max(self.obsdataset.input[0] - residuals)
        fuess_comp_contrast = cand_extracted['con']
        psf1=(self.obsdataset.input[0] - residuals)/guess_flux

        with fits.open(cand_extracted['PSFref']) as hdul:
            psf2=(hdul['SCI'].data)/np.max(hdul['SCI'].data)

        if path2iso_interp is not None:
            ## Load interpolated isochrones
            with open(path2iso_interp, 'rb') as file_handle:
                getLogger(__name__).info(f'Loading isochrones interpolator dictionary from file: {path2iso_interp}')
                interp = pickle.load(file_handle)

            if self.primary.logSPacc is not None:
                self.candidate.mass = 10**interp[filter]['logmass'](self.candidate.mag - 5 * np.log10(self.primary.dist / 10), self.primary.age, self.primary.logSPacc)
                self.candidate.teff = interp[filter]['teff'](self.candidate.mag - 5 * np.log10(self.primary.dist / 10), self.primary.age, self.primary.logSPacc)
                self.candidate.R = 10**interp[filter]['logR'](self.candidate.mag - 5 * np.log10(self.primary.dist / 10), self.primary.age, self.primary.logSPacc)
            else:
                self.candidate.mass = 10 ** interp[filter]['logmass'](self.candidate.mag - 5 * np.log10(self.primary.dist / 10), self.primary.age)
                self.candidate.teff = interp[filter]['teff'](self.candidate.mag - 5 * np.log10(self.primary.dist / 10),self.primary.age)
                self.candidate.R = 10 ** interp[filter]['logR'](self.candidate.mag - 5 * np.log10(self.primary.dist / 10), self.primary.age)

            cand_extracted['mass'] = float(self.candidate.mass)
            cand_extracted['teff'] = float(self.candidate.teff)
            cand_extracted['R'] = float(self.candidate.R)

        with open(outputdir + f"/extracted_candidate/{filter}_extracted.yaml", "w") as f:
            yaml.dump(cand_extracted, f,  sort_keys=False)

        if len(path2yalms) != 0:
            q_cand = path2loadyalm.split('_')[-2]

        shutil.copy(test_outputdir + f'{filter}_cand_{q_cand}_residuals.png',
                    outputdir + f'/extracted_candidate/{filter}_cand_residuals.png')
        shutil.copy(test_outputdir + f'{filter}_cand_{q_cand}_corner.png',
                    outputdir + f'/extracted_candidate/{filter}_cand_corner.png')
        pass

    def candidate_masked(self, filter, outputdir, residuals, min_corr=None):
        res = np.copy(residuals)
        os.makedirs(outputdir + '/masked_candidates', exist_ok=True)
        getLogger(__name__).info( f'Loading candidate dictionary from file: {outputdir}/extracted_candidate/{filter}_extracted.yaml')
        with open(outputdir+f"/extracted_candidate/{filter}_extracted.yaml", "r") as f:
            cand_extracted = yaml.safe_load(f)
            self.xcomp = cand_extracted['x']-1
            self.ycomp = cand_extracted['y']-1

        # self.maskeddataset = copy.deepcopy(self.obsdataset)
        # self.maskedpsflib =  copy.deepcopy(self.obspsflib)
        # self.get_temp_sep_and_posang()
        # fakes.inject_planet(self.maskeddataset.input, self.maskeddataset.centers, [-self.obsPSF * np.nanmax(self.obsdataset.input[0])*cand_extracted['con']],
        #                     self.obsdataset.wcs, cand_extracted['sep'], cand_extracted['PA'],
        #                     fwhm=self.fwhm)
        fakes.inject_planet(res, self.obsdataset.centers, [-self.obsPSF * np.nanmax(self.obsdataset.input[0])*cand_extracted['con']],
                            self.obsdataset.wcs, cand_extracted['sep'], cand_extracted['PA'],
                            fwhm=self.fwhm)

        # Save the FITS file
        # hdu = fits.PrimaryHDU(self.maskeddataset.input[0])
        hdu = fits.PrimaryHDU([res])
        hdul = fits.HDUList([hdu])
        hdul.writeto(outputdir + f'/masked_candidates/{filter}-masked.fits', overwrite=True)
        hdul.close()

        # # self.maskedpsflib.save_correlation(outputdir + f'/masked_candidates//{filter}_masked_corr_matrix.fits', overwrite=True)
        # self.maskedpsflib.prepare_library(self.maskeddataset)
        # self.maskedpsflib.isgoodpsf = self.maskedpsflib.isgoodpsf[self.maskedpsflib.correlation[0][1:] > min_corr]
        #
        # parallelized.klip_dataset(self.maskeddataset,
        #                           mode='RDI',
        #                           outputdir=outputdir + '/masked_candidates',
        #                           fileprefix=f"{filter}-res_masked",
        #                           annuli=1,
        #                           subsections=1,
        #                           movement=0.,
        #                           numbasis=self.numbasis,
        #                           maxnumbasis=np.nanmax(self.numbasis),
        #                           calibrate_flux=False,
        #                           aligned_center=self.maskeddataset._centers[0],
        #                           psf_library=self.maskedpsflib,
        #                           corr_smooth=0,
        #                           verbose=False)
        return res

    def mk_contrast_curves(self, filter, residuals, outputdir, seps, mask_candidate, klstep, min_corr=None, KLdetect=1, arc_sec=False,ylim=[1e-4, 1],xlim=None,r=2):
        getLogger(__name__).info(f'Making contrast curves.')
        os.makedirs(outputdir, exist_ok=True)
        getLogger(__name__).info(f'Loading residuas from file: {outputdir}/masked_candidates/{filter}-res_masked-KLmodes-all.fits')

        if mask_candidate:
            # getLogger(__name__).info(f'Loading candidate dictionary from file: {outputdir}/extracted_candidate/{filter}_extracted.yaml')
            # with open(outputdir+ f"/extracted_candidate/{filter}_extracted.yaml", "r") as f:
            #     cand_extracted = yaml.safe_load(f)
            # getLogger(__name__).info(f'Loading file: {cand_extracted["PSFref"]} as best PSF model')
            # with fits.open(cand_extracted['PSFref']) as kl_hdulist:
            #     ref = kl_hdulist[f'MODEL{KLdetect}'].data
            # self.obsPSF = get_MODEL_from_data(ref/np.nanmax(ref), self.obsdataset._centers[0])
            # self.candidate_masked(filter, outputdir, res, min_corr=min_corr)

            res = self.candidate_masked(filter, outputdir, residuals, min_corr=min_corr)


        # else:
            # self.maskeddataset=deepcopy(self.obsdataset)

        fig1, ax1 = plt.subplots(figsize=(12,6))
        cc_dict={}

        for elno in range(len(self.numbasis[::klstep])):
            KL = self.numbasis[::klstep][elno]
            # klframe = residuals[elno]/np.nanmax(self.maskeddataset.input[0])
            # contrast_seps, contrast = klip.meas_contrast(klframe, 0, int(seps[-1])+1, self.fwhm,
            #                                              center=self.maskeddataset._centers[0], low_pass_filter=False)
            klframe = res[elno]/np.nanmax(self.obsdataset.input[0])
            contrast_seps, contrast = klip.meas_contrast(klframe, 0, int(seps[-1])+1, self.fwhm,
                                                         center=self.obsdataset._centers[0], low_pass_filter=False)

            med_interp = interp1d(contrast_seps,
                                  contrast,
                                  fill_value=(contrast[0], contrast[-1]),
                                  bounds_error=False,
                                  kind='slinear')
            contrast_curve_iterp=[]
            for i, sep in enumerate(contrast_seps):
                contrast_curve_iterp.append(med_interp(sep))

            if KL == self.KLdetect:
                if arc_sec:
                    ax1.plot(contrast_seps*self.pixelscale, contrast_curve_iterp, '-', color='k', label=f'KL = {KL}', linewidth=3.0, zorder=5)
                else:
                    ax1.plot(contrast_seps, contrast_curve_iterp, '-',color = 'k', label=f'KL = {KL}',linewidth=3.0, zorder=5)
            else:
                if arc_sec:
                    ax1.plot(contrast_seps*self.pixelscale, contrast_curve_iterp, '-.', label=f'KL = {KL}', linewidth=3.0)
                else:
                    ax1.plot(contrast_seps, contrast_curve_iterp, '-.',label=f'KL = {KL}',linewidth=3.0)

            cc_dict[KL] = med_interp(contrast_seps)
            cc_dict['sep'] = contrast_seps
            # if arc_sec:
            #     cc_dict['sep_arcsec'] = [i * self.pixelscale for i in contrast_seps]

        getLogger(__name__).info(f'Saving contrast curves plot in: {outputdir}')
        ax1.set_yscale('log')
        ax1.set_ylabel('5$\sigma$ Contrast')
        if arc_sec:
            ax1.set_xlabel('Separation ["]')
            if xlim is None:
                ax1.set_xlim(np.nanmin([i * self.pixelscale for i in cc_dict['sep']]), np.nanmax([i * self.pixelscale for i in cc_dict['sep']]))
            else:
                ax1.set_xlim(xlim)
        else:
            ax1.set_xlabel('Separation [pix]')
            if xlim is None:
                ax1.set_xlim(int(np.nanmin(cc_dict['sep'])), int(np.nanmax(cc_dict['sep'])))
            else:
                ax1.set_xlim(xlim)
        ax1.minorticks_on()
        if ylim is None:
            ax1.set_ylim(np.min(contrast_curve_iterp)*0.5, 1)
        else:
            ax1.set_ylim(ylim)
        fig1.legend(ncols=3, loc=1)
        fig1.savefig(outputdir + f'/{filter}-raw.png', bbox_inches='tight')
        plt.close()

        getLogger(__name__).info(f'residual masked plot in: {outputdir}')
        fig2 = plt.figure(figsize=(6,6))
        plt.ylabel('y')
        plt.xlabel('x')
        plt.imshow(residuals[np.where(self.numbasis==self.KLdetect)[0][0]], cmap='gray',origin='lower')
        plt.minorticks_on()
        plt.colorbar()
        fig2.savefig(outputdir + f'/{filter}-masked.png', bbox_inches='tight')
        plt.close()

        getLogger(__name__).info(f'Saving contrast curves dictionary to file: {outputdir}/{filter}_contrast_curves.pkl')
        with open(outputdir + f"/{filter}_contrast_curves.pkl", "wb") as f:
            pickle.dump(cc_dict, f)

    def mk_cal_contrast_curves(self, filter, outputdir, inject_fake, mask_candidate, seps, pa_list, klstep, min_corr=None, KLdetect=1, arc_sec=False,ylim=[1e-4, 1],xlim=None):
        getLogger(__name__).info(f'Making corrected contrast curves.')
        os.makedirs(outputdir, exist_ok=True)
        if mask_candidate:
            getLogger(__name__).info(f'Loading masked candidate data.')
            hdul = fits.open(outputdir + f'/masked_candidates/{filter}-masked.fits')
            data = hdul[0].data
            self.fkdataset = copy.deepcopy(self.obsdataset)
        else:
            self.fkdataset = copy.deepcopy(self.obsdataset)
            data = self.fkdataset.input[0].copy()
        self.fkpsflib = copy.deepcopy(self.obspsflib)

        getLogger(__name__).info(f'Loading contrast curves dictionary from file: {outputdir}/{filter}_contrast_curves.pkl')
        with open(outputdir + f"/{filter}_contrast_curves.pkl", "rb") as f:
            cc_dict = pickle.load(f)

        seps = cc_dict['sep']

        input_contrast_list = [cc_dict[KL] for KL in self.numbasis]
        input_planet_fluxes =  np.median(input_contrast_list, axis=0)*np.nanmax(self.fkdataset.input[0])
        if inject_fake:
            if mask_candidate:
                getLogger(__name__).info(f'Loading candidate dictionary from file: {outputdir}/extracted_candidate/{filter}_extracted.yaml')
                with open(outputdir + f"/extracted_candidate/{filter}_extracted.yaml", "r") as f:
                    cand_extracted = yaml.safe_load(f)
                getLogger(__name__).info(f'Loading file: {cand_extracted["PSFref"]} as best PSF model')
                with fits.open(cand_extracted['PSFref']) as kl_hdulist:
                    ref = kl_hdulist[f'MODEL{KLdetect}'].data
                self.obsPSF = get_MODEL_from_data(ref / np.nanmax(ref), self.obsdataset._centers[0])

            os.makedirs(outputdir + '/inj_candidates', exist_ok=True)
            elno=1
            ### TO DO: interpolate over a smaller sample of separations instead of using all the one from the raw contrast curves
            # print(seps)
            # print(pa_list)
            getLogger(__name__).info(f'Getting ready to inject {int(len(seps)*len(pa_list))} for filter {filter}')
            for input_planet_flux, sep in zip(input_planet_fluxes, seps):
                for pa in pa_list:
                    getLogger(__name__).info(f'Injecting fake {elno} at pa: {pa}, sep: {sep}, flux: {input_planet_flux}')
                    self.fkdataset.input[0] = data.copy()
                    fakes.inject_planet(self.fkdataset.input,self.fkdataset.centers, [self.obsPSF*input_planet_flux], self.fkdataset.wcs, sep, pa,
                                        fwhm=self.fwhm)

                    # self.fkpsflib.save_correlation(outputdir + f'/inj_candidates/{filter}_fake_corr_matrix-withfakes.fits', overwrite=True)
                    self.fkpsflib.prepare_library(self.fkdataset)
                    self.fkpsflib.isgoodpsf = self.fkpsflib.isgoodpsf[self.fkpsflib.correlation[0][1:] > min_corr]

                    parallelized.klip_dataset(self.fkdataset,
                                              mode='RDI',
                                              outputdir=outputdir+'/inj_candidates',
                                              fileprefix=f"{filter}-withfakes_{elno}",
                                              annuli=1,
                                              subsections=1,
                                              movement=0.,
                                              numbasis=self.numbasis,
                                              maxnumbasis=np.nanmax(self.numbasis),
                                              calibrate_flux=False,
                                              aligned_center=self.fkdataset.centers[0],
                                              psf_library=self.fkpsflib,
                                              corr_smooth=0,
                                              verbose=False)
                    elno+=1

        fig2, ax2 = plt.subplots(figsize=(12, 6))
        fig3, ax3 = plt.subplots(figsize=(12, 6))
        ccc_dict={}
        ccc_dict['sep'] = seps
        # if arc_sec:
        #     ccc_dict['sep_arcsec'] = [i * self.pixelscale for i in seps]

        for elno in range(len(self.numbasis[::klstep])):
            KL = self.numbasis[::klstep][elno]
            retrieved_fluxes = []  # will be populated, one for each separation
            input_fluxes = []  # will be populated, one for each separation
            std_fluxes = []
            elno2=1
            for input_planet_flux, sep in zip(input_planet_fluxes, seps):
                fake_planet_fluxes = []
                for pa in pa_list:
                    kl_hdulist = fits.open(f"{outputdir}/inj_candidates/{filter}-withfakes_{elno2}-KLmodes-all.fits")
                    dat_with_fakes = kl_hdulist[0].data[elno]
                    dat_with_fakes_centers = [kl_hdulist[0].header['PSFCENTX'], kl_hdulist[0].header['PSFCENTY']]

                    fake_flux = fakes.retrieve_planet_flux(dat_with_fakes, dat_with_fakes_centers, self.fkdataset.wcs[0],
                                                           sep,
                                                           pa,
                                                           searchrad=5,
                                                           guesspeak=input_planet_flux,
                                                           guessfwhm=self.fwhm,
                                                           refinefit=True)
                    fake_planet_fluxes.append(fake_flux)
                    elno2+=1

                fake_planet_fluxes=np.array(fake_planet_fluxes)[np.array(fake_planet_fluxes)>0]
                median_flux=np.nanmedian(fake_planet_fluxes)
                if median_flux >= 0:
                    retrieved_fluxes.append(median_flux)
                    input_fluxes.append(input_planet_flux)
                    std_fluxes.append(np.nanstd(input_planet_flux))


            # fake planet output / fake planet input = throughput of KLIP
            algo_throughput = np.round(np.array(retrieved_fluxes) / np.array(input_fluxes),2)  # a number less than 1 probably
            if np.any(algo_throughput>1):
                getLogger(__name__).warning(f"Algorithm throughput above 1 in KLmode {KL}: {algo_throughput}")

            if arc_sec:
                seps *= self.pixelscale

            if KL == self.KLdetect:
                ax3.plot(seps, algo_throughput, '-', color='k', ms=2, label=f'KL = {KL}', linewidth=3.0, zorder=5)
            else:
                ax3.plot(seps, algo_throughput, '-.', ms=2, label=f'KL = {KL}', linewidth=3.0)

            ccc_dict[KL] = cc_dict[KL]/algo_throughput
            if KL == self.KLdetect:
                ax2.plot(seps, ccc_dict[KL], '-',color='k', label=f'KL = {KL}',linewidth=3.0, zorder=5)
            else:
                ax2.plot(seps, ccc_dict[KL], '-.',label=f'KL = {KL}',linewidth=3.0)


        getLogger(__name__).info(f'Saving corrected contrast curves plot in: {outputdir}')
        ax2.set_yscale('log')
        ax2.set_ylabel('5$\sigma$ Contrast')
        if arc_sec:
            ax2.set_xlabel('Separation ["]')
            if xlim is None:
                ax2.set_xlim(np.nanmin([i * self.pixelscale for i in ccc_dict['sep']]), np.nanmax([i * self.pixelscale for i in ccc_dict['sep']]))
            else:
                ax2.set_xlim(xlim)
        else:
            ax2.set_xlabel('Separation [pix]')
            if xlim is None:
                ax2.set_xlim(int(np.nanmin(ccc_dict['sep'])), int(np.nanmax(ccc_dict['sep'])))
            else:
                ax2.set_xlim(xlim)
        ax2.minorticks_on()
        if ylim is None:
            ax2.set_ylim(np.min([ccc_dict[KL] for KL in self.numbasis])*0.5, 1)
        else:
            ax2.set_ylim(ylim)
        fig2.legend(ncols=3, loc=1)
        fig2.savefig(outputdir+f'/{filter}-calcc.png',bbox_inches='tight')

        getLogger(__name__).info(f'Saving throughput correction curves plot in: {outputdir}')
        ax3.set_ylabel('Throughput')
        if arc_sec:
            ax3.set_xlabel('Separation ["]')
            if xlim is None:
                ax3.set_xlim(np.nanmin([i * self.pixelscale for i in cc_dict['sep']]), np.nanmax([i * self.pixelscale for i in cc_dict['sep']]))
            else:
                ax3.set_xlim(xlim)
        else:
            ax3.set_xlabel('Separation [pix]')
            if xlim is None:
                ax3.set_xlim(int(np.nanmin(cc_dict['sep'])), int(np.nanmax(cc_dict['sep'])))
            else:
                ax3.set_xlim(xlim)
        ax3.minorticks_on()
        if ylim is None:
            ax3.set_ylim(np.min(algo_throughput) * 0.5, np.max(algo_throughput) )
        else:
            ax3.set_ylim(ylim)
        fig3.legend(ncols=3, loc=1)
        fig3.savefig(outputdir+f'/{filter}-throughput.png',bbox_inches='tight')

        plt.close()

        getLogger(__name__).info(f'Saving corrected contrast curves dictionary to file: {outputdir}/{filter}_corr_contrast_curves.pkl')
        with open(outputdir + f"/{filter}_corr_contrast_curves.pkl", "wb") as f:
            pickle.dump(ccc_dict, f)

    def mk_mass_sensitivity_curves(self,filter, outputdir, path2iso_interp, klstep=1, arc_sec=False,ylim=None,xlim=None, ext='_contrast_curves.pkl'):
        getLogger(__name__).info(f'Making mass sensitivity curves.')
        os.makedirs(outputdir, exist_ok=True)
        getLogger(__name__).info( f'Loading contrast curves dictionary from file: {outputdir}/{filter}{ext}')
        with open(outputdir + f"/{filter}{ext}", "rb") as f:
            cc_dict = pickle.load(f)

        input_contrast_list = [cc_dict[KL] for KL in self.numbasis]
        seps = cc_dict['sep']

        ## Load interpolated isochrones
        with open(path2iso_interp, 'rb') as file_handle:
            getLogger(__name__).info(f'Loading isochrones interpolator dictionary from file: {path2iso_interp}')
            interp = pickle.load(file_handle)

        fig4, ax4 = plt.subplots(figsize=(12, 6))
        # convert specific contrast to magnitude and then to mass given an age and distance for the system
        mass_sensitivity_curves = []
        for elno in range(len(self.numbasis[::klstep])):
            KL = self.numbasis[::klstep][elno]
            contrast_curve = input_contrast_list[::klstep][elno]
            mass_sensitivity_curve = []
            for contrast in contrast_curve:
                delmag = -2.5 * np.log10(contrast)  # mag
                appmag = self.primary.mag + delmag
                if self.primary.logSPacc is not None:
                    mass = 10**interp[filter]['logmass'](appmag - 5 * np.log10(self.primary.dist / 10)-self.primary.av*self.DF.Av[self.primary.ccd][f'm_{filter}'], self.primary.age, self.primary.logSPacc)
                else:
                    mass = 10**interp[filter]['logmass'](appmag - 5 * np.log10(self.primary.dist / 10)-self.primary.av*self.DF.Av[self.primary.ccd][f'm_{filter}'], self.primary.age, self.primary.logSPacc)
                mass_sensitivity_curve.append(mass)
            if arc_sec:
                ax4.plot([i * self.pixelscale for i in cc_dict['sep']],mass_sensitivity_curve, '-.',label=f'KL = {KL}',linewidth=3.0)
            else:
                ax4.plot(cc_dict['sep'],mass_sensitivity_curve, '-.',label=f'KL = {KL}',linewidth=3.0)
            mass_sensitivity_curves.append(mass_sensitivity_curve)  # vegamag

        getLogger(__name__).info(f'Saving corrected contrast curves plot in: {outputdir}')

        ax4.set_yscale('log')
        ax4.set_ylabel('Mass sensitivity')
        if arc_sec:
            ax4.set_xlabel('Separation ["]')
            if xlim is None:
                ax4.set_xlim(np.nanmin([i * self.pixelscale for i in cc_dict['sep']]), np.nanmax([i * self.pixelscale for i in cc_dict['sep']]))
            else:
                ax4.set_xlim(xlim)
        else:
            ax4.set_xlabel('Separation [pix]')
            if xlim is None:
                ax4.set_xlim(int(np.nanmin(cc_dict['sep'])), int(np.nanmax(cc_dict['sep'])))
            else:
                ax4.set_xlim(xlim)
        ax4.minorticks_on()
        if ylim is None:
            ax4.set_ylim(np.min(mass_sensitivity_curves), 1)
        else:
            ax4.set_ylim(ylim)
        fig4.legend(ncols=3, loc=1)
        fig4.savefig(outputdir + f'/{filter}-masssens.png', bbox_inches='tight')
        plt.close()
        pass

def run_analysis(DF, unq_id, mvs_id, filter, numbasis, fwhm, dataset, obsdataset, residuals, outputdir, xycomp_list=[None, None],
             extract_candidate=True, contrast_curves=True, cal_contrast_curves=True,mass_sensitivity_curves=True, mask_candidate=True, inject_fake = True,
             guess_contrast=1e-1, pxsc_arcsec = 0.04, KLdetect = 7, klstep = 1  ,min_corr = 0.8, ext='_contrast_curves.pkl',
             pa_list = [0, 45, 90, 135, 180, 225, 270, 315], seps = [1, 2, 3, 4, 5, 10, 15],overwrite=True,
             subtract_companion=False, arc_sec=False,ylim=[1e-4, 1],xlim=None,path2iso_interp=None,age=1,accr=1e-5,distance=400,av=0,logSPacc=-5,kwargs=None):

    print_readme_to_file(output_path=outputdir+f"/analysis/")
    psflib, PSF = generate_psflib(DF, mvs_id, obsdataset, filter,
                                  KL=KLdetect,
                                  dir=outputdir + f"/{filter}_corr_matrix.fits",
                                  min_corr=dataset.pipe_cfg.analysis['min_corr'],
                                  epsf=dataset.pipe_cfg.analysis['epsf'])

    # xycomp_list is a 1-index array, e.g. coordinates from ds9, while the extraction need a 0-index array to work
    if isinstance(xycomp_list, dict):
        xcomp, ycomp = np.array(xycomp_list[str(mvs_id)]) - 1
    else:
        xcomp, ycomp = np.array(xycomp_list) - 1

    if np.all([x is None for x in [xcomp, ycomp]]) and np.any([extract_candidate,mask_candidate]):
        max_value = np.nanmax(np.median(residuals, axis=0))
        max_index = np.where(np.median(residuals, axis=0) == max_value)
        xcomp, ycomp = [max_index[1][0], max_index[0][0]]


    analysistools = AnalysisTools(DF=DF,
                                  pixelscale=pxsc_arcsec,
                                  dataset=obsdataset,
                                  KLdetect = KLdetect,
                                  obspsflib = psflib,
                                  obsPSF = PSF,
                                  xcomp = xcomp,
                                  ycomp = ycomp,
                                  guess_contrast=guess_contrast,
                                  numbasis = numbasis,
                                  extract_candidate = extract_candidate,
                                  mask_candidate = mask_candidate,
                                  contrast_curves = contrast_curves,
                                  cal_contrast_curves = cal_contrast_curves,
                                  fwhm = fwhm,
                                  subtract_companion=subtract_companion)

    analysistools.primary = Primary(
        mag=DF.mvs_targets_df.loc[DF.mvs_targets_df.mvs_ids == mvs_id, f'm_{filter.lower()}'].values[0],
        emag=DF.mvs_targets_df.loc[DF.mvs_targets_df.mvs_ids == mvs_id, f'e_{filter.lower()}'].values[0],
        age=DF.unq_targets_df.loc[DF.unq_targets_df.unq_ids == unq_id, age].values[0] if isinstance(age,str) else age,
        dist=DF.unq_targets_df.loc[DF.unq_targets_df.unq_ids == unq_id, distance].values[0] if isinstance(distance,str) else distance,
        ccd=DF.mvs_targets_df.loc[DF.mvs_targets_df.mvs_ids == mvs_id, 'ext'].values[0],
        av=DF.unq_targets_df.loc[DF.unq_targets_df.unq_ids == unq_id, av].values[0] if isinstance(av,str) else av,
        logSPacc = DF.unq_targets_df.loc[DF.unq_targets_df.unq_ids == unq_id, logSPacc].values[0] if isinstance(logSPacc, str) else logSPacc)

    analysistools.candidate = Candidate(input=obsdataset,
                                        chi2=[],
                                        ext='cand')
    if extract_candidate:
        analysistools.candidate_extraction(filter,residuals[np.where(np.array(DF.kmodes)==KLdetect)[0][0]], outputdir+f"/analysis/ID{mvs_id}",overwrite=overwrite,path2iso_interp=path2iso_interp,arc_sec=arc_sec,kwargs=kwargs)

    if contrast_curves:
        analysistools.mk_contrast_curves(filter, residuals, outputdir+f"/analysis/ID{mvs_id}", seps, mask_candidate, klstep, min_corr=min_corr, KLdetect=KLdetect,arc_sec=arc_sec,ylim=ylim,xlim=xlim)

    if cal_contrast_curves:
        analysistools.mk_cal_contrast_curves(filter, outputdir+f"/analysis/ID{mvs_id}", inject_fake, mask_candidate, seps,  pa_list, klstep, min_corr=min_corr, KLdetect=KLdetect, arc_sec=arc_sec,ylim=ylim,xlim=xlim)

    if mass_sensitivity_curves and path2iso_interp is not None:
        analysistools.mk_mass_sensitivity_curves(filter,  outputdir+f"/analysis/ID{mvs_id}", path2iso_interp=path2iso_interp, klstep=klstep, arc_sec=arc_sec, ylim=ylim, xlim=xlim, ext=ext)
    else:
        getLogger(__name__).warning(f'Cannot convert contrast curves in mass sensistivity curves without an interpolator!')


# if __name__ == "steps.analysis":

def run(packet):
    getLogger(__name__).info(f'Running Analysis step')
    DF = packet['DF']
    dataset = packet['dataset']
    numbasis = np.array(DF.kmodes)
    outputdir =dataset.pipe_cfg.paths['out']
    config.make_paths(config=None, paths=outputdir + '/analysis/')

    if len(dataset.pipe_cfg.unq_ids_list) == 0:
        unq_ids_list = DF.unq_targets_df.unq_ids.unique()
    else:
        unq_ids_list = dataset.pipe_cfg.unq_ids_list

    for unq_id in unq_ids_list:
        for filter in dataset.pipe_cfg.analysis['filter']:
            mvs_ids_lit = DF.crossmatch_ids_df.loc[DF.crossmatch_ids_df.unq_ids == unq_id].mvs_ids.values
            for mvs_id in mvs_ids_lit:
                if DF.mvs_targets_df.loc[DF.mvs_targets_df.mvs_ids==mvs_id,[f'flag_{filter}']].values[0] != 'rejected':
                    if not DF.mvs_candidates_df.loc[DF.mvs_candidates_df.mvs_ids.isin(mvs_ids_lit), f'flag_{filter}'].empty and DF.mvs_candidates_df.loc[DF.mvs_candidates_df.mvs_ids.isin(mvs_ids_lit), f'flag_{filter}'].values[0] != 'rejected':
                        if dataset.pipe_cfg.analysis['candidate']['coords'] is not None:
                            xycomp_list = dataset.pipe_cfg.analysis['candidate']['coords'],
                            KLdetect = int(dataset.pipe_cfg.analysis['candidate']['KLdetect'])
                        elif dataset.pipe_cfg.analysis['candidate']['coords'] is None:
                            xycomp_list = DF.mvs_candidates_df.loc[DF.mvs_candidates_df.mvs_ids==mvs_id,[f'x_tile_{filter}',f'y_tile_{filter}']].values[0]+1 #xycomp_list is a 1-index array
                            KLdetect = int(DF.mvs_candidates_df.loc[DF.mvs_candidates_df.mvs_ids == mvs_id, f'kmode_{filter}'].values[0])
                        # else:
                        #     xycomp_list = np.round(np.nanmedian(DF.mvs_candidates_df.loc[DF.mvs_candidates_df.mvs_ids.isin(mvs_ids_lit), [ f'x_tile_{filter}', f'y_tile_{filter}']].values, axis=0))+1 #xycomp_list is a 1-index array
                        #     KLdetect = int(np.nanmin(DF.mvs_candidates_df.loc[DF.mvs_candidates_df.mvs_ids.isin(mvs_ids_lit), f'kmode_{filter}'].values))
                    else:
                        xycomp_list = [np.nan, np.nan]
                        KLdetect = np.nanmax(DF.kmodes)
                else:
                    continue
                # if not np.isnan(KLdetect):
                obsdataset, residuals = setup_DATASET(DF, unq_id, mvs_id, filter, dataset.pipe_cfg)
                fwhm = dataset.pipe_cfg.instrument['fwhm'][filter]
                run_analysis(DF, unq_id, mvs_id, filter.lower(), numbasis, fwhm, dataset, obsdataset, residuals, outputdir,
                         extract_candidate=dataset.pipe_cfg.analysis['steps']['extract_candidate'],
                         contrast_curves=dataset.pipe_cfg.analysis['steps']['contrast_curves'],
                         cal_contrast_curves=dataset.pipe_cfg.analysis['steps']['cal_contrast_curves'],
                         mass_sensitivity_curves=dataset.pipe_cfg.analysis['steps']['mass_sensitivity_curves'],
                         mask_candidate=dataset.pipe_cfg.analysis['mask_candidate'],
                         inject_fake=dataset.pipe_cfg.analysis['inject_fake'],
                         pxsc_arcsec=dataset.pipe_cfg.instrument['pixelscale'],
                         klstep=dataset.pipe_cfg.analysis['klstep'],
                         min_corr=dataset.pipe_cfg.analysis['min_corr'],
                         ext=dataset.pipe_cfg.analysis['ext'],
                         pa_list=dataset.pipe_cfg.analysis['pa_list'],
                         seps=dataset.pipe_cfg.analysis['seps'],
                         overwrite=dataset.pipe_cfg.analysis['overwrite'],
                         arc_sec=dataset.pipe_cfg.analysis['arc_sec'],
                         ylim=dataset.pipe_cfg.analysis['ylim'],
                         xlim=dataset.pipe_cfg.analysis['xlim'],
                         path2iso_interp=dataset.pipe_cfg.analysis['path2iso_interp'],
                         age=dataset.pipe_cfg.analysis['primay']['age'],
                         distance=dataset.pipe_cfg.analysis['primay']['dist'],
                         av=dataset.pipe_cfg.analysis['primay']['av'],
                         logSPacc=dataset.pipe_cfg.analysis['primay']['logSPacc'],
                         guess_contrast = dataset.pipe_cfg.analysis['candidate']['guess_contrast'],
                         xycomp_list = xycomp_list,
                         KLdetect = KLdetect,
                         subtract_companion = dataset.pipe_cfg.analysis['candidate']['subtract_companion'],
                         kwargs=dataset.pipe_cfg.analysis['kwargs'])

    DF.save_dataframes(__name__)