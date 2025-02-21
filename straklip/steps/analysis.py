import os, copy, sys, corner,yaml,shutil,warnings
from copy import deepcopy

import pyklip.fakes as fakes
import matplotlib.pyplot as plt
import pyklip.klip as klip
import pyklip.parallelized as parallelized
from scipy.interpolate import interp1d
import matplotlib.pylab as plt
from astropy.io import fits
import astropy.io.fits as pyfits
from pyklip.kpp.utils.mathfunc import *
from pyklip.instruments.Instrument import GenericData
import pyklip.rdi as rdi
import pyklip.fmlib.fmpsf as fmpsf
import pyklip.fm as fm
import pyklip.fitpsf as fitpsf
from stralog import getLogger
from io import StringIO
import scipy.ndimage.interpolation as sinterp

def get_MODEL_from_data(psf, centers, d=3):
    psf[psf < 0] = 0
    PSF = psf[centers[0] - d:centers[0] + d + 1, centers[1] - d:centers[1] + d + 1].copy()
    PSF = np.tile(PSF, (1, 1))
    return (PSF)


def setup_DATASET(DF, id, filter, numbasis):  # , remove_candidate=False):
    getLogger(__name__).info(f'Setting up Dataset for observation')
    filename = DF.path2out + f'/mvs_tiles/{filter}/tile_ID{id}.fits'
    hdulist = pyfits.open(filename)
    data = hdulist['SCI'].data
    data[data < 0] = 0
    centers = [int((data.shape[1] - 1) / 2), int((data.shape[0] - 1) / 2)]

    residuals = np.array([hdulist[f'KMODE{nb}'].data for nb in numbasis])
    dataset = GenericData([data], [centers], filenames=[filename])
    return (dataset, residuals)

def generate_psflib(DF, id, dataset, filter, d=3, KL=1, dir='./', min_corr=None, badfiles=None):
    getLogger(__name__).info(f'Generating PSF library')
    data = dataset.input[0]
    centers = dataset._centers[0]
    psf_list = [data]
    psf_names = [DF.path2out + f'/mvs_tiles/{filter}/tile_ID{id}.fits']
    models_list = []

    for psfid in DF.mvs_targets_df.loc[
        (DF.mvs_targets_df[f'flag_{filter}'] == 'good_psf') & (DF.mvs_targets_df.mvs_ids != id)].mvs_ids.unique():
        hdul = fits.open(DF.path2out + f'/mvs_tiles/{filter}/tile_ID{psfid}.fits')
        data = hdul['SCI'].data
        data[data < 0] = 0
        model = hdul[f'MODEL{KL}'].data
        model[model < 0] = 0
        model_peak = np.nanmax(model)
        models_list.append(model / model_peak)
        psf_list.append(data)
        psf_names.append(DF.path2out + f'/mvs_tiles/{filter}/tile_ID{psfid}.fits')
        hdul.close()

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

class AnalysisTools():
    def __init__(self,dataset=None,resdataset=None,maskeddataset=None, KLdetect=None, obspsflib = None, obsPSF = None,
                 xcomp=None, ycomp=None, guess_contrast=None, pxsc_arcsec = None, numbasis = None, extract_candidate = None,
                 mask_candidate = None, contrast_curves = None, cal_contrast_curves = None, fwhm=None):
        self.obsdataset = dataset
        self.obspsflib = obspsflib
        self.obsPSF = obsPSF
        self.res = resdataset
        self.masked = maskeddataset
        self.KLdetect = KLdetect
        self.xcomp = xcomp
        self.ycomp = ycomp
        self.guess_contrast = guess_contrast
        self.pxsc_arcsec = pxsc_arcsec
        self.numbasis = numbasis
        self.extract_candidate = extract_candidate
        self.mask_candidate = mask_candidate
        self.contrast_curves = contrast_curves
        self.cal_contrast_curves = cal_contrast_curves
        self.fwhm = fwhm


    def run_FMAstrometry(self,filter,PSF,chaindir,boxsize,dr,fileprefix,outputdir,fitkernel,corr_len_guess,corr_len_range,xrange,yrange,frange,delta_x,delta_y,nwalkers,nburn,nsteps,nthreads,wkl):
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
                                     sep=self.separation_pixels,
                                     pa=self.position_angle,
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
        fma = fitpsf.FMAstrometry(guess_sep=self.separation_pixels,
                                  guess_pa=self.position_angle,
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

        self.con = np.round(fma.fit_flux.bestfit * self.guess_contrast,3)
        self.econ = np.round(fma.fit_flux.error_2sided * self.guess_contrast,3)
        xshift = -(fma.raw_RA_offset.bestfit - delta_x)
        yshift = fma.raw_Dec_offset.bestfit - delta_y
        getLogger(__name__).info(f'dx: {xshift}, dy: {yshift}, contrast: {self.con}, error: {self.econ}')
        self.fma = fma

    def get_sep_and_posang(self):
        # Coordinates of the primary and candidate in pixels
        x1, y1 = self.obsdataset._centers[0][0], self.obsdataset._centers[0][1]  # Primary star's coordinates

        # Calculate separation in pixels
        self.separation_pixels = np.sqrt((self.xcomp - x1)**2 + (self.ycomp - y1)**2)

        # Convert separation to arcseconds
        self.separation_arcsec = self.separation_pixels * self.pxsc_arcsec

        # Calculate position angle in degrees
        # Position angle is measured from north to east
        self.delta_x = x1 - self.xcomp
        self.delta_y = self.ycomp - y1
        self.position_angle = np.degrees(np.arctan2(self.delta_x, self.delta_y))# % 360

        getLogger(__name__).info(f"Separation: {self.separation_arcsec:.2f} arcsec")
        getLogger(__name__).info(f"Position Angle: {self.position_angle:.2f} degrees")

    def candidate_extraction(self, filter, outputdir):
        getLogger(__name__).info(f'Extracting candidate from: {self.obsdataset.filenames}')
        os.makedirs(outputdir, exist_ok=True)
        chaindir = f'{outputdir}/extracted_candidate/chains'
        os.makedirs(chaindir, exist_ok=True)
        self.get_sep_and_posang()
        self.guess_flux = np.nanmax(self.obsdataset.input[0]) * self.guess_contrast
        self.chi2=[]
        for elno in range(len(self.obspsflib.master_library)):
            test_outputdir=f'{outputdir}/extracted_candidate/testPSF4extraction/'
            os.makedirs(test_outputdir, exist_ok=True)
            PSF = get_MODEL_from_data(self.obspsflib.master_library[elno]/np.nanmax(self.obspsflib.master_library[elno]), self.obsdataset._centers[0])
            self.run_FMAstrometry(filter, PSF, chaindir, boxsize=7, dr=5,
                                  fileprefix=f"{filter}", outputdir=test_outputdir,
                                  fitkernel='diag', corr_len_guess=3, corr_len_range=2,
                                  xrange=1, yrange=1, delta_x=self.delta_x, delta_y=self.delta_y, frange=1, nwalkers=100,
                                  nburn=500, nsteps=1000, nthreads=4,
                                  wkl=np.where(self.KLdetect == np.array(self.numbasis))[0])

            self.fma.sampler.flatchain[:, 2] *= self.guess_contrast

            getLogger(__name__).info(f'Saving MCMC plots in: {test_outputdir}')
            # Plot the MCMC fit results.
            all_labels = [r"x", r"y", r"$\alpha$"]
            all_labels = np.append(all_labels, self.fma.covar_param_labels)
            fig = corner.corner(self.fma.sampler.flatchain, labels=all_labels, quantiles=[0.16, 0.5, 0.84], show_titles=True,
                                title_fmt='.4f')
            path = os.path.join(test_outputdir, f'{filter}-corner-{elno}.png')
            fig.savefig(path)
            plt.close(fig)

            fig = self.fma.best_fit_and_residuals()
            path = os.path.join(test_outputdir, f'{filter}-residuals-{elno}.png')
            fig.savefig(path)
            plt.close(fig)

            dx = self.fma.fit_x.bestfit - self.fma.data_stamp_x_center
            dy = self.fma.fit_y.bestfit - self.fma.data_stamp_y_center

            fm_bestfit = self.fma.fit_flux.bestfit * sinterp.shift(self.fma.fm_stamp, [dy, dx])
            if self.fma.padding > 0:
                fm_bestfit = fm_bestfit[self.fma.padding:-self.fma.padding, self.fma.padding:-self.fma.padding]

            # make residual map
            residual_map = self.fma.data_stamp - fm_bestfit
            self.chi2.append(np.nansum(residual_map)**2)
            comp_extracted = {}
            comp_extracted['con'] = self.con
            comp_extracted['econ'] = list(self.econ)
            comp_extracted['x'] = self.fma.fit_x.bestfit
            comp_extracted['y'] = self.fma.fit_y.bestfit
            comp_extracted['PSFref'] = self.obspsflib.master_filenames[elno]

            getLogger(__name__).info(f'Saving candidate dictionary to file: {test_outputdir}/{filter}_comp_extracted_{elno}.pkl')
            with open(test_outputdir + f"{filter}_comp_extracted_{elno}.yaml", "w") as f:
                yaml.dump(comp_extracted, f)

        q=np.where(self.chi2 == np.nanmin(self.chi2))[0][0]
        shutil.copy(test_outputdir + f"{filter}_comp_extracted_{q}.pkl", outputdir + f"/extracted_candidate/{filter}_comp_extracted.pkl")
        shutil.copy(test_outputdir + f'{filter}-residuals-{q}.png',outputdir + f'/extracted_candidate/{filter}-residuals.png')
        shutil.copy(test_outputdir + f'{filter}-corner-{q}.png',outputdir + f'/extracted_candidate/{filter}-corner.png')

    def candidate_masked(self, filter, outputdir, min_corr=None):
        os.makedirs(outputdir + '/masked_candidates', exist_ok=True)
        with open(outputdir+f"/extracted_candidate/{filter}_comp_extracted.yaml", "r") as f:
            comp_extracted = yaml.safe_load(f)
            self.xcomp = comp_extracted['x']
            self.ycomp = comp_extracted['y']

        self.maskeddataset = copy.deepcopy(self.obsdataset)
        self.maskedpsflib =  copy.deepcopy(self.obspsflib)
        self.get_sep_and_posang()
        fakes.inject_planet(self.maskeddataset.input, self.maskeddataset.centers, [-self.obsPSF * np.nanmax(self.obsdataset.input[0])*comp_extracted['con']],
                            self.obsdataset.wcs, self.separation_pixels, self.position_angle,
                            fwhm=self.fwhm)

        # Save the FITS file
        hdu = fits.PrimaryHDU(self.maskeddataset.input[0])
        hdul = fits.HDUList([hdu])
        hdul.writeto(outputdir + f'/masked_candidates/{filter}-masked.fits', overwrite=True)
        hdul.close()

        # self.maskedpsflib.save_correlation(outputdir + f'/masked_candidates//{filter}_masked_corr_matrix.fits', overwrite=True)
        self.maskedpsflib.prepare_library(self.maskeddataset)
        self.maskedpsflib.isgoodpsf = self.maskedpsflib.isgoodpsf[self.maskedpsflib.correlation[0][1:] > min_corr]

        parallelized.klip_dataset(self.maskeddataset,
                                  mode='RDI',
                                  outputdir=outputdir + '/masked_candidates',
                                  fileprefix=f"{filter}-res_masked",
                                  annuli=1,
                                  subsections=1,
                                  movement=0.,
                                  numbasis=self.numbasis,
                                  maxnumbasis=np.nanmax(self.numbasis),
                                  calibrate_flux=False,
                                  aligned_center=self.maskeddataset._centers[0],
                                  psf_library=self.maskedpsflib,
                                  corr_smooth=0,
                                  verbose=False)

    def mk_contrast_curves(self, filter, residuals, outputdir, seps, mask_candidate, klstep, min_corr=None, KLdetect=1):
        getLogger(__name__).info(f'Making contrast curves.')
        os.makedirs(outputdir, exist_ok=True)

        if mask_candidate:
            getLogger(__name__).info(f'Loading candidate dictionary from file: {outputdir}/extracted_candidate/{filter}_comp_extracted.pkl')
            with open(outputdir + f"/extracted_candidate/{filter}_comp_extracted.yaml", "r") as f:
                comp_extracted = yaml.safe_load(f)
            getLogger(__name__).info(f'Loading file: {comp_extracted["PSFref"]} as best PSF model')
            with fits.open(comp_extracted['PSFref']) as kl_hdulist:
                ref = kl_hdulist[f'MODEL{KLdetect}'].data
            self.obsPSF = get_MODEL_from_data(ref/np.nanmax(ref), self.obsdataset._centers[0])
            self.candidate_masked(filter, outputdir, min_corr=min_corr)

            getLogger(__name__).info(f'Loading residuas from file: {outputdir}/masked_candidates/{filter}-res_masked-KLmodes-all.fits')
            with fits.open(f"{outputdir}/masked_candidates/{filter}-res_masked-KLmodes-all.fits") as kl_hdulist:
                residuals =  kl_hdulist[0].data
        else:
            self.maskeddataset=deepcopy(self.obsdataset)

        fig1, ax1 = plt.subplots(figsize=(12,6))
        cc_dict={}
        cc_dict['sep'] = seps
        for elno in range(len(self.numbasis[::klstep])):
            KL = self.numbasis[::klstep][elno]
            klframe = residuals[elno]/np.nanmax(self.maskeddataset.input[0])
            contrast_seps, contrast = klip.meas_contrast(klframe, 0.5, seps[-1]+self.fwhm, self.fwhm,
                                                         center=self.maskeddataset._centers[0], low_pass_filter=False)

            med_interp = interp1d(contrast_seps,
                                  contrast,
                                  fill_value=(contrast[0], contrast[-1]),
                                  bounds_error=False,
                                  kind='slinear')
            contrast_curve_iterp=[]
            for i, sep in enumerate(seps):
                contrast_curve_iterp.append(med_interp(sep))

            if KL == self.KLdetect:
                ax1.plot(seps, contrast_curve_iterp, '-',color = 'k', label=f'KL = {KL}',linewidth=3.0, zorder=5)
            else:
                ax1.plot(seps, contrast_curve_iterp, '-.',label=f'KL = {KL}',linewidth=3.0)

            cc_dict[KL] = med_interp(seps)

        getLogger(__name__).info(f'Saving contrast curves plot in: {outputdir}')
        ax1.set_yscale('log')
        ax1.set_ylabel('5$\sigma$ Contrast')
        ax1.set_xlabel('Separation [pix]')
        ax1.minorticks_on()
        ax1.set_xlim(1, int(np.nanmax(contrast_seps)))
        ax1.set_ylim(1e-2, 1)
        fig1.legend(ncols=3, loc=1)
        fig1.savefig(outputdir + f'/{filter}-raw.png', bbox_inches='tight')
        plt.close()

        getLogger(__name__).info(f'Saving contrast curves dictionary to file: {outputdir}/{filter}_contrast_curves.pkl')
        with open(outputdir + f"/{filter}_contrast_curves.yaml", "w") as f:
            yaml.dump(cc_dict, f)

    def mk_cal_contrast_curves(self, filter, outputdir, inject_fake, mask_candidate, pa_list, klstep, min_corr=None, KLdetect=1):
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
        with open(outputdir + f"/{filter}_contrast_curves.yaml", "r") as f:
            cc_dict = yaml.safe_load(f)

        seps = cc_dict['sep']
        input_contrast_list = [cc_dict[KL] for KL in self.numbasis]
        input_planet_fluxes =  np.median(input_contrast_list, axis=0)*np.nanmax(self.fkdataset.input[0])
        if inject_fake:
            if mask_candidate:
                getLogger(__name__).info(f'Loading candidate dictionary from file: {outputdir}/{filter}_comp_extracted.pkl')
                with open(outputdir + f"/{filter}_comp_extracted.yaml", "r") as f:
                    comp_extracted = yaml.safe_load(f)
                getLogger(__name__).info(f'Loading file: {comp_extracted["PSFref"]} as best PSF model')
                with fits.open(comp_extracted['PSFref']) as kl_hdulist:
                    ref = kl_hdulist[f'MODEL{KLdetect}'].data
                self.obsPSF = get_MODEL_from_data(ref / np.nanmax(ref), self.obsdataset._centers[0])

            os.makedirs(outputdir + '/inj_candidates', exist_ok=True)
            os.makedirs(outputdir + '/inj_candidates/with_fakes', exist_ok=True)
            os.makedirs(outputdir + '/inj_candidates/residuals_with_fakes', exist_ok=True)
            elno=1
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
        ccc_dict['sep'] = cc_dict['sep']
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

            if KL == self.KLdetect:
                ax3.plot(seps, algo_throughput, '-', color='k', ms=2, label=f'KL = {KL}', linewidth=3.0, zorder=5)
            else:
                ax3.plot(seps, algo_throughput, '-.', ms=2, label=f'KL = {KL}', linewidth=3.0)

            ccc_dict[KL] = cc_dict[KL]/algo_throughput
            if KL == self.KLdetect:
                ax2.plot(ccc_dict['sep'], ccc_dict[KL], '-',color='k', label=f'KL = {KL}',linewidth=3.0, zorder=5)
            else:
                ax2.plot(ccc_dict['sep'], ccc_dict[KL], '-.',label=f'KL = {KL}',linewidth=3.0)


        getLogger(__name__).info(f'Saving corrected contrast curves plot in: {outputdir}')
        ax2.set_yscale('log')
        ax2.set_ylabel('5$\sigma$ Contrast')
        ax2.set_xlabel('Separation [pix]')
        ax2.set_xlim(1, int(np.nanmax(cc_dict['sep'])))
        ax2.minorticks_on()
        ax2.set_ylim(1e-2, 1)
        fig2.legend(ncols=3, loc=1)
        fig2.savefig(outputdir+f'/{filter}-calcc.png',bbox_inches='tight')

        getLogger(__name__).info(f'Saving throughput correction curves plot in: {outputdir}')
        ax3.set_ylabel('Throughput')
        ax3.set_xlabel('Separation [pix]')
        ax3.set_xlim(1, int(np.nanmax(seps)))
        ax3.minorticks_on()
        fig3.legend(ncols=3, loc=1)
        fig3.savefig(outputdir+f'/{filter}-throughput.png',bbox_inches='tight')

        plt.close()

        getLogger(__name__).info(f'Saving corrected contrast curves dictionary to file: {outputdir}/{filter}_corr_contrast_curves.pkl')
        with open(outputdir + f"/{filter}_corr_contrast_curves.yaml", "w") as f:
            yaml.dump(ccc_dict, f)

def run_analysis(DF, id, filter, numbasis, fwhm, dataset, obsdataset, residuals, outputdir, xycomp_list=[None, None],
             extract_candidate=True, contrast_curves=True, cal_contrast_curves=True, mask_candidate=True, inject_fake = True,
             guess_contrast=1e-1, pxsc_arcsec = 0.04, KLdetect = 7, klstep = 1  ,min_corr = 0.8,
             pa_list = [0, 45, 90, 135, 180, 225, 270, 315], seps = [1, 2, 3, 4, 5, 10, 15]):

    psflib, PSF = generate_psflib(DF, id, obsdataset, filter,
                                  KL=KLdetect,
                                  dir=outputdir + f"/{filter}_corr_matrix.fits",
                                  min_corr=dataset.pipe_cfg.analysis['min_corr'])

    xcomp, ycomp = xycomp_list
    if np.all([x is None for x in [xcomp, ycomp]]) and np.any([extract_candidate,mask_candidate]):
        max_value = np.nanmax(np.median(residuals, axis=0))
        max_index = np.where(np.median(residuals, axis=0) == max_value)
        xcomp, ycomp = [max_index[1][0], max_index[0][0]]


    analysistools = AnalysisTools(dataset=obsdataset,
                                  KLdetect = KLdetect,
                                  obspsflib = psflib,
                                  obsPSF = PSF,
                                  xcomp = xcomp,
                                  ycomp = ycomp,
                                  guess_contrast=guess_contrast,
                                  pxsc_arcsec = pxsc_arcsec,
                                  numbasis = numbasis,
                                  extract_candidate = extract_candidate,
                                  mask_candidate = mask_candidate,
                                  contrast_curves = contrast_curves,
                                  cal_contrast_curves = cal_contrast_curves,
                                  fwhm = fwhm)

    if extract_candidate:
        analysistools.candidate_extraction(filter, outputdir+f"/candidates/ID{id}")

    if contrast_curves:
        analysistools.mk_contrast_curves(filter, residuals, outputdir+f"/contrast_curves/ID{id}", seps, mask_candidate, klstep, min_corr=min_corr, KLdetect=KLdetect)

    if cal_contrast_curves:
        analysistools.mk_cal_contrast_curves(filter, outputdir+f"/contrast_curves/ID{id}", inject_fake, mask_candidate, pa_list, klstep, min_corr=min_corr, KLdetect=KLdetect)

if __name__ == "steps.analysis":

    def run(packet):
        getLogger(__name__).info(f'Running Analysis step')
        DF = packet['DF']
        dataset = packet['dataset']
        numbasis = np.array(DF.kmodes)

        for id in dataset.pipe_cfg.analysis['id_list']:
            outputdir = f"/Users/gstrampelli/PycharmProjects/FFP_binaries/out"

            for filter in dataset.pipe_cfg.analysis['filter']:
                obsdataset, residuals = setup_DATASET(DF, id, filter, dataset.pipe_cfg.psfsubtraction['kmodes'])
                fwhm = dataset.pipe_cfg.instrument['fwhm'][filter]
                run_analysis(DF, id, filter.lower(), numbasis, fwhm, dataset, obsdataset, residuals, outputdir,
                         xycomp_list=dataset.pipe_cfg.analysis['xycomp_list'] if len(dataset.pipe_cfg.analysis['xycomp_list']) >0 else [None, None],
                         extract_candidate=dataset.pipe_cfg.analysis['extract_candidate'],
                         contrast_curves=dataset.pipe_cfg.analysis['contrast_curves'],
                         cal_contrast_curves=dataset.pipe_cfg.analysis['cal_contrast_curves'],
                         mask_candidate=dataset.pipe_cfg.analysis['mask_candidate'],
                         inject_fake=dataset.pipe_cfg.analysis['inject_fake'],
                         guess_contrast=dataset.pipe_cfg.analysis['guess_contrast'],
                         pxsc_arcsec=dataset.pipe_cfg.instrument['pixelscale'],
                         KLdetect=dataset.pipe_cfg.analysis['KLdetect'] if dataset.pipe_cfg.analysis['KLdetect'] is not None else np.max(dataset.pipe_cfg.psfsubtraction['kmodes']),
                         klstep=dataset.pipe_cfg.analysis['klstep'],
                         min_corr=dataset.pipe_cfg.analysis['min_corr'],
                         pa_list=dataset.pipe_cfg.analysis['pa_list'],
                         seps=dataset.pipe_cfg.analysis['seps'])
