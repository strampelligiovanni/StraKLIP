import os
import pyklip.fakes as fakes
import matplotlib.pyplot as plt
import pyklip.klip as klip
import pyklip.parallelized as parallelized
import copy
from scipy.interpolate import interp1d
from datetime import datetime
import config, input_tables
import matplotlib.pylab as plt
from astropy.io import fits
import astropy.io.fits as pyfits
from pyklip.kpp.utils.mathfunc import *
from pyklip.instruments.Instrument import GenericData
import pyklip.rdi as rdi
import pyklip.fmlib.fmpsf as fmpsf
import pyklip.fm as fm
import pyklip.fitpsf as fitpsf
import corner,pickle

def get_MODEL_from_data(psf,centers, d):
    psf[psf < 0] = 0
    PSF = psf[centers[0] - d:centers[0] + d + 1, centers[1] - d:centers[1] + d + 1].copy()
    PSF = np.tile(PSF, (1, 1))
    return(PSF)

def setup_DATASET(DF, id, filter, numbasis):  # , remove_companion=False):
    filename = DF.path2out + f'/mvs_tiles/{filter}/tile_ID{id}.fits'
    hdulist = pyfits.open(filename)
    data = hdulist['SCI'].data
    data[data < 0] = 0
    centers = [int((data.shape[1] - 1) / 2), int((data.shape[0] - 1) / 2)]

    residuals=np.array([hdulist[f'KMODE{nb}'].data for nb in numbasis])
    dataset = GenericData([data], [centers], filenames=[filename])
    return(dataset, residuals)

def generate_psflib(DF,id,dataset,filter,d=3,KL=1,dir='./'):
    data = dataset.input[0]
    centers = dataset._centers[0]
    psf_list = [data]
    psf_names = [DF.path2out + f'/mvs_tiles/{filter}/tile_ID{id}.fits']
    models_list = []

    for psfid in DF.mvs_targets_df.loc[(DF.mvs_targets_df[f'flag_{filter}'] == 'good_psf')&(DF.mvs_targets_df.mvs_ids != id)].mvs_ids.unique():
        hdul = fits.open(DF.path2out + f'/mvs_tiles/{filter}/tile_ID{psfid}.fits')
        data = hdul['SCI'].data
        data[data < 0] = 0
        model = hdul[f'MODEL{KL}'].data
        model[model < 0] = 0
        model_peak = np.nanmax(model)
        models_list.append(model/model_peak)
        psf_list.append(data)
        psf_names.append(DF.path2out + f'/mvs_tiles/{filter}/tile_ID{psfid}.fits')


    PSF = get_MODEL_from_data(np.median(models_list,axis=0), centers, d)

    # make the PSF library
    # we need to compute the correlation matrix of all images vs each other since we haven't computed it before
    psflib = rdi.PSFLibrary(np.array(psf_list), centers, np.array(psf_names), compute_correlation=True)

    # save the correlation matrix to disk so that we also don't need to recomptue this ever again
    # In the future we can just pass in the correlation matrix into the PSFLibrary object rather than having it compute it
    psflib.save_correlation(dir,overwrite=True)

    # now we need to prepare the PSF library to reduce this dataset
    # what this does is tell the PSF library to not use files from this star in the PSF library for RDI
    psflib.prepare_library(dataset)
    # return(psflib,psf_list)
    return(psflib,psf_list,PSF)

def run_FMAstrometry(dataset,PSF,psflib,filter,separation_pixels,position_angle,guess_flux,guess_contrast,numbasis,chaindir,boxsize,dr,fileprefix,outputdir,fitkernel,corr_len_guess,corr_len_range,xrange,yrange,frange,delta_x,delta_y,nwalkers,nburn,nsteps,nthreads,wkl):
    # ####################################################################################################
    # setup FM guesses
    # initialize the FM Planet PSF class
    fm_class = fmpsf.FMPlanetPSF(inputs_shape=dataset.input.shape,
                                 input_wvs=[1],
                                 numbasis=numbasis,
                                 sep=separation_pixels,
                                 pa=position_angle,
                                 dflux=guess_flux,
                                 input_psfs=np.array([PSF]))

    # PSF subtraction parameters
    # run KLIP-FM
    fm.klip_dataset(dataset,
                    fm_class,
                    mode='RDI',
                    outputdir=outputdir,
                    fileprefix=fileprefix,
                    annuli=1,
                    subsections=1,
                    movement=1.,
                    numbasis=numbasis,
                    maxnumbasis=np.nanmax(numbasis),
                    aligned_center=dataset._centers[0],
                    psf_library=psflib,
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
    fma = fitpsf.FMAstrometry(guess_sep=separation_pixels,
                              guess_pa=position_angle,
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

    con = np.round(fma.fit_flux.bestfit * guess_contrast,3)
    econ = np.round(fma.fit_flux.error_2sided * guess_contrast,3)
    xshift = -(fma.raw_RA_offset.bestfit - delta_x)
    yshift = fma.raw_Dec_offset.bestfit - delta_y
    print(f'dx: {xshift}, dy: {yshift}, contrast: {con}, error: {econ}')
    return(fma,con,econ)

def get_sep_and_posang(dataset,x2,y2,pxsc_arcsec):
    # Coordinates of the primary and companion in pixels
    x1, y1 = dataset._centers[0][0], dataset._centers[0][1]  # Primary star's coordinates

    # Calculate separation in pixels
    separation_pixels = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    # Convert separation to arcseconds
    separation_arcsec = separation_pixels * pxsc_arcsec

    # Calculate position angle in degrees
    # Position angle is measured from north to east
    delta_x = x1 - x2
    delta_y = y2 - y1
    position_angle = np.degrees(np.arctan2(delta_x, delta_y))# % 360

    print(f"Separation: {separation_arcsec:.2f} arcsec")
    print(f"Position Angle: {position_angle:.2f} degrees")
    return(separation_pixels, separation_arcsec, position_angle, delta_x, delta_x)


def companion_extraction(filter, outputdir, dataset, xcomp, ycomp, pxsc_arcsec, guess_contrast, numbasis, KLdetect):
    chaindir = f'{outputdir}/extracted_companion/chains'
    os.makedirs(chaindir, exist_ok=True)
    separation_pixels, separation_arcsec, position_angle, delta_x, delta_y = get_sep_and_posang(dataset, xcomp, ycomp,
                                                                                                pxsc_arcsec)
    guess_flux = np.nanmax(dataset.input[0]) * guess_contrast
    fma, con, econ = run_FMAstrometry(dataset, PSF, psflib, filter, separation_pixels, position_angle, guess_flux,
                                      guess_contrast,
                                      numbasis, chaindir, boxsize=7, dr=5,
                                      fileprefix=f"{filter}", outputdir=outputdir + '/extracted_companion',
                                      fitkernel='diag', corr_len_guess=3, corr_len_range=2,
                                      xrange=1, yrange=1, delta_x=delta_x, delta_y=delta_y, frange=1, nwalkers=100,
                                      nburn=500, nsteps=1000, nthreads=4,
                                      wkl=np.where(KLdetect == np.array(numbasis))[0])

    fma.sampler.flatchain[:, 2] *= guess_contrast
    # Plot the MCMC fit results.
    all_labels = [r"x", r"y", r"$\alpha$"]
    all_labels = np.append(all_labels, fma.covar_param_labels)
    fig = corner.corner(fma.sampler.flatchain, labels=all_labels, quantiles=[0.16, 0.5, 0.84], show_titles=True,
                        title_fmt='.4f')
    path = os.path.join(outputdir + '/extracted_companion', f'{filter}-corner.png')
    fig.savefig(path)
    plt.close(fig)

    fig = fma.best_fit_and_residuals()
    path = os.path.join(outputdir + '/extracted_companion', f'{filter}-residuals.png')
    fig.savefig(path)
    plt.close(fig)

    comp_extracted = {}
    comp_extracted['con'] = con
    comp_extracted['econ'] = list(econ)
    comp_extracted['x'] = fma.fit_x.bestfit
    comp_extracted['y'] = fma.fit_y.bestfit

    with open(outputdir + f"/extracted_companion/{filter}_comp_extracted.pkl", "wb") as f:
        pickle.dump(comp_extracted, f)

def companion_masked(filter, outputdir, dataset, pxsc_arcsec, PSF, dataset_fwhm, numbasis):
    with open(outputdir+f"/extracted_companion/{filter}_comp_extracted.pkl", "rb") as f:
        comp_extracted = pickle.load(f)

    dataset_masked_companion = copy.deepcopy(dataset)
    separation_pixels, separation_arcsec, position_angle, delta_x, delta_y = get_sep_and_posang(dataset_masked_companion, comp_extracted['x'], comp_extracted['y'], pxsc_arcsec)
    fakes.inject_planet(dataset_masked_companion.input, dataset_masked_companion.centers, [-PSF * np.nanmax(dataset.input[0])*comp_extracted['con']],
                        dataset.wcs, separation_pixels, position_angle,
                        fwhm=dataset_fwhm)

    os.makedirs(outputdir + '/masked_companions', exist_ok=True)
    # Save the FITS file
    hdu = fits.PrimaryHDU(dataset_masked_companion.input[0])
    hdul = fits.HDUList([hdu])
    hdul.writeto(outputdir + f'/masked_companions/{filter}-masked.fits', overwrite=True)
    hdul.close()

    psflib.prepare_library(dataset_masked_companion)
    parallelized.klip_dataset(dataset_masked_companion,
                              mode='RDI',
                              outputdir=outputdir + '/masked_companions',
                              fileprefix=f"{filter}-res_masked",
                              annuli=1,
                              subsections=1,
                              movement=0.,
                              numbasis=numbasis,
                              maxnumbasis=np.nanmax(numbasis),
                              calibrate_flux=False,
                              aligned_center=dataset_masked_companion._centers[0],
                              psf_library=psflib,
                              corr_smooth=0,
                              verbose=False)

    kl_hdulist = fits.open(f"{outputdir}/masked_companions/{filter}-res_masked-KLmodes-all.fits")
    return kl_hdulist[0].data

def mk_contrast_curves(dataset, residuals, seps, numbasis, dataset_fwhm):
    fig1, ax1 = plt.subplots(figsize=(12,6))
    cc_dict={}
    cc_dict['sep'] = seps
    for elno in range(len(numbasis[::klstep])):
        KL = numbasis[::klstep][elno]
        klframe = residuals[elno]/np.nanmax(dataset.input[0])
        contrast_seps, contrast = klip.meas_contrast(klframe, 0.5, seps[-1]+dataset_fwhm, dataset_fwhm,
                                                     center=dataset._centers[0], low_pass_filter=False)

        med_interp = interp1d(contrast_seps,
                              contrast,
                              fill_value=(contrast[0], contrast[-1]),
                              bounds_error=False,
                              kind='slinear')
        contrast_curve_iterp=[]
        for i, sep in enumerate(seps):
            contrast_curve_iterp.append(med_interp(sep))

        if KL == KLdetect:
            ax1.plot(seps, contrast_curve_iterp, '-',color = 'k', label=f'KL = {KL}',linewidth=3.0, zorder=5)
        else:
            ax1.plot(seps, contrast_curve_iterp, '-.',label=f'KL = {KL}',linewidth=3.0)

        cc_dict[KL] = med_interp(seps)

    ax1.set_yscale('log')
    ax1.set_ylabel('5$\sigma$ Contrast')
    ax1.set_xlabel('Separation [pix]')
    ax1.minorticks_on()
    ax1.set_xlim(1, int(np.nanmax(contrast_seps)))
    ax1.set_ylim(1e-2, 1)
    fig1.legend(ncols=3, loc=1)
    fig1.savefig(outputdir + f'/{filter}-raw.png', bbox_inches='tight')

    with open(outputdir + f"/extracted_companion/{filter}_contrast_curves.pkl", "wb") as f:
        pickle.dump(cc_dict, f)

def mk_cal_contrast_curves(dataset, psflib, filter, outputdir, numbasis, dataset_fwhm, inject_fake, dataset_with_fkcompanion):
    with open(outputdir + f"/extracted_companion/{filter}_contrast_curves.pkl", "rb") as f:
        cc_dict = pickle.load(f)

    input_contrast_list = [cc_dict[KL] for KL in numbasis]
    input_planet_fluxes =  np.median(input_contrast_list, axis=0)*np.nanmax(dataset.input[0])
    if inject_fake:
        os.makedirs(outputdir + '/inj_companions', exist_ok=True)
        elno=1
        for input_planet_flux, sep in zip(input_planet_fluxes, seps):
            for pa in pa_list:
                print(f'Injecting fake {elno} at pa: {pa}, sep: {sep}, flux: {input_planet_flux}')
                fakes.inject_planet(dataset_with_fkcompanion.input,dataset_with_fkcompanion.centers, [PSF*input_planet_flux], dataset.wcs, sep, pa,
                                    fwhm=dataset_fwhm)

                psflib.prepare_library(dataset_with_fkcompanion)

                parallelized.klip_dataset(dataset_with_fkcompanion,
                                          mode='RDI',
                                          outputdir=outputdir+'/inj_companions',
                                          fileprefix=f"{filter}-withfakes_{elno}",
                                          annuli=1,
                                          subsections=1,
                                          movement=0.,
                                          numbasis=numbasis,
                                          maxnumbasis=np.nanmax(numbasis),
                                          calibrate_flux=False,
                                          aligned_center=dataset_with_fkcompanion._centers[0],
                                          psf_library=psflib,
                                          corr_smooth=0,
                                          verbose=False)
                elno+=1

    fig2, ax2 = plt.subplots(figsize=(12, 6))
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    ccc_dict={}
    ccc_dict['sep'] = cc_dict['sep']
    for elno in range(len(numbasis[::klstep])):
        KL = numbasis[::klstep][elno]
        retrieved_fluxes = []  # will be populated, one for each separation
        input_fluxes = []  # will be populated, one for each separation
        std_fluxes = []
        elno2=1
        for input_planet_flux, sep in zip(input_planet_fluxes, seps):
            fake_planet_fluxes = []
            for pa in pa_list:
                kl_hdulist = fits.open(f"{outputdir}/inj_companions/{filter}-withfakes_{elno2}-KLmodes-all.fits")
                dat_with_fakes = kl_hdulist[0].data[elno]
                dat_with_fakes_centers = [kl_hdulist[0].header['PSFCENTX'], kl_hdulist[0].header['PSFCENTY']]

                fake_flux = fakes.retrieve_planet_flux(dat_with_fakes, dat_with_fakes_centers, dataset.wcs[0],
                                                       sep,
                                                       pa,
                                                       searchrad=5,
                                                       guesspeak=input_planet_flux,
                                                       guessfwhm=dataset_fwhm,
                                                       refinefit=True)
                fake_planet_fluxes.append(fake_flux)
                elno2+=1

            fake_planet_fluxes=np.array(fake_planet_fluxes)[np.array(fake_planet_fluxes)>0]
            median_flux=np.nanmedian(fake_planet_fluxes)
            if median_flux >= 0:
                retrieved_fluxes.append(median_flux)
                # retrieved_seps.append(sep)
                input_fluxes.append(input_planet_flux)
                std_fluxes.append(np.nanstd(input_planet_flux))


        # fake planet output / fake planet input = throughput of KLIP
        algo_throughput = np.round(np.array(retrieved_fluxes) / np.array(input_fluxes),2)  # a number less than 1 probably
        algo_throughput[algo_throughput>1] = 1

        if KL == KLdetect:
            ax3.plot(seps, algo_throughput, '-', color='k', ms=2, label=f'KL = {KL}', linewidth=3.0, zorder=5)
        else:
            ax3.plot(seps, algo_throughput, '-.', ms=2, label=f'KL = {KL}', linewidth=3.0)

        ccc_dict[KL] = cc_dict[KL]/algo_throughput
        if KL == KLdetect:
            ax2.plot(ccc_dict['sep'], ccc_dict[KL], '-',color='k', label=f'KL = {KL}',linewidth=3.0, zorder=5)
        else:
            ax2.plot(ccc_dict['sep'], ccc_dict[KL], '-.',label=f'KL = {KL}',linewidth=3.0)


    ax2.set_yscale('log')
    ax2.set_ylabel('5$\sigma$ Contrast')
    ax2.set_xlabel('Separation [pix]')
    ax2.set_xlim(1, int(np.nanmax(cc_dict['sep'])))
    ax2.minorticks_on()
    ax2.set_ylim(1e-2, 1)
    fig2.legend(ncols=3, loc=1)
    fig2.savefig(outputdir+f'/{filter}-calcc.png',bbox_inches='tight')

    ax3.set_ylabel('Throughput')
    ax3.set_xlabel('Separation [pix]')
    ax3.set_xlim(1, int(np.nanmax(seps)))
    ax3.minorticks_on()
    fig3.legend(ncols=3, loc=1)
    fig3.savefig(outputdir+f'/{filter}-throughput.png',bbox_inches='tight')

    plt.show()

    with open(outputdir + f"/extracted_companion/{filter}_corr_contrast_curves.pkl", "wb") as f:
        pickle.dump(ccc_dict, f)

if __name__ == "__main__":
    id = 52 #id of the star to test
    xycomp_list= [[None, None]]
    extract_companion = True
    contrast_curves = True
    cal_contrast_curves = True

    mask_companion=True
    inject_fake = False

    filters = ['f850lp']
    # filters = 'f814w'

    guess_contrast=1e-1
    pxsc_arcsec = 0.04
    KLdetect =7
    klstep=1 #klip mod step to test (pick a klipmode every klstep)

    # three sets, planets get fainter as contrast gets better further out
    pa_list = [0, 45, 90, 135, 180, 225, 270, 315]
    seps = [1,2,3,4,5,10,15]

    pipe_cfg = '/Users/gstrampelli/PycharmProjects/FFP_binaries/pipeline_logs/pipe.yaml'
    data_cfg = '/Users/gstrampelli/PycharmProjects/FFP_binaries/pipeline_logs/data.yaml'
    pipe_cfg = config.configure_pipeline(pipe_cfg, pipe_cfg=pipe_cfg, data_cfg=data_cfg,
                                         dt_string=datetime.now().strftime("%d/%m/%Y %H:%M:%S"))

    data_cfg = config.configure_data(data_cfg, pipe_cfg)

    dataset = input_tables.Tables(data_cfg, pipe_cfg)
    DF = config.configure_dataframe(dataset, load=True)
    numbasis = np.array(DF.kmodes)

    for elno in range(len(filters)):
        filter=filters[elno]
        fileprefix = f"{filter}-fmpsf"
        outputdir = f"/Users/gstrampelli/PycharmProjects/FFP_binaries/out/contrast_curves/ID{id}"
        os.makedirs(outputdir+'/inj_companions/', exist_ok=True)
        dataset_fwhm = pipe_cfg.instrument['fwhm'][filter]

        xcomp, ycomp = xycomp_list[elno]
        dataset, residuals = setup_DATASET(DF, id, filter, pipe_cfg.psfsubtraction['kmodes'])
        psflib, psf_list, PSF = generate_psflib(DF, id, dataset, filter, KL=KLdetect, dir =outputdir+f"/{filter}_corr_matrix.fits")

        if np.all([x is None for x in [xcomp, ycomp]]) and mask_companion:
            max_value = np.nanmax(np.median(residuals, axis=0))
            max_index = np.where(np.median(residuals, axis=0) == max_value)
            xcomp, ycomp = [max_index[1][0], max_index[0][0]]

        if extract_companion:
            companion_extraction(filter, outputdir, dataset, xcomp, ycomp, pxsc_arcsec, guess_contrast, numbasis, KLdetect)

        if mask_companion:
            residuals = companion_masked(filter, outputdir, dataset, pxsc_arcsec, PSF, dataset_fwhm, numbasis)

        if contrast_curves:
            mk_contrast_curves(dataset, residuals, seps, numbasis, dataset_fwhm)

        if mask_companion:
            hdul = fits.open(outputdir + f'/masked_companions/{filter}-masked.fits')
            data = hdul[0].data
            dataset_with_fkcompanion = GenericData([data], [dataset._centers[0]], filenames=[DF.path2out + f'/mvs_tiles/{filter}/tile_ID{id}.fits'])
            psflib, psf_list, PSF = generate_psflib(DF, id, dataset, filter, KL=KLdetect,
                                                    dir=outputdir + f"/{filter}_corr_matrix.fits")
        else:
            dataset_with_fkcompanion = copy.deepcopy(dataset)

        if cal_contrast_curves:
            mk_cal_contrast_curves(dataset, psflib, filter, outputdir, numbasis, dataset_fwhm, inject_fake,
                                   dataset_with_fkcompanion)


    print()