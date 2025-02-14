import os

import pyklip.fakes as fakes
from extract_companion import setup_DATASET, generate_psflib, get_sep_and_posang, run_FMAstrometry
import astropy.io.fits as fits
import config, input_tables
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import pyklip.klip as klip
import pyklip.parallelized as parallelized
import pickle
import copy
from scipy.interpolate import interp1d
import corner
import scipy.ndimage.interpolation as sinterp

def uniform_list(a, b, n):
    if n <= 0:
        return []
    step = (b - a) / (n - 1) if n > 1 else 0
    return [a + i * step for i in range(n)]

def mk_raw_contrast_curves(id,normalization, residuals, numbasis=[1], klstep=1, path2dir='./', dataset_iwa = 1, dataset_owa = 10, fwhm = 1.460, filename=None):
    fig, ax1 = plt.subplots(figsize=(12,6))
    contrasts=[]
    for KL, klframe in zip(numbasis[::klstep],residuals[::klstep]):
        klframe /= normalization
        contrast_seps, contrast = klip.meas_contrast(klframe, dataset_iwa, dataset_owa, fwhm,
                                                     center=[(klframe.shape[0]-1)//2,(klframe.shape[1]-1)//2])

        ax1.plot(contrast_seps, contrast, '-.',label=f'KL = {KL}',linewidth=3.0)
        contrasts.append(contrast)

    ax1.set_ylim([np.nanmin(contrasts),np.nanmax(contrasts)])
    ax1.set_yscale('log')
    ax1.set_ylabel('5$\sigma$ Contrast')
    ax1.set_xlabel('Separation [pix]')
    fig.legend(ncols=3,loc=1)
    plt.tight_layout()
    if filename is None:
        filename = f'tile_ID{id}_raw_cc.png'
    plt.savefig(path2dir+f'/{filename}',bbox_inches='tight')
    plt.close()

def mk_cal_contrast_curves(id,normalization, residuals, input_planet_fluxes, seps, numbasis=[1], klstep=1, path2dir='./', dataset_iwa = 1, dataset_owa = 10, fwhm = 1.460, filename=None):
    fig, ax1 = plt.subplots(figsize=(12, 6))

    cor_contrasts = []
    for KL, klframe in zip(numbasis[::klstep], residuals[::klstep]):
        klframe /= normalization
        contrast_seps, contrast = klip.meas_contrast(klframe, dataset_iwa, dataset_owa, fwhm,
                                                     center=[(klframe.shape[0] - 1) // 2, (klframe.shape[1] - 1) // 2])
        retrieved_fluxes = []  # will be populated, one for each separation

        for input_planet_flux, sep in zip(input_planet_fluxes, seps):
            fake_planet_fluxes = []
            for pa in [0, 90, 180, 270]:
                fake_flux = fakes.retrieve_planet_flux(klframe, dataset._centers[0],
                                                       dataset.output_wcs[0], sep,
                                                       pa, searchrad=7)
                fake_planet_fluxes.append(fake_flux)
            retrieved_fluxes.append(np.nanmean(fake_planet_fluxes))

        # fake planet output / fake planet input = throughput of KLIP
        algo_throughput = np.array(retrieved_fluxes) / np.array(input_planet_fluxes)  # a number less than 1 probably

        corrected_contrast_curve = np.copy(contrast)
        for i, sep in enumerate(contrast_seps):
            closest_throughput_index = np.argmin(np.abs(sep - seps))
            corrected_contrast_curve[i] /= algo_throughput[closest_throughput_index]

        cor_contrasts.append(corrected_contrast_curve)
        ax1.plot(contrast_seps, corrected_contrast_curve, '-.', label=f'KL = {KL}', linewidth=3.0)

    # ax1.set_ylim([np.nanmin(cor_contrasts), np.nanmax(cor_contrasts)])
    ax1.set_yscale('log')
    ax1.set_ylabel('5$\sigma$ Contrast')
    ax1.set_xlabel('Separation [pix]')
    fig.legend(ncols=3, loc=1)
    plt.tight_layout()
    if filename is None:
        filename = f'tile_ID{id}_raw_cc.png'
    plt.savefig(path2dir + f'/{filename}', bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    id = 52 #id of the star to test
    xycomp_list= [[None, None]]
    mask_companion=True
    inject_fake = True

    filters = ['f850lp']
    # filters = 'f814w'

    guess_contrast=1e-1
    pxsc_arcsec = 0.04
    KLdetect =7
    klstep=1 #klip mod step to test (pick a klipmode every klstep)

    # three sets, planets get fainter as contrast gets better further out
    pa_list = [0, 45, 90, 135, 180, 225, 270, 315]
    # pa_list = [0, 90, 180, 270]
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
        os.makedirs(outputdir+'inj_companions/{filter}', exist_ok=True)


        dataset_fwhm = pipe_cfg.instrument['fwhm'][filter]

        xcomp, ycomp = xycomp_list[elno]
        dataset, residuals = setup_DATASET(DF, id, filter, pipe_cfg.psfsubtraction['kmodes'])

        input_planet_fluxes = uniform_list(np.nanmax(dataset.input[0]) * 0.5e-1, np.nanmax(dataset.input[0]) * 0.9,len(seps))[::-1]
        if np.all([x is None for x in [xcomp, ycomp]]) and mask_companion:
            max_value = np.nanmax(np.median(residuals, axis=0))
            max_index = np.where(np.median(residuals, axis=0) == max_value)
            xcomp, ycomp = [max_index[1][0], max_index[0][0]]

        if mask_companion:
            chaindir = f'{outputdir}/extracted_companion/chains'
            dictdir = f"/Users/gstrampelli/PycharmProjects/FFP_binaries/out/extraction/ID{id}/"
            os.makedirs(chaindir, exist_ok=True)
            psflib, psf_list, PSF = generate_psflib(DF, id, dataset, filter, KL=KLdetect, dir =outputdir+f"/extracted_companion/{filter}_corr_matrix.fits")

            with open(dictdir + f"/{filter}_comp_extracted.pkl", "rb") as f:
                comp_extracted = pickle.load(f)

            dataset_masked_companion = copy.deepcopy(dataset)
            separation_pixels, separation_arcsec, position_angle, delta_x, delta_y = get_sep_and_posang(dataset_masked_companion, comp_extracted['x'], comp_extracted['y'], pxsc_arcsec)
            fakes.inject_planet(dataset_masked_companion.input, dataset_masked_companion.centers, [-PSF * np.nanmax(dataset.input[0])*comp_extracted['con']],
                                dataset.wcs, separation_pixels, position_angle,
                                fwhm=dataset_fwhm)

        # if inject_fake:
        #     os.makedirs(outputdir + '/inj_companions', exist_ok=True)
        #     elno=1
        #     for input_planet_flux, sep in zip(input_planet_fluxes, seps):
        #         for pa in pa_list:
        #             print(f'Injecting fake {elno} at pa: {pa}, sep: {sep}, flux: {input_planet_flux}')
        #             if mask_companion:
        #                 dataset_with_fkcompanion = copy.deepcopy(dataset_masked_companion)
        #             else:
        #                 dataset_with_fkcompanion = copy.deepcopy(dataset)
        #
        #             fakes.inject_planet(dataset_with_fkcompanion.input,dataset_with_fkcompanion.centers, [PSF*input_planet_flux], dataset.wcs, sep, pa,
        #                                 fwhm=dataset_fwhm)
        #
        #             psflib.prepare_library(dataset_with_fkcompanion)
        #
        #             parallelized.klip_dataset(dataset_with_fkcompanion,
        #                                       mode='RDI',
        #                                       outputdir=outputdir+'/inj_companions',
        #                                       fileprefix=f"{filter}-withfakes_{elno}",
        #                                       annuli=1,
        #                                       subsections=1,
        #                                       movement=0.,
        #                                       numbasis=numbasis,
        #                                       maxnumbasis=np.nanmax(numbasis),
        #                                       calibrate_flux=False,
        #                                       aligned_center=dataset_with_fkcompanion._centers[0],
        #                                       psf_library=psflib,
        #                                       corr_smooth=0,
        #                                       verbose=False)
        #             elno+=1

        fig1, ax1 = plt.subplots(figsize=(12,6))
        # fig2, ax2 = plt.subplots(figsize=(12, 6))
        # fig3, ax3 = plt.subplots(figsize=(12, 6))
        contrast_list=[]
        # contrast_corr_list=[]
        # algo_throughput_dict={}
        # algo_throughput_error_dict={}
        if mask_companion:
            os.makedirs(outputdir + '/masked_companions', exist_ok=True)
            psflib.prepare_library(dataset_masked_companion)
            parallelized.klip_dataset(dataset_masked_companion,
                                      mode='RDI',
                                      outputdir=outputdir + '/masked_companions',
                                      fileprefix=f"{filter}-masked",
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
            kl_hdulist = fits.open(f"{outputdir}/masked_companions/{filter}-masked-KLmodes-all.fits")
            residuals = kl_hdulist[0].data

        for elno in range(len(numbasis[::klstep])):
            KL = numbasis[::klstep][elno]
            # algo_throughput_dict[KL] = {}
            # algo_throughput_error_dict[KL] = {}
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
                contrast_curve_iterp.append( med_interp(sep))

            if KL == KLdetect:
                ax1.plot(seps, contrast_curve_iterp, '-',color = 'k', label=f'KL = {KL}',linewidth=3.0, zorder=5)
            else:
                ax1.plot(seps, contrast_curve_iterp, '-.',label=f'KL = {KL}',linewidth=3.0)

        #     contrast_list.append(contrast)
        #     retrieved_seps=[]
        #     retrieved_fluxes = []  # will be populated, one for each separation
        #     input_fluxes = []  # will be populated, one for each separation
        #     std_fluxes = []
        #     elno2=1
        #     for input_planet_flux, sep in zip(input_planet_fluxes, seps):
        #         fake_planet_fluxes = []
        #         for pa in pa_list:
        #             kl_hdulist = fits.open(f"{outputdir}/inj_companions/{filter}-withfakes_{elno2}-KLmodes-all.fits")
        #             dat_with_fakes = kl_hdulist[0].data[elno]
        #             dat_with_fakes_centers = [kl_hdulist[0].header['PSFCENTX'], kl_hdulist[0].header['PSFCENTY']]
        #
        #             fake_flux = fakes.retrieve_planet_flux(dat_with_fakes, dat_with_fakes_centers, dataset.wcs[0],
        #                                                    sep,
        #                                                    pa,
        #                                                    searchrad=5,
        #                                                    guesspeak=input_planet_flux,
        #                                                    guessfwhm=dataset_fwhm,
        #                                                    refinefit=True)
        #             fake_planet_fluxes.append(fake_flux)
        #             elno2+=1
        #
        #         fake_planet_fluxes=np.array(fake_planet_fluxes)[np.array(fake_planet_fluxes)>0]
        #         median_flux=np.nanmedian(fake_planet_fluxes)
        #         if median_flux >= 0:
        #             retrieved_fluxes.append(median_flux)
        #             retrieved_seps.append(sep)
        #             input_fluxes.append(input_planet_flux)
        #             std_fluxes.append(np.nanstd(input_planet_flux))
        #
        #
        #     # fake planet output / fake planet input = throughput of KLIP
        #     algo_throughput = np.round(np.array(retrieved_fluxes) / np.array(input_fluxes),2)  # a number less than 1 probably
        #     algo_throughput[algo_throughput>1] = 1
        #     algo_throughput_plus = np.round((np.array(retrieved_fluxes)+np.array(std_fluxes)) / np.array(input_fluxes),2)  # a number less than 1 probably
        #     algo_throughput_minus = np.round((np.array(retrieved_fluxes)-np.array(std_fluxes)) / np.array(input_fluxes),2)  # a number less than 1 probably
        #     algo_throughput_error = [algo_throughput_plus-algo_throughput,algo_throughput_minus-algo_throughput]
        #
        #     algo_throughput_dict[KL]['throughput'] = algo_throughput
        #     algo_throughput_dict[KL]['cseps'] = retrieved_seps
        #     if KL == KLdetect:
        #         ax3.plot(seps, algo_throughput, '-', color='k', ms=2, label=f'KL = {KL}', linewidth=3.0, zorder=5)
        #     else:
        #         ax3.plot(seps, algo_throughput, '-.', ms=2, label=f'KL = {KL}', linewidth=3.0)
        #
        #
        #     algo_throughput_error_dict[KL]['throughput'] = algo_throughput_error
        #     algo_throughput_error_dict[KL]['cseps'] = retrieved_seps
        #
        #
        #     algo_med_interp = interp1d(retrieved_seps,
        #                           algo_throughput,
        #                           fill_value=(algo_throughput[0], algo_throughput[-1]),
        #                           bounds_error=False,
        #                           kind='slinear')
        #
        #     cont_med_interp = interp1d(contrast_seps,
        #                           contrast,
        #                           fill_value=(contrast[0], contrast[-1]),
        #                           bounds_error=False,
        #                           kind='slinear')
        #
        #     corrected_contrast_curve = []
        #     for i, sep in enumerate(retrieved_seps):
        #         corrected_contrast_curve.append(cont_med_interp(sep)/algo_med_interp(sep)) #algo_throughput[closest_throughput_index]
        #
        #     if KL == KLdetect:
        #         ax2.plot(retrieved_seps, corrected_contrast_curve, '-',color='k', label=f'KL = {KL}',linewidth=3.0, zorder=5)
        #     else:
        #         ax2.plot(retrieved_seps, corrected_contrast_curve, '-.',label=f'KL = {KL}',linewidth=3.0)
        #
        #     contrast_corr_list.append(corrected_contrast_curve)
        #
        # with open(outputdir+f"/{filter}_algo_throughput_dict.pkl", "wb") as f:
        #     pickle.dump(algo_throughput_dict, f)
        # with open(outputdir+f"/{filter}_algo_throughput_error_dict.pkl", "wb") as f:
        #     pickle.dump(algo_throughput_error_dict, f)

        # wkl = np.where(numbasis  == KLdetect)
        # ax1.plot(contrast_seps, contrast_list[wkl], '-',color='k', linewidth=3.0)
        ax1.set_yscale('log')
        ax1.set_ylabel('5$\sigma$ Contrast')
        ax1.set_xlabel('Separation [pix]')
        ax1.minorticks_on()
        ax1.set_xlim(1,int(np.nanmax(contrast_seps)))
        ax1.set_ylim(1e-2, 1)
        fig1.legend(ncols=3, loc=1)
        fig1.savefig(outputdir+f'/{filter}-raw.png',bbox_inches='tight')

        # # ax2.plot(contrast_seps, contrast_corr_list[wkl], '-',color='k', linewidth=3.0)
        # ax2.set_yscale('log')
        # ax2.set_ylabel('5$\sigma$ Contrast')
        # ax2.set_xlabel('Separation [pix]')
        # ax2.set_xlim(1, int(np.nanmax(contrast_seps)))
        # ax2.minorticks_on()
        # ax2.set_ylim(1e-2, 1)
        # fig2.legend(ncols=3, loc=1)
        # fig2.savefig(outputdir+f'/{filter}-calcc.png',bbox_inches='tight')
        #
        # # ax2.plot(contrast_seps, algo_throughput_dict[KLdetect], '-',color='k', linewidth=3.0)
        # ax3.set_ylabel('Throughput')
        # ax3.set_xlabel('Separation [pix]')
        # ax3.set_xlim(1, int(np.nanmax(seps)))
        # # ax3.set_ylim(0, 1.2)
        # ax3.minorticks_on()
        # fig3.legend(ncols=3, loc=1)
        # fig3.savefig(outputdir+f'/{filter}-throughput.png',bbox_inches='tight')

        plt.show()
    print()