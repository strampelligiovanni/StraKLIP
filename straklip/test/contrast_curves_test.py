import sys,os
# sys.path.append('/Users/gstrampelli/PycharmProjects/Giovanni/src/StraKLIP/straklip/test/')
# sys.path.append('/')
# sys.path.append('/')
# sys.path.append('//')

import pyklip.fakes as fakes
from extract_companion_test import setup_DATASET, generate_psflib, get_sep_and_posang
import astropy.io.fits as fits
import config, input_tables
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import pyklip.klip as klip
import pyklip.parallelized as parallelized
import pickle
import copy

def mk_raw_contrast_curves(id,normalization, residuals, numbases=[1], klstep=1, path2dir='./', dataset_iwa = 1, dataset_owa = 10, fwhm = 1.460, filename=None):
    fig, ax1 = plt.subplots(figsize=(12,6))
    contrasts=[]
    for KL, klframe in zip(numbases[::klstep],residuals[::klstep]):
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

def mk_cal_contrast_curves(id,normalization, residuals, input_planet_fluxes, seps, numbases=[1], klstep=1, path2dir='./', dataset_iwa = 1, dataset_owa = 10, fwhm = 1.460, filename=None):
    fig, ax1 = plt.subplots(figsize=(12, 6))

    cor_contrasts = []
    for KL, klframe in zip(numbases[::klstep], residuals[::klstep]):
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
    id = 51 #id of the star to test
    d = 3 #stamp base for the PSF and the searchrad for fakes.retrieve_planet_flux
    klstep=1 #klip mod step to test (pick a klipmode every klstep)
    x2, y2 = 18, 21 #None, None  # Companion's coordinates if present to suppress
    pxsc_arcsec = 0.04

    inject_fake = False

    filter = 'f850lp'
    # filter = 'f814w'
    fileprefix = f"{filter}-fmpsf"
    outputdir = f"/Users/gstrampelli/PycharmProjects/Giovanni/work/analysis/FFP_drc/test/data/ID{id}/contrast_curves/inj_companions/{filter}"
    plotdir = f"/Users/gstrampelli/PycharmProjects/Giovanni/work/analysis/FFP_drc/test/data/ID{id}/contrast_curves/"
    os.makedirs(outputdir, exist_ok=True)

    pipe_cfg = '/Users/gstrampelli/PycharmProjects/Giovanni/work/pipeline_logs/FFP_drc/pipe.yaml'
    data_cfg = '/Users/gstrampelli/PycharmProjects/Giovanni/work/pipeline_logs/FFP_drc/data.yaml'
    pipe_cfg = config.configure_pipeline(pipe_cfg, pipe_cfg=pipe_cfg, data_cfg=data_cfg,
                                         dt_string=datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
    dataset_fwhm = pipe_cfg.instrument['fwhm'][filter]

    data_cfg = config.configure_data(data_cfg, pipe_cfg)
    numbases = np.array(pipe_cfg.psfsubtraction['kmodes'])
    numbases = numbases[numbases>=7]

    dataset = input_tables.Tables(data_cfg, pipe_cfg)
    DF = config.configure_dataframe(dataset, load=True)

    dataset, PSF, residuals = setup_DATASET(DF, id, filter, d, pipe_cfg.psfsubtraction['kmodes'], KL=numbases[-1])
    psflib = generate_psflib(DF, id, dataset, filter)

    low_pass_size = 1.  # pixel, corresponds to the sigma of the Gaussian

    # three sets, planets get fainter as contrast gets better further out
    pa_list = [0, 60, 120, 180, 240, 315]
    seps = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
    input_planet_fluxes = np.sort(np.random.uniform(0.01, 1, 12))[::-1]

    fluxes = np.nanmax(dataset.input[0]) * input_planet_fluxes
    if np.all([x is not None for x in [x2, y2]]):
        separation_pixels, separation_arcsec, position_angle, _, _ = get_sep_and_posang(dataset, x2, y2, pxsc_arcsec)
        closest_sep_index = np.argmin(np.abs(separation_pixels - seps))
        closest_posang_index = np.argmin(np.abs(position_angle - pa_list))

        # I don't want to inject and retrive close to an actual companion
        closest_sep_index_list=[]
        closest_posang_index_list=[]
        for x in [-1,0,1]:
            if closest_sep_index+x >= 0 and closest_sep_index+x < len(seps):
                closest_sep_index_list.append(closest_sep_index+x)
            if closest_posang_index+x < 0:
                closest_posang_index_list.append(len(pa_list)-1)
            if closest_posang_index+x > len(pa_list):
                closest_posang_index_list.append(0)
            else:
                closest_posang_index_list.append(closest_posang_index+x)

    if inject_fake:
        elno=1
        for input_planet_flux, sep in zip(fluxes, seps):
            for pa in pa_list:
                if np.all([x is not None for x in [x2,y2]]) and np.any(sep in np.array(seps)[closest_sep_index_list]) and np.any(pa in np.array(pa_list)[closest_posang_index_list]):
                    continue
                else:
                    # Make copy of the original pyKLIP dataset.
                    dataset_with_companion = copy.deepcopy(dataset)
                    fakes.inject_planet(dataset_with_companion.input,dataset_with_companion.centers, [PSF/np.nanmax(PSF)*input_planet_flux], dataset.wcs, sep, pa,
                                        fwhm=dataset_fwhm)
                    psflib.prepare_library(dataset_with_companion)

                    parallelized.klip_dataset(dataset_with_companion,
                                              mode='RDI',
                                              outputdir=outputdir,
                                              fileprefix=f"{filter}-withfakes_{elno}",
                                              annuli=1,
                                              subsections=1,
                                              movement=0.,
                                              numbasis=numbases,
                                              maxnumbasis=np.nanmax(numbases),
                                              calibrate_flux=False,
                                              aligned_center=dataset_with_companion._centers[0],
                                              psf_library=psflib,
                                              corr_smooth=0,
                                              verbose=False)

                    elno+=1

    fig1, ax1 = plt.subplots(figsize=(12,6))
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    contrast_list=[]
    contrast_corr_list=[]
    algo_throughput_dict={}
    algo_throughput_error_dict={}
    for elno in range(len(numbases[::klstep])):
        KL = numbases[::klstep][elno]
        klframe = residuals[elno]/np.nanmax(dataset.input[0])
        # now mask the data
        ydat, xdat = np.indices(klframe.shape)
        if np.all([x is not None for x in [x2,y2]]):
            distance_from_planet = np.sqrt((xdat - x2) ** 2 + (ydat - y2) ** 2)
            klframe[np.where(distance_from_planet <= 2 * dataset_fwhm)] = 0#np.nan

        contrast_seps, contrast = klip.meas_contrast(klframe, 1, klframe.shape[1]//2, dataset_fwhm,
                                                     center=dataset._centers[0], low_pass_filter=False)

        ax1.plot(contrast_seps, contrast, '-.',label=f'KL = {KL}',linewidth=3.0)
        contrast_list.append(contrast)

        retrieved_fluxes = []  # will be populated, one for each separation
        std_fluxes = []

        elno2=1
        for input_planet_flux, sep in zip(input_planet_fluxes, seps):
            fake_planet_fluxes = []
            for pa in pa_list:
                if np.all([x is not None for x in [x2,y2]]) and np.any(sep in np.array(seps)[closest_sep_index_list]) and np.any(pa in np.array(pa_list)[closest_posang_index_list]):
                    pass
                else:
                    kl_hdulist = fits.open(f"{outputdir}/{filter}-withfakes_{elno2}-KLmodes-all.fits")
                    dat_with_fakes = kl_hdulist[0].data[elno] / np.nanmax(dataset.input[0])
                    dat_with_fakes_centers = [kl_hdulist[0].header['PSFCENTX'], kl_hdulist[0].header['PSFCENTY']]

                    fake_flux = fakes.retrieve_planet_flux(dat_with_fakes, dat_with_fakes_centers, dataset.wcs[0],
                                                           sep,
                                                           pa, searchrad=d, guessfwhm=dataset_fwhm)
                    fake_planet_fluxes.append(fake_flux)
                    elno2+=1

            if np.nanmean(fake_planet_fluxes) >= 0:
                retrieved_fluxes.append(np.nanmedian(fake_planet_fluxes))
                std_fluxes.append(np.nanstd(fake_planet_fluxes))
            else:
                retrieved_fluxes.append(0)

        # fake planet output / fake planet input = throughput of KLIP
        algo_throughput = np.round(np.array(retrieved_fluxes) / np.array(input_planet_fluxes),2)  # a number less than 1 probably
        algo_throughput_plus = np.round((np.array(retrieved_fluxes)+np.array(std_fluxes)) / np.array(input_planet_fluxes),2)  # a number less than 1 probably
        algo_throughput_minus = np.round((np.array(retrieved_fluxes)-np.array(std_fluxes)) / np.array(input_planet_fluxes),2)  # a number less than 1 probably
        algo_throughput_error = [algo_throughput_plus-algo_throughput,algo_throughput_minus-algo_throughput]

        algo_throughput[algo_throughput>1] = 1
        algo_throughput_dict[KL] = algo_throughput
        algo_throughput_error_dict[KL] = algo_throughput_error
        corrected_contrast_curve = np.copy(contrast)
        for i, sep in enumerate(contrast_seps):
            closest_throughput_index = np.argmin(np.abs(sep - seps))
            corrected_contrast_curve[i] /= algo_throughput[closest_throughput_index]

        ax2.plot(contrast_seps, corrected_contrast_curve, '-.',label=f'KL = {KL}',linewidth=3.0)
        contrast_corr_list.append(corrected_contrast_curve)

    with open(outputdir+f"/{filter}_algo_throughput_dict.pkl", "wb") as f:
        pickle.dump(algo_throughput_dict, f)
    with open(outputdir+f"/{filter}_algo_throughput_error_dict.pkl", "wb") as f:
        pickle.dump(algo_throughput_error_dict, f)

    ax1.plot(contrast_seps, np.nanmedian(contrast_list,axis=0), '-',color='k', label=f'Median', linewidth=3.0)
    ax1.set_yscale('log')
    ax1.set_ylabel('5$\sigma$ Contrast')
    ax1.set_xlabel('Separation [pix]')
    ax1.minorticks_on()
    ax1.set_xlim(int(np.nanmin(contrast_seps)),int(np.nanmax(contrast_seps)))
    # ax1.set_ylim(np.nanmin(contrast_list), np.nanmax(contrast_list))
    ax1.set_ylim(1e-2, 1)
    fig1.legend(ncols=3, loc=1)
    fig1.savefig(plotdir+f'/{filter}-raw.png',bbox_inches='tight')

    ax2.plot(contrast_seps, np.nanmedian(contrast_corr_list,axis=0), '-',color='k', label=f'Median', linewidth=3.0)
    ax2.set_yscale('log')
    ax2.set_ylabel('5$\sigma$ Contrast')
    ax2.set_xlabel('Separation [pix]')
    ax2.set_xlim(int(np.nanmin(contrast_seps)), int(np.nanmax(contrast_seps)))
    # ax2.set_ylim(np.nanmin(corrected_contrast_curve), np.nanmax(corrected_contrast_curve))
    ax2.set_ylim(1e-2, 1)
    fig2.legend(ncols=3, loc=1)
    fig2.savefig(plotdir+f'/{filter}-calcc.png',bbox_inches='tight')


    plt.close()
    print()