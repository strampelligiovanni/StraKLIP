import sys
sys.path.append('/')
from glob import glob
from utils_dataframe import fk_writing,mk_fakes_df
from utils_fpanalysis import task_fake_reference_infos,task_fake_infos
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

import numpy as np
from stralog import getLogger
from random import choice
from astropy.io import fits
from ancillary import parallelization_package

def psf_scale(psfdata):
    psfdata[psfdata<0]=0
    # psfdata+=(1-np.sum(psfdata))/(psfdata.shape[1]*psfdata.shape[0])
    psfdata/=np.sum(psfdata)
    return(psfdata)

def update_fk_dataframes(DF, parallel_runs=False, workers=None, NPSFstars=300, NPSFsample=30, inner_shift=0.25,
                    path2data=None,  showplot=False, aptype='4pixels', delta=1,
                    suffix='',multiply_by_exptime=False,skip_filter=[]):
    '''
    Update the fake dataframe.

    Parameters
    ----------
    # zpt_list_in : TYPE, optional
    #     zero point input list for photometry. The default is [].
    # exptime_list_in : TYPE, optional
    #     exposure time input list for photmetry. The default is [].
    workers : int, optional
        number of workers for parallelization. The default is None.
    NPSFstars : int, optional
        number of PSF to create. The default is 300.
    NPSFsample : int, optional
        number of PSF to select to creat the sample for the PSF libray. The default is 30.
    inner_shift : float, optional
        subpixel shift around satr injection coordinate. The default is 0.45.
    path2psfdir: str, optional
         default dir for psfs. If None, use default path. The default is None
    aptype : (circular,square,4pixels), optional
        defin the aperture type to use during aperture photometry.
        The default is '4pixels'.
    delta : int, optional
        step to create the square mask in range -delta, x, +delt and -delta, y, +delta. The default is 1.

    Returns
    -------
    None.

    '''
    getLogger(__name__).info('Updating the fake injection dataframe with detections from fake binaries')

    for filter in DF.filters:
        if filter not in skip_filter:
            getLogger(__name__).info(f'Collecting background values for filter: {filter}')
            path2psfdir = path2data+f'/mvs_tiles/{filter}/'


            mvs_psf_ids_list = DF.mvs_targets_df.loc[
                DF.mvs_targets_df[f'flag_{filter}'] == 'good_psf'].mvs_ids.unique()

            bkg_list = DF.mvs_targets_df[f'sky_{filter}' ].values.astype(float) / DF.mvs_targets_df[
                f'exptime_{filter}'].values.astype(float)
            ebkg_list = DF.mvs_targets_df[f'esky_{filter}' ].values.astype(float) / DF.mvs_targets_df[
                f'exptime_{filter}'].values.astype(float)
            nbkg_list = DF.mvs_targets_df[f'nsky_{filter}'].values.astype(float)

            bgk_sel = ~np.isnan(bkg_list)
            bkg_list = bkg_list[bgk_sel]
            ebkg_list = ebkg_list[bgk_sel]
            nbkg_list = nbkg_list[bgk_sel]

            magmin = DF.mvs_targets_df.loc[(DF.mvs_targets_df[f'flag_{filter}'] == 'good_psf'), f'm_{filter}{suffix}'].min()  # &(DF.mvs_targets_df['m%s%s'%(filter[1:4],suffix)]>0)
            magmax = DF.mvs_targets_df.loc[(DF.mvs_targets_df[f'flag_{filter}'] == 'good_psf'), f'm_{filter}{suffix}'].max()
            zpt = DF.zpt[filter]

            fk_ids_list = [i for i in range(0, NPSFstars)]

            # getLogger(__name__).info(f'Generating the fake PSF library stars for filter {filter}')
            psf_list = []
            psf_ids_list = []
            for fk_ids in fk_ids_list:
                ID = int(choice(mvs_psf_ids_list))
                getLogger(__name__).info(f'Loading PSF from tile_ID{ID}.fits')
                exptime = DF.mvs_targets_df.loc[DF.mvs_targets_df.mvs_ids == ID, f'exptime_{filter}'].values[0]
                # if len(psf) == 0:
                with fits.open(str(path2psfdir) + f'/tile_ID{ID}.fits', memmap=False) as hdul:
                    if multiply_by_exptime:
                        target_psf = hdul[-1].data / exptime
                    else:
                        target_psf = hdul[-1].data
                    psf_ids_list.append(ID)
                # else:
                #     target_psf = psf.copy()

                psf_list.append(
                    task_fake_reference_infos(DF, fk_ids, target_psf, magmin, magmax, zpt, exptime, bkg_list, ebkg_list,
                                              inner_shift))

            magbin_list = DF.fk_targets_df.loc[filter].index.get_level_values('magbin').unique().astype(float)
            dmag_list = DF.fk_targets_df.loc[filter].index.get_level_values('dmag').unique().astype(float)
            sep_list = DF.fk_targets_df.loc[filter].index.get_level_values('sep').unique().astype(float)

            if np.max(DF.kmodes)>NPSFsample:
                getLogger(__name__).warning(f'NPSFsample ({NPSFsample}) less than maximum number of Klip mode requested ({np.max(DF.kmodes)})'
                                            f' Limiting Klip modes to {NPSFsample}')
                kmodes_list=[i for i in  DF.kmodes if i <=NPSFsample]
            else:
                kmodes_list=DF.kmodes

            columns_no_injection = ['x', 'y', 'm', 'exptime'] + np.array(
                [[f'nsigma_kmode{kmode}', f'counts_kmode{kmode}', f'noise_kmode{kmode}', f'm_kmode{kmode}']
                 for kmode in kmodes_list]).ravel().tolist()
            columns = ['x', 'y', 'counts', 'm', 'exptime'] + np.array(
                [[f'nsigma_kmode{kmode}', f'counts_kmode{kmode}', f'noise_kmode{kmode}', f'm_kmode{kmode}']
                 for kmode in kmodes_list]).ravel().tolist()

            for magbin in magbin_list:
                for dmag in dmag_list:
                    if parallel_runs:
                        workers, chunksize, ntarget = parallelization_package(workers, len(sep_list), verbose=False)
                        with ProcessPoolExecutor(max_workers=workers) as executor:
                            for out, out_no_injection in executor.map(task_fake_infos, repeat(DF), repeat(magbin),
                                                                      repeat(dmag), sep_list, repeat(filter), repeat(zpt),
                                                                      repeat(psf_list), repeat(psf_ids_list),
                                                                      repeat(NPSFsample), repeat(bkg_list),
                                                                      repeat(ebkg_list), repeat(nbkg_list),
                                                                      repeat(inner_shift), repeat(path2psfdir),
                                                                      repeat(multiply_by_exptime), repeat(showplot),
                                                                      repeat(aptype), repeat(delta), chunksize=chunksize):
                                DF=fk_writing(DF, filter, out_no_injection, 'fk_targets_df', columns_no_injection)
                                DF=fk_writing(DF, filter, out, 'fk_candidates_df', columns)
                                del out, out_no_injection
                    else:
                        for sep in sep_list:
                            out, out_no_injection = task_fake_infos(DF, magbin, dmag, sep, filter, zpt, psf_list,
                                                                    psf_ids_list, NPSFsample, bkg_list, ebkg_list,
                                                                    nbkg_list, inner_shift, path2psfdir,
                                                                    multiply_by_exptime, showplot, aptype, delta)
                            DF=fk_writing(DF, filter, out_no_injection, 'fk_targets_df', columns_no_injection)
                            DF=fk_writing(DF, filter, out, 'fk_candidates_df', columns)
                            del out, out_no_injection
            del psf_list, psf_ids_list, bkg_list, ebkg_list
    return(DF)

def make_fk_dataframes(DF,dataset):
    '''
    This is a wrapper for the creation of basic dataframes for the pipeline

    Parameters
    ----------


    Returns
    -------
    None.

    '''
    getLogger(__name__).info(f'Creating the fake injection dataframe')
    mk_fakes_df(DF,
                MagBin_list=dataset.pipe_cfg.buildfphdf['magbins'],
                Dmag_list=dataset.pipe_cfg.buildfphdf['dmags'],
                Sep_range=dataset.pipe_cfg.buildfphdf['sep_range'],
                Nstar=dataset.pipe_cfg.buildfphdf['nstars'],
                filters=dataset.data_cfg.filters,
                skip_filters=dataset.pipe_cfg.buildfphdf['skip_filters'])

def run(packet):
    DF = packet['DF']
    dataset = packet['dataset']

    NPSFstars=dataset.pipe_cfg.buildfphdf['nstars']
    NPSFsample=dataset.pipe_cfg.buildfphdf['NPSFsample']
    if NPSFstars > NPSFstars:
        getLogger(__name__).warning(
            f'NPSFsample ({NPSFsample}) greather than maximum number of stars to be generated ({NPSFstars})'
            f' Limiting NPSFsample to {NPSFstars}')
        NPSFsample = NPSFstars

    if dataset.pipe_cfg.buildfphdf['redo'] or len(glob(dataset.pipe_cfg.paths['out']+'/fk_*'))==0:
        make_fk_dataframes(DF, dataset)
        DF=update_fk_dataframes(DF,
                                parallel_runs=dataset.pipe_cfg.buildfphdf['parallel_runs'],
                                workers=dataset.pipe_cfg.ncpu,
                                NPSFstars=NPSFstars,
                                NPSFsample=NPSFsample,
                                inner_shift=dataset.pipe_cfg.buildfphdf['inner_shift'],
                                path2data=dataset.pipe_cfg.paths['out'],
                                showplot=dataset.pipe_cfg.buildfphdf['showplot'],
                                aptype=dataset.pipe_cfg.klipphotometry['aptype'],
                                delta=dataset.pipe_cfg.klipphotometry['delta'],
                                suffix=dataset.pipe_cfg.buildfphdf['suffix'],
                                multiply_by_exptime=dataset.pipe_cfg.mktiles['multiply_by_exptime'],
                                skip_filter=dataset.pipe_cfg.buildfphdf['skip_filters']
                                )
        DF.save_dataframes(__name__)
    else:
        getLogger(__name__).info(f'Fetching false positive dataframes from {dataset.pipe_cfg.paths["out"]}')

