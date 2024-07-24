import numpy as np
from stralog import getLogger
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from ancillary import parallelization_package
from utils_photometry import mvs_aperture_photometry,unq_aperture_photometry
import os
def get_ee_df(dataset):
    """
    Create the Encircled Energy Dataframe dictionary from file.

    It si expected one file for each ccd with the following format:
    - The first column must bet named FILTER and contain the filter name.
    - Each subsequent columns must bet named with a value indicating the Aperture Radius (arcsec)

    Example:

    FILTER	0.04	0.08	0.12	0.16	0.2	0.24	0.28	0.32	0.36	0.4	0.44	0.48	0.51	0.55	0.59	0.63	0.67	0.71	0.75	0.79	0.83	0.87	0.91	0.95	0.99	1.03	1.07	1.11	1.15	1.19	1.23	1.27	1.31	1.35	1.39	1.43	1.47	1.5	1.54	1.58	1.62	1.66	1.7	1.74	1.78	1.82	1.86	1.9	1.94	1.98
    FFFFF	0.5713	0.6311	0.6845	0.7255	0.752	0.7757	0.7946	0.8086	0.8201	0.8312	0.8435	0.8558	0.8636	0.873	0.8817	0.8913	0.902	0.9129	0.9237	0.9337	0.9424	0.9499	0.9561	0.9614	0.966	0.9699	0.9733	0.9763	0.9788	0.981	0.9828	0.9843	0.9855	0.9865	0.9873	0.988	0.9886	0.989	0.9895	0.99	0.9904	0.9909	0.9913	0.9917	0.9921	0.9924	0.9928	0.9931	0.9935	0.9938

    :param dataset:
    :return:
    """
    getLogger(__name__).info(f'Fetching encircled energy dataframe for filters {dataset.data_cfg.filters}')
    ee_dict={}
    ee_cfg_dict=dataset.pipe_cfg.instrument['ee_name']
    try:
        pixels = ee_cfg_dict.pop('pixels')
    except:
        pixels = None
    for key in list(ee_cfg_dict):
        ee_df=pd.read_csv(dataset.pipe_cfg.paths['database']+'/'+ee_cfg_dict[key])
        ee_df=ee_df.rename(columns={'Filter':'FILTER'})
        ee_df['FILTER'] = ee_df['FILTER'].str.lower()
        # dataset.pipe_cfg.instrument['ee_name']
        if pixels :
            ee_df=ee_df.rename(columns={'%s'%i:'%s'%(np.round(float(i)*dataset.pipe_cfg.instrument['pixelscale'],2)) for i in ee_df.columns[1:]})
        ee_dict[key] = ee_df
    return ee_dict

def task_mvs_photometry(DF,fitsname,ids_list,filter,ee_dict,zpt,la_cr_remove,cr_radius,multiply_by_exptime,
                        multiply_by_gain,multiply_by_PAM,bpx_list,spx_list,radius_ap,radius_sky_inner,radius_sky_outer,sat_thr,
                        kill_plots,grow_curves,r_in,p,gstep, path2savefile):
    '''
    parallelized task for the update_mvs_tiles.
    '''
    phot=[]
    for id in ids_list:
        phot.append(mvs_aperture_photometry(DF,filter,ee_dict,zpt,fitsname=fitsname,
                                            mvs_ids_list_in=[id],bpx_list=bpx_list,spx_list=spx_list,
                                            la_cr_remove=la_cr_remove,cr_radius=cr_radius,radius_ap=radius_ap,
                                            radius_sky_inner=radius_sky_inner,radius_sky_outer=radius_sky_outer,sat_thr=sat_thr,
                                            kill_plots=kill_plots,grow_curves=grow_curves,r_in=r_in,p=p,
                                            gstep=gstep,multiply_by_exptime=multiply_by_exptime,
                                            multiply_by_gain=multiply_by_gain,multiply_by_PAM=multiply_by_PAM,
                                            path2savefile=path2savefile))

    return(phot)

def make_mvs_photometry(DF,filter,mvs_ids_test_list=[],ee_dict=None,workers=None,
                      parallel_runs=True,la_cr_remove=False,cr_radius=3,chunksize = None,
                      multiply_by_exptime=False,multiply_by_gain=False,multiply_by_PAM=False, bpx_list=[], spx_list=[],
                      zpt=0,radius_ap=10,radius_sky_inner=10,radius_sky_outer=15, sat_thr=np.inf, kill_plots=True,
                      grow_curves=True, r_in=1,p=100,gstep=0.1,skip_flags=['rejected'],path2savefile=None):
    '''
    update the multi-visits dataframe with the tile photometry for each source

    '''
    DF.radius_ap = radius_ap
    DF.radius_sky_inner = radius_sky_inner
    DF.radius_sky_outer = radius_sky_outer
    DF.sat_thr = sat_thr
    DF.grow_curves = grow_curves
    getLogger(__name__).info(f'Make photometry for multi-visits targets on filter {filter}')
    fits_dict = {}
    skip_IDs_list_list = DF.mvs_targets_df.loc[DF.mvs_targets_df[f'flag_{filter}'].isin(skip_flags)].mvs_ids.unique()
    if path2savefile is not None:
        if not os.path.exists(path2savefile):
            os.makedirs(path2savefile)
            getLogger(__name__).info(f'Making {path2savefile} directory')

    if len(mvs_ids_test_list) != 0:
        fitsname_test_list = DF.mvs_targets_df.loc[
            DF.mvs_targets_df.mvs_ids.isin(mvs_ids_test_list) & ~DF.mvs_targets_df.mvs_ids.isin(
                skip_IDs_list_list), 'fits_%s' % (filter)].tolist()
        for index, row in DF.mvs_targets_df.loc[
            DF.mvs_targets_df['fits_%s' % (filter)].isin(fitsname_test_list) & DF.mvs_targets_df.mvs_ids.isin(
                    mvs_ids_test_list) & ~DF.mvs_targets_df.mvs_ids.isin(skip_IDs_list_list)].groupby(
                'fits_%s' % (filter)):
            fits_dict[index] = row.mvs_ids.tolist()
    else:
        fitsname_test_list = DF.mvs_targets_df.loc[
            ~DF.mvs_targets_df.mvs_ids.isin(skip_IDs_list_list), 'fits_%s' % (filter)].unique().tolist()
        for index, row in DF.mvs_targets_df.loc[
            DF.mvs_targets_df['fits_%s' % (filter)].isin(fitsname_test_list) & ~DF.mvs_targets_df.mvs_ids.isin(
                    skip_IDs_list_list)].groupby('fits_%s' % (filter)):
            fits_dict[index] = row.mvs_ids.tolist()

    fits_dict = (dict(sorted(fits_dict.items(), key=lambda item: item[1])))
    ids_list_of_lists = list(fits_dict.values())
    fitsname_list = list(fits_dict.keys())

    if parallel_runs:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            workers, chunksize, ntarget = parallelization_package(workers, len(fitsname_list), chunksize=chunksize)

            for phot in executor.map(task_mvs_photometry, repeat(DF), fitsname_list, ids_list_of_lists, repeat(filter), repeat(ee_dict), repeat(zpt), repeat(la_cr_remove), repeat(cr_radius),
                                    repeat(multiply_by_exptime),
                                    repeat(multiply_by_gain), repeat(multiply_by_PAM), repeat(bpx_list), repeat(spx_list), repeat(radius_ap), repeat(radius_sky_inner),
                                    repeat(radius_sky_outer), repeat(sat_thr),
                                    repeat(kill_plots), repeat(grow_curves), repeat(r_in), repeat(p), repeat(gstep), repeat(path2savefile),
                                  chunksize=chunksize):


                phot=np.array(phot)
                for elno in range(len(phot)):
                    sel=(DF.mvs_targets_df.mvs_ids==phot[elno,0].astype(float))&(DF.mvs_targets_df.ext==phot[elno,1].astype(float))
                    DF.mvs_targets_df.loc[
                        sel, ['counts_%s' % filter, 'ecounts_%s' % filter, 'nap_%s' % filter,
                      'm_%s' % filter, 'e_%s' % filter, 'spx_%s' % filter,
                      'bpx_%s' % filter, 'r_%s' % filter, 'rsky1_%s' % filter, 'rsky2_%s' % filter,
                      'sky_%s' % filter, 'esky_%s' % filter, 'nsky_%s' % filter,
                      'grow_corr_%s' % filter]] = phot[elno, 2:].astype(float)
                    # DF.mvs_targets_df.loc[sel,['flag_%s'%filter]]=phot[elno,-1:]

    else:
        for elno in range(len(fitsname_list)):
            phot=task_mvs_photometry(DF, fitsname_list[elno], ids_list_of_lists[elno], filter, ee_dict, zpt, la_cr_remove, cr_radius,
                                multiply_by_exptime,
                                multiply_by_gain, multiply_by_PAM, bpx_list, spx_list, radius_ap, radius_sky_inner,
                                radius_sky_outer, sat_thr,
                                kill_plots, grow_curves, r_in, p, gstep, path2savefile)
            phot=np.array(phot)
            for elno in range(len(phot)):
                sel=(DF.mvs_targets_df.mvs_ids==phot[elno,0].astype(float))&(DF.mvs_targets_df.ext==phot[elno,1].astype(float))
                DF.mvs_targets_df.loc[
                    sel, ['counts_%s' % filter, 'ecounts_%s' % filter, 'nap_%s' % filter,
                          'm_%s' % filter, 'e_%s' % filter, 'spx_%s' % filter,
                          'bpx_%s' % filter, 'r_%s' % filter, 'rsky1_%s' % filter, 'rsky2_%s' % filter,
                          'sky_%s' % filter, 'esky_%s' % filter, 'nsky_%s' % filter,
                          'grow_corr_%s' % filter]] = phot[elno, 2:].astype(float)
                # DF.mvs_targets_df.loc[sel,['flag_%s'%filter]]=phot[elno,-1:]
    return(DF)

def make_median_photometry(DF,filter,unq_ids_list=[],workers=None,parallel_runs=True,suffix='',goodness_phot_label='e', skip_flag='rejected',chunksize=None):
    getLogger(__name__).info(f'Make photometry for average targets on filter {filter}')
    if len(unq_ids_list) == 0: unq_ids_list = DF.unq_targets_df.unq_ids.unique()
    if parallel_runs:
        workers, chunksize, ntarget = parallelization_package(workers, len(unq_ids_list), chunksize=chunksize)
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for phot_out in executor.map(unq_aperture_photometry, repeat(DF), unq_ids_list, repeat(filter),
                                              repeat(goodness_phot_label), repeat(suffix), repeat(skip_flag), chunksize=chunksize):
                phot_out = np.array(phot_out)
                DF.unq_targets_df.loc[
                    DF.unq_targets_df.unq_ids == phot_out[0], ['m_%s' % filter, 'e_%s' % filter,
                                                                 'spx_%s' % filter,
                                                                 'bpx_%s' % filter]] = phot_out[1:]

    else:
        for id in unq_ids_list:
            phot_out = unq_aperture_photometry(DF, id, filter,goodness_phot_label,suffix,skip_flag)
            phot_out = np.array(phot_out)
            DF.unq_targets_df.loc[
                DF.unq_targets_df.unq_ids == phot_out[0], ['m_%s' % filter, 'e_%s' % filter,
                                                                 'spx_%s' % filter, 'bpx_%s' % filter]] = phot_out[1:]
    return(DF)

def run(packet):
    DF = packet['DF']
    dataset = packet['dataset']
    zpt = DF.zpt
    ee_dict=get_ee_df(dataset)
    for filter in dataset.data_cfg.filters:
        DF=make_mvs_photometry(DF, filter,
                            mvs_ids_test_list=[],
                            ee_dict=ee_dict,
                            workers=dataset.pipe_cfg.ncpu,
                            parallel_runs=dataset.pipe_cfg.mkphotometry['parallel_runs'],
                            la_cr_remove=dataset.pipe_cfg.mktiles['la_cr_remove'],
                            cr_radius=dataset.pipe_cfg.mktiles['cr_radius'],
                            multiply_by_exptime=dataset.pipe_cfg.mktiles['multiply_by_exptime'],
                            multiply_by_gain=dataset.pipe_cfg.mktiles['multiply_by_gain'],
                            multiply_by_PAM=dataset.pipe_cfg.mktiles['multiply_by_PAM'],
                            zpt=zpt,
                            radius_ap=dataset.pipe_cfg.mkphotometry['radius_ap']/DF.pixscale,
                            radius_sky_inner=dataset.pipe_cfg.mkphotometry['radius_sky_inner']/DF.pixscale,
                            radius_sky_outer=dataset.pipe_cfg.mkphotometry['radius_sky_outer']/DF.pixscale,
                            kill_plots=not dataset.pipe_cfg.mkphotometry['debug'],
                            grow_curves=dataset.pipe_cfg.mkphotometry['grow_curves'],
                            p=dataset.pipe_cfg.mkphotometry['p'],
                            gstep=dataset.pipe_cfg.mkphotometry['gstep'],
                            bpx_list=dataset.pipe_cfg.mkphotometry['bad_pixel_flags'],
                            spx_list=dataset.pipe_cfg.mkphotometry['sat_pixel_flags'],
                            skip_flags=dataset.pipe_cfg.mkphotometry['skip_flags'],
                            path2savefile = dataset.pipe_cfg.paths['database'] + f'/targets_photometry_tiles/{filter}')


        DF=make_median_photometry(DF,filter,
                               unq_ids_list=[],
                               parallel_runs=dataset.pipe_cfg.mkphotometry['parallel_runs'],
                               workers=dataset.pipe_cfg.ncpu)


    DF.save_dataframes(__name__)
