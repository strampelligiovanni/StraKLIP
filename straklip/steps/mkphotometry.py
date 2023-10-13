import numpy as np
from stralog import getLogger
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from ancillary import parallelization_package
from utils_photometry import mvs_aperture_photometry,avg_aperture_photometry

def get_ee_df(dataset):
    getLogger(__name__).info(f'Fetching encircled energy dataframe for filters {dataset.data_cfg.filters}')
    ee_df=pd.read_csv(dataset.pipe_cfg.paths['database']+'/'+dataset.data_cfg.target['ee_name'])
    ee_df=ee_df.rename(columns={'Filter':'FILTER'})
    ee_df['FILTER'] = ee_df['FILTER'].str.lower()
    ee_df=ee_df.rename(columns={'%s'%i:'%s'%(np.round(float(i)*dataset.pipe_cfg.instrument['pixelscale'],2)) for i in ee_df.columns[1:]})
    return ee_df

def task_mvs_photometry(DF,fitsname,ids_list,filter,ee_df,zpt,la_cr_remove,cr_radius,multiply_by_exptime,
                        multiply_by_gain,multiply_by_PAM,bpx_list,spx_list,radius_in,radius1_in,radius2_in,sat_thr,
                        kill_plots,grow_curves,r_in,p,gstep):
    '''
    parallelized task for the update_mvs_tiles.
    '''
    phot=[]
    for id in ids_list:
        type_flag = DF.avg_targets_df.loc[DF.avg_targets_df.avg_ids == DF.crossmatch_ids_df.loc[
            DF.crossmatch_ids_df.mvs_ids == id].avg_ids.unique()[0]].type.values[0]
        flag = 'good_target'
        if type_flag == 2:
            flag = 'unresolved_double'
        phot.append(mvs_aperture_photometry(DF,filter,ee_df,zpt,fitsname=fitsname,
                                            mvs_ids_list_in=[id],bpx_list=bpx_list,spx_list=spx_list,
                                            la_cr_remove=la_cr_remove,cr_radius=cr_radius,radius_in=radius_in,
                                            radius1_in=radius1_in,radius2_in=radius2_in,sat_thr=sat_thr,
                                            kill_plots=kill_plots,grow_curves=grow_curves,r_in=r_in,p=p,
                                            gstep=gstep,flag=flag,multiply_by_exptime=multiply_by_exptime,
                                            multiply_by_gain=multiply_by_gain,multiply_by_PAM=multiply_by_PAM))

    return(phot)

def make_mvs_photometry(DF,filter,mvs_ids_test_list=[],ee_df=None,workers=None,
                      parallel_runs=True,la_cr_remove=False,cr_radius=3,chunksize = None,
                      multiply_by_exptime=False,multiply_by_gain=False,multiply_by_PAM=False, bpx_list=[], spx_list=[],
                      zpt=0,radius_in=10,radius1_in=10,radius2_in=15, sat_thr=np.inf, kill_plots=True,
                      grow_curves=True, r_in=1,p=100,gstep=0.1):
    '''
    update the multi-visits dataframe with the tile photometry for each source

    '''
    DF.radius_in = radius_in
    DF.radius1_in = radius1_in
    DF.radius2_in = radius2_in
    DF.sat_thr = sat_thr
    DF.grow_curves = grow_curves
    getLogger(__name__).info(f'Make photometry for multi-visits targets on filter {filter}')
    fits_dict = {}
    skip_IDs_list_list = []

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

            for phot in executor.map(task_mvs_photometry, repeat(DF), fitsname_list, ids_list_of_lists, repeat(filter), repeat(ee_df), repeat(zpt), repeat(la_cr_remove), repeat(cr_radius),
                                    repeat(multiply_by_exptime),
                                    repeat(multiply_by_gain), repeat(multiply_by_PAM), repeat(bpx_list), repeat(spx_list), repeat(radius_in), repeat(radius1_in),
                                    repeat(radius2_in), repeat(sat_thr),
                                    repeat(kill_plots), repeat(grow_curves), repeat(r_in), repeat(p), repeat(gstep),
                                  chunksize=chunksize):


                phot=np.array(phot)
                for elno in range(len(phot)):
                    sel=(DF.mvs_targets_df.mvs_ids==phot[elno,0].astype(float))&(DF.mvs_targets_df.ext==phot[elno,1].astype(float))
                    DF.mvs_targets_df.loc[
                        sel, ['counts_%s' % filter, 'ecounts_%s' % filter, 'nap_%s' % filter,
                      'm_%s' % filter, 'e_%s' % filter, 'spx_%s' % filter,
                      'bpx_%s' % filter, 'r_%s' % filter, 'rsky1_%s' % filter, 'rsky2_%s' % filter,
                      'sky_%s' % filter, 'esky_%s' % filter, 'nsky_%s' % filter,
                      'grow_corr_%s' % filter]] = phot[elno, 2:-1].astype(float)
                    DF.mvs_targets_df.loc[sel,['%s_flag'%filter]]=phot[elno,-1:]

    else:
        for elno in range(len(fitsname_list)):
            phot=task_mvs_photometry(DF, fitsname_list[elno], ids_list_of_lists[elno], filter, ee_df, zpt, la_cr_remove, cr_radius,
                                multiply_by_exptime,
                                multiply_by_gain, multiply_by_PAM, bpx_list, spx_list, radius_in, radius1_in,
                                radius2_in, sat_thr,
                                kill_plots, grow_curves, r_in, p, gstep)
            phot=np.array(phot)
            for elno in range(len(phot)):
                sel=(DF.mvs_targets_df.mvs_ids==phot[elno,0].astype(float))&(DF.mvs_targets_df.ext==phot[elno,1].astype(float))
                DF.mvs_targets_df.loc[
                    sel, ['counts_%s' % filter, 'ecounts_%s' % filter, 'nap_%s' % filter,
                          'm_%s' % filter, 'e_%s' % filter, 'spx_%s' % filter,
                          'bpx_%s' % filter, '%s_r' % filter, '%s_rsky1' % filter, '%s_rsky2' % filter,
                          'sky_%s' % filter, 'esky_%s' % filter, 'nsky_%s' % filter,
                          'grow_corr%s' % filter]] = phot[elno, 2:-1].astype(float)
                DF.mvs_targets_df.loc[sel,['%s_flag'%filter]]=phot[elno,-1:]

def make_median_photometry(DF,filter,avg_ids_list=[],workers=None,parallel_runs=True,suffix='',goodness_phot_label='e', skip_flag='rejected',chunksize=None):
    getLogger(__name__).info(f'Make photometry for average targets on filter {filter}')
    if len(avg_ids_list) == 0: avg_ids_list = DF.avg_targets_df.avg_ids.unique()
    if parallel_runs:
        workers, chunksize, ntarget = parallelization_package(workers, len(avg_ids_list), chunksize=chunksize)
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for phot_out in executor.map(avg_aperture_photometry, repeat(DF), avg_ids_list, repeat(filter),
                                              repeat(goodness_phot_label), repeat(suffix), repeat(skip_flag), chunksize=chunksize):
                phot_out = np.array(phot_out)
                DF.avg_targets_df.loc[
                    DF.avg_targets_df.avg_ids == phot_out[0], ['m_%s' % filter, 'e_%s' % filter,
                                                                 'spx_%s' % filter,
                                                                 'bpx_%s' % filter]] = phot_out[1:]

    else:
        for id in avg_ids_list:
            phot_out = avg_aperture_photometry(DF, id, filter,goodness_phot_label,suffix,skip_flag)
            phot_out = np.array(phot_out)
            DF.avg_targets_df.loc[
                DF.avg_targets_df.avg_ids == phot_out[0], ['m_%s' % filter, 'e_%s' % filter,
                                                                 'spx_%s' % filter, 'bpx_%s' % filter]] = phot_out[1:]

def run(packet):
    DF = packet['DF']
    dataset = packet['dataset']
    zpt = DF.zpt
    ee_df=get_ee_df(dataset)
    for filter in dataset.data_cfg.filters:
        make_mvs_photometry(DF, filter,
                            mvs_ids_test_list=[],
                            ee_df=ee_df,
                            workers=dataset.pipe_cfg.ncpu,
                            parallel_runs=dataset.pipe_cfg.mkphotometry['parallel_runs'],
                            la_cr_remove=dataset.pipe_cfg.mktiles['la_cr_remove'],
                            cr_radius=dataset.pipe_cfg.mktiles['cr_radius'],
                            multiply_by_exptime=dataset.pipe_cfg.mktiles['multiply_by_exptime'],
                            multiply_by_gain=dataset.pipe_cfg.mktiles['multiply_by_gain'],
                            multiply_by_PAM=dataset.pipe_cfg.mktiles['multiply_by_PAM'],
                            zpt=zpt,
                            radius_in=dataset.pipe_cfg.mkphotometry['radius_in']/DF.pixscale,
                            radius1_in=dataset.pipe_cfg.mkphotometry['radius1_in']/DF.pixscale,
                            radius2_in=dataset.pipe_cfg.mkphotometry['radius2_in']/DF.pixscale,
                            kill_plots=dataset.pipe_cfg.mkphotometry['kill_plots'],
                            grow_curves=dataset.pipe_cfg.mkphotometry['grow_curves'],
                            p=dataset.pipe_cfg.mkphotometry['p'],
                            gstep=dataset.pipe_cfg.mkphotometry['gstep'])

        make_median_photometry(DF,filter,
                               avg_ids_list=[],
                               parallel_runs=dataset.pipe_cfg.mkphotometry['parallel_runs'],
                               workers=dataset.pipe_cfg.ncpu)


    DF.save_dataframes(__name__)