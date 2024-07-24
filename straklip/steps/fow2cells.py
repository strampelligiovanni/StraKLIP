from utils_plot import fow_stamp
from utils_dataframe import update_flags,update_type
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from ancillary import parallelization_package
from stralog import getLogger

def break_FOW_in_cells(DF, filter, suffix='', goodness_phot_label='e', showplot=False, path2savedir='./',
                       workers=None, chunksize=None, psf_nmin=5, qx=10, qy=10, sep_wide=2, sat_px=3,
                       psf_sat_px=3, bad_px=3, psf_bad_px=3, mag_limit=10,
                       psf_goodness_limit=0.01, goodness_limit=0.1,
                       add_flags=True):
    '''
    This is a wrapper for the updating of multi-visits targets dataframe

    '''

    getLogger(__name__).info(f'Braking {filter} FOW in {qx*qy} cells.')

    if add_flags:
        # type_flag = DF.unq_targets_df.loc[DF.unq_targets_df.unq_ids == DF.crossmatch_ids_df.loc[
        #     DF.crossmatch_ids_df.mvs_ids == mvs_ids].unq_ids.unique()[0]].type.values[0]
        # if type_flag == 2:
        #     flag = 'unresolved_double'
        # else:
        #     flag = DF.mvs_targets_df.loc[(DF.mvs_targets_df.mvs_ids == mvs_ids), f'flag_{filter}'].values[0]
        #     if np.isnan(flag):
        #         flag = 'good_target'

        getLogger(__name__).info(f'Updating flag for mvs detections.')
        unq_ids_list = DF.unq_targets_df.unq_ids.unique()
        workers, chunksize, ntarget = parallelization_package(workers, len(unq_ids_list), chunksize=chunksize)

        with ProcessPoolExecutor(max_workers=workers) as executor:
            for out in executor.map(update_flags, repeat(DF), repeat(filter), unq_ids_list, repeat(suffix),
                                 repeat(goodness_phot_label), repeat(sat_px),
                                 repeat(psf_sat_px), repeat(bad_px), repeat(psf_bad_px),
                                 repeat(mag_limit), repeat(psf_goodness_limit),
                                 repeat(goodness_limit), repeat(sep_wide), chunksize=chunksize):
                for elno in range(len(out)):
                    DF.mvs_targets_df.loc[
                        DF.mvs_targets_df.mvs_ids == float(out[elno][0]), ['flag_%s' % filter]] = out[elno][1]

        # for unq_ids in unq_ids_list:
        #     out = update_flags(DF, filter, unq_ids, suffix,
        #                          goodness_phot_label, sat_px,
        #                          psf_sat_px, bad_px, psf_bad_px,
        #                          mag_limit, psf_goodness_limit,
        #                          goodness_limit, sep_wide)
        #     for elno in range(len(out)):
        #         DF.mvs_targets_df.loc[
        #             DF.mvs_targets_df.mvs_ids == float(out[elno][0]), ['flag_%s' % filter]] = out[elno][1]

    fow_stamp(DF, filter, qx, qy, n=20, no_sel=False, path2savedir=path2savedir, psf_nmin=psf_nmin,
              showplot=showplot)
    return(DF)


def run(packet):
    elno=0
    DF = packet['DF']
    dataset = packet['dataset']
    for filter in dataset.data_cfg.filters:
        DF=break_FOW_in_cells(DF, filter,
                           qx=dataset.pipe_cfg.fow2cells['qx'],
                           qy=dataset.pipe_cfg.fow2cells['qy'],
                           psf_nmin=dataset.pipe_cfg.fow2cells['psf_nmin'],
                           sep_wide=dataset.pipe_cfg.mktiles['max_separation'],
                           sat_px=dataset.pipe_cfg.fow2cells['sat_px'],
                           psf_sat_px=dataset.pipe_cfg.fow2cells['psf_sat_px'],
                           bad_px=dataset.pipe_cfg.fow2cells['bad_px'],
                           psf_bad_px=dataset.pipe_cfg.fow2cells['psf_bad_px'],
                           mag_limit=dataset.pipe_cfg.fow2cells['mag_limit'][elno],
                           psf_goodness_limit=dataset.pipe_cfg.fow2cells['psf_goodness_limit'],
                           goodness_limit=dataset.pipe_cfg.fow2cells['goodness_limit'],
                           path2savedir=dataset.pipe_cfg.paths['database'] + '/',
                           add_flags=dataset.pipe_cfg.fow2cells['add_flags'])
        elno+=1

    getLogger(__name__).info(f'Updating type for unique detections.')
    for unq_ids in DF.crossmatch_ids_df.unq_ids.unique():
        DF=update_type(DF, unq_ids)

    DF.save_dataframes(__name__)