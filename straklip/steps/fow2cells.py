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

    getLogger(__name__).info(f'Working on {filter} to braking FOW in {qx*qy} cells.')

    if add_flags:
        getLogger(__name__).info(f'Updating flag for mvs detections.')
        avg_ids_list = DF.avg_targets_df.avg_ids.unique()
        workers, chunksize, ntarget = parallelization_package(workers, len(avg_ids_list), chunksize=chunksize)
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for out in executor.map(update_flags, repeat(DF), repeat(filter), avg_ids_list, repeat(suffix),
                                 repeat(goodness_phot_label), repeat(sat_px),
                                 repeat(psf_sat_px), repeat(bad_px), repeat(psf_bad_px),
                                 repeat(mag_limit), repeat(psf_goodness_limit),
                                 repeat(goodness_limit), repeat(sep_wide), chunksize=chunksize):
                for elno in range(len(out)):
                    DF.mvs_targets_df.loc[
                        DF.mvs_targets_df.mvs_ids == float(out[elno][0]), ['flag_%s' % filter]] = out[elno][1]
    fow_stamp(DF, filter, qx, qy, n=20, no_sel=False, path2savedir=path2savedir, psf_nmin=psf_nmin,
              showplot=showplot)


def run(packet):
    elno=0
    DF = packet['DF']
    dataset = packet['dataset']
    for filter in dataset.data_cfg.filters:
        break_FOW_in_cells(DF, filter,
                           qx=dataset.pipe_cfg.fow2cells['qx'],
                           qy=dataset.pipe_cfg.fow2cells['qy'],
                           psf_nmin=dataset.pipe_cfg.fow2cells['psf_nmin'],
                           sep_wide=dataset.pipe_cfg.mktiles['max_separation'],
                           sat_px=dataset.pipe_cfg.fow2cells['sat_px'],
                           psf_sat_px=dataset.pipe_cfg.fow2cells['psf_sat_px'],
                           bad_px=dataset.pipe_cfg.fow2cells['bad_px'],
                           psf_bad_px=dataset.pipe_cfg.fow2cells['psf_bad_px'],
                           mag_limit=dataset.pipe_cfg.fow2cells['mag_limit'][elno],
                           psf_goodness_limit=dataset.pipe_cfg.fow2cells[
                               'psf_goodness_limit'],
                           goodness_limit=dataset.pipe_cfg.fow2cells['goodness_limit'],
                           path2savedir=dataset.pipe_cfg.paths['database'] + '/')
        elno+=1

    getLogger(__name__).info(f'Updating type for unique detections.')
    for avg_ids in DF.crossmatch_ids_df.avg_ids.unique():
        update_type(DF, avg_ids)

    DF.save_dataframes(__name__)