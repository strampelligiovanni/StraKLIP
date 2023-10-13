from utils_plot import fow_stamp
from utils_dataframe import add_flag2dataframe
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from ancillary import parallelization_package
from stralog import getLogger

def break_FOW_in_cells(DF, filter, suffix='', goodness_phot_label='e', showplot=False, path2savedir='./',
                       workers=None, chunksize=None, psf_nmin=5, qx=10, qy=10, sep_wide=2, default_sat_px=3,
                       default_psf_sat_px=3, default_bad_px=3, default_psf_bad_px=3, default_mag_limit=10,
                       default_psf_goodness_limit=0.01, default_goodness_limit=0.1,
                       add_flags=True):
    '''
    This is a wrapper for the updating of multi-visits targets dataframe

    '''

    getLogger(__name__).info(f'Working on {filter} to braking FOW in {qx*qy} cells.')

    if add_flags:
        getLogger(__name__).info(f'Updating flag for stars.')
        avg_ids_list = DF.avg_targets_df.avg_ids.unique()
        workers, chunksize, ntarget = parallelization_package(workers, len(avg_ids_list), chunksize=chunksize)
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for out in executor.map(add_flag2dataframe, repeat(DF), repeat(filter), avg_ids_list, repeat(suffix),
                                 repeat(goodness_phot_label), repeat(default_sat_px),
                                 repeat(default_psf_sat_px), repeat(default_bad_px), repeat(default_psf_bad_px),
                                 repeat(default_mag_limit), repeat(default_psf_goodness_limit),
                                 repeat(default_goodness_limit), repeat(sep_wide), chunksize=chunksize):
                for elno in range(len(out)):
                    DF.mvs_targets_df.loc[
                        DF.mvs_targets_df.mvs_ids == float(out[elno][0]), ['flag_%s' % filter]] = out[elno][1]
    fow_stamp(DF, filter, qx, qy, n=20, no_sel=False, path2savedir=path2savedir, psf_nmin=psf_nmin,
              showplot=showplot)


def run(packet):
    DF = packet['DF']
    dataset = packet['dataset']
    for filter in dataset.data_cfg.filters:
        break_FOW_in_cells(DF, filter,
                           qx=dataset.data_cfg.fow2cells['qx'],
                           qy=dataset.data_cfg.fow2cells['qy'],
                           psf_nmin=dataset.data_cfg.fow2cells['psf_nmin'],
                           sep_wide=dataset.data_cfg.tiles['max_separation'],
                           default_sat_px=dataset.pipe_cfg.fow2cells['default_sat_px'],
                           default_psf_sat_px=dataset.pipe_cfg.fow2cells['default_psf_sat_px'],
                           default_bad_px=dataset.pipe_cfg.fow2cells['default_bad_px'],
                           default_psf_bad_px=dataset.pipe_cfg.fow2cells['default_psf_bad_px'],
                           default_mag_limit=dataset.pipe_cfg.fow2cells['default_mag_limit'],
                           default_psf_goodness_limit=dataset.pipe_cfg.fow2cells[
                               'default_psf_goodness_limit'],
                           default_goodness_limit=dataset.pipe_cfg.fow2cells['default_goodness_limit'],
                           path2savedir=dataset.pipe_cfg.paths['database'] + '/')

    DF.save_dataframes(__name__)