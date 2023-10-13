from stralog import getLogger
from utils_straklip import perform_KLIP_PSF_subtraction_on_tiles

def KLIP_PSF_subtraction(DF, filter, label, mvs_ids_list=[], kmodes_list=[], workers=None, parallel_runs=True,
                         skip_flags=['rejected', 'known_double'],overwrite=False, chunksize=None):
    '''
    This is a wrapper for KLIP PSF subtraction step

    '''
    DF.kmodes_list = kmodes_list
    perform_KLIP_PSF_subtraction_on_tiles(DF, filter, label,
                                          workers=workers,
                                          parallel_runs=parallel_runs,
                                          mvs_ids_list=mvs_ids_list,
                                          kmodes_list=kmodes_list,
                                          skip_flags=skip_flags,
                                          chunksize=chunksize,
                                          overwrite=overwrite)

def run(packet):
    dataset = packet['dataset']
    DF = packet['DF']
    if dataset.pipe_cfg.mktiles['cr_remove'] or dataset.pipe_cfg.mktiles['la_cr_remove']:
        label='crclean_data'
        getLogger(__name__).info(f'Performing KLIP PSF subtraction on CR cleaned tiles.')
    else:
        label='data'
        getLogger(__name__).info(f'Performing KLIP PSF subtraction on tiles.')

    for filter in dataset.data_cfg.filters:
        KLIP_PSF_subtraction(DF, filter,
                             label=label,
                             mvs_ids_list=dataset.pipe_cfg.psfsubtraction['mvs_ids_list'],
                             kmodes_list=dataset.pipe_cfg.psfsubtraction['kmodes_list'],
                             workers=dataset.pipe_cfg.ncpu,
                             parallel_runs=dataset.pipe_cfg.psfsubtraction['parallel_runs'],
                             skip_flags=dataset.pipe_cfg.psfsubtraction['skip_flags'],
                             overwrite=dataset.pipe_cfg.psfsubtraction['overwrite'])
    DF.save_dataframes(__name__)
