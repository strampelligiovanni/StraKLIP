from stralog import getLogger
from utils_dataframe import mk_crossmatch_ids_df,mk_avg_targets_df,mk_mvs_targets_df,mk_avg_candidates_df,mk_mvs_candidates_df,mk_fakes_df

def mk_targets_df(DF,dataset):
    '''
    This is a wrapper for the creation of the targets dataframe

    '''
    getLogger(__name__).info(f'Creating the targets dataframe')
    mk_mvs_targets_df(DF,dataset)
    mk_avg_targets_df(DF,dataset)

def make_candidates_dataframes(DF):
    '''
    This is a wrapper for the creation of the candidates dataframes for the pipeline

    Returns
    -------
    None.

    '''
    getLogger(__name__).info(f'Creating the candidates dataframe')
    mk_avg_candidates_df(DF)
    mk_mvs_candidates_df(DF)

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
                MagBin_list=dataset.pipe_cfg.buildhdf['fp_table']['magbins'],
                Dmag_list=dataset.pipe_cfg.buildhdf['fp_table']['dmags'],
                Sep_range=dataset.pipe_cfg.buildhdf['fp_table']['sep_range'],
                Nstar=dataset.pipe_cfg.buildhdf['fp_table']['nstar'],
                filters=dataset.data_cfg.filters,
                skip_filters=dataset.pipe_cfg.buildhdf['fp_table']['skip_filters'])


def run(packet):
    dataset = packet['dataset']
    getLogger(__name__).info(f'Initializing new dataframes.')
    DF = packet['DF']
    mk_crossmatch_ids_df(DF, dataset.crossmatch_ids_table)
    mk_targets_df(DF,dataset)
    make_candidates_dataframes(DF)
    make_fk_dataframes(DF,dataset)
    DF.save_dataframes(__name__)

