from stralog import getLogger
from utils_dataframe import mk_crossmatch_ids_df,mk_avg_targets_df,mk_mvs_targets_df,mk_avg_candidates_df,mk_mvs_candidates_df
def mk_targets_df(DF,dataset):#,multiple_visit_id_label='id',average_visits_id_label='id',avg_coords_labels=['ra','dec'],average_type_label='type',max_separation=2):
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

def run(packet):
    dataset = packet['dataset']
    getLogger(__name__).info(f'Initializing new dataframes.')
    DF = packet['DF']
    mk_crossmatch_ids_df(DF, dataset.crossmatch_ids_table)
    mk_targets_df(DF,dataset)
    make_candidates_dataframes(DF)
    DF.save_dataframes(__name__)

