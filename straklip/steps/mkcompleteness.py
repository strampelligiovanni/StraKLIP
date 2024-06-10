import sys
sys.path.append('/')
from utils_dataframe import mk_fk_completeness_df
from utils_fpanalysis import mk_completeness_from_fakes,mvs_plot_completness


import numpy as np
from stralog import getLogger





def evaluate_completeness_from_fk_inj(DF, filters_list=None, Nvisit_range=None,
                                          path2savedir=None, AUC_lim=0., FP_lim=0.001, DF_fk=None,
                                          skip_filters=[], parallel_runs=False, workers=None):
    '''
    This is a wrapper for the mk_completeness_from_fakes

    Parameters
    ----------
    Nvisit_range : list, or int
        If a list, make a completness map for each of these numbers of visitis.
        If None, take it from catalogue. The default is None.
    path2savedir : str, optional
        path where to save the cell selection pdf. The default is './Plots/'.
    AUC_lim : float
        minimum AUC to consider for detection. The default is 0.75.
    FP_lim : float
        minimum percentage of False Positive. The default is 0.001.
    DF_fk : dataframe class, optional
        dataframe class containing the fake dataframe. If None look into DF. The default is None.
    filter : str
        target firter where to evaluate the tong plot.
    Kmodes_list : list, optional
        list of KLIPmode to use to evaluate the median completness for the tong plot. The default is [].
    num_of_chunks : int, optional
        number of chunk to split the targets. The default is None.

    Returns
    -------
    None.
    '''

    if path2savedir == None: path2savedir = './Plots/'
    if isinstance(Nvisit_range, str):
        nvisit_list = np.arange(int(Nvisit_range.split('-')[0]),int(Nvisit_range.split('-')[1])+1,1)
    elif isinstance(Nvisit_range, int):
        nvisit_list = [Nvisit_range]
    elif isinstance(Nvisit_range, (list, np.ndarray)):
        nvisit_list = Nvisit_range
    else:
        nvisit_list = DF.fk_completeness_df.index.get_level_values('Nvisit').unique()
    DF=mk_fk_completeness_df(DF, nvisit_list, skip_filters=skip_filters)

    for filter in filters_list:
        if filter not in skip_filters:
            getLogger(__name__).info(f'Evaluating completeness map for filter {filter}')
            DF=mk_completeness_from_fakes(DF,
                                          filter,
                                          nvisit_list,
                                          AUC_lim=AUC_lim,
                                          FP_lim=FP_lim,
                                          DF_fk=DF_fk,
                                          skip_filters=skip_filters,
                                          parallel_runs=parallel_runs,
                                          workers=workers)
        else:
            getLogger(__name__).info(f'Skipping filter {filter}. Check klipphotometry.skip_filters in pipe.yaml')

    return(DF)



def run_false_positive_analysis(DF,path2savedir=None, filters_list=[], Nvisit_range=None, AUC_lim=0.5,
                                FP_lim = 0.001, parallel_runs=True, Kmodes_list = [], skip_filters = [], workers=1,
                                save_figure=False, ticks=np.arange(0.3, 1., 0.1),
                                ):

    DF=evaluate_completeness_from_fk_inj(DF,
                                        path2savedir = path2savedir,
                                        filters_list = filters_list,
                                        Nvisit_range = Nvisit_range,
                                        AUC_lim = AUC_lim,
                                        FP_lim = FP_lim,
                                        DF_fk = None,
                                        skip_filters = skip_filters,
                                        parallel_runs = parallel_runs,
                                        workers = workers
                                        )

    mvs_plot_completness(DF,
                            save_figure=save_figure,
                            path2savedir=path2savedir,
                            showplot=False,
                            filters_list=None,
                            avg_ids_list=None,
                            Nvisit_list=None,
                            MagBin_list=None,
                            Kmodes_list=None,
                            ticks=ticks)


    return(DF)

def run(packet):
    DF = packet['DF']
    dataset = packet['dataset']

    DF=run_false_positive_analysis(DF,
                                   path2savedir=dataset.pipe_cfg.paths['out'],
                                   filters_list=DF.filters,
                                   Nvisit_range=dataset.pipe_cfg.mkcompleteness['nvisits'],
                                   AUC_lim=dataset.pipe_cfg.mkcompleteness['auc_limit'],
                                   FP_lim=dataset.pipe_cfg.mkcompleteness['fp_limit'],
                                   Kmodes_list=dataset.pipe_cfg.psfsubtraction['kmodes'],
                                   skip_filters=dataset.pipe_cfg.klipphotometry['skip_filters'],
                                   parallel_runs=dataset.pipe_cfg.mkcompleteness['parallel_runs'],
                                   workers=dataset.pipe_cfg.ncpu,
                                   save_figure=dataset.pipe_cfg.mkcompleteness['save_figure']
                                   )

    DF.save_dataframes(__name__)