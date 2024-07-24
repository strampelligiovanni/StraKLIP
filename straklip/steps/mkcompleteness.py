import sys
sys.path.append('/')
from utils_dataframe import mk_fk_completeness_df
from utils_fpanalysis import mk_completeness_from_fakes,mvs_plot_completness,unq_plot_completness
from utils_completeness import flatten_matrix_completeness_curves_df,mk_completeness_curves_df,make_matrix_completeness_curves_df
from ancillary import interpND

import numpy as np
from stralog import getLogger
import pandas as pd




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
        nvisit_list = DF.fk_completeness_df.index.get_level_values('nvisit').unique()
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



def make_mvs_completeness_maps(DF,path2savedir=None, filters_list=[], Nvisit_range=None, AUC_lim=0.5,
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
                            unq_ids_list=None,
                            Nvisit_list=None,
                            MagBin_list=None,
                            Kmodes_list=None,
                            ticks=ticks)


    return(DF)

# def evaluate_average_completeness_from_fk_inj(DF, mag2mass_interpolator, mass2mag_interoplator, mass_list=[0.001, 2],
#                                 skip_filters=[], filters_list=[], Nvisit_list=[], KLIPmodes_list=[],
#                                 sma_interp=True, parallel_runs=True, workers=None):
def evaluate_average_completeness_from_fk_inj(DF, skip_filters=[], filters_list=[], Nvisit_list=[], KLIPmodes_list=[],
                                              sep_step=0.01, parallel_runs=True, workers=None):

    DCC_list_df = []
    # QCC_list_df = []
    MDCC_list_df = []
    # MQCC_list_df = []
    key_filters_list = []
    if len(filters_list) == 0: filters_list = DF.filters_list
    getLogger(__name__).info(f'Making average completeness dataframe from fake injections')
    for filter in filters_list:
        if filter not in skip_filters:
            # DCC_df, QCC_df = mk_completeness_curves_df(DF, 'MagBin%s' % filter[1:4], filter,
            #                                            [mag2mass_interpolator, mass2mag_interoplator],
            #                                            pmass_range_list=mass_list)
            # MDCC_df, MQCC_df = make_matrix_completeness_curves_df(DCC_df, QCC_df, sma_interp=sma_interp,
            #                                                       Nvisit_list=Nvisit_list,
            #                                                       KLIPmodes_list=KLIPmodes_list,
            #                                                       parallel_runs=parallel_runs, workers=workers)
            DCC_df = mk_completeness_curves_df(DF, filter)
            MDCC_df = make_matrix_completeness_curves_df(DCC_df, Nvisit_list=Nvisit_list, KLIPmodes_list=KLIPmodes_list,
                                                         sep_step=sep_step, parallel_runs=parallel_runs, workers=workers)

            DCC_list_df.append(DCC_df)
            # QCC_list_df.append(QCC_df)
            MDCC_list_df.append(MDCC_df)
            # MQCC_list_df.append(MQCC_df)
            key_filters_list.append(filter)
        else:
            getLogger(__name__).info(f'Skipping filter {filter}. Check klipphotometry.skip_filters in pipe.yaml')

    DF.dmag_completeness_curves_df = pd.concat(DCC_list_df, keys=key_filters_list, names=['Filter'], axis=0)
    # DF.q_completeness_curves_df = pd.concat(QCC_list_df, keys=key_filters_list, names=['Filter'], axis=0)
    DF.matrix_dmag_completeness_curves_df = pd.concat(MDCC_list_df, keys=key_filters_list, names=['Filter'],
                                                      axis=0)
    # DF.matrix_q_completeness_curves_df = pd.concat(MQCC_list_df, keys=key_filters_list, names=['Filter'], axis=0)

    return(DF)

def flatten_completeness_curves_df(DF,filters_list=[],sma_interp=True,skip_filters=[],Nvisit_list=None,KLIPmodes_list=None):
    FMDCC_list_df=[]
    # FMQCC_list_df=[]
    key_filters_list=[]
    if len(filters_list)==0: filters_list=DF.filters_list
    getLogger(__name__).info(f'Making flat completeness curve dataframe')
    for filter in filters_list:
        if filter not in skip_filters:
            # FMDCC_df,FMQCC_df=flatten_matrix_completeness_curves_df(DF,filter,sma_interp=sma_interp,Nvisit_list=Nvisit_list,KLIPmodes_list=KLIPmodes_list)
            FMDCC_df=flatten_matrix_completeness_curves_df(DF,filter,sma_interp=sma_interp,Nvisit_list=Nvisit_list,KLIPmodes_list=KLIPmodes_list)
            FMDCC_list_df.append(FMDCC_df)
            # FMQCC_list_df.append(FMQCC_df)
            key_filters_list.append(filter)
        else:
            getLogger(__name__).info(f'Skipping filter {filter}. Check klipphotometry.skip_filters in pipe.yaml')

    DF.flatten_matrix_dmag_completeness_curves_df=pd.concat(FMDCC_list_df,keys=key_filters_list,names=['Filter'],axis=0)
    # DF.flatten_matrix_q_completeness_curves_df=pd.concat(FMQCC_list_df,keys=key_filters_list,names=['Filter'],axis=0)
    return(DF)

def mk_interpolations(DF,iso_path='',iso_name='', smooth=0.001,method='linear',showplot=False):

    iso_df = pd.read_csv(iso_path+iso_name,sep='\s+')
    mag_label_list = np.array(['m%s' % filter for filter in DF.filters])
    node_label_list = list(mag_label_list)
    node_list = [iso_df[label].values.ravel() for label in node_label_list]
    x = np.log10(iso_df['mass'].values).ravel()

    interp_iso_dict = interpND([x, node_list], method=method, showplot=showplot, smooth=smooth, z_label=node_label_list,
                               workers=0, progress=False, fx=1500, fy=700, x_label='mass', nrows=1)

    mag_label_list = np.array(['m%s' % filter for filter in DF.filters])
    node_label_list = list(mag_label_list)
    interp_iso_mass_dict = dict()
    for label in node_label_list:
        x = iso_df[label].values.ravel()

        interp_btsettl_mass_new_dict = interpND([x, [np.log10(iso_df['mass'].values).ravel()]], method=method,
                                                showplot=showplot, smooth=smooth, z_label=['mass%s' % label[1:]],
                                                x_label=label, workers=10, fx=600, fy=600, nrows=1)
        interp_iso_mass_dict.update(interp_btsettl_mass_new_dict)

    return(interp_iso_mass_dict,interp_iso_dict)

def make_unq_completeness_maps(DF,filters_list=[],sma_interp=True,skip_filters=[],Nvisit_list=None,KLIPmodes_list=None,
                               sep_step=0.01, mass_list=[0.001, 2], parallel_runs=True, workers=None,save_figure=''):


    nvisit_list = DF.fk_completeness_df.index.get_level_values('nvisit').unique()

    DF=evaluate_average_completeness_from_fk_inj(DF, skip_filters=skip_filters, filters_list=filters_list,
                                                 Nvisit_list=nvisit_list, KLIPmodes_list=KLIPmodes_list,
                                                 sep_step=sep_step, parallel_runs=parallel_runs, workers=workers)

    DF=flatten_completeness_curves_df(DF,filters_list=filters_list,sma_interp=sma_interp,skip_filters=skip_filters,
                                      Nvisit_list=nvisit_list,KLIPmodes_list=KLIPmodes_list)

    unq_plot_completness(DF,ylim=[],xlim=[],dist=DF.dist,df_ylabel='Dmag',df_xlabel='SMA',
                         xlabel='projected SMA [arcsec]',ylabel='Dmag',log_y=False,
                         show_plot=False,c_lim=0.1,collapsed=True,path2savedir=save_figure)
    return(DF)



def run(packet):
    DF = packet['DF']
    dataset = packet['dataset']

    DF=make_mvs_completeness_maps(DF,
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

    DF=make_unq_completeness_maps(DF,
                                  filters_list=DF.filters,
                                  sma_interp=dataset.pipe_cfg.mkcompleteness['sma_interp'],
                                  skip_filters=dataset.pipe_cfg.klipphotometry['skip_filters'],
                                  Nvisit_list=dataset.pipe_cfg.mkcompleteness['nvisits'],
                                  KLIPmodes_list=dataset.pipe_cfg.psfsubtraction['kmodes'],
                                  sep_step=dataset.pipe_cfg.mkcompleteness['sep_step'],
                                  # mass_list=dataset.pipe_cfg.mkcompleteness['mass_list'],
                                  parallel_runs=dataset.pipe_cfg.mkcompleteness['parallel_runs'],
                                  workers=dataset.pipe_cfg.ncpu,
                                  save_figure=dataset.pipe_cfg.mkcompleteness['save_figure'])

    DF.save_dataframes(__name__)