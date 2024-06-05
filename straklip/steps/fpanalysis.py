import sys
sys.path.append('/')

from utils_straklip import update_mvs_targets,mk_median_tiles_and_photometry,mk_mvs_tiles_and_photometry,update_photometry_after_KLIP_subtraction,check4duplicants,update_median_candidates_tile,perform_KLIP_PSF_subtraction_on_tiles,update_candidates,update_candidates_photometry,perform_FP_analysis,mk_completeness_from_fakes,update_fakes_df
from utils_dataframe import mk_header_df,mk_targets_df,add_flag2dataframe,add_type2dataframe,mk_fakes_df,mk_fk_completeness_df
from utils_plot import mvs_completeness_plots

import numpy as np
from tqdm import tqdm



def evaluate_completeness_from_fk_inj(self, filters_list=None, Nvisit_range=None, magbin_list=None,
                                      path2savedir=None, AUC_lim=0., FP_lim=0.001, DF_fk=None, Kmodes_list=[],
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
        dataframe class containing the fake dataframe. If None look into self. The default is None.
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
    if isinstance(Nvisit_range, int):
        nvisit_list = [Nvisit_range]
    elif isinstance(Nvisit_range, (list, np.ndarray)):
        nvisit_list = Nvisit_range
    else:
        nvisit_list = self.fk_completeness_df.index.get_level_values('Nvisit').unique()
    mk_fk_completeness_df(self, nvisit_list, skip_filters=skip_filters)

    # if filter!=None and not isinstance(Nvisit_range, (list,np.ndarray)):  filters_list=[filter]
    if not isinstance(filters_list, (list, np.ndarray)): filters_list = self.filters_list

    for filter in filters_list:
        if filter not in skip_filters:
            if isinstance(Nvisit_range, int):
                Nvisit_range_out = [Nvisit_range]
            elif not isinstance(Nvisit_range, (list, np.ndarray)):
                Nvisit_range_out = np.sort(self.avg_candidates_df['N%s' % filter[1:4]].loc[
                                               ~self.avg_candidates_df['N%s' % filter[1:4]].isna()].unique())
            else:
                Nvisit_range_out = Nvisit_range.copy()
            if filter not in skip_filters:
                mk_completeness_from_fakes(self, filter, Nvisit_range_out, magbin_list=magbin_list, AUC_lim=AUC_lim,
                                           FP_lim=FP_lim, DF_fk=DF_fk, skip_filters=skip_filters,
                                           parallel_runs=parallel_runs, workers=workers)


def update_fk_dataframes(self,filter,workers=None,NPSFstars=300,NPSFsample=30,path2psfdir=None,psf_filename='',parallel_runs=False,showplot=False,aptype='4pixels',delta=1,use_median_sky=False,suffix=''):
        '''
        Tis is a wrapper for the updating of the fake dataframes

        Parameters
        ----------
        workers : int, optional
            number of workers to split the work accross multiple CPUs. The default is 3.
        NPSFstars : int, optional
            number of PSF to create. The default is 300.
        NPSFsample : int, optional
            number of PSF to select to creat the sample for the PSF libray. The default is 30.
        path2psfdir: str, optional
            default dir for psfs. If None, use default path. The default is None
        aptype : (circular,square,4pixels), optional
            defin the aperture type to use during aperture photometry.
            The default is '4pixels'.
        delta : int, optional
            step to create the square mask in range -delta, x, +delt and -delta, y, +delta. The default is 1.
        Returns
        -------
        None.

        '''
        update_fakes_df(self,filter,parallel_runs=parallel_runs,workers=workers,NPSFstars=NPSFstars,NPSFsample=NPSFsample,path2psfdir=path2psfdir,psf_filename=psf_filename,showplot=showplot,aptype=aptype,delta=delta,use_median_sky=use_median_sky,suffix=suffix)


def mvs_plot_completness(self, filters_list=[], path2savedir=None, MagBin_list=[], Nvisit_list=[], avg_ids_list=[],
                         Kmodes_list=[], title=None, fx=7, fy=7, fz=20, ncolumns=4, xnew=None, ynew=None,
                         ticks=np.arange(0.3, 1., 0.1), show_IDs=False, save_completeness=False, save_figure=False,
                         skip_filters=[], showplot=False, parallel_runs=True, suffix=''):
    '''
    This is a wrapper for the mvs_completeness_plots

    Parameters
    ----------
    Nvisit_range : list, or int
        If a list, make a completness map for each of these numbers of visitis.
        If None, take it from catalogue. The default is None.
    path2savedir : str, optional
        path where to save the cell selection pdf. The default is './Plots/'.
    Kmodes_list : list, optional
        list of KLIPmode to use to evaluate the median completness for the tong plot. The default is [].
    title : str, optional
        title of the tong plot. The default is ''.
    fx : int, optional
        x dimension of each subplots. The default is 7.
    fy : int, optional
        y dimension of each subplots. The default is 7.
    fz : int, optional
        font size for the title. The default is 20.
    ncolumns : int, optional
        number of colum for the subplots. The default is 4.
    xnew : list or None, optional
        list of values for interpolate on the X axis. If None use the default X axixs of the dataframe. The default is None.
    ynew : list or None, optional
        list of values for interpolate on the Y axis. If None use the default X axixs of the dataframe. The default is None.
    ticks : list, optional
        mark a contourn line corresponding to these values on the tong plot. The default is np.arange(0.3,1.,0.1).
    show_IDs : bool, optional
        choose to show IDs for each candidate on the tong plot. The default is False.
    save_completeness : bool, optional
        chose to save the completeness of each candidate in the dataframe. The default is True.
    save_figure : bool, optional
        chose to save tong plots on HD. The default is True.
    num_of_chunks : int, optional
        number of chunk to split the targets. The default is None.

    Returns
    -------
    None.
    '''
    # if not isinstance(filters_list, (list,np.ndarray)):
    if len(MagBin_list) == 0: MagBin_list = self.fk_completeness_df.index.get_level_values('magbin').unique()
    if len(filters_list) == 0: filters_list = self.fk_completeness_df.index.get_level_values('filter').unique()
    if len(Nvisit_list) == 0: Nvisit_list = self.fk_completeness_df.index.get_level_values('nvisit').unique()
    if len(Kmodes_list) == 0: Kmodes_list = self.header_df.loc['Kmodes_list', 'Values']
    if path2savedir == None: path2savedir = './Plots/'

    print('Working on filter:')
    for filter in tqdm(filters_list):
        if filter not in skip_filters:
            # print('> ',filter)
            if len(MagBin_list) == 0: MagBin_list = self.fk_completeness_df.index.get_level_values('magbin').unique()
            mvs_completeness_plots(self, filter=filter, path2savedir=path2savedir, MagBin_list=MagBin_list,
                                   Nvisit_list=Nvisit_list, avg_ids_list=avg_ids_list, Kmodes_list=Kmodes_list,
                                   title=title, fx=fx, fy=fy, fz=fz, ncolumns=ncolumns, xnew=xnew, ynew=ynew,
                                   ticks=ticks, show_IDs=show_IDs, save_completeness=save_completeness,
                                   save_figure=save_figure, showplot=showplot, suffix=suffix)


def update_candidates_photometry_dataframes(self,avg_ids_list=[],label='data',aptype='4pixels',verbose=False,noBGsub=False,sigma=2.5,DF_fk=None,delta=3,skip_filters=[],sat_thr=np.inf,suffix=''):
    '''
    This is a wrapper for the photometry update for candidates dataframes after estimating the throughput

    Parameters
    ----------
    avg_ids_list : list, optional
        list of average ids to test. The default is [].
    aptype : (circular,square,4pixels), optional
        defin the aperture type to use during aperture photometry.
        The default is '4pixels'.
    verbose : bool, optional
        choose to show prints and plots.
    noBGsub : TYPE, optional
        DESCRIPTION. The default is False.
    sigma : TYPE, optional
        DESCRIPTION. The default is 2.5.
    DF_fk : dataframe class, optional
        dataframe class containing the fake dataframe. If None look into self. The default is None.

    Returns
    -------
    None.

    '''
    update_candidates_photometry(self,avg_ids_list=avg_ids_list,label=label,aptype=aptype,verbose=verbose,noBGsub=noBGsub,sigma=sigma,DF_fk=DF_fk,delta=delta,skip_filters=skip_filters,sat_thr=sat_thr,suffix=suffix)

def false_positive_analysis(self,avg_ids_list=[],showplot=False,verbose=False,DF_fk=None,skip_filters=[],nbins=10,suffix=''):
    '''
    This is a wrapper for the false positive analysis on candidates dataframes

    Parameters
    ----------
    avg_ids_list : list, optional
        list of average ids to test. The default is [].
    showplot : bool, optional
        choose to show plots.
    verbose : bool, optional
        choose to show prints.
    DF_fk : dataframe class, optional
        dataframe class containing the fake dataframe. If None look into self. The default is None.

    Returns
    -------
    None.

    '''
    perform_FP_analysis(self,avg_ids_list=avg_ids_list,showplot=showplot,verbose=verbose,DF_fk=DF_fk,skip_filters=skip_filters,nbins=nbins,suffix=suffix)


make_fk_dataframes(MagBin_list=MagBin_list,Dmag_list=Dmag_list,Sep_list=Sep_list,Nstar=Nstar)
update_fk_dataframes(filter,parallel_runs=True,workers=workers,aptype=aptype,delta=delta)
evaluate_completeness_from_fk_inj(path2savedir=path2dir,Nvisit_range=Nvisit_range,AUC_lim=0.5,parallel_runs=True,workers=workers)
mvs_plot_completness(save_figure=False,showplot=True,avg_ids_list=None,Nvisit_list=[...],MagBin_list=[...],Kmodes_list=[...],ticks=np.arange(0.3,1.,0.1))
update_candidates_photometry_dataframes(label='data',aptype='4pixels',verbose=True,delta=2)
false_positive_analysis(avg_ids_list=[],showplot=True,verbose=True)
def run(packet):
    DF = packet['DF']
    dataset = packet['dataset']


    DF=ooooo()

    DF.save_dataframes(__name__)