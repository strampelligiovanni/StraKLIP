"""
utilities functions that can be use by or with the dataframe class
"""

import sys
sys.path.append('/')
# from pipeline_config import path2data
from ancillary import distances_cube
from stralog import getLogger

import pandas as pd
import numpy as np
######################
# Ancillary routines #
######################

def create_empty_df(row_list,columns_list,set_index2=[],int_columns=None,object_columns=None,str_columns=None,flt_columns=None,multy_index=False,levels=[]):
    '''
    Create an empty dataframe

    Parameters
    ----------
    row_list : list
        list of index ofr the empty dataframe.
    columns_list : list
        list of columns for the empty datarame.
    set_index2 : str, optional
        name of a column to use as index. The default is []. It is used to create multi-index dataframe
    int_columns : list, optional
        list of columns to which assign int format. The default is None.
    object_columns : list, optional
        list of columns to which assign object format. The default is None.
    str_columns : list, optional
        list of columns to which assign object format. The default is None.
    flt_columns : list, optional
        list of columns to which assign float format. The default is None.
    multy_index : bool, optional
        choose to create a multindex dataframe or not. The default is False.
    levels : list, optional
        list of lists of indexs to populate the multiindex index part of the datafrane. The default is [].

    Examples:
        1) Create a general dataframe:
            
        >create_empty_df([0,1],['a','b','c','d'])
        
             a    b    c    d
        ---------------------
        0    0    0    0    0
        1    0    0    0    0

        2) Create a multi_index dataframe:
            
        >create_empty_df(['x','y'],['a','b','c','d'],multy_index=True,levels=[[0,1],[0,1,2,3]])
        
                 a    b    c    d
        x    y                
        --------------------------
        0    0    0    0    0    0
             1    0    0    0    0
             2    0    0    0    0
             3    0    0    0    0
        1    0    0    0    0    0
             1    0    0    0    0
             2    0    0    0    0
             3    0    0    0    0

    Returns
    -------
    None.

    '''
    if multy_index == True:
        if levels:
            my_index = pd.MultiIndex.from_product([level for level in levels], names=row_list)
        else:
            empty_list = [list([]) for _ in np.arange(len(row_list))]
            my_index = pd.MultiIndex(levels=empty_list,
                                      labels=empty_list,
                                      names=row_list)
        empty_df = pd.DataFrame(index=my_index, columns=columns_list)
    else:
        empty_df = pd.DataFrame(index=row_list,columns=columns_list)

    if len(set_index2)>0: empty_df=empty_df.set_index(set_index2)

    if int_columns != None: empty_df=empty_df.astype({key: int for key in int_columns})
    if object_columns != None: empty_df=empty_df.astype({key: object for key in object_columns})
    if str_columns != None: empty_df=empty_df.astype({key: str for key in str_columns})
    if flt_columns != None: empty_df=empty_df.astype({key: float for key in flt_columns})
    
    return(empty_df)

def update_flags(DF,filter,avg_ids,suffix='',goodness_phot_label='e',sat_px=3,psf_sat_px=3,bad_px=3,psf_bad_px=3,mag_limit=10,psf_goodness_limit=0.01,goodness_limit=0.5,sep_wide=2):
    '''
    This is a wrapper for ta add flags to the targets dataframe

    Parameters
    ----------
    suffix: str, optional
        suffix to append to mag label. For example, if original photometry is present in the catalog, it canbe use with suffix='_o'.
        Default is ''.
    goodness_phot_label: str, optional
        label to look for to mesure the goodness of the photometry, for example 'e' for the error or 'chi' for the chi square.
        Default is 'e'.
    sat_px : int, optional
        default number of saturated pixels in the tile to limit target selection. 
        The default is 3.
    psf_sat_px : int, optional
        default number of saturated pixels in the tile to limit PSF target selection. 
        The default is 3.
    bad_px : int, optional
        default number of bad pixels in the tile to limit target selection. 
        The default is 3.
    psf_bad_px : int, optional
        default number of bad pixels in the tile to limit PSF target selection. 
        The default is 3.
    psf_goodness_limit : int, optional
        default magnitude uncertanties or chi square to limit PSF selection. 
        The default is 0.01.
    goodness_limit : int, optional
        default magnitude uncertanties or chi square to limit PSF/Candidate selection. 
        The default is 0.5.
    sep_wide : list, optional
        separation between sources to limit PSF/Candidate selection and define the wide doubles already present in the input catalog. 
        The default is 2 arcsecond.

    Returns
    -------
    None.

    '''
    # for filter in filters_list:
    mvs_ids_list=DF.crossmatch_ids_df.loc[DF.crossmatch_ids_df.avg_ids==avg_ids].mvs_ids.unique()    
    if DF.avg_targets_df.loc[DF.avg_targets_df.avg_ids==avg_ids,'type'].values[0]==0:
        DF.avg_targets_df.loc[DF.avg_targets_df.avg_ids==avg_ids,'type']=0
        DF.mvs_targets_df.loc[DF.mvs_targets_df.mvs_ids.isin(mvs_ids_list),'flag_%s'%filter]='rejected'
    elif DF.avg_targets_df.loc[DF.avg_targets_df.avg_ids==avg_ids,'type'].values[0]==2:
        DF.avg_targets_df.loc[DF.avg_targets_df.avg_ids==avg_ids,'type']=2
        DF.mvs_targets_df.loc[DF.mvs_targets_df.mvs_ids.isin(mvs_ids_list),'flag_%s'%filter]='unresolved_double'
    else:
        DF.avg_targets_df.loc[DF.avg_targets_df.avg_ids==avg_ids,'type']=1
        DF.mvs_targets_df.loc[DF.mvs_targets_df.mvs_ids.isin(mvs_ids_list),'flag_%s'%filter]='good_target'

        
    avg_df_sel=DF.avg_targets_df.loc[DF.avg_targets_df.avg_ids==avg_ids]
    ID_good_list=avg_df_sel.loc[(avg_df_sel.FirstDist>sep_wide)].avg_ids.values#&~(avg_df_sel.type.isin([0,2]))
    ID_wide_list=avg_df_sel.loc[(avg_df_sel.FirstDist<=sep_wide)].avg_ids.values #|(avg_df_sel.type==3)
    ID_double_list=avg_df_sel.loc[(avg_df_sel.type==2)].avg_ids.values
    
    flag_list=[]
    for mvs_ids in mvs_ids_list:
        mvs_df_sel=DF.mvs_targets_df.loc[DF.mvs_targets_df.mvs_ids==mvs_ids]
        crossmatch_sel=DF.crossmatch_ids_df.loc[DF.crossmatch_ids_df.mvs_ids==mvs_ids]
        x_sel=~(mvs_df_sel['x_%s'%filter].isna())&(mvs_df_sel['x_%s'%filter]>=(DF.tilebase-1)/2)&(mvs_df_sel['x_%s'%filter]<=DF.xyaxis['x']-(DF.tilebase-1)/2)
        y_sel=~(mvs_df_sel['x_%s'%filter].isna())&(mvs_df_sel['y_%s'%filter]>=(DF.tilebase-1)/2)&(mvs_df_sel['y_%s'%filter]<=DF.xyaxis['y']-(DF.tilebase-1)/2)
        if (mvs_df_sel['m_%s%s'%(filter,suffix)].values[0] < mag_limit) or (mvs_df_sel['%s_%s%s'%(goodness_phot_label,filter,suffix)].values[0] > goodness_limit) or (mvs_df_sel['spx_%s'%filter].values[0] > sat_px) or (mvs_df_sel['bpx_%s'%filter].values[0] > bad_px) or not x_sel.values[0] or not y_sel.values[0]:#or crossmatch_sel.avg_ids.isin(ID_bad_list).values[0]
            flag='rejected'
        else:
            if crossmatch_sel.avg_ids.isin(ID_wide_list).values[0]:
                flag='known_double'
            elif crossmatch_sel.avg_ids.isin(ID_double_list).values[0]:
                flag='unresolved_double'
            else:
                if (mvs_df_sel['spx_%s'%filter].values[0] <=psf_sat_px) and (mvs_df_sel['bpx_%s'%filter].values[0] <=psf_bad_px) and (mvs_df_sel['%s_%s%s'%(goodness_phot_label,filter,suffix)].values[0] <= psf_goodness_limit): #crossmatch_sel.avg_ids.isin(ID_psf_list).values[0] and
                    flag='good_psf'
                elif crossmatch_sel.avg_ids.isin(ID_good_list).values[0]:
                    flag='good_target'
        flag_list.append([float(mvs_ids),flag])

    return(flag_list)

def update_type(DF,avg_ids):
    mvs_ids_list=DF.crossmatch_ids_df.loc[DF.crossmatch_ids_df.avg_ids==avg_ids].mvs_ids.unique()

    if DF.mvs_targets_df.loc[DF.mvs_targets_df.mvs_ids.isin(mvs_ids_list),['flag_%s'%(filter) for filter in DF.filters]].apply(lambda x: x.str.contains('rejected',case=False)).all(axis=1).all(axis=0):
        DF.avg_targets_df.loc[DF.avg_targets_df.avg_ids==avg_ids,['type']]=0
    elif DF.mvs_targets_df.loc[DF.mvs_targets_df.mvs_ids.isin(mvs_ids_list),['flag_%s'%(filter) for filter in DF.filters]].apply(lambda x: x.str.contains('known',case=False)).any(axis=1).any(axis=0):
        if DF.mvs_targets_df.loc[DF.mvs_targets_df.mvs_ids.isin(mvs_ids_list),['flag_%s'%(filter) for filter in DF.filters]].apply(lambda x: x.str.contains('known|rejected',case=False)).all(axis=1).all(axis=0):
            if DF.avg_targets_df.loc[DF.avg_targets_df.avg_ids==avg_ids].type.isin([0,1,2,3]).values[0]:
                DF.avg_targets_df.loc[DF.avg_targets_df.avg_ids==avg_ids,['type']]=3
        else:
            raise ValueError('All flags should be the same for a known_double. Please check avg_ids %s'%avg_ids)
    elif DF.mvs_targets_df.loc[DF.mvs_targets_df.mvs_ids.isin(mvs_ids_list),['flag_%s'%(filter) for filter in DF.filters]].apply(lambda x: x.str.contains('unresolved',case=False)).any(axis=1).any(axis=0):
        if DF.mvs_targets_df.loc[DF.mvs_targets_df.mvs_ids.isin(mvs_ids_list),['flag_%s'%(filter) for filter in DF.filters]].apply(lambda x: x.str.contains('unresolved|rejected',case=False)).all(axis=1).all(axis=0):
            if DF.avg_targets_df.loc[DF.avg_targets_df.avg_ids==avg_ids].type.isin([0,1,2,3]).values[0]:
                DF.avg_targets_df.loc[DF.avg_targets_df.avg_ids==avg_ids,['type']]=2
        else:
            raise ValueError('All flags should be the same for an unresolved_double. Please check avg_ids %s'%avg_ids)
    elif DF.mvs_targets_df.loc[DF.mvs_targets_df.mvs_ids.isin(mvs_ids_list),['flag_%s'%(filter) for filter in DF.filters]].apply(lambda x: x.str.contains('good',case=False)).any(axis=1).any(axis=0):
        if DF.mvs_targets_df.loc[DF.mvs_targets_df.mvs_ids.isin(mvs_ids_list),['flag_%s'%(filter) for filter in DF.filters]].apply(lambda x: x.str.contains('good|rejected',case=False)).all(axis=1).all(axis=0):
            if DF.avg_targets_df.loc[DF.avg_targets_df.avg_ids==avg_ids].type.isin([0,1,2,3]).values[0]:
                DF.avg_targets_df.loc[DF.avg_targets_df.avg_ids==avg_ids,['type']]=1
        else:
            raise ValueError('All flags should be the good for a psf/target. Please check avg_ids %s'%avg_ids)

    else:
       raise ValueError('No type/flag match. Please check avg_ids %s'%avg_ids)
    return(DF)

def fk_writing(DF,filter,out,df_label,labels):
    '''
    Wrapper to help handle the writing of the fake dataframe

    Parameters
    ----------
    filter : str
        filter name.
    out : list
        list of index and values to save in dataframe.
    df_label : str
        name of dataframe to update.
    labels : list
        list of clumns in dataframe to update.

    Returns
    -------
    None.

    '''
    if not isinstance(out, np.ndarray): out=np.array(out)
    sel0=(getattr(DF,df_label).index.get_level_values('filter')==filter)
    sel1=(getattr(DF,df_label).index.get_level_values('magbin').isin(out[:,0]))
    sel2=(getattr(DF,df_label).index.get_level_values('dmag').isin(out[:,1]))
    sel3=(getattr(DF,df_label).index.get_level_values('sep').isin(out[:,2]))
    sel4=(getattr(DF,df_label).index.get_level_values('fk_ids').isin(out[:,3]))
    for elno in range(len(labels)):
        getattr(DF,df_label).loc[(sel0&sel1&sel2&sel3&sel4),[labels[elno]]]=out[:,elno+4]
    return(DF)



#########################################################
# Header and cross match ids dataframe related routines #
#########################################################

def mk_crossmatch_ids_df(DF,ids_table):
    '''
    generate a cross match dataframe of IDs

    Parameters
    ----------
    ids_list : list
        list of lists of ids to fill the columns.
    columns_list : list
        list of lists of columns name.

    Returns
    -------
    None.

    '''

    columns_list = ids_table.columns.values
    getLogger(__name__).info(f'Creating the cross match ids dataframe')
    df=create_empty_df(np.arange(len(ids_table[columns_list[0]])),columns_list)

    for column in columns_list:
        df[column] = ids_table[column].values
    DF.crossmatch_ids_df = df


def mk_avg_targets_df(DF, dataset):
    # average_visits_id_label='id',avg_coords_labels=['ra','dec'],average_type_label='type',showplot=False,minsep=0,max_separation=2):
    '''
    Create the standard template for the averaged targets dataframe (single avg_targets)


    Parameters
    ----------
    df : pandas dataframe
        input dataframe.
    average_visits_id_label: str, optional
        unique identifier for each star in the mean catalog. The default is 'id'
    avg_coords_labels: list, optional
        list of labels for sources coordiantes. The default is ['ra','dec']
    average_type_label: str, optional
        type identifier for each star in the mean catalog. The default is 'type'
    showplot : float, optional
        show distance cube plots.
    minsep : float, optional
        minimum separation in arcsec for Known doubles. The default is 2.
    max_separation : float, optional
        minimum separation in arcsec for Known doubles. The default is 2.

    Returns
    -------
    None.

    '''
    getLogger(__name__).info(f'Creating the multi-visit targets dataframe')
    df_new = create_empty_df([], [dataset.pipe_cfg.buildhdf['default_avg_table']['id']] + DF.radec + ['m_%s' % i for i in
                             DF.filters] + ['e_%s' % i for i in DF.filters] + ['type', 'FirstDist', 'SecondDist',
                             'ThirdDist', 'FirstID', 'SecondID','ThirdID'],
                             int_columns=['avg_ids', 'type', 'FirstID', 'SecondID','ThirdID'],
                             flt_columns=DF.radec + ['m_%s' % i for i in DF.filters] + ['e_%s' % i for i in DF.filters] +
                             ['FirstDist', 'SecondDist','ThirdDist'])

    df_new[dataset.pipe_cfg.buildhdf['default_avg_table']['id']] = dataset.avg_table[
        dataset.pipe_cfg.buildhdf['default_avg_table']['id']].values
    df_new[DF.radec] = dataset.avg_table[DF.radec].values
    df_new[dataset.pipe_cfg.buildhdf['default_avg_table']['type']] = dataset.avg_table[
        dataset.pipe_cfg.buildhdf['default_avg_table']['type']].values

    try:
        df_new[['x_%s' % i for i in DF.filters]] = dataset.avg_table[['x_%s' % i for i in DF.filters]]
        df_new[['y_%s' % i for i in DF.filters]] = dataset.avg_table[['y_%s' % i for i in DF.filters]]
    except:
        pass
    try:
        df_new[['m_%s_o' % i for i in DF.filters]] = dataset.avg_table[
            ['m_%s' % i for i in DF.filters]]
        df_new[['e_%s_o' % i for i in DF.filters]] = dataset.avg_table[
            ['e_%s' % i for i in DF.filters]]
    except:
        pass

    DF.avg_targets_df = df_new
    DF.avg_targets_df = distances_cube(df_new, coords_labels=DF.radec,
                                       showplot=dataset.pipe_cfg.buildhdf['distance_cube']['shwoplot'], pixelscale=DF.pixscale,
                                       min_separation=DF.minsep, max_separation=DF.maxsep)


def mk_mvs_targets_df(DF, dataset):
    '''
    Create the standard template for the multiple visitis targets dataframe


    Parameters
    ----------
    df : pandas dataframe
        input dataframe.
    unique_id_lable: str, optional
        unique identifier for each star in the multivisits catalog . The default is 'id'

    Returns
    -------
    None.

    '''
    getLogger(__name__).info(f'Creating the average targets dataframe')
    df_new = create_empty_df([], ['mvs_ids'] + ['x_%s' % i for i in DF.filters] + ['y_%s' % i for i in DF.filters]
                             + ['vis', 'ext'] + ['counts_%s' % i for i in DF.filters] + ['ecounts_%s' % i for i in DF.filters] +
                             ['m_%s' % i for i in DF.filters] + [
                                 'e_%s' % i for i in DF.filters] + ['spx_%s' % i for i in DF.filters] + [
                                 'bpx_%s' % i for i in DF.filters] + ['nap_%s' % i for i in DF.filters] + [
                                 'sky_%s' % i for i in DF.filters] + ['esky_%s' % i for i in DF.filters] + [
                                 'nsky_%s' % i for i in DF.filters] + ['chi_%s' % i for i in DF.filters] + [
                                 'grow_corr_%s' % i for i in DF.filters] + ['r_%s' % i for i in
                                                                                 DF.filters] + ['rsky1_%s' % i for
                                                                                                     i in
                                                                                                     DF.filters] + [
                                 'rsky2_%s' % i for i in DF.filters] + ['flag_%s' % i for i in DF.filters] + [
                                 'exptime_%s' % i for i in DF.filters] + ['cell_%s' % i for i in
                                                                               DF.filters] + ['rota_%s' % i for i
                                                                                                   in
                                                                                                   DF.filters] + [
                                 'pav3_%s' % i for i in DF.filters] + ['fits_%s' % i for i in DF.filters],
                             int_columns=['mvs_ids', 'ext'] + ['bpx_%s' % i for i in DF.filters]+ ['cell_%s' % i for i in
                                                                               DF.filters] ,
                             flt_columns=['x_%s' % i for i in DF.filters] + ['y_%s' % i for i in
                                                                                  DF.filters] + ['counts_%s' % i
                                                                    for i in DF.filters] + ['ecounts_%s' % i for i in
                                                                              DF.filters]  + ['m_%s' % i for i
                                                                                                      in
                                                                                                      DF.filters] + [
                                             'e_%s' % i for i in DF.filters] + ['r_%s' % i for i in
                                                                                     DF.filters] + ['rsky1_%s' % i
                                                                                                         for i in
                                                                                                         DF.filters] + [
                                             'rsky2_%s' % i for i in DF.filters] + ['spx_%s' % i for i in
                                                                                         DF.filters] + [
                                             'bpx_%s' % i for i in DF.filters] + ['nap_%s' % i for i in DF.filters] + ['exptime_%s' % i for i
                                                                    in DF.filters] + ['rota_%s' % i for i in
                                                                                           DF.filters] + [
                                             'pav3_%s' % i for i in DF.filters]+ [
                                 'sky_%s' % i for i in DF.filters] + ['esky_%s' % i for i in DF.filters] + [
                                 'nsky_%s' % i for i in DF.filters] + ['chi_%s' % i for i in DF.filters] + [
                                 'grow_corr_%s' % i for i in DF.filters] + ['r_%s' % i for i in
                                                                                 DF.filters] + ['rsky1_%s' % i for
                                                                                                     i in
                                                                                                     DF.filters] + [
                                 'rsky2_%s' % i for i in DF.filters],
                             str_columns=['vis'] + ['fits_%s' % i for i in DF.filters] + ['flag_%s' % i for i in
                                                                                               DF.filters])

    try:
        df_new['mvs_ids'] = dataset.mvs_table[dataset.pipe_cfg.buildhdf['default_mvs_table']['id']].values
    except:
        getLogger(__name__).critical('mvs_ids is required as input to build the dataframe')
        raise ValueError('mvs_ids is required as input to build the dataframe')
    try:
        df_new[ 'ext'] = dataset.mvs_table['ext'].values
    except:
        getLogger(__name__).critical('ext is required as input to build the dataframe')
        raise ValueError('ext is required as input to build the dataframe')
    try:
        df_new['vis'] = dataset.mvs_table['vis'].values
    except:
        getLogger(__name__).critical('vis is required as input to build the dataframe')
        raise ValueError('vis is required as input to build the dataframe')
    try:
        df_new[['x_%s' % i for i in DF.filters]] = dataset.mvs_table[['x_%s' % i for i in DF.filters]].values
    except:
        getLogger(__name__).critical('x is required as input to build the dataframe')
        raise ValueError('x is required as input to build the dataframe')
    try:
        df_new[['y_%s' % i for i in DF.filters]] = dataset.mvs_table[['y_%s' % i for i in DF.filters]].values
    except:
        getLogger(__name__).critical('y is required as input to build the dataframe')
        raise ValueError('y is required as input to build the dataframe')

    try:
        df_new[['rota_%s' % i for i in DF.filters]] = dataset.mvs_table[['rota_%s' % i for i in DF.filters]].values
    except:
        df_new[['rota_%s' % i for i in DF.filters]] = np.nan
    try:
        df_new[['pav3_%s' % i for i in DF.filters]] = dataset.mvs_table[['pav3_%s' % i for i in DF.filters]].values
    except:
        df_new[['pav3_%s' % i for i in DF.filters]] = np.nan
    try:
        df_new[['exptime_%s' % i for i in DF.filters]] = dataset.mvs_table[
            ['exptime_%s' % i for i in DF.filters]].values
    except:
        df_new[['exptime_%s' % i for i in DF.filters]] = np.nan
    try:
        df_new[['fits_%s' % i for i in DF.filters]] = dataset.mvs_table[
            ['fits_%s' % i for i in DF.filters]].values
    except:
        getLogger(__name__).critical('fitsroot name is required as input to build the dataframe')
        raise ValueError('fitsroot name is required as input to build the dataframe')

    try:
        df_new[['flag_%s' % i for i in DF.filters]] = dataset.mvs_table[
            ['flag_%s' % i for i in DF.filters]].values
    except:
        df_new[['flag_%s' % i for i in DF.filters]] = 'rejected'
        for filter in DF.filters:
            df_new.loc[~df_new['x_%s' % filter].isna(), ['flag_%s' % filter]] = 'good_target'
    df_new[['cell_%s' % i for i in DF.filters]] = np.nan
    DF.mvs_targets_df = df_new


########################################
# Candidate dataframe related routines #
########################################

def mk_avg_candidates_df(DF):
    '''
    Create the standard template for the multiple visitis candidate dataframe


    Parameters
    ----------
    avg_ids_list : list, optional
        list of ids from the average dataframe to test. The default is [].


    Returns
    -------
    None.

    '''

    df_new=create_empty_df([],['avg_ids','mass','emass','sep','mkmode']+['n_%s'%i for i in DF.filters]+['nsigma_%s'%i for i in DF.filters]+['m_%s'%i for i in DF.filters]+['e_%s'%i for i in DF.filters]+['th_%s'%i for i in DF.filters]+['magbin_%s'%i for i in DF.filters]+['tp_above_th_%s'%i for i in DF.filters]+['tp_above_nsigma_%s'%i for i in DF.filters]+['fp_above_th_%s'%i for i in DF.filters]+['fp_above_nsigma_%s'%i for i in DF.filters]+['auc_%s'%i for i in DF.filters],
                           int_columns=['avg_ids','mkmode']+['magbin_%s'%i for i in DF.filters],
                           flt_columns=['mass','emass','sep']+['n_%s'%i for i in DF.filters]+['nsigma_%s'%i for i in DF.filters]+['m_%s'%i for i in DF.filters]+['e_%s'%i for i in DF.filters]+['th_%s'%i for i in DF.filters]+['tp_above_th_%s'%i for i in DF.filters]+['tp_above_nsigma_%s'%i for i in DF.filters]+['fp_above_th_%s'%i for i in DF.filters]+['fp_above_nsigma_%s'%i for i in DF.filters]+['auc_%s'%i for i in DF.filters])
    df_new['avg_ids']=DF.avg_targets_df.avg_ids.unique()
    DF.avg_candidates_df=df_new

def mk_mvs_candidates_df(DF):
    '''
    Create the standard template for the average candidate dataframe


    Parameters
    ----------
    None

    Returns
    -------
    None.

    '''
    df_new=create_empty_df([],['mvs_ids']+['x_tile_%s'%i for i in DF.filters]+['y_tile_%s'%i for i in DF.filters]+['x_rot_%s'%i for i in DF.filters]+['y_rot_%s'%i for i in DF.filters]+['counts_%s'%i for i in DF.filters]+['ecounts_%s'%i for i in DF.filters]+['m_%s'%i for i in DF.filters]+['e_%s'%i for i in DF.filters]+['rota_%s'%i for i in DF.filters]+['pav3_%s'%i for i in DF.filters]+['flag_%s'%i for i in DF.filters]+['kmode_%s'%i for i in DF.filters]+['sep_%s'%i for i in DF.filters]+['std_%s'%i for i in DF.filters]+['nsigma_%s'%i for i in DF.filters]+['th_%s'%i for i in DF.filters]+['tp_above_th_%s'%i for i in DF.filters]+['tp_above_nsigma_%s'%i for i in DF.filters]+['fp_above_th_%s'%i for i in DF.filters]+['fp_above_nsigma_%s'%i for i in DF.filters]+['auc_%s'%i for i in DF.filters],
                           int_columns=['mvs_ids']+['kmode_%s'%i for i in DF.filters]+['x_tile_%s'%i for i in DF.filters]+['y_tile_%s'%i for i in DF.filters],
                           flt_columns=['x_rot_%s'%i for i in DF.filters]+['y_rot_%s'%i for i in DF.filters]+['counts_%s'%i for i in DF.filters]+['ecounts_%s'%i for i in DF.filters]+['m_%s'%i for i in DF.filters]+['e_%s'%i for i in DF.filters]+['rota_%s'%i for i in DF.filters]+['pav3_%s'%i for i in DF.filters]+['flag_%s'%i for i in DF.filters]+['sep_%s'%i for i in DF.filters]+['std_%s'%i for i in DF.filters]+['nsigma_%s'%i for i in DF.filters]+['th_%s'%i for i in DF.filters]+['tp_above_th_%s'%i for i in DF.filters]+['tp_above_nsigma_%s'%i for i in DF.filters]+['fp_above_th_%s'%i for i in DF.filters]+['fp_above_nsigma_%s'%i for i in DF.filters]+['auc_%s'%i for i in DF.filters])
    df_new['mvs_ids']=DF.mvs_targets_df.mvs_ids.unique()
    df_new[['flag_%s'%i for i in DF.filters]]='rejected'
    DF.mvs_candidates_df=df_new



def mk_fk_references_df(DF,Nstar):
    '''
    Create the empty tile data frame for average visits targets

    Parameters
    ----------

    Returns
    -------
    None.

    '''
    df_new=create_empty_df([],['fk_ids']+['x_%s'%i for i in DF.filters]+['y_%s'%i for i in DF.filters]+['counts%s'%i for i in DF.filters]+['m%s'%i for i in DF.filters],int_columns=['fk_ids'],flt_columns=['x_%s'%i for i in DF.filters]+['y_%s'%i for i in DF.filters]+['counts%s'%i for i in DF.filters]+['m%s'%i for i in DF.filters])
    df_new['fk_ids']=[i for i in range(Nstar)]
    DF.fk_references_df=df_new

def mk_fakes_df(DF,MagBin_list,Dmag_list,Sep_range,Nstar,filters=None,skip_filters=[]):
    '''
    Create the empty tile data frame for average visits targets

    Parameters
    ----------

    Returns
    -------
    None.

    '''
    counts_KLIP_list=[]
    noise_KLIP_list=[]
    Nsigma_KLIP_list=[]
    mag_KLIP_list=[]
    elno=0
    df_targets_list=[]
    df_candidates_list=[]
    if isinstance(Sep_range, str):
        seps_levels = np.arange(int(Sep_range.split('-')[0]),int(Sep_range.split('-')[1])+1,1)
    elif isinstance(Sep_range, (list, np.ndarray)):
        seps_levels = Sep_range
    else:
        seps_levels=[i for i in range(Sep_range[0],Sep_range[1]+1)]

    nstar_levels=[i for i in range(Nstar)]
    for Kmode in [f'_kmode{i}' for i in DF.kmodes]:
        counts_KLIP_list.extend(['counts%s'%(Kmode)])
        noise_KLIP_list.extend(['noise%s'%Kmode])
        Nsigma_KLIP_list.extend(['nsigma%s'%Kmode])
        mag_KLIP_list.extend(['m%s'%(Kmode)])

    # if filters==None:filters=DF.filters

    if len(MagBin_list)==1 and len(filters)>1:
        MagBin_list*=len(filters)
    if len(Dmag_list)==1 and len(filters)>1:
        Dmag_list*=len(filters)

    for filter in filters:
        if filter not in skip_filters:
            if len(MagBin_list[elno])==1: magbins_levels=[i for i in range(MagBin_list[elno][0],MagBin_list[elno][0]+1)]
            else: magbins_levels=[i for i in range(MagBin_list[elno][0],MagBin_list[elno][1]+1)]
            if len(Dmag_list[elno])==1: dmags_levels=[i for i in range(Dmag_list[elno][0],Dmag_list[elno][0]+1)]
            else: dmags_levels=[i for i in range(Dmag_list[elno][0],Dmag_list[elno][1]+1)]
            elno+=1
            df_targets_new=create_empty_df(['filter','magbin','dmag','sep','fk_ids'],[],multy_index=True,levels=[[filter],magbins_levels,dmags_levels,seps_levels,nstar_levels])
            df_candidates_new=create_empty_df(['filter','magbin','dmag','sep','fk_ids'],[],multy_index=True,levels=[[filter],magbins_levels,dmags_levels,seps_levels,nstar_levels])
            df_targets_list.append(df_targets_new)
            df_candidates_list.append(df_candidates_new)
    DF.fk_targets_df=pd.concat(df_targets_list)    
    DF.fk_candidates_df=pd.concat(df_candidates_list) 

def mk_fk_tiles_df(DF):
    '''
    Create the empty tile data frame for average visits targets

    Parameters
    ----------

    Returns
    -------
    None.

    '''
    MagBin_list=DF.fk_targets_df.index.get_level_values('magbin').unique()
    Dmag_list=DF.fk_targets_df.index.get_level_values('dmag').unique()
    Sep_list=DF.fk_targets_df.index.get_level_values('sep').unique()
    Nstar_list=DF.fk_targets_df.index.get_level_values('fk_ids').unique()
    counts_KLIP_list=[]
    for filter in DF.filters:
        for Kmode in DF.kmodes:
            counts_KLIP_list.append(f'kmode{Kmode}_{filter}')
            counts_KLIP_list.append(f'Kmode{Kmode}_no_injection_{filter}')


    columns=['%s_data'%i for i in DF.filters]+[f'data_no_injection_{i}' for i in DF.filters]+counts_KLIP_list
    df_new=create_empty_df(['magbin','dmag','sep','fk_ids'],[columns],multy_index=True,levels=[MagBin_list,Dmag_list,Sep_list,Nstar_list])
    DF.fk_tiles_df=df_new

def mk_fk_references_tiles_df(DF):
    '''
    Create the empty tile data frame for average visits targets

    Parameters
    ----------

    Returns
    -------
    None.

    '''
    df_new=create_empty_df([],['fk_ids']+['%s_data'%i for i in DF.filters],int_columns=['fk_ids'])
    df_new['fk_ids']=DF.fk_references_df.fk_ids.unique()
    DF.fk_references_tiles_df=df_new


def mk_fk_completeness_df(DF,nvisit_list,skip_filters=[]):
    '''
    Create the standard template for the fake completeness dataframe


    Parameters
    ----------
    df : pandas dataframe
        input dataframe.
    unique_id_lable: unique identifier for each star in the multivisits catalog . The default is 'id'

    Returns
    -------
    None.

    '''
    df_list=[]
    for filter in DF.filters:
        if filter not in skip_filters:
            if isinstance(nvisit_list, int):
                nvisit_list=[nvisit_list]
            if not isinstance(nvisit_list, (list,np.ndarray)):
                nvisit_list=np.sort(DF.avg_candidates_df['N%s'%filter].loc[~DF.avg_candidates_df['N%s'%filter].isna()].unique())

            magbin_list=DF.fk_candidates_df.loc[filter].index.get_level_values('magbin').unique()
            dmag_list=DF.fk_candidates_df.loc[filter].index.get_level_values('dmag').unique()
            sep_list=DF.fk_candidates_df.loc[filter].index.get_level_values('sep').unique()
            df=create_empty_df(['filter','nvisit','magbin','dmag','sep'],[],multy_index=True,levels=[[filter],nvisit_list,magbin_list,dmag_list,sep_list])
            df_list.append(df)

    DF.fk_completeness_df=pd.concat(df_list)
    return(DF)