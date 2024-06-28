"""
utilities functions for the complteness analysis
"""

import sys
sys.path.append('/')
from ancillary import round2closerint,parallelization_package
from utils_dataframe import create_empty_df

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from itertools import repeat
from concurrent.futures import ProcessPoolExecutor
from IPython.display import display
from dynesty import NestedSampler,plotting,utils
from scipy.interpolate import interp2d
from scipy.integrate import quad
from stralog import getLogger


def P_s(s):
    if s<=1: x=1.3*s
    elif s>1 and s<=1.8: x=-35./32.*(s-9./5.)
    else: x=0
    return(x)

def integrand(s,P_a,D,y):
    a=D/s
    return(P_a(a,y)*P_s(s)/s)

def smaorsep_interpolations_task(y,f,xnew,amin,amax):
    DP_D=[]
    for D in xnew:
        s1=np.min([2,D/amax])
        s2=np.min([2,D/amin])
        P_D=quad(integrand, s1, s2,args=(f,D,y))[0]
        DP_D.append(D*P_D)
    return(DP_D)

def mk_smaorsep_interpolations(X,Y,Z,ystep,sep_step=0.01,log=False,sma_interp=False,amin=0.01,amax=2,workers=None,parallel_runs=True,num_of_chunks=None,chunksize = None):
    sep_min,sep_max=[np.nanmin(X),np.nanmax(X)]
    ymin,ymax=[np.nanmin(Y),np.nanmax(Y)]

    ############# Iterpolation function of the completeness over the grid od sep and mass ################           
    f=interp2d(X, Y, Z)

    ############# Build a finer grid in Mass and Sep ######################   
          
    xnew=np.arange(sep_min,sep_max+sep_step,sep_step)
    if log==True:  ynew=np.arange(np.log10(ymin),np.log10(ymax+ystep),ystep)            
    else: ynew=np.arange(ymin,ymax+ystep,ystep)            

    # ############# New completeness obtained from the interpolation over the finer grid ################           
   
    if sma_interp:
        DP_D_list=[]
        if parallel_runs:
            ######## Split the workload over different CPUs ##########
            workers,chunksize,ntarget=parallelization_package(workers,len(ynew),verbose=False)
            with ProcessPoolExecutor(max_workers=workers) as executor:
                for DP_D in executor.map(smaorsep_interpolations_task,ynew,repeat(f),repeat(xnew),repeat(amin),repeat(amax),chunksize=chunksize): #the task routine is where everithing is done!
                    DP_D_list.append(DP_D)
            DP_D_list=np.array(DP_D_list)

        else:
            for y in ynew:
                DP_D=smaorsep_interpolations_task(y,f,xnew,amin,amax)
                DP_D_list.append(DP_D)
            DP_D_list=np.array(DP_D_list)

        return(xnew,ynew,DP_D_list)
    
    else:     

        return(xnew,ynew,f(xnew, ynew))
    
def mk_pivot_df(hdf,hdfpivot_columns=[],pindex='',pcolumn='',level=1):
    hdfreset = hdf.reset_index()
    hdfpivot=hdfreset[hdfpivot_columns].pivot(index=pindex,columns=pcolumn)
    X=hdfpivot.columns.levels[level].values.astype(float)
    Y=hdfpivot.index.values.astype(float)
    Z=hdfpivot['Ratio'].values.astype(float)
    Xi,Yi = np.meshgrid(X, Y)
    return(X,Y,Z,Xi,Yi)

def mk_completeness_curves_task(DF,Nvisit,filter,Kmode):#,pmass_range_list,q_range_list,splines):
    # DM=5*np.log10(DF.dist/10)
    # mag_label1='m%s'%filter[1:4]
    # str_s=splines[0]
    # inv_s=splines[1]
    # pmass_d=pmass_range_list[0]
    # pmass_u=pmass_range_list[1]
    # magp_u=round2closerint([inv_s[mag_label1](np.log10(pmass_u))],base=1,integer=True)[0]+DM
    # magp_d=round2closerint([inv_s[mag_label1](np.log10(pmass_d))],base=1,integer=True)[0]+DM
    # magp_u=DF.fk_completeness_df.loc[filter].index.get_level_values('magbin').unique()[0]
    # magp_d=DF.fk_completeness_df.loc[filter].index.get_level_values('magbin').unique()[-1]
    list_of_dmag_ordered_df=[]
    # list_of_q_ordered_df=[]
    
    # for magp_bin in np.arange(int(round(magp_u)),int(round(magp_d))+1,1):
    for magp_bin in DF.fk_completeness_df.loc[filter].index.get_level_values('magbin').unique():
        list_of_mass_ref_df=[]
        ref_dmag_column_list=[]
        
        # list_of_q_ref_df=[]
        # ref_q_column_list=[]

        # if magp_bin in DF.fk_completeness_df.loc[filter].index.get_level_values('magbin').unique() and magp_bin >= magp_u:
        for dmag in DF.fk_completeness_df.loc[filter].index.get_level_values('dmag').unique():
            df=DF.fk_completeness_df.loc[(filter,Nvisit,magp_bin,dmag),'ratio_kmode%s'%Kmode]
            df.rename('Ratio%i'%magp_bin,inplace=True)
            list_of_mass_ref_df.append(pd.concat([df],axis=1))
            ref_dmag_column_list.append(dmag)

            # pmass=10**str_s['mass%s'%filter[1:4]](magp_bin-DM)
            # if pmass <= pmass_u:
                # for q in q_range_list:
                #     massc_conv=pmass*q
                #     magc_bin=round2closerint([inv_s[mag_label1](np.log10(massc_conv))]+DM,base=1,integer=True)[0]
                #     dmag=magc_bin-magp_bin
                #     sep_list=DF.fk_completeness_df.loc[filter].index.get_level_values('sep').unique()
                #     if dmag in DF.fk_completeness_df.loc[filter].index.get_level_values('dmag').unique():
                #         df=DF.fk_completeness_df.loc[(filter,Nvisit,magp_bin,dmag),'ratio_Kmode%s'%Kmode]
                #         df.rename('Ratio%i'%magp_bin,inplace=True)
                #         list_of_q_ref_df.append(pd.concat([df],axis=1))
                #         ref_q_column_list.append(q)
                #     else:
                #         df=create_empty_df(sep_list,['Ratio%i'%magp_bin])
                #         df['Ratio%i'%magp_bin].fillna(value=0, inplace=True)
                #         list_of_q_ref_df.append(pd.concat([df],axis=1))
                #         ref_q_column_list.append(q)
                        

        if len(list_of_mass_ref_df)>0:
            df=pd.concat(list_of_mass_ref_df,keys=ref_dmag_column_list)
            df.index.names=['Dmag','Sep']
            list_of_dmag_ordered_df.append(df)

        # if len(list_of_q_ref_df)>0:
        #     df_q=pd.concat(list_of_q_ref_df,keys=ref_q_column_list)
        #     df_q.index.names=['qBin','Sep']
        #     list_of_q_ordered_df.append(df_q)

    ###################################################################################
    df=pd.concat(list_of_dmag_ordered_df,axis=1)
    df['wRatio']=np.nan
    for index in list(df.index.get_level_values('Dmag').unique()):
        indices = np.where(~np.isnan(df.loc[index].values.astype('float64')))
        if len(list(set(indices[1])))>1:
            colnames=df.columns[list(set(indices[1]))]
            median=np.median(df.loc[index,colnames].values.astype('float64'),axis=1)
            median[np.isnan(median)]=0
            median[median>1]=1
            df.loc[index,'wRatio']=median
        else:
            colname=df.columns[list(set(indices[1]))][0]
            df.loc[index,['wRatio']]=df.loc[index,[colname]].values.astype('float64').ravel()

    df=df[['wRatio']].round(3)
    df=df.reset_index()
    df['Sep']=df['Sep']*DF.pixscale
    df=df.set_index(['Dmag','Sep'])
    return(df)
    ###################################################################################

    # df_q=pd.concat(list_of_q_ordered_df,axis=1)
    # df_q['wRatio']=np.nan
    # for index in list(df_q.index.get_level_values('qBin').unique()):
    #     indices = np.where(~np.isnan(df_q.loc[index].values.astype('float64')))
    #     if len(list(set(indices[1])))>1:
    #         colnames=df_q.columns[list(set(indices[1]))]
    #         median=np.median(df_q.loc[index,colnames].values.astype('float64'),axis=1)
    #         median[np.isnan(median)]=0
    #         median[median>1]=1
    #         df_q.loc[index,'wRatio']=median
    #     else:
    #         colname=df_q.columns[list(set(indices[1]))][0]
    #         df_q.loc[index,['wRatio']]=df_q.loc[index,[colname]].values.astype('float64').ravel()
    #
    # df_q=df_q[['wRatio']].round(3)
    #
    # df_q=df_q.reset_index()
    # df_q['Sep']=df_q['Sep']*DF.pixscale
    # df_q=df_q.set_index(['qBin','Sep'])

    # return(df,df_q)

# def mk_completeness_curves_df(self,MagBin_label,filter,splines,pmass_range_list=[0.01,2],save=True,workers=None):
def mk_completeness_curves_df(DF, filter):
    """
    For each visit, Kmode, mass converted photometry of primary and companion pull a contrast curve and evaluate the completeness as a function of separation for this specific configuration. 
    """
    # pmass_range_list=np.array(pmass_range_list)
    # q_range_list=np.arange(0.010,1+0.005,0.005).round(3)
    # df=[]
    getLogger(__name__).info(f'Generating average completeness dataframes for filter {filter}')
    Nvisit_list=np.sort(DF.fk_completeness_df.index.get_level_values('nvisit').unique())
    Kmodes_list=DF.kmodes
    Nvisit_mass_df_list=[]
    # Nvisit_q_df_list=[]
    for Nvisit in Nvisit_list:
        Kmode_mass_df_list=[]
        # Kmode_q_df_list=[]

        for Kmode in Kmodes_list:
            # df,df_q=mk_completeness_curves_task(DF,Nvisit,pmass_range_list,q_range_list,splines,MagBin_label,filter,Kmode) #the task routine is where everithing is done!
            # Kmode_mass_df_list.append(df)
            # Kmode_q_df_list.append(df_q)
            df = mk_completeness_curves_task(DF, Nvisit, filter, Kmode)  # the task routine is where everithing is done!
            Kmode_mass_df_list.append(df)

        if len(Kmode_mass_df_list)>0:
             df=pd.concat(Kmode_mass_df_list,keys=DF.kmodes)
             # df.index.names=['Kmode','MprimBin','Dmag','Sep']
             df.index.names=['Kmode','Dmag','Sep']
             Nvisit_mass_df_list.append(df)
        # if len(Kmode_q_df_list)>0:
        #      df=pd.concat(Kmode_q_df_list,keys=DF.Kmodes_list)
        #      # df.index.names=['Kmode','MprimBin','qBin','Sep']
        #      df.index.names=['Kmode','q','Sep']
        #      Nvisit_q_df_list.append(df)
    if len(Nvisit_mass_df_list)>0:
         DC_df=pd.concat(Nvisit_mass_df_list,keys=Nvisit_list)
         # DC_df.index.names=['Nvisit','Kmode','MprimBin','Dmag','Sep']
         DC_df.index.names=['Nvisit','Kmode','Dmag','Sep']
    # if len(Nvisit_q_df_list)>0:
    #      QC_df=pd.concat(Nvisit_q_df_list,keys=Nvisit_list)
    #      # QC_df.index.names=['Nvisit','Kmode','MprimBin','qBin','Sep']
    #      QC_df.index.names=['Nvisit','Kmode','q','Sep']
    DC_df.columns=['Ratio']
    # QC_df.columns=['Ratio']
    # return(DC_df,QC_df)
    return(DC_df)

# def mk_matrix_completeness_curves_task(DCC_sel_df,QCC_sel_df,sep_step,KLIPmode,sma_interp,parallel_runs,workers):
def mk_matrix_completeness_curves_task(DCC_sel_df, sep_step, KLIPmode, sma_interp, parallel_runs):

    X1,Y1,Z1,Xi1,Yi1=mk_pivot_df(DCC_sel_df.loc[KLIPmode,'Ratio'],hdfpivot_columns=['Sep','Dmag','Ratio'],pindex='Dmag',pcolumn='Sep')
    xnew,ynew,znew=mk_smaorsep_interpolations(X1,Y1,Z1,0.05,sep_step=sep_step,sma_interp=sma_interp,parallel_runs=parallel_runs)
    try:
        if len(znew)>1: znew[znew>1]=1
    except: 
        display(DCC_sel_df.loc[KLIPmode,'Ratio'])
        print(znew)
        sys.exit()
    df=pd.DataFrame(znew,columns=np.round(xnew,3),index=np.round(ynew,3))

    # X1,Y1,Z1,Xi1,Yi1=mk_pivot_df(QCC_sel_df.loc[KLIPmode,'Ratio'],hdfpivot_columns=['Sep','q','Ratio'],pindex='q',pcolumn='Sep')
    # xnew,ynew,znew=mk_smaorsep_interpolations(X1,Y1,Z1,0.1,sep_step=sep_step,sma_interp=sma_interp,parallel_runs=parallel_runs)
    # try:
    #     if len(znew)>1: znew[znew>1]=1
    # except:
    #     display(QCC_sel_df.loc[KLIPmode,'Ratio'])
    #     print(znew)
    #     sys.exit()
    # df_q=pd.DataFrame(znew,columns=np.round(xnew,3),index=np.round(ynew,3))
    # return(df,df_q)
    return(df)

 
# def make_matrix_completeness_curves_df(DCC_df,QCC_df,Nvisit_list=[],KLIPmodes_list=[],sep_step=0.01,cmap='Oranges_r',workers=None,save=True,sma_interp=True,parallel_runs=True):
def make_matrix_completeness_curves_df(DCC_df, Nvisit_list=[], KLIPmodes_list=[], sep_step=0.01,
                                       workers=None, sma_interp=True, parallel_runs=True):

    """
    For each bin of mass of the primary and visits, convert the mass completeness curves in matrix MassCompanionXSep and then convert the separation in projected Semi-major axis.
    """
    if len(Nvisit_list)==0:Nvisit_list=DCC_df.index.get_level_values('Nvisit').unique().values
    if len(KLIPmodes_list)==0:KLIPmodes_list=DCC_df.index.get_level_values('Kmode').unique().values
    Nvisit_mass_df_list=[]
    # Nvisit_q_df_list=[]
    for Nvisit in Nvisit_list:
        getLogger(__name__).info(f'Making matrix completeness curves for Nvisits:  {Nvisit}')
        klipmode_dmag_df_list=[]
        # klipmode_q_df_list=[]

        for KLIPmode in KLIPmodes_list: #the task routine is where everithing is done!
            # df,df_q=mk_matrix_completeness_curves_task(DCC_df.loc[(Nvisit)],QCC_df.loc[(Nvisit)],sep_step,KLIPmode,sma_interp,parallel_runs,workers)
            df=mk_matrix_completeness_curves_task(DCC_df.loc[int(Nvisit)],sep_step,KLIPmode,sma_interp,parallel_runs)
            klipmode_dmag_df_list.append(df)
            # klipmode_q_df_list.append(df_q)

        df=pd.concat(klipmode_dmag_df_list,keys=KLIPmodes_list)#.sort_index()
        Nvisit_mass_df_list.append(df)
        # dfq=pd.concat(klipmode_q_df_list,keys=KLIPmodes_list)#.sort_index()
        # Nvisit_q_df_list.append(dfq)
    MDCC_df=pd.concat(Nvisit_mass_df_list,keys=Nvisit_list)#.sort_index()
    # MQCC_df=pd.concat(Nvisit_q_df_list,keys=Nvisit_list)#.sort_index()
    if sma_interp: xlabel='SMA'
    else:  xlabel='Sep'
    
    MDCC_df.index.set_names(['Nvisit','KLIPmode','Dmag'],inplace=True)  
    MDCC_df.columns.set_names([xlabel],inplace=True)  
    # MQCC_df.index.set_names(['Nvisit','KLIPmode','q'],inplace=True)
    # MQCC_df.columns.set_names([xlabel],inplace=True)
       
    # return(MDCC_df,MQCC_df)
    return(MDCC_df)

def flatten_matrix_completeness_curves_df(DF,filter,sma_interp=True,Nvisit_list=None,KLIPmodes_list=None):
    MDCC_df=DF.matrix_dmag_completeness_curves_df.loc[filter]
    # MQCC_df=DF.matrix_q_completeness_curves_df.loc[filter]
    
    if sma_interp: xlabel='SMA'
    else:  xlabel='Sep'
    if Nvisit_list is None or len(Nvisit_list)==0: Nvisit_list=MDCC_df.index.get_level_values('Nvisit').unique()
    if KLIPmodes_list is None or len(KLIPmodes_list)==0: KLIPmodes_list=MDCC_df.index.get_level_values('KLIPmode').unique()
    
    nvisit_mass_list_df=[]
    nvisit_q_list_df=[]
    for nvisit in Nvisit_list:
        getLogger(__name__).info(f'Flattening the matrix for completeness curves for Nvisits:  {nvisit}')
        KLIPmode_dmag_list_df=[]
        # KLIPmode_q_list_df=[]
        for idx,row in MDCC_df.loc[nvisit,slice(None)].groupby(['Dmag']):
            if isinstance(idx,tuple):
                idx=int(idx[0])
            else:
                idx=int(idx)

            kmode_idx=MDCC_df.index.isin(KLIPmodes_list,level='KLIPmode')
            median=np.median(MDCC_df.loc[nvisit,kmode_idx,idx].values.astype('float64'),axis=0)
            median[np.isnan(median)]=0
            median[median>1]=1
            df=create_empty_df([MDCC_df.columns.get_level_values(xlabel).unique()],[idx])
            df[idx]=median
            df=df.T
            KLIPmode_dmag_list_df.append(df)
        
        # for idx,row in MQCC_df.loc[nvisit,slice(None)].groupby(['q']):
        #     kmode_idx=MQCC_df.index.isin(KLIPmodes_list,level='KLIPmode')
        #     median=np.median(MQCC_df.loc[nvisit,kmode_idx,idx].values.astype('float64'),axis=0)
        #     median[np.isnan(median)]=0
        #     median[median>1]=1
        #     df=create_empty_df([MQCC_df.columns.get_level_values(xlabel).unique()],[idx])
        #     df[idx]=median
        #     df=df.T
        #     KLIPmode_q_list_df.append(df)
        
        nvisit_mass_list_df.append(pd.concat(KLIPmode_dmag_list_df))
        # nvisit_q_list_df.append(pd.concat(KLIPmode_q_list_df))
    
    # df_mass=pd.concat(nvisit_mass_list_df,keys=Nvisit_list)
    # df_mass.index.set_names(['Nvisit','Dmag'],inplace=True)  

    # df_q=pd.concat(nvisit_q_list_df,keys=Nvisit_list)
    # df_q.index.set_names(['Nvisit','q'],inplace=True)
    # mean_dmag_list_df=[]
    # display(df_q)
    # for idx,row in df_mass.groupby(['Dmag']):
    #     median=np.median(row.values,axis=0)
    #     df_new=utils_dataframe.create_empty_df([row.columns.get_level_values(xlabel).unique()],[idx])
    #     median[np.isnan(median)]=0
    #     median[median>1]=1
    #     df_new[idx]=median
    #     df_new=df_new.T
    #     mean_dmag_list_df.append(df_new)

    # mean_q_list_df=[]
    # for idx,row in df_q.groupby(['q']):
    #     median=np.median(row.values,axis=0)#,weights=wNvisits)
    #     df_new=utils_dataframe.create_empty_df([row.columns.get_level_values(xlabel).unique()],[idx])
    #     median[np.isnan(median)]=0
    #     median[median>1]=1
    #     df_new[idx]=median
    #     df_new=df_new.T
    #     mean_q_list_df.append(df_new)



    # FMDCC_df=pd.concat(mean_dmag_list_df)
    # FMDCC_df.index.set_names(['Dmag'],inplace=True) 
    # # mean_dmag_def_list.append(df)

    # FMQCC_df=pd.concat(mean_q_list_df)
    # FMQCC_df.index.set_names(['q'],inplace=True) 
    # # mean_q_def_list.append(df_q)

    FMDCC_df=pd.concat(nvisit_mass_list_df,keys=Nvisit_list)
    FMDCC_df.index.set_names(['Nvisit','Dmag'],inplace=True)  
    
    # FMQCC_df=pd.concat(nvisit_q_list_df,keys=Nvisit_list)
    # FMQCC_df.index.set_names(['Nvisit','q'],inplace=True)
    
    # FMDCC_df=pd.concat(mean_mass_def_list,keys=pmass_range_list)
    # FMDCC_df.columns=list(FMDCC_df.columns.get_level_values(xlabel).unique())
    # FMDCC_df.rename_axis(xlabel,axis=1,inplace=True)

    # FMQCC_df=pd.concat(mean_q_def_list,keys=pmass_range_list)
    # FMQCC_df.columns=list(FMQCC_df.columns.get_level_values(xlabel).unique())
    # FMQCC_df.rename_axis(xlabel,axis=1,inplace=True)
    # return(FMDCC_df,FMQCC_df)
    return(FMDCC_df)

# def filter_median_flatten_matrix_completeness_curves_df(FMDCC_df,FMQCC_df,mass_list,sma_interp=True):
#     if sma_interp: xlabel='SMA'
#     else:  xlabel='Sep'

#     np.nanmedian(FMDCC_df)

#     MDCC_df=pd.concat(mean_mass_def_list,keys=MPrim_list)
#     MDCC_df.columns=list(MDCC_df.columns.get_level_values(xlabel).unique())
#     MDCC_df.rename_axis(xlabel,axis=1,inplace=True)

#     MQCC_df=pd.concat(mean_q_def_list,keys=MPrim_list)
#     MQCC_df.columns=list(MQCC_df.columns.get_level_values(xlabel).unique())
#     MQCC_df.rename_axis(xlabel,axis=1,inplace=True)
#     return(MDCC_df,MQCC_df)
    
def ptform(u):
    return 6. * u - 3

def loglpoissontoy(theta):
    rate_fit = np.power(10,theta[0])        
    lpx = - mean_comp*Ntest_global*rate_fit + Ndetect*np.log(mean_comp*rate_fit)- (-(Ndetect+1)*np.log(Ntest_global) + 1/2*np.log(2*np.pi*Ndetect) + Ndetect*np.log(Ndetect) - Ndetect )
    return lpx

# def mk_histogram_from_completeness(binaries_df,CFQMCC_df,sep_bins,q_bins,Ntest=0,x_list=[],y_list=[],eyd_list=[],eyu_list=[],data=[],N=0,x=0,showplot=False,nlive=500,verbose=True):
def mk_histogram_from_completeness(binaries_df,sep_bins,q_bins,apply_completeness_correction=False,Ntest=0,x_list=[],y_list=[],eyd_list=[],eyu_list=[],data=[],N=0,x=0,showplot=False,nlive=500,verbose=True):
    # if isinstance(CFQMCC_df,pd.DataFrame): 
    if apply_completeness_correction: 
        Ncomp,VNcomp_d,VNcomp_u=completeness_correction(binaries_df,Ntest,sep_bins,q_bins,showplot=showplot,nlive=nlive,verbose=verbose)
        return(Ncomp,VNcomp_d,VNcomp_u)
    else:return(0,0,0)

def completeness_correction(binaries_df,Ntest,sep_bins,q_bins,showplot=False,nlive=500,step=5,verbose=True):
    Ncomp=0
    VNcomp_u=0
    VNcomp_d=0
    # if showplot==True:print(binaries_df[['avg_ids_p','mass_p','mass_c','q','sep']+binaries_df.columns[binaries_df.columns.str.contains('Completeness')].tolist()])

    for elq in range(len(q_bins[:-1])):
        global Ntest_global
        Ntest_global=Ntest
        for elsep in range(len(sep_bins[:-1])):
            global mean_comp,Ndetect
            # selected_columns=CFQMCC_df.columns[(np.array([i[0] for i in CFQMCC_df.columns.values])>=sep_bins[elsep])&(np.array([i[0] for i in CFQMCC_df.columns.values])<=sep_bins[elsep+1])]
            # selected_index=CFQMCC_df.index[(CFQMCC_df.index>=q_bins[elq])&(CFQMCC_df.index<=q_bins[elq+1])]                
            # q_completeness_curves_sel_df=CFQMCC_df.loc[(selected_index),selected_columns]
            sel=(binaries_df.sep>=sep_bins[elsep])&(binaries_df.sep<sep_bins[elsep+1])&(binaries_df.q>=q_bins[elq])&(binaries_df.q<q_bins[elq+1])
            N=binaries_df.loc[(sel)].avg_ids_p.nunique()
            if N>0:
                # mean_comp=q_completeness_curves_sel_df.mean(axis=1).mean()
                mean_comp=np.nanmean(binaries_df.loc[sel,binaries_df.columns[binaries_df.columns.str.contains('Completeness')].tolist()])
                ndim=1
                Ndetect = N
    
                sampler = NestedSampler(loglpoissontoy, ptform, ndim,nlive=nlive)
    
                sampler.run_nested(print_progress=verbose)
    
                res1 = sampler.results
    
                samples = res1.samples # samples
                weights = np.exp(res1.logwt - res1.logz[-1])  # normalized weights
                # Compute quantiles.
                quantiles = utils.quantile(samples[:, 0] , [0.025,0.5, 0.975], weights=weights)
                x=quantiles[1]
                ex_u=quantiles[2]-quantiles[1]
                ex_d=quantiles[0]-quantiles[1]
                Ncomp+=Ntest*10**(x)
                VNcomp_u+=(Ntest*10**(x+ex_u)-Ntest*10**(x))**2
                VNcomp_d+=(Ntest*10**(x)-Ntest*10**(x+ex_d))**2
    
                if showplot == True:
                    print(r'q bin [m$_c$/m$_p$]= %.2f-%.2f'%(q_bins[elq],q_bins[elq+1]))
                    print('Separation bin [\'\']= %.2f-%.2f'%(sep_bins[elsep],sep_bins[elsep+1]))
                    display(binaries_df.loc[sel].reset_index(drop=True))
                    res1.summary()
                    print('x=%.2f ex=(%.2f, %.2f)'%(x,ex_u,ex_d))
                    print('Ntarget = %i'%Ntest)
                    print('Det = %i'%N)
                    print('Ratio = %.3f'%float(N/Ntest))
                    print('MCompleteness = %.3f'%mean_comp)
                    print('Ncomp = %.2f ENcomp=(-%.2f,+%.2f)'%(Ntest*10**(x),Ntest*10**(x)-Ntest*10**(x+ex_d),Ntest*10**(x+ex_u)-Ntest*10**(x)))
                    # print('Ncomp = %.2f ENcomp=(-%.2f,+%.2f)'%(Ncomp,np.sqrt(VNcomp_d),np.sqrt(VNcomp_u)))
                    plotting.runplot(res1) 
                    fig, axes = plotting.traceplot(res1, truths=np.zeros(ndim),truth_color='black', show_titles=True,trace_cmap='viridis', connect=True,connect_highlight=range(5))
                    plt.show()
    
                del mean_comp,Ndetect
        # del Ntest
    return(Ncomp,VNcomp_d,VNcomp_u)

