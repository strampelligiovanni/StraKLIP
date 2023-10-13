"""
utilities functions for the false positives analysis
"""

import sys
sys.path.append('/')
from ancillary import frac_above_thresh,find_closer

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
from sklearn import metrics

def FP_analysis(self,avg_ids,filter,AUC_lim,FP_lim=0.001,step=10,nbins=10,showplot=False,DF_fk=None,fig=None,ax=None,suffix=''):
    '''
    Read the True Positive (signal where we inject the companion) and 
    False Positive (signla where we do not inject the companion) from the
    the fake injection dataframe to perform the False Positive Analysis on 
    one single target through the construction of ROC curves.

    Parameters
    ----------
    avg_ids : int
        average id for the single target.
    filter : str
        DESCRIPTION.name of the filter
    FP_lim : float, optional
        minimum false posive % accepted for detection. The default is 0.001.
    step : float, optional
        binsize for the finer interpolation of the dataframe. The default is 0.5.
    showplot : bool, optional
        choose to show plots. The default is False.
    DF_fk : pandas DataFrame, optional
        fake injection dataframe. If None, look in self. The default is None.

    Returns
    -------
    None.

    '''
    if not isinstance(DF_fk,pd.DataFrame):DF_fk=self
    ncols=self.crossmatch_ids_df.loc[self.crossmatch_ids_df.avg_ids==avg_ids].mvs_ids.count()
    FP_sel=FP_lim**(1/(ncols))
    if fig==None or not isinstance(ax, (list,np.ndarray)):fig,ax=plt.subplots(2,ncols,squeeze=False,figsize=(7*ncols,14))
    elno=0
    
    out=[]
    check_list=[]
    for mvs_ids in self.mvs_candidates_df.loc[self.mvs_candidates_df.mvs_ids.isin(self.crossmatch_ids_df.loc[self.crossmatch_ids_df.avg_ids==avg_ids].mvs_ids)].mvs_ids.unique():
        magbin=self.mvs_targets_df.loc[self.mvs_targets_df.mvs_ids==mvs_ids,'m%s%s'%(filter[1:4],suffix)].values[0]
        dmag=self.mvs_candidates_df.loc[self.mvs_candidates_df.mvs_ids==mvs_ids,'m%s'%filter[1:4]].values[0]-magbin
        sep=self.mvs_candidates_df.loc[self.mvs_candidates_df.mvs_ids==mvs_ids,'%s_sep'%filter].astype(float).values[0]
        Kmode=self.mvs_candidates_df.loc[self.mvs_candidates_df.mvs_ids==mvs_ids,'%s_Kmode'%filter].astype(float).values[0]
        Nsigma=self.mvs_candidates_df.loc[self.mvs_candidates_df.mvs_ids==mvs_ids,'%s_Nsigma'%filter].astype(float).values[0]
        # Nsigma=self.mvs_candidates_df.loc[self.mvs_candidates_df.mvs_ids==mvs_ids,'Nsigma%s'%filter[1:4]].astype(float).values[0]
        if (self.mvs_targets_df.loc[self.mvs_targets_df.mvs_ids==mvs_ids,'%s_flag'%filter].values[0]!='rejected'):
            if np.isfinite(magbin) and np.isfinite(dmag) and np.isfinite(sep) and np.isfinite(Kmode) and np.isfinite(Nsigma) :
                magbin=int(magbin)
                dmag=int(dmag)
                sep=int(sep)
                Kmode=int(Kmode)
                Nsigma=int(Nsigma)
                
                if magbin not in DF_fk.fk_candidates_df.loc[(filter)].index.get_level_values('magbin').unique():
                    magbin,_=find_closer(DF_fk.fk_candidates_df.loc[(filter)].index.get_level_values('magbin').unique(),magbin)
                if dmag not in DF_fk.fk_candidates_df.loc[(filter)].index.get_level_values('dmag').unique():
                    dmag,_=find_closer(DF_fk.fk_candidates_df.loc[(filter)].index.get_level_values('dmag').unique(),dmag)
                if sep not in DF_fk.fk_candidates_df.loc[(filter)].index.get_level_values('sep').unique():
                    sep,_=find_closer(DF_fk.fk_candidates_df.loc[(filter)].index.get_level_values('sep').unique(),sep)
                out,check_list=ROC_plot(DF_fk,mvs_ids,filter,magbin,dmag,sep,Kmode,Nsigma,FP_sel,ncols,AUC_lim=AUC_lim,out=out,check_list=check_list,nbins=30,step=1,fig=fig,ax=ax,elno=elno)

            else:
                check_list.append(False)
                out.append([np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,'rejected'])
        else:
            check_list.append(False)
            out.append([np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,'rejected'])

        elno+=1
    if showplot and np.any(check_list):
        plt.tight_layout()
        plt.show()
    else: plt.close()
    return(np.array(out))

def get_roc_curve(data1, data2,std=False,nbins=1000):
    """
    Get the ROC for testing if the population of data2 is drawn from the population of data1
    data1: null hypothesis
    data2: alternative hypothesis
    Returns:
        fpf: false positive fraction
        tpf: true positive fraction
    """
    if std == True: 
        std1=np.std(data1)
        std2=np.std(data2)
    else: 
        std1=1
        std2=1
    data1 = np.sort(data1)/std1
    data2 = np.sort(data2)/std2
    # FPF thresholds
    fpf = np.linspace(0,1.,nbins+1)
    # interpolate the data1 values that correspond to the FPF thresholds
    func1 = interp1d(np.linspace(0,1., data1.size), data1)
    # fpf=np.insert(fpf,0,0)
    fpf_thresh = func1(fpf)
    tpf = frac_above_thresh(data2, fpf_thresh[::-1])
    # tpf=np.insert(tpf,0,0)
    return(fpf, tpf, fpf_thresh[::-1])

def ROC_plot(self,mvs_ids,filter,magbin,dmag,sep,Kmode,Nsigma,FP_sel,ncols,AUC_lim=0.5,out=[],check_list=[],nbins=30,step=1,fig=None,ax=None,elno=0,shwoplot=True):
    if fig==None or not isinstance(ax, (list,np.ndarray)):fig,ax=plt.subplots(2,ncols,squeeze=False,figsize=(7*ncols,14))

    TPnsigma_inj_list=self.fk_candidates_df.loc[(filter,magbin,dmag,sep),['Nsigma_Kmode%i'%(Kmode)]].values.ravel()
    FPnsigma_list=self.fk_targets_df.loc[(filter,magbin,dmag,sep),['Nsigma_Kmode%i'%(Kmode)]].values.ravel()
    TPnsigma_inj_list=TPnsigma_inj_list[~np.isnan(TPnsigma_inj_list)]
    FPnsigma_list=FPnsigma_list[~np.isnan(FPnsigma_list)]
    if len(TPnsigma_inj_list[TPnsigma_inj_list>0])<=1:
        out.append([np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,'rejected'])
    
    else:
        if shwoplot:
            if nbins==0: bins=np.arange(int(min(np.append(TPnsigma_inj_list,FPnsigma_list))),int(max(np.append(TPnsigma_inj_list,FPnsigma_list)))+step,step)
            else: bins=bins=np.histogram(np.hstack((TPnsigma_inj_list,FPnsigma_list)), bins=nbins)[1] #get the bin edges
            ax[0][elno].set_title('ID %i %s MagBin %i-%i DeltaMag %i \nKLIPmode %i Sep %i'%(mvs_ids,filter,magbin,magbin+1,dmag,Kmode,sep))
            if len(TPnsigma_inj_list)>0: ax[0][elno].hist(TPnsigma_inj_list,bins=bins,edgecolor='black', linewidth=1.2,label='Inj Cand ',color='#FFA500')
            if len(FPnsigma_list)>0: ax[0][elno].hist(FPnsigma_list,bins=bins,edgecolor='black', linewidth=1.2,label='Not Inj Cand ',color='#483D8B')
            ax[0][elno].axvline(Nsigma,linestyle='-.',lw=2,color='b')
            ax[0][elno].set_xlabel('S/N')
            ax[0][elno].set_ylabel('N')
            ax[0][elno].legend()

        X,Y,th=get_roc_curve(FPnsigma_list,TPnsigma_inj_list,nbins=10000)
        X=np.insert(X,0,0)
        Y=np.insert(Y,0,0)
 
        w=min(np.where(abs(X-FP_sel)==min(abs(X-FP_sel)))[0])       
        ax[0][elno].axvline(th[w],linestyle='-.',lw=2,color='k')
        AUC=round(metrics.auc(X, Y),3)
        Ratio_TP_above_th=len(TPnsigma_inj_list[TPnsigma_inj_list>=th[w]])/len(TPnsigma_inj_list)
        Ratio_FP_above_th=len(FPnsigma_list[FPnsigma_list>=th[w]])/len(FPnsigma_list)
        Ratio_TP_above_Nsigma=len(TPnsigma_inj_list[TPnsigma_inj_list>=Nsigma])/len(TPnsigma_inj_list)
        Ratio_FP_above_Nsigma=len(FPnsigma_list[FPnsigma_list>=Nsigma])/len(FPnsigma_list)
        if AUC>=AUC_lim:FP_flag='accepted'
        else:FP_flag='rejected'
        out.append([round(th[w],3),Ratio_TP_above_th,Ratio_TP_above_Nsigma,Ratio_FP_above_th,Ratio_FP_above_Nsigma,AUC,FP_flag])
        if shwoplot:
            ax[1][elno].set_title('TPR[th %.3f] %.3f FPR[th %.3f] %.3f\nAUC %.3f'%(th[w],Y[w],th[w],X[w],AUC))
            ax[1][elno].plot(X,Y,'-k')
            ax[1][elno].axvline(X[w], color='k', lw=2, linestyle='-.')
            ax[1][elno].axhline(Y[w], color='k', lw=2, linestyle='-.')
            ax[1][elno].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            ax[1][elno].set_ylim([-0.05,1.05])
            ax[1][elno].set_ylabel('TPR')
            ax[1][elno].set_xlabel('FPR')
        check_list.append(Nsigma>=round(th[w],3))
    return(out,check_list)


