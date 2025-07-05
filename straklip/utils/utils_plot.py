"""
generic utilities functions to create plots by the pipeline or final analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
import pyklip.klip as klip
from straklip.utils.ancillary import auc,dataframe_2D_finer_interpolator,KDE,find_closer,print_mean_median_and_std_sigmacut,power_law_fitting,latex_table
from straklip.utils.utils_completeness import mk_histogram_from_completeness
from straklip.stralog import getLogger
from astropy import units as u
from pathlib import PurePath
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.stats import norm
from mpl_toolkits.axes_grid1 import make_axes_locatable

def add_subplot_axes(ax,rect,axisbg='w'):
    fig = plt.gcf()
    box = ax.get_position()
    width = box.width
    height = box.height
    inax_position  = ax.transAxes.transform(rect[0:2])
    transFigure = fig.transFigure.inverted()
    infig_position = transFigure.transform(inax_position)    
    x = infig_position[0]
    y = infig_position[1]
    width *= rect[2]
    height *= rect[3]  # <= Typo was here
    #subax = fig.add_axes([x,y,width,height],facecolor=facecolor)  # matplotlib 2.0+
    subax = fig.add_axes([x,y,width,height])
    x_labelsize = subax.get_xticklabels()[0].get_size()
    y_labelsize = subax.get_yticklabels()[0].get_size()
    x_labelsize *= rect[2]**0.5
    y_labelsize *= rect[3]**0.5
    subax.xaxis.set_tick_params(labelsize=x_labelsize)
    subax.yaxis.set_tick_params(labelsize=y_labelsize)
    return(subax)

def unq_completeness_plots(DF,inst=None,fig=None,axes=None,completeness_curves_df=None,binary_df=None,w_pad=0,fx=7,fy=7,dfx=2,show_plot=False,dist=None,pixelscale=None,filters_list=None,df_ylabel='q',df_xlabel='SMA',cmap='Oranges_r',xlabel='SMA [arcsec]',ylabel='q [Mass$_{c}$/Mass$_{p}$]',ylim=[0.01,1],xlim=[0,1],title=False,log_y=False,log_x=False,invert_y=False,c_sel=None,c_lim=0.3,cbar_step=0.1,r=2,collapsed=False,path2savedir=None,show_candidates=False,Nvisits_list=None,select_candidate_by_visit=False,save_completeness=False):
    if inst==None: inst=DF.inst
    if not isinstance(completeness_curves_df, pd.DataFrame):   
        if df_ylabel == 'Dmag':completeness_curves_df=DF.flatten_matrix_dmag_completeness_curves_df
        elif df_ylabel == 'q':completeness_curves_df=DF.flatten_matrix_q_completeness_curves_df
        else:raise ValueError('df_ylabel MUST be eiter Dmag or q')
    if filters_list==None or len(filters_list)==0: filters_list=completeness_curves_df.index.get_level_values('Filter').unique()
    if Nvisits_list==None or len(Nvisits_list)==0:Nvisits_list=completeness_curves_df.index.get_level_values('Nvisit').unique()
    if collapsed:
        df_list_collapsedd=[]

        for filter in filters_list:
        # for Nvisit in Nvisits_list:
            df_list=[]
            # for filter in filters_list:df_list.append(completeness_curves_df.loc[filter,Nvisit])
            for Nvisit in Nvisits_list:
                df_list.append(completeness_curves_df.loc[filter,Nvisit])

            df_list_collapsedd.append(pd.concat(df_list).groupby(level=0).mean())
        completeness_curves_df=pd.concat(df_list_collapsedd,keys=filters_list)
        nv=1
    else:
        nv=len(Nvisits_list)
    n=len(filters_list)
    if pixelscale==None: pixelscale=DF.pixscale
    if show_candidates and not isinstance(binary_df, pd.DataFrame): 
        binary_df=DF.unq_candidates_df.copy()
        use_pipeline_df=True
    else:
        use_pipeline_df=False

    for elno2 in range(nv):
        Nvisit=Nvisits_list[elno2]
        fig, axes=plt.subplots(1,n,figsize=(fx*n+dfx,fy),squeeze=False,sharex=True,sharey=True)
        show_cbar=False
        set_ylabel=True
        chek4NaN=[]
        for elno in range(n):
            if elno==n-1:show_cbar=True
            if elno>0:set_ylabel=False
            filter=filters_list[elno]
            if not collapsed: CFMCC_df=completeness_curves_df.loc[filter,Nvisit]
            else: CFMCC_df=completeness_curves_df.loc[filter]
            sma_list=np.sort(CFMCC_df.columns.get_level_values(df_xlabel).unique())
            MComp_list=np.sort(CFMCC_df.index.get_level_values(df_ylabel).unique())
            p_m_list=CFMCC_df.values

            completeness_plot(fig,axes[0][elno],sma_list,MComp_list,p_m_list,show_cbar=show_cbar,set_ylabel=set_ylabel,cmap=cmap,log_y=log_y,log_x=log_x,xlabel=xlabel,ylabel=ylabel,xlim=xlim,ylim=ylim,c_lim=c_lim,cbar_step=cbar_step,title=title,d=dist,invert_y=invert_y,c_sel=c_sel)
            if show_candidates:
                if select_candidate_by_visit:
                    sel_visits=(binary_df['N%s'%filter[1:4]]==Nvisit)
                    
                    if not binary_df.loc[sel_visits].empty:
                        if use_pipeline_df:
                            sel_FP=(binary_df['FPa_flag%s'%filter[1:4]]=='accepted')
                            #sep here is in pixels
                            if df_ylabel =='q': axes[0][elno].plot(binary_df.loc[sel_FP&sel_visits,'sep'].values*pixelscale,binary_df.loc[sel_FP&sel_visits,'q'].values,'ok')
                            else:axes[0][elno].plot(binary_df.loc[sel_FP&sel_visits,'sep'].values*pixelscale,(binary_df.loc[sel_FP&sel_visits,'m%s'%filter[1:4]]-binary_df.loc[sel_FP&sel_visits,'MagBin%s'%filter[1:4]]).values,'ok')
                            chek4NaN.append(False)
    
                        else:
                            #sep here is in arcsec
                            if df_ylabel =='q': axes[0][elno].plot(binary_df.loc[sel_visits,'sep'].values,binary_df.loc[sel_visits,'q'].values,'ok')
                            else:axes[0][elno].plot(binary_df.loc[sel_visits,'sep'].values,(binary_df.loc[sel_visits,'m%s_c'%filter[1:4]]-binary_df.loc[sel_visits,'m%s_p%s'%filter[1:4]]).values,'ok')
                            chek4NaN.append(False)
                            if df_ylabel =='q' and save_completeness: 
                                for idx,row in binary_df.loc[sel_visits].iterrows():
                                    X=find_closer(sma_list,row['sep'])[-1][0]
                                    Y=find_closer(MComp_list,row['q'])[-1][0]
                                    binary_df.loc[binary_df.unq_ids_p==row.unq_ids_p,'Completeness%s'%filter[1:4]]=p_m_list[Y][X]

                    else:
                        chek4NaN.append(False)

                else:
                    if not isinstance(binary_df, pd.DataFrame):
                        #sep here is in pixels
                        sel_FP=(binary_df['FPa_flag%s'%filter[1:4]]=='accepted')
                        if df_ylabel =='q': axes[0][elno].plot(binary_df.loc[sel_FP,'sep'].values*pixelscale,binary_df.loc[sel_FP,'q'].values,'ok')
                        else:axes[0][elno].plot(binary_df.loc[sel_FP,'sep'].values*pixelscale,(binary_df.loc[sel_FP,'m%s'%filter[1:4]]-binary_df.loc[sel_FP,'MagBin%s'%filter[1:4]]).values,'ok')
                        chek4NaN.append(False)
    
                    else:
                        #sep here is in arcsec
                        if df_ylabel =='q': axes[0][elno].plot(binary_df['sep'].values,binary_df['q'].values,'ok')
                        else:axes[0][elno].plot(binary_df['sep'].values*pixelscale,(binary_df['m%s'%filter[1:4]]-binary_df['MagBin%s'%filter[1:4]]).values,'ok')
                        chek4NaN.append(False)
                        if df_ylabel =='q' and save_completeness: 
                            for idx,row in binary_df.iterrows():
                                X=find_closer(sma_list,row['sep'])[-1][0]
                                Y=find_closer(MComp_list,row['q'])[-1][0]
                                binary_df.loc[binary_df.unq_ids_p==row.unq_ids_p,'Completeness%s'%filter[1:4]]=p_m_list[Y][X]

            else:
                chek4NaN.append(False)
            elno+=1
        if not collapsed: save_name='%s_N%s_average_completeness_plot.pdf'%(inst,Nvisit)
        else: save_name='%s_collapsed_average_completeness_plot.pdf'%(inst)
        if not np.all(chek4NaN):plt.tight_layout(w_pad=w_pad)
        if isinstance(path2savedir, str) and not np.all(chek4NaN):
            print('Saving %s in %s'%(save_name,path2savedir))
            fig.savefig(path2savedir+'/'+save_name)
        if show_plot and not np.all(chek4NaN):
            if not collapsed:print('> Nvisit: ',Nvisit)
            plt.show()
        else:plt.close('all')
    if isinstance(binary_df, pd.DataFrame):return(binary_df)

def completeness_plot(fig,axes,X, Y, Z, cmap='Greys_r',manual_locations=[],set_xlabel=True,set_ylabel=True,invert_y=False,log_y=False,log_x=False,xlim=[],ylim=[],xlabel='',ylabel='',cx_list=[],cy_list=[],peculiar_cx=[],peculiar_cy=[],title_label='',c_lim=0.3,c_sel=None,cbar_step=0.1,title=True,d=None,fontsize=15,show_cbar=True):
    if isinstance(cbar_step,list):
        ticks=cbar_step
    else:
        ticks=np.append(np.arange(c_lim,1.0,cbar_step),[1])
    caxx=axes.contourf(X, Y, Z, ticks, cmap=cmap,linewidths=2,extend='both')
    if c_sel!=None:
        caxx1 = axes.contour(caxx,levels=ticks[ticks!=c_sel], colors='k')
        caxx1 = axes.contour(caxx,levels=[c_sel], colors='#FF00FF',linewidths = 2)
    else:
        caxx1 = axes.contour(caxx,levels=ticks, colors='k')
    
    if title==True:axes.set_title(title_label)

    if set_xlabel:axes.set_xlabel(xlabel)
    if set_ylabel:axes.set_ylabel(ylabel)

    if len(xlim)>0: axes.set_xlim(xlim)
    if len(ylim)>0: axes.set_ylim(ylim)

    if log_y==True:
        axes.set_yscale('log')
    if log_x==True:
        axes.set_xscale('log')
    if invert_y==True:
        axes.set_ylim(axes.get_ylim()[::-1])
    if d!=None:
        axticks = axes.get_xticks()
        axes2 = axes.twiny()
        axes2.xaxis.set_major_locator(ticker.FixedLocator(axticks*d))
        axes2.set_xlim([axes.get_xlim()[0]*d,axes.get_xlim()[1]*d])
        axes2.set_xlabel('Separation [AU]',labelpad=20)#,fontsize=fontsize)
    if show_cbar:
        divider = make_axes_locatable(axes)
        cax = divider.append_axes('right', size='5%', pad=0.5)
        cbar = plt.colorbar(caxx, cax=cax, orientation='vertical')
        cbar.set_label('Completeness', fontsize=20)
        tick_locator = ticker.MaxNLocator(nbins=len(ticks))
        cbar.locator = tick_locator
        cbar.ax.tick_params(labelsize=12)
        cbar.update_ticks()

    
def cumulative(x,bins):
    hist, bin_edges = np.histogram(x, bins=bins)
    return(np.cumsum(hist))
     
            
def fow_stamp(self,filter,qx,qy,fig=None,axes=None,path2savedir='./Plots/',title='Quad',no_sel=False,psf_nmin=10,n=1,ms=5,fz=15,psf_c='b',iso_c='g',double_c=['r','y'],bad_c='gray',text_c='k',save=True,update_cell=True,showplot=False,show_text=True,no_number=False):
    '''
    create the field of view foot stamp with all the multiple visits detections. It needs to be use in conjunction with a dataframe class object

    Parameters
    ----------
    qx : int, optional
        number of cells on the x axis. The default is 10.
    qy : int, optional
        number of cells on the y axis. The default is 10.
    filter : TYPE
        DESCRIPTION.
    fig : matplotlib.fig, optional
        figure object for plot. The default is None.
    axes : matplotlib.axes, optional
        axis for plot. The default is None.
    path2savedir : str, optional
        path where to save the cell selection pdf. The default is './Plots/'.
    title : str, optional
        title of the plot. The default is 'Quad'.
    no_sel : bool, optional
        choos if apply a selection based on the number of psf s
    bad_c : str, optional
        color of bad stars for plot. The default is 'grey'.
    iso_c : str, optional
        color of isolated stars for plot. The default is 'green'.
    double_c : list, optional
        color of double stars for plot. The default is 'red' for close, 'yellow' for wide.
    text_c : str, optional
        color for text in plot. The default is 'k'.
    save : bool, optional
        choose to save the plot. The default is True.
    update_cell : bool, optional
        choose to update the cell columns with the values. The default is True.
    showplot : bool, optional
        chose to show the plot. The default is True.
    show_text : bool, optional
        choose to show the text in the upper part of the plot. The default is True.
    no_number : bool, optional
        chose to not show the cell number in cells. The default is False.

    Returns
    -------
    None.

    '''
    xlen=self.xyaxis['x']
    ylen=self.xyaxis['y']
    x=xlen/qx
    y=ylen/qy
    x_label='x_%s'%filter
    y_label='y_%s'%filter
    quad_label='cell_%s'%filter
    flag_label='flag_%s'%filter
    if fig==None:fig,axes=plt.subplots(1,1,figsize=(n,n*ylen/xlen))
    axes.set_xlim([0,xlen])
    axes.set_ylim([0,ylen])
    elno=0
    self.mvs_targets_df[quad_label]=np.nan

    for elnoy in np.arange(0,ylen,y):
        for elnox in np.arange(0,xlen,x):
            if update_cell==True:
                self.mvs_targets_df.loc[(self.mvs_targets_df[x_label] >= elnox) & (self.mvs_targets_df[x_label] < elnox+x) & (self.mvs_targets_df[y_label] >= elnoy) & (self.mvs_targets_df[y_label] < elnoy+y),quad_label]=elno
            df_sel=self.mvs_targets_df[(self.mvs_targets_df[x_label] >= elnox) & (self.mvs_targets_df[x_label] < elnox+x) & (self.mvs_targets_df[y_label] >= elnoy) & (self.mvs_targets_df[y_label] < elnoy+y)]
            if no_sel==True:
                df_no_sel=df_sel[(df_sel[quad_label]==elno)]
                axes.scatter(df_no_sel[x_label],df_no_sel[y_label],color=iso_c,s=ms)
                axes.axhline(elnoy)
                axes.axvline(elnox)
            else:
                df_bad_sel=df_sel.loc[(df_sel[quad_label]==elno) & (df_sel[flag_label].str.contains('rejected'))]
                df_iso_sel=df_sel.loc[(df_sel[quad_label]==elno) & (df_sel[flag_label].str.contains('target'))]
                df_psf_sel=df_sel.loc[(df_sel[quad_label]==elno) & (df_sel[flag_label].str.contains('psf'))]
                df_cdouble_sel=df_sel.loc[(df_sel[quad_label]==elno) & (df_sel[flag_label].str.contains('unresolved'))]
                df_wdouble_sel=df_sel.loc[(df_sel[quad_label]==elno) & (df_sel[flag_label].str.contains('known'))]
                axes.plot(df_bad_sel[x_label],df_bad_sel[y_label],'o',color=bad_c,ms=ms)
                axes.plot(df_iso_sel[x_label],df_iso_sel[y_label],'o',color=iso_c,ms=ms)
                axes.plot(df_cdouble_sel[x_label],df_cdouble_sel[y_label],'o',color=double_c[0],ms=ms)
                axes.plot(df_wdouble_sel[x_label],df_wdouble_sel[y_label],'o',color=double_c[1],ms=ms)
                axes.plot(df_psf_sel[x_label],df_psf_sel[y_label],'o',color=psf_c,ms=ms)
                axes.axhline(elnoy)
                axes.axvline(elnox)
                if df_psf_sel.mvs_ids.count()<psf_nmin :
                    getLogger(__name__).critical(f'Need more than {psf_nmin} PSF stars in quadrant {elno}. Found {df_psf_sel.mvs_ids.count()}')
                    raise ValueError

            if (elnoy+0.5*y <= ylen) and (elnox+0.5*x <= xlen): 
                # if showplot == True and no_number==False:
                axes.text(elnox+0.5*x,elnoy+0.5*y,'%s'%elno,color=text_c,horizontalalignment='center',verticalalignment='center',fontsize=fz)
                elno+=1
    bad_el_count=self.mvs_targets_df[(self.mvs_targets_df['flag_%s'%filter].str.contains('rejected'))].mvs_ids.count()/elno
    good_el_count=self.mvs_targets_df[(self.mvs_targets_df['flag_%s'%filter].str.contains('target'))].mvs_ids.count()/elno
    psf_el_count=self.mvs_targets_df[(self.mvs_targets_df['flag_%s'%filter].str.contains('psf'))].mvs_ids.count()/elno
    cdouble_el_count=self.mvs_targets_df[(self.mvs_targets_df['flag_%s'%filter].str.contains('unresolved'))].mvs_ids.count()/elno
    wdouble_el_count=self.mvs_targets_df[(self.mvs_targets_df['flag_%s'%filter].str.contains('known'))].mvs_ids.count()/elno
    # if showplot == True:
    if show_text==True:
        axes.text(0.5,1.08,'mean number of bad objects = %s\n'%('{:3.1f}'.format(bad_el_count)),fontsize=fz,color=bad_c, horizontalalignment='center', verticalalignment='center', transform=axes.transAxes)
        axes.text(0.5,1.06,'mean number of target objects = %s\n'%('{:3.1f}'.format(good_el_count)),fontsize=fz,color=iso_c, horizontalalignment='center', verticalalignment='center', transform=axes.transAxes)
        axes.text(0.5,1.04,'mean number of psf objects = %s\n'%('{:3.1f}'.format(psf_el_count)),fontsize=fz,color=psf_c, horizontalalignment='center', verticalalignment='center', transform=axes.transAxes)
        axes.text(0.5,1.02,'mean number of unresolved pairs objects = %s\n'%('{:3.1f}'.format(cdouble_el_count)),fontsize=fz,color=double_c[0], horizontalalignment='center', verticalalignment='center', transform=axes.transAxes)
        axes.text(0.5,1.0,'mean number of known pairs objects = %s\n'%('{:3.1f}'.format(wdouble_el_count)),fontsize=fz,color=double_c[1], horizontalalignment='center', verticalalignment='center', transform=axes.transAxes)
        fig.tight_layout()
    if save:
        filename=quad_label+'.png'
        fig.savefig(path2savedir+filename,bbox_inches="tight")
        getLogger(__name__).info(f'Saving {filename} in {path2savedir}')

    if showplot:
        plt.show()
        plt.close('all')
    else:plt.close('all')
    
def mk_arrows(xa,ya,theta_0,PAV3_0,plt,L=1,Lp=None,dtx=0.3,dty=0.15,head_width=0.5, head_length=0.5,width=0.15,fz=15, fc='k', ec='k',tc='k',north=True,east=False,roll=True):
    '''
    Create an arrow on the plot

    Parameters
    ----------
    xa : float
        x arrow anchoring point
    ya : float
        y arrow anchoring point
    theta_0 : flot
        arrow angle..
    PAV3_0 : float
        PAV3 of the telescope.
    plt : matplotlib.pylab
        plot instance.
    L : float, optional
        length of the arrow. The default is 1.
    Lp : float, optional
        length of the PA arrow. The default is None.
    dtx : float, optional
        add x space between arrow and text. The default is 0.3.
    dty : float, optional
        add y space between arrow and text. The default is 0.15.
    head_width : float, optional
        arrow head width. The default is 0.5.
    head_length : float, optional
        arrow head length. The default is 0.5.
    width : float, optional
        arrow body width. The default is 0.15.
    fz : int, optional
        text font size. The default is 15.
    fc : str, optional
        arrow face color. The default is 'k'.
    ec : str, optional
        arrow face color. The default is 'k'.
    tc : str, optional
        arrow face color. The default is 'k'.
    north : bool, optional
        choose to show the north arrow. The default is True.
    east : bool, optional
        choose to show the east arrow. The default is False.
    roll : bool, optional
        choose to show the PA arrow. The default is True.

    Returns
    -------
    None.

    '''
    if Lp==None:
        Lp=L

    if theta_0==None: theta_0=0
    theta=90-theta_0
    PAV3=theta+PAV3_0

    xEnd1a = L*np.cos(np.deg2rad(theta))         # X coordinate of arrow end
    yEnd1a = L*np.sin(np.deg2rad(theta))         # Y coordinate of arrow end

    xEnd1b = L*np.cos(np.deg2rad(theta+90))         # X coordinate of arrow end
    yEnd1b = L*np.sin(np.deg2rad(theta+90))         # Y coordinate of arrow end

    xEnd2 = Lp*np.cos(np.deg2rad(PAV3))         # X coordinate of arrow end
    yEnd2 = Lp*np.sin(np.deg2rad(PAV3))         # Y coordinate of arrow end

    if north==True:
        plt.arrow(xa,ya, xEnd1a,yEnd1a, head_width=head_width, head_length=head_length,width=width, fc=fc, ec=ec)        # Plot arrow
        plt.text(xa+xEnd1a+dtx,ya+yEnd1a+dty,'N',color=tc,fontsize=fz)
    if east==True:
        plt.arrow(xa,ya, xEnd1b,yEnd1b, head_width=head_width, head_length=head_length,width=width, fc=fc, ec=ec)        # Plot arrow
        plt.text(xa+xEnd1b+dtx,ya+yEnd1b+dty,'E',color=tc,fontsize=fz)
    if roll==True:
        plt.arrow(xa,ya, xEnd2,yEnd2, head_width=head_width, head_length=head_length,width=width, fc=fc, ec=ec)        # Plot arrow
        plt.text(xa+xEnd2+dtx,ya+yEnd2+dty,'PA',color=tc,fontsize=fz)


def mk_raw_contrast_curves(id,normalization, residuals, klstep=5, path2dir='./', dataset_iwa = 1, dataset_owa = 10, fwhm = 1.460, filename=None):
    fig, ax1 = plt.subplots(figsize=(12,6))
    contrasts=[]
    for KL in residuals.columns.values[::klstep]:
        klframe = residuals[KL].values[0]/normalization
        contrast_seps, contrast = klip.meas_contrast(klframe, dataset_iwa, dataset_owa, fwhm,
                                                     center=[(klframe.shape[0]-1)//2,(klframe.shape[1]-1)//2])

        ax1.plot(contrast_seps, contrast, '-.',label=f'KL = {KL}',linewidth=3.0)
        contrasts.append(contrast)

    ax1.set_ylim([np.nanmin(contrasts),np.nanmax(contrasts)])
    ax1.set_yscale('log')
    ax1.set_ylabel('5$\sigma$ Contrast')
    ax1.set_xlabel('Separation [pix]')
    fig.legend(ncols=3,loc=1)
    plt.tight_layout()
    if filename is None:
        filename = f'tile_ID{id}_raw_cc.png'
    plt.savefig(path2dir+f'/{filename}',bbox_inches='tight')
    plt.close()

def mk_residual_tile_plots(SCI,RES,id,pat2savedir):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(f'tile_ID{id}_res.png')
    ax[0].imshow(SCI, origin='lower', cmap='gray')
    ax[1].imshow(RES, origin='lower', cmap='gray')
    plt.savefig(f'{pat2savedir}/tile_ID{id}_res.png')
    plt.close()

def mk_qmass_plot(axScatter,xo,yo,fig=None,x=None,y=None,axScHistx=None,axScHisty=None,color='grey',xmin0=None,xmax0=None,rotation_m1=-82,rotation_m2=-82,xm1=0.055,xm2=0.0095,ym1=0.9,ym2=0.9,path2saveimage=None,fmt='%.1f',mass_list=None,q_list=None,n=None,label_line=False,binwidthx = 0.25,binwidthy = 0.1,levels=None,labelsize=15,countour_plot=False,mk='+',ms=2,mscolor='grey',linewidth=3,edgecolors=None,mass_limit=None,hollow=False,xlog=True):
    if xmin0==None: xmin0=min(xo)
    if xmax0==None: xmax0=max(xo)
    xmin, xmax = xmin0,xmax0
    ymin, ymax = 0,1
    if countour_plot==True:
        xx,yy,f=KDE(x,y,xmin,xmax,ymin,ymax)
        if n!=None:levels=np.linspace(min(f.ravel()),max(f.ravel()),n+2)
        # Contourf plot
        axScatter.contourf(xx, yy, f, cmap='Blues',levels=levels)
        ## Or kernel density estimate plot instead of the contourf plot
    
        # Contour plot
        cset = axScatter.contour(xx, yy, f, colors='k',levels=levels)
    if hollow==True:
        facecolors='none'
    else:facecolors=mscolor
    if edgecolors==None:
        edgecolors=facecolors
    axScatter.scatter(xo, yo,edgecolors=edgecolors,marker=mk,s=ms,facecolors=facecolors,linewidth=linewidth,zorder=100)

    #set countour plot limits
    axScatter.set_ylim(ymin, ymax)
    axScatter.set_xlim(xmin, xmax)

    # Plot the companion mass lines
    if mass_limit!=None:
        mass_range= np.arange(xmin,xmax,0.0001)
        q_lim_comp_line=[mass_limit/x for x in mass_range]
        q_BD_comp_line=[0.075/x for x in mass_range]
        q_PL_comp_line=[0.013/x for x in mass_range]
        axScatter.plot(mass_range,q_BD_comp_line,color=color,linestyle='--',lw=5)
        axScatter.plot(mass_range,q_PL_comp_line,color=color,linestyle='--',lw=5)
        axScatter.text(xm1,ym1,'M$_c$ = 0.075',fontsize=20,rotation=rotation_m1,color=color)
        axScatter.text(xm2,ym2,'M$_c$ = 0.013',fontsize=20,rotation=rotation_m2,color=color)
        new_mass_range=np.insert(mass_range,0,0.001)
        new_mass_range=np.insert(new_mass_range,0,0.001)
        axScatter.fill(new_mass_range,[0,1]+q_lim_comp_line,'grey',alpha=0.5)
        axScatter.text(xmin+0.001,0.02,'Detection threshold',fontsize=20,rotation=0,color='white')
    
    if q_list!=None and mass_list!=None:axScatter.plot(np.log10(mass_list), q_list,linestyle='dotted',lw=5,color='k')

    # Label plot
    if label_line==True:axScatter.clabel(cset, inline=1, fontsize=12,fmt=fmt)

    #plot completness obstruction

    # now determine nice limits by hand:
    binsx=np.arange(min(xo), max(xo) + binwidthx, binwidthx)
    binsy=np.arange(min(yo), max(yo) + binwidthy, binwidthy)

    axScatter.set_ylabel('q [M$_{c}$/M$_{p}$]')
#    axScatter.set_xlabel(r'log(M$_{p}$ [M$_{\odot}$]) ')
    axScatter.set_xlabel('M$_{p}$ [M$_{\odot}$]')
#    axScatter.tick_params(labelsize=labelsize)

    if axScHistx != None: 
        axScHistx.hist(xo, bins=binsx)
        axScHistx.set_xlim(axScatter.get_xlim())
        axScHistx.set_ylabel('N')
        axScHistx.tick_params(labelsize=labelsize)
    if axScHisty != None:
        axScHisty.hist(yo, bins=binsy, orientation='horizontal')
        axScHisty.set_ylim(axScatter.get_ylim())
        axScHisty.set_xlabel('N')
        axScHisty.tick_params(labelsize=labelsize)
    if xlog==True:
        axScatter.set_xscale('log')
    if path2saveimage!= None:
        print('Saving %s'%path2saveimage)
        fig.savefig(path2saveimage,bbox_inches='tight')
        
def mvs_completeness_plots(DF,filter,path2savedir='./Plots/',MagBin_list=[],Nvisit_list=[],unq_ids_list=None,Kmodes_list=[],title=None,fx=14,fy=10,fz=20,ncolumns=4,xnew=None,ynew=None,ticks=np.arange(0.3,1.,0.1),show_IDs=False,save_completeness=False,save_figure=False,showplot=True,suffix=''):
    '''
    make completenese tong plot for each candidate visit and bin of magnitude of the primaries

    Parameters
    ----------
    filter : str
        target firter where to evaluate the tong plot.
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

    Returns
    -------
    None.
    '''

    if unq_ids_list==None:
        save_completeness=False
        for index in Nvisit_list:

            elno1=0
            elno2=0
            cc=0
            Tot=len(Kmodes_list)*len(MagBin_list)
            nrows =1
            if Tot > ncolumns:
                if Tot % ncolumns ==0: nrows =0
                nrows += Tot // ncolumns 
            else: ncolumns=Tot
            fig,ax=plt.subplots(nrows,ncolumns,figsize=(fx*ncolumns,fy*nrows),squeeze=False)
            for index2 in MagBin_list:
                for index3 in Kmodes_list:
                    if title==None: 
                        title_in='%s N %i MagBin %i Kmode %i'%(filter,index,index2,index3)
                    if index2 not in DF.fk_completeness_df.loc[(filter)].index.get_level_values('magbin').unique():
                        index2,_=find_closer(DF.fk_completeness_df.loc[(filter)].index.get_level_values('magbin').unique(),index2)
                    tong_plot(DF,index,index2,DF.dist,filter,ax=ax[elno1][elno2],Kmodes_list=[int(index3)],title=title_in,fz=fz,xnew=xnew,ynew=ynew,ticks=ticks,show_IDs=show_IDs,save_completeness=save_completeness,suffix=suffix)
                    elno2+=1
                    cc+=1
                    if elno2==ncolumns:
                        elno2=0
                        elno1+=1
            try:
                for elno in range(elno2,ncolumns):ax[elno1][elno].axis('off')
            except:pass
            plt.tight_layout()
            if save_figure: 
                filename=f'{filter}_Tong_plots_N{index}.png'
                path2dir=path2savedir+'/'+filename
                fig.savefig(PurePath(path2dir))
            if showplot: plt.show()
            else: plt.close('all')        
    
    else:

        if len(unq_ids_list)==0:unq_ids_list=DF.unq_candidates_df.unq_ids.unique()
        if len(Nvisit_list)==0:
            Nvisit_list=DF.unq_candidates_df.loc[~DF.unq_candidates_df[f'n_{filter}'].isna(),f'n_{filter}'].unique()
        if len(Kmodes_list)==0:
            Kmodes_list=DF.unq_candidates_df.loc[~DF.unq_candidates_df['mkmode'].isna(),'mkmode'].unique()
        for index,row in DF.unq_candidates_df.loc[~DF.unq_candidates_df[f'n_{filter}'].isna()&(DF.unq_candidates_df.unq_ids.isin(unq_ids_list))].groupby(f'n_{filter}'):
            if index in Nvisit_list:
                Tot=0
                for index2,row2 in DF.unq_candidates_df.loc[DF.unq_candidates_df.unq_ids.isin(row.unq_ids.unique())].groupby(f'magbin_{filter}'):
                    for index3,row3 in DF.unq_candidates_df.loc[DF.unq_candidates_df.unq_ids.isin(row2.unq_ids.unique())].groupby('mkmode'):Tot+=1
                elno1=0
                elno2=0
                nrows =1
                if Tot > ncolumns:
                    if Tot % ncolumns ==0: nrows =0
                    nrows += Tot // ncolumns 
                check4NaN=[]
                fig,ax=plt.subplots(nrows,ncolumns,figsize=(fx*ncolumns,fy*nrows),squeeze=False)
                for index2,row2 in DF.unq_candidates_df.loc[DF.unq_candidates_df.unq_ids.isin(row.unq_ids.unique())].groupby(f'magbin_{filter}'):
                    for index3,row3 in DF.unq_candidates_df.loc[DF.unq_candidates_df.unq_ids.isin(row2.unq_ids.unique())].groupby('mkmode'):
                        if index3 in Kmodes_list:
                            if title==None: 
                                title_in='%s N %i MagBin %i Kmode %i'%(filter,index,index2,index3)
                            if index2 not in DF.fk_completeness_df.loc[(filter)].index.get_level_values('magbin').unique():
                                index2,_=find_closer(DF.fk_completeness_df.loc[(filter)].index.get_level_values('magbin').unique(),index2)
                            sel_unq_ids_list=row3.unq_ids.unique()
                            if np.any(~np.isnan(row3[f'm_{filter}'].values)):
                                check4NaN.append(False)
                                tong_plot(DF,index,index2,DF.dist,filter,ax=ax[elno1][elno2],unq_ids_list=sel_unq_ids_list,Kmodes_list=[int(index3)],title=title_in,fz=fz,xnew=xnew,ynew=ynew,ticks=ticks,show_IDs=show_IDs,save_completeness=save_completeness,suffix=suffix)
                                elno2+=1
                                if elno2==ncolumns:
                                    elno2=0
                                    elno1+=1
                            else:check4NaN.append(True)
                if np.all(check4NaN):plt.close()    
                else: plt.tight_layout()

                try:
                    for elno in range(elno2,ncolumns):ax[elno1][elno].axis('off')
                except:pass                        
                if save_figure and not np.all(check4NaN):
                    filename = f'{filter}_Tong_plots_N{index}.png'
                    try:path2dir=str(path2savedir/filename)
                    except: path2dir=path2savedir+filename
                    fig.savefig(PurePath(path2dir))
                if showplot and not np.all(check4NaN): plt.show()
                else: plt.close('all')

def plot_cumulative_distributions(ax,xc_list,yc_list,xw_list,yw_list,sep_list,lwide_list,lclose_list,lwide_elist,lclose_elist,ls='-',label1='unresolved',label2='known'):
    ax.plot(xc_list,yc_list,ls,color='#B22222',lw=4,label=label1)
    ax.plot(xw_list,yw_list,ls,color='#483D8B',lw=4,label=label2)
    if len(lclose_elist)>0:ax.errorbar(sep_list,lclose_list,yerr=lclose_elist,color='#B22222',fmt='--o',lw=3,label=label1+' CA')
    if len(lwide_elist)>0:ax.errorbar(sep_list,lwide_list,yerr=lwide_elist,color='#483D8B',fmt='--o',lw=3,label=label2+' CA')
    ax.set_xlim(0,1100)
    custom_lines = [Line2D([0], [0], color='#B22222', lw=4),
                    Line2D([0], [0], color='#483D8B', lw=4)]
    ax.legend(custom_lines, ['unresolved', 'known'],loc=2)
    ax.legend(loc=2)
    ax.set_xlabel(r'Distance to $\theta^1$ Ori C [arcsec]')
    ax.set_ylabel(r'N')
    ax.minorticks_on()
    ax.tick_params(which="both", bottom=True,left=True)

def plot_dynamical_evoution(ax,ax2,x_list,yr_list,ylim=[0,1],xlim=[0,900],mu=None,ls='-',verbose=False,xmax=500,xvert=250,tick_range=[200,400,600,800]):
    ########## mean velocity dispertion in the cluster from Kim 2019 in km/s ##############
    if mu==None: mu=np.sqrt((1.57)**2+(2.12)**2)*u.km/u.s

    ax.plot(x_list,yr_list ,'k',linestyle=ls,lw=4)
    a=print_mean_median_and_std_sigmacut(yr_list[x_list>xmax],verbose=verbose)
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)

    ax2.set_xticks(tick_range)
    ax2.set_xticklabels(['%.1e'%((2*x/(mu.value*402))*u.s.to(u.yr)*(u.pc.to(u.km))) for x in tick_range])
    ax2.set_xlim(ax.get_xlim())

    ax2.set_xlabel(r'$\tau_{cros}$ [yr]',labelpad=15)
    ax.set_xlabel(r'Distance to $\theta^1$ Ori C [arcsec]')
    ax.set_ylabel(r'N$_{wide}$/N$_{close}$')
    ax.minorticks_on()
    ax.tick_params(which="both", bottom=True,left=True)
    ax.axvline(xvert,linestyle='-.',c='k',lw=3)
    ax.annotate("",
            xy=(10, ylim[1]-0.08), xycoords='data',
            xytext=(230, ylim[1]-0.08), textcoords='data',
            arrowprops=dict(arrowstyle="<->",
                            connectionstyle="arc3", color='r', lw=2),
            )
    ax.text(100, ylim[1]-0.05, 'Heavily', 
            fontsize = 16,
            ha= 'center')
    ax.text(120, ylim[1]-0.15, 'processed', 
            fontsize = 16,
            ha= 'center')
    
    ax.annotate("",
            xy=(300, ylim[1]-0.08), xycoords='data',
            xytext=(900, ylim[1]-0.08), textcoords='data',
            arrowprops=dict(arrowstyle="<->",
                            connectionstyle="arc3", color='r', lw=2),
            )
    ax.text(600, ylim[1]-0.15, 'Partially processed', 
            fontsize = 16,
            ha= 'center')
    
    if verbose:
        print('%.1e'%((2*xvert/(mu.value*402))*u.s.to(u.yr)*(u.pc.to(u.km))))
        print('%.1e'%((2*xmax/(mu.value*402))*u.s.to(u.yr)*(u.pc.to(u.km))))
        print(a)

# def plot_completed_histogram(completeness_curves_df,df,xlabel='sep',show_unq_completeness=True,show_hist=True,verbose=True,Ntest=0,axes=None,xcomp_lim=[],sep_bins=[0.1,0.35],q_bins=[0.1,0.3,1],d=402,binwidth=0.25,AUm=0,AUM=600,pixelscale=None,alpha=1,showylabel=True,showxlabel=True,visible_upper_axis=True,fz=25,lfz=22,m_up=None,m_down=None,mass_p_label=None,color_hist='#6A5ACD',color_complete='#7B68EE',axl=None,legend=None,xm=None,xM=None,ym=None,yM=None,set_axes=False,density=False,showplot=False,completeness=False,nlive=500,hollow=False,linewidth=3,xmin=None,xmax=None,set_xlogscale=False,set_ylogscale=False,show_errorbars=True,broken_plaw_fit=False,logNbins=11,edgecolor='k',histtype='step',loc='best',ncol=1,kill_plots=False,zorder=0,show_fit=False):    
    # if isinstance(completeness_curves_df, pd.DataFrame):
    #     if show_unq_completeness: 
    #         fig, ax=plt.subplots(1,1,figsize=(10,10),squeeze=False)
    #         _=unq_completeness_plots(completeness_curves_df,fig=fig,axes=ax ,dist=d,log_y=True,df_ylabel='q',df_xlabel='SMA',xlabel='SMA [arcsec]',ylabel='q [mass$_c$/mass$_p$]',invert_y=False,r=20,bin_df=df.loc[(df.unq_ids_c.isna())],collapsed=True,xlim=xcomp_lim)
    #         for elsep in range(len(sep_bins)): ax[0][0].axvline(sep_bins[elsep],c='k',linestyle='-.',lw=3)
    #         for elq in range(len(q_bins)): ax[0][0].axhline(q_bins[elq],c='k',linestyle='-.',lw=3)
            
    #     median_list_df=[]
    #     for idx,row in completeness_curves_df.groupby(['q']):
    #         median=np.median(row.values,axis=0)
    #         df_new=utils_dataframe.create_empty_df([row.columns.unique()],[idx])
    #         median[np.isnan(median)]=0
    #         median[median>1]=1
    #         df_new[idx]=median
    #         df_new=df_new.T
    #         median_list_df.append(df_new)
        
    #     CFQMCC_df=pd.concat(median_list_df)
    #     CFQMCC_df.index.set_names(['q'],inplace=True) 
    # else:CFQMCC_df=[]
        
def plot_completed_histogram(df,apply_completeness_correction=False,xlabel='sep',show_hist=True,verbose=True,Ntest=0,axes=None,xcomp_lim=[],sep_bins=[0.1,0.35],q_bins=[0.1,0.3,1],d=402,binwidth=0.25,AUm=0,AUM=600,pixelscale=None,alpha=1,showylabel=True,showxlabel=True,visible_upper_axis=True,fz=25,lfz=22,m_up=None,m_down=None,mass_p_label=None,color_hist='#6A5ACD',color_complete='#7B68EE',axl=None,legend=None,xm=None,xM=None,ym=None,yM=None,set_axes=False,density=False,showplot=False,nlive=500,hollow=False,linewidth=3,xmin=None,xmax=None,set_xlogscale=False,set_ylogscale=False,show_errorbars=True,broken_plaw_fit=False,logNbins=11,edgecolor='k',histtype='step',loc='best',ncol=1,kill_plots=False,zorder=0,show_fit=False):    
    if axes ==None:fig,axes=plt.subplots(1,1,figsize=(10,10))
    if apply_completeness_correction:
        data=[]
        x_list=[]
        y_list=[]
        eyd_list=[]
        eyu_list=[]
        N_list=[]

        if set_xlogscale: 
            if xmin==None:xmin=df[xlabel].min()
            if xmax==None:xmax=df[xlabel].max()+0.01
            bins=np.logspace(np.log10(xmin), np.log10(xmax), logNbins)
        else: 
            if xmin==None:xmin=df[xlabel].min()
            if xmax==None:xmax=df[xlabel].max()+binwidth
            bins=np.arange(xmin,xmax+binwidth,binwidth)
        data=[]
        for elno in range(len(bins[:-1])):
            x=(bins[elno]+bins[elno+1])/2
            comp_sel=(df.unq_ids_c.isna())
            sep_sel=(df[xlabel]>=bins[elno])&(df[xlabel]<bins[elno+1])
            binaries_sel_df=df.loc[sep_sel&comp_sel]
            N=df.loc[(sep_sel)].unq_ids_p.nunique()
            Nb=binaries_sel_df.unq_ids_p.nunique()
            # Ncomp,VNcomp_d,VNcomp_u=mk_histogram_from_completeness(binaries_sel_df,CFQMCC_df,sep_bins,q_bins,Ntest=Ntest,N=0,nlive=nlive,verbose=verbose)
            Ncomp,VNcomp_d,VNcomp_u=mk_histogram_from_completeness(binaries_sel_df,sep_bins=sep_bins,q_bins=q_bins,apply_completeness_correction=apply_completeness_correction,Ntest=Ntest,N=0,showplot=False,nlive=nlive,verbose=verbose)

            N_list.append(N)
            x_list.append(x)
            if Ncomp>Nb:y_list.append(int(N+Ncomp-Nb))
            else:y_list.append(int(N+Nb))
            eyd_list.append(int(VNcomp_d))
            eyu_list.append(int(VNcomp_u))
            for dumb in range(y_list[-1]):                
                data.append(x)

    else: 
        # completeness=False
        data=df[xlabel].values
        if set_xlogscale: 
            if xmin==None:xmin=np.nanmin(data)
            if xmax==None:xmax=np.nanmax(data+0.01)
            bins=np.logspace(np.log10(xmin), np.log10(xmax), logNbins)
        else: 
            if xmin==None:xmin=np.nanmin(data)
            if xmax==None:xmax=np.nanmax(data+binwidth)
            bins=np.arange(xmin,xmax,binwidth)
        x_list=[]
        for elno in range(len(bins[:-1])):
            x_list.append((bins[elno]+bins[elno+1])/2)

    data2=[]
    
    if apply_completeness_correction:
        n, bins, rectangles=axes.hist(data,fill=True,facecolor=color_complete,bins=bins,edgecolor=edgecolor,hatch='/',alpha=0.5,label=legend,density=density,linewidth=linewidth,histtype=histtype,zorder=1)
        axes.hist(data,fill=False,facecolor=color_complete,bins=bins,edgecolor=edgecolor,hatch='/',alpha=1,label=legend,density=density,linewidth=linewidth,histtype=histtype,zorder=2)
        for elx in range(len(x_list)):
            x=x_list[elx]
            N=N_list[elx]
            data2.extend([x]*N)
        _=axes.hist(data2,fill=True,bins=bins,alpha=1,label=legend,density=density,linewidth=linewidth,edgecolor='k',color=color_hist,histtype=histtype,zorder=30)
    else:
        if hollow==False: n, bins, rectangles=axes.hist(data,color=color_hist,bins=bins,linewidth=3,hatch='/',fill=True,alpha=0.5,density=density,histtype=histtype,zorder=zorder)
        n, bins, rectangles=axes.hist(data,facecolor='None',bins=bins,edgecolor=edgecolor,alpha=1,label=legend,density=density,linewidth=linewidth,histtype=histtype,zorder=zorder)
        y_list=n
        eyd_list=np.sqrt(y_list)
        eyu_list=np.sqrt(y_list)
    
    if show_errorbars:axes.errorbar(x_list,y_list,yerr=[eyd_list,eyu_list], fmt='k.',zorder=zorder)
    
    
    if broken_plaw_fit:
        axes.plot(x_list,y_list,'ok')
        w=np.where(y_list==np.nanmax(y_list))[0][0]
        x_list=np.array(x_list)
        y_list=np.array(y_list)
        yerr_list=np.array([np.mean(i) for i in zip(eyd_list,eyu_list)])
        
        powerlaw_list1,xnewdata1,amp1, ampErr1, index1, indexErr1=power_law_fitting(x_list[0:w+1][y_list[0:w+1]>0],y_list[0:w+1][y_list[0:w+1]>0],yerr_list[0:w+1][y_list[0:w+1]>0],showplot=False,verbose=verbose)
        index1=np.round(index1,2)
        indexErr1=np.round(indexErr1,2)
        if abs(index1)/abs(indexErr1)>1 and np.isfinite(indexErr1) and indexErr1>0 and show_fit:
            if legend!=None: label=r'$\alpha_{1}$ = %.2f $\pm$ %.2f'%(index1,indexErr1)
            else: label=None
            axes.plot(xnewdata1,powerlaw_list1,'-g',label=label,lw=3)
        
        powerlaw_list2,xnewdata2,amp2, ampErr2, index2, indexErr2=power_law_fitting(x_list[w:][y_list[w:]>0],y_list[w:][y_list[w:]>0],yerr_list[w:][y_list[w:]>0],showplot=False,verbose=verbose)
        index2=np.round(index2,2)
        indexErr2=np.round(indexErr2,2)

        if abs(index2)/abs(indexErr2)>1 and np.isfinite(indexErr2) and indexErr2>0 and show_fit:
            if legend!=None: label=r'$\alpha_{1}$ = %.2f $\pm$ %.2f'%(index2,indexErr2)
            else: label=None
            axes.plot(xnewdata2,powerlaw_list2,'-y',label=label,lw=3)
    
        print('################################################################')
        print('> breaking point: ',np.round(x_list[w],2),np.round(np.diff(bins),2)[w])
        print(r'$\alpha_{1}$ = %.2f $\pm$ %.2f'%(index1,indexErr1))
        print(r'$\alpha_{1}$ = %.2f $\pm$ %.2f'%(index2,indexErr2))

    if legend != None: 
        axes.legend(loc=loc,ncol=ncol)
    if set_axes == True:
        if ym!=None and yM!=None:
        # if len(yhist_lim)>0:
            axes.set_ylim(ym,yM)
            # axes.set_ylim(yhist_lim)
        if xm!=None and xM!=None:
        # if len(xhist_lim)>0:
            axes.set_ylim(xm,xM)
            # axes.set_ylim(xhist_lim)
    else:
        if pixelscale==None: raise ValueError('Please define pixelscale option.')
        pmax=AUM/(d*pixelscale)
        pmin=AUm/(d*pixelscale)
        sepARCSECM=pmax*pixelscale
        sepARCSECm=pmin*pixelscale
        xlabels=[round(p,1) for p in np.arange(sepARCSECm,sepARCSECM+0.1,0.1)]
        axes.set_xticks(xlabels)
        axes.set_xticklabels(xlabels)
        axes.set_xlim(sepARCSECm,sepARCSECM)

    if showylabel:
        if density == True:axes.set_ylabel('Prob Density')
        else: axes.set_ylabel('N')#,fontsize=fz)
    if showxlabel:
        if xlabel=='sep': axes.set_xlabel('Separation [arcsec]')#,fontsize=fz)
        elif xlabel=='q': axes.set_xlabel('q [mass$_c$/mass$_p$]')#,fontsize=fz)
        elif xlabel=='mass_p': 
            label='mass$_p$ [M$_\odot$]'
            # if set_xlogscale:'log$_{10}$'+label
            axes.set_xlabel(label)#,fontsize=fz)
        elif xlabel=='mass_c': 
            label='mass$_c$ [M$_\odot$]'
            # if set_xlogscale:'log$_{10}$'+label
            axes.set_xlabel(label)#,fontsize=fz)
        elif xlabel=='MCMC_mass': 
            label='mass [M$_\odot$]'
            # if set_xlogscale:'log$_{10}$'+label
            axes.set_xlabel(label)#,fontsize=fz)
        if set_xlogscale:axes.set_xscale('log')
        if set_ylogscale:axes.set_yscale('log')

    if axl!=None: axes.axvline(axl,linestyle='--',color='k',lw=3)
    if visible_upper_axis==True and xlabel=='sep':
        axes2 = axes.twiny()
        axes2.set_xlabel('Separation [AU]',labelpad=20)#,fontsize=fz)
        xm,xM=axes.get_xlim()
        axes2.set_xlim([xm,xM*d])
    if show_hist and not kill_plots:plt.show()
    if kill_plots:plt.close('all')
    if broken_plaw_fit:return(data,y_list,x_list,index1,indexErr1,amp1,ampErr1,index2,indexErr2,amp2,ampErr2)
    else:return(data,y_list,x_list)


def plot_histograms(vars_dict_list,bins_dict_list,fig=None,ax=None,labels_list=None,colors_list=['#6A5ACD','#FFD700'],log=False,label='Orion',fy=5,fx=15,verbose=False,sharey=True,sharex=False,logx=False,logy=False,twinx=False,twin_vars_dict_list=[],twin_bins_dict_list=[],twin_colors_list=[],twin_labels_list=[],alpha=0.1,histtype='step',sigma=3,fz=13,lw=2,overplot=False,path2saveimage=None,wspace=0,labelpad=1,fontsize=20):
    m=len(vars_dict_list[0].keys())
    if overplot: m=1
    if not isinstance(ax,np.ndarray) and fig == None: 
        fig,ax=plt.subplots(1,m,figsize=(fx,fy),sharey=sharey,sharex=sharex,squeeze=False)
    # fig.subplots_adjust(wspace=wspace)

    if labels_list==None: labels_list=list(vars_dict_list[0].keys())
    for elvar in range(len(vars_dict_list)):
        elno = 0
        elno2 = 0
        handles_list = []
        row_line = [label]
        for key in list(vars_dict_list[elvar].keys()):
            if key==list(labels_list)[0]: handles_list.append(mpatches.Patch(color=colors_list[elvar], label=label))
            a=print_mean_median_and_std_sigmacut(vars_dict_list[elvar][key],verbose=verbose,pre=key,r=2,sigma=sigma,log=log)
            row_line.append('%s $\pm$ %s'%(a[1],a[2]))

            ax[0][elno].hist(vars_dict_list[elvar][key],bins=bins_dict_list[elvar][key],linewidth=3,hatch='/',alpha=0.5, edgecolor='k',fill=True,zorder=-0.5,facecolor=colors_list[elvar],histtype=histtype)
            ax[0][elno].hist(vars_dict_list[elvar][key],bins=bins_dict_list[elvar][key],alpha=1,color='k', histtype=histtype, fill=False,linewidth=3,zorder=-0.6)

            ax[0][elno].set_xlabel(labels_list[elno2],fontsize=fontsize, labelpad=labelpad)
            if elno==0: ax[0][elno].set_ylabel('N',fontsize=fontsize)
            ax[0][elno].tick_params(which="both", bottom=True,left=True)
            ax[0][elno].minorticks_on()
            if logy: ax[0][elno].set_yscale('log')
            if logx: ax[0][elno].set_xscale('log')
            if twinx:
                axTwin = ax[0][elno].twiny()
                # Set ax's patch invisible
                ax[0][elno].patch.set_visible(False)
                # Set axtwin's patch visible and colorize it in grey
                axTwin.patch.set_visible(True)
                # move ax in front
                ax[0][elno].set_zorder(axTwin.get_zorder() + 1)

                axTwin.hist(twin_vars_dict_list[elvar][key],bins=twin_bins_dict_list[elvar][key],linewidth=3,hatch='/',alpha=0.5, edgecolor='k',fill=True,zorder=0,facecolor=twin_colors_list[elvar],histtype=histtype)
                axTwin.hist(twin_vars_dict_list[elvar][key],bins=twin_bins_dict_list[elvar][key],alpha=1,color='k', histtype=histtype, fill=False,linewidth=3,zorder=0.1)

                axTwin.set_xlabel(twin_labels_list[elno2], fontsize=fontsize, labelpad=labelpad)
                axTwin.tick_params(which="both", bottom=True, left=True)
                axTwin.minorticks_on()
                if logx: axTwin.set_xscale('log')

            if not overplot: elno+=1
            elno2+=1


        vars_line=[['']+list(vars_dict_list[elvar].keys())]
        vars_line.append(row_line)
        alignment=["c"]*(len(vars_dict_list[elvar].keys())+1)
        latex_table(vars_line,alignment=alignment)

    plt.tight_layout(w_pad=wspace)

    if path2saveimage!=None:
        print('> Saving %s'%path2saveimage)
        plt.savefig(path2saveimage, bbox_inches='tight')

    plt.show()

def plot_cmd(fig,ax,mean_df,filter1,filter2,filter3,yrange,dy,dx,y,x,ex,DM,Avs,xlim,ylim,errmin=None,iso_interp=[],iso_mass_list=np.arange(0.01,2,0.01),iso_age_list=[1],iso_logSPacc_list=[-5],iso_label=[],ID_list=[],ID_label='unq_ids',mean_df2=[],mass_list=[0.8,0.5,0.3,0.1,0.05],marker_list=['o','d','D','v','^'],line_style_list=['-','--','-.',':'],iso_color='k',iso_color_list=['b','r','g','y'],sat_th=None,spx_lim=3,cbar_label='',label_data='Data',label_iso='',label_iso_old='',xlabel=None,ylabel=None,colorbar=False,cmap='Greys_r',cmd_color='gray',iso_df=None,iso_old=None,bin_df=[],IDms=15,s=4,fz=20,color='k',label='',label_p='_p',label_c='_c',sat_min=0,cluster=False,onlycluster=False,legend=False,cm=0,cM=10,cmin=None,cmax=None,show_sat_line=False,show_minorticks=False,show_gird=False,set_tight_layout=False,show_plot=False,path2saveimage=None):
    if onlycluster:cluster=True

    mag1_list=mean_df['m_%s%s'%(filter1,label)].values
    mag2_list=mean_df['m_%s%s'%(filter2,label)].values
    mag3_list=mean_df['m_%s%s'%(filter3,label)].values
    emag1_list=mean_df['e_%s%s'%(filter1,label)].values
    emag2_list=mean_df['e_%s%s'%(filter2,label)].values
    emag3_list=mean_df['e_%s%s'%(filter3,label)].values
    if show_sat_line:
        if sat_th==None:
            sat_th=np.nan
            if isinstance(mean_df2,pd.DataFrame): 
                mag1_list2=mean_df2['m_%s%s'%(filter1,label)].values
                mag2_list2=mean_df2['m_%s%s'%(filter2,label)].values
                mag3_list2=mean_df2['m_%s%s'%(filter3,label)].values
                spx=mean_df2['spx_%s'%filter1].values
                # ax.scatter(mag2_list2-mag3_list2,mag1_list2,marker='o',c='k',s=5*s,label=label_data,zorder=-1)
                ax.scatter(mag2_list2-mag3_list2,mag1_list2,marker='o',c='gray',edgecolors='k',lw=1,s=s,zorder=-1)
                sat_th,_,_,_=print_mean_median_and_std_sigmacut(mag1_list2[(spx>=1)&(spx<=spx_lim)])
                
            else:
                if mean_df.columns.str.contains('spx_%s'%filter1).any():
                    spx=mean_df['spx_%s'%filter1].values
                    if len(mag1_list[(spx>=1)&(spx<=spx_lim)])>0:
                        sat_th,_,_,_=print_mean_median_and_std_sigmacut(mag1_list[(spx>=1)&(spx<=spx_lim)])
                elif mean_df.columns.str.contains('f_%s'%filter1).any():
                    spx=mean_df['f_%s'%filter1].values
                    if len(mag1_list[(spx==spx_lim)])>0:
                        sat_th,_,_,_=print_mean_median_and_std_sigmacut(mag1_list[(spx==spx_lim)])
                    
        print('%s sat_th: %.1f'%(filter1,sat_th))
        color_list=np.arange(cm,cM+0.25,0.25)
        sat_list=[sat_th for i in color_list]
        ax.plot(color_list,sat_list,'--k',lw=2,label='Sat. Thr.',zorder=80)

    if not onlycluster:
        ax.scatter(mag2_list-mag3_list,mag1_list,marker='o',c='k',s=5*s,label=label_data,zorder=0)
        ax.scatter(mag2_list-mag3_list,mag1_list,marker='o',c='w',s=s,zorder=1)
        im=ax.scatter(mag2_list-mag3_list,mag1_list,marker='o',c=cmd_color,s=s,zorder=10)
        if cmin!=None and cmax!=None:im.set_clim(cmin, cmax)
        if colorbar:
            axins = inset_axes(ax,
                       width="5%",  # width = 5% of parent_bbox width
                       height="60%",  # height : 50%
                       loc='lower left',
                       bbox_to_anchor=(1.05, 0., 1, 1),
                       bbox_transform=ax.transAxes,
                       borderpad=0)
            cbar=plt.colorbar(im,cax=axins,orientation='vertical')
            cbar.set_label(cbar_label, rotation=90,labelpad=5)


    if isinstance(bin_df,pd.DataFrame): 
        ID_not_nan=~(bin_df['m%s%s'%(filter1[1:4],label_p)].isna())&~(bin_df['m%s%s'%(filter1[1:4],label_c)].isna())&~(bin_df['m%s%s'%(filter2[1:4],label_p)].isna())&~(bin_df['m%s%s'%(filter2[1:4],label_c)].isna())&~(bin_df['m%s%s'%(filter3[1:4],label_p)].isna())&~(bin_df['m%s%s'%(filter3[1:4],label_c)].isna())
        eID_not_nan=~(bin_df['e%s%s'%(filter1[1:4],label_p)].isna())&~(bin_df['e%s%s'%(filter1[1:4],label_c)].isna())&~(bin_df['e%s%s'%(filter2[1:4],label_p)].isna())&~(bin_df['e%s%s'%(filter2[1:4],label_c)].isna())&~(bin_df['e%s%s'%(filter3[1:4],label_p)].isna())&~(bin_df['e%s%s'%(filter3[1:4],label_c)].isna())
        mag1_p_list=bin_df.loc[ID_not_nan&eID_not_nan,'m%s%s'%(filter1[1:4],label_p)].values
        mag2_p_list=bin_df.loc[ID_not_nan&eID_not_nan,'m%s%s'%(filter2[1:4],label_p)].values
        mag3_p_list=bin_df.loc[ID_not_nan&eID_not_nan,'m%s%s'%(filter3[1:4],label_p)].values
        emag1_p_list=bin_df.loc[ID_not_nan&eID_not_nan,'e%s%s'%(filter1[1:4],label_p)].values
        emag2_p_list=bin_df.loc[ID_not_nan&eID_not_nan,'e%s%s'%(filter2[1:4],label_p)].values
        emag3_p_list=bin_df.loc[ID_not_nan&eID_not_nan,'e%s%s'%(filter3[1:4],label_p)].values

        mag1_c_list=bin_df.loc[ID_not_nan&eID_not_nan,'m%s%s'%(filter1[1:4],label_c)].values
        mag2_c_list=bin_df.loc[ID_not_nan&eID_not_nan,'m%s%s'%(filter2[1:4],label_c)].values
        mag3_c_list=bin_df.loc[ID_not_nan&eID_not_nan,'m%s%s'%(filter3[1:4],label_c)].values
        emag1_c_list=bin_df.loc[ID_not_nan&eID_not_nan,'e%s%s'%(filter1[1:4],label_c)].values
        emag2_c_list=bin_df.loc[ID_not_nan&eID_not_nan,'e%s%s'%(filter2[1:4],label_c)].values
        emag3_c_list=bin_df.loc[ID_not_nan&eID_not_nan,'e%s%s'%(filter3[1:4],label_c)].values

        
        for elno in range(len(mag2_p_list)):ax.plot([mag2_p_list[elno]-mag3_p_list[elno],mag2_c_list[elno]-mag3_c_list[elno]],[mag1_p_list[elno],mag1_c_list[elno]],linestyle='-.',color='b',lw=5)
        ax.errorbar(mag2_p_list-mag3_p_list,mag1_p_list, yerr=emag1_p_list, xerr=np.sqrt(np.array(emag2_p_list**2+emag3_p_list**2,dtype=float)),ecolor='b',fmt='.')
        ax.errorbar(mag2_c_list-mag3_c_list,mag1_c_list, yerr=emag1_c_list, xerr=np.sqrt(np.array(emag2_c_list**2+emag3_c_list**2,dtype=float)),ecolor='g',fmt='.')

    if len(iso_interp):
        ddd=0
        # eee=0
        if len(iso_label)!=0:iso_label_final=iso_label
        else:iso_label_final=[]
        for logSPacc in iso_logSPacc_list:
            ccc=0            
            for age in iso_age_list:
                imag_list=[]
                icol_list=[]
                if len(iso_label)==0: iso_label_final.append('Iso: age %s, logSPacc %s'%(age,logSPacc))
                for elno in range(len(mass_list)):
                    mass=mass_list[elno]
                    marker=marker_list[elno]
                    imag1_mass=iso_interp['m%s'%(filter1[1:4])](np.log10(mass),np.log10(age),logSPacc)
                    imag2_mass=iso_interp['m%s'%(filter2[1:4])](np.log10(mass),np.log10(age),logSPacc)
                    imag3_mass=iso_interp['m%s'%(filter3[1:4])](np.log10(mass),np.log10(age),logSPacc)
                    if ccc==0 and ddd==0:ax.plot(imag2_mass-imag3_mass,imag1_mass+DM,'%s%s'%(iso_color_list[ddd],marker),ms=13,zorder=200,label='%s M$_{\odot}$'%mass)
                    else:ax.plot(imag2_mass-imag3_mass,imag1_mass+DM,'%s%s'%(iso_color_list[ddd],marker),ms=13,zorder=200)#,label='%s M_{sun}'%mass,zorder=11)

                for mass in iso_mass_list:
                    imag1=iso_interp['m%s'%(filter1[1:4])](np.log10(mass),np.log10(age),logSPacc)
                    imag2=iso_interp['m%s'%(filter2[1:4])](np.log10(mass),np.log10(age),logSPacc)
                    imag3=iso_interp['m%s'%(filter3[1:4])](np.log10(mass),np.log10(age),logSPacc)
                    imag_list.append(imag1+DM)
                    icol_list.append(imag2-imag3)
                ax.plot(icol_list,imag_list,'%s'%iso_color_list[ddd],linestyle=line_style_list[ccc],linewidth=3,label=iso_label_final[ccc],zorder=100)
                
                ccc+=1
            ddd+=1
            
    if isinstance(iso_df,pd.DataFrame):
        imag1=iso_df['m%s'%(filter1[1:4])].values
        imag2=iso_df['m%s'%(filter2[1:4])].values
        imag3=iso_df['m%s'%(filter3[1:4])].values
        ax.plot(imag2-imag3,imag1+DM,'-',color=iso_color,linewidth=3,label=label_iso,zorder=10000)
        
        for elno in range(len(mass_list)):
            mass=mass_list[elno]
            marker=marker_list[elno]
            imag1_mass=iso_df.loc[(iso_df.mass==mass),'m%s'%(filter1[1:4])].values
            imag2_mass=iso_df.loc[(iso_df.mass==mass),'m%s'%(filter2[1:4])].values
            imag3_mass=iso_df.loc[(iso_df.mass==mass),'m%s'%(filter3[1:4])].values
            ax.plot(imag2_mass-imag3_mass,imag1_mass+DM,'%s'%marker,color=iso_color,ms=13,label='%s M$_{\odot}$'%mass,zorder=1001)

    if isinstance(iso_old,pd.DataFrame):
        imag1=iso_old['m%s'%(filter1[1:4])].values
        imag2=iso_old['m%s'%(filter2[1:4])].values
        imag3=iso_old['m%s'%(filter3[1:4])].values
        ax.plot(imag2-imag3,imag1+DM,'-.k',linewidth=3,label=label_iso_old,zorder=1000)
        for elno in range(len(mass_list)):
            mass=mass_list[elno]
            marker=marker_list[elno]
            imag1_mass=iso_old.loc[(iso_old.mass==mass),'m%s'%(filter1[1:4])].values
            imag2_mass=iso_old.loc[(iso_old.mass==mass),'m%s'%(filter2[1:4])].values
            imag3_mass=iso_old.loc[(iso_old.mass==mass),'m%s'%(filter3[1:4])].values
            ax.plot(imag2_mass-imag3_mass,imag1_mass+DM,'%s'%marker,color=iso_color,ms=13,zorder=1001)#,label='%s M_{sun}'%mass,zorder=11)

    if Avs>0:
        ax.arrow(x,y,Avs*dx,Avs*dy, head_width=0.05, head_length=0.1, fc='r', ec='k')
        ax.text(x+0.25,y+0.5,'Av = %s'%Avs,fontsize=fz)

    if cluster:
        smag1_list=mean_df.loc[mean_df.Membership=='Cluster','m_%s%s'%(filter1,label)].values
        smag2_list=mean_df.loc[mean_df.Membership=='Cluster','m_%s%s'%(filter2,label)].values
        smag3_list=mean_df.loc[mean_df.Membership=='Cluster','m_%s%s'%(filter3,label)].values
        ax.scatter(smag2_list-smag3_list,smag1_list,marker='o',color='k',s=5*s,label='Cluster',zorder=20)

    if len(ID_list)>0:   
        IDmag1_list=mean_df.loc[mean_df[ID_label].isin(ID_list),'m_%s%s'%(filter1,label)].values
        IDmag2_list=mean_df.loc[mean_df[ID_label].isin(ID_list),'m_%s%s'%(filter2,label)].values
        IDmag3_list=mean_df.loc[mean_df[ID_label].isin(ID_list),'m_%s%s'%(filter3,label)].values
        ax.scatter(IDmag2_list-IDmag3_list,IDmag1_list,marker='o',edgecolors='k',facecolors='k',s=IDms*s,zorder=10)
        # ax.scatter(IDmag2_list-IDmag3_list,IDmag1_list,marker='o',color='r',s=10*s,zorder=500)
    
    mag_bins=[i for i in yrange]
    skip=False
    for elno in range(len(mag_bins)-1):
        esel=(mag1_list>mag_bins[elno])&(mag1_list<=mag_bins[elno+1])&~(np.isnan(mag1_list))
        yerr=np.nanmedian(emag1_list[esel])
        if errmin!=None and yerr>=errmin and not skip:
            skip=True
            print(filter1,mag_bins[elno])
        xerr=np.nanmedian(np.sqrt(emag2_list[esel]**2+emag3_list[esel]**2))
        ax.errorbar(ex, mag_bins[elno], yerr=yerr, xerr=xerr,ecolor='k')
    
    if show_minorticks:
        ax.minorticks_on()
        ax.tick_params(which="both", bottom=True,left=True)
    
    if show_gird: ax.grid(True)
        
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    if xlabel==None: ax.set_xlabel('m%s-m%s'%(filter2[1:4],filter3[1:4]))
    else: ax.set_xlabel(xlabel)
    if ylabel==None: ax.set_ylabel('m%s'%(filter1[1:4]))
    else: ax.set_ylabel(ylabel)
    if legend: ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    if set_tight_layout: plt.tight_layout()
    
    if path2saveimage!=None: 
        print('> Saving %s'%(path2saveimage))
        plt.savefig(path2saveimage)
    if show_plot: plt.show()
                
def tong_plot(DF,N,magbin,d,filter,ax=None,unq_ids_list=[],Kmodes_list=[],title='',fz=20,xnew=None,ynew=None,ticks=np.arange(0.3,1.,0.1),show_IDs=False,save_completeness=False,suffix=''):
    '''
    Plot the tong plot for the candidates for a given number of visits, bin of magnitude of the primary and a specific filter.

    Parameters
    ----------
    N : int
        number of visits for the targets.
    magbin : int
        bin of magnitude for the targets.
    d : int
        distance in parcec of the targets.
    filter : str
        target firter where to evaluate the tong plot.
    ax : matplotlib subplots axes, optional
        matplotlib subplots axes. The default is None.
    unq_ids_list : list, optional
        list of average ids of candidates to evaluate completness for. The default is [].
    Kmodes_list : list, optional
        list of KLIPmode to use to evaluate the median completness for the tong plot. The default is [].
    title : str, optional
        title of the tong plot. The default is ''.
    fz : int, optional
        font size for the title. The default is 20.
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

    Returns
    -------
    None.

    '''
    Z_list=[]
    if len(Kmodes_list)==0:Kmodes_list=DF.kmodes
    for kmode in Kmodes_list:
        xnew,ynew,znew,X_index_label,Y_index_label=dataframe_2D_finer_interpolator(DF.fk_completeness_df.loc[(filter,N,magbin,slice(None),slice(None)),[f'ratio_kmode{kmode}']],xnew=None,ynew=None,Z_columns_label=f'ratio_kmode{kmode}')
        hdfpivot=pd.DataFrame(znew, columns = ynew, index = xnew)
        hdfpivot.columns.name=Y_index_label
        hdfpivot.index.name=X_index_label
        X=hdfpivot.columns.values.astype(float)
        Y=hdfpivot.index.values.astype(float)
        Z=hdfpivot.values.astype(float)
        Xi,Yi = np.meshgrid(X, Y)
        Z_list.append(Z)
    
    Z=np.median(Z_list,axis=0)
    print(ticks)
    if ax ==None: fig,ax=plt.subplots(1,1,figsize=(10,7))
    caxx=ax.contourf(Yi, Xi, Z, ticks, cmap='Oranges_r',linewidths=4,extend='both')
    caxx1 = ax.contour(caxx,levels=ticks, colors='k')
    ax.clabel(caxx1, fmt='%.2f', colors='k', fontsize=10)
    ax.invert_yaxis()
    ax.set_ylabel('Dmag [mag]',fontsize=20)
    ax.set_xlabel('Sep [px]',fontsize=20)
    ax.tick_params(labelsize=12) 
    ax.set_title(title,fontsize=fz)

    ax2 = ax.twiny()
    new_labels=[int(d*p*DF.pixscale) for p in ax.get_xticks()]
    ax2.set_xticks(ax.get_xticks())
    ax2.set_xticklabels(new_labels)
    ax2.set_xlabel('Sep [AU]',labelpad=20,fontsize=20)
    # ax2.set_xlim(ax.get_xlim())
    ax2.tick_params(labelsize=12) 

    ax3 = ax.twiny()
    ax3.spines['top'].set_position(("axes", 1.2))
    new_labels=[np.round(float(p*DF.pixscale),3) for p in ax.get_xticks()]
    ax3.set_xticks(ax.get_xticks())
    ax3.set_xticklabels(new_labels)
    ax3.set_xlabel('Sep [arcsec]',labelpad=20,fontsize=20)
    # ax3.set_xlim(ax.get_xlim())
    ax3.tick_params(labelsize=12) 
    # unq_ids_list2check=DF.unq_candidates_df.loc[(DF.unq_candidates_df['MagBin%s'%filter[1:4]]==magbin)&(DF.unq_candidates_df['N%s'%filter[1:4]]==N)].unq_ids.unique()
    # if len(unq_ids_list)==0: unq_ids_list=unq_ids_list2check
    for unq_ids in unq_ids_list:
        dmag=DF.unq_candidates_df.loc[DF.unq_candidates_df.unq_ids==unq_ids,f'm_{filter}'].values[0]-DF.unq_targets_df.loc[DF.unq_targets_df.unq_ids==unq_ids,f'm_{filter}{suffix}'].values[0]
        # print(DF.unq_candidates_df.loc[DF.unq_candidates_df.unq_ids==unq_ids,'m%s'%filter[1:4]].values[0],DF.unq_targets_df.loc[DF.unq_targets_df.unq_ids==unq_ids,'m%s%s'%(filter[1:4],suffix)].values[0])
        sep=DF.unq_candidates_df.loc[DF.unq_candidates_df.unq_ids==unq_ids].sep.values[0]
        if not np.isnan(dmag):
            if save_completeness:
                completeness=dataframe_2D_finer_interpolator(DF.fk_completeness_df.loc[(filter,N,magbin,slice(None),slice(None)),[f'ratio_kmode{kmode}']],xnew=sep,ynew=dmag,Z_columns_label=f'ratio_kmode{kmode}')[2][0]
                DF.unq_candidates_df.loc[DF.unq_candidates_df.unq_ids==unq_ids,'Completeness%s'%filter[1:4]]=completeness
                # print(unq_ids,completeness)
                    
            if show_IDs:
                ax.text(sep-0.03,dmag-0.1,'ID %i'%unq_ids,fontsize=10)
            ax.scatter(sep,dmag,c='k')

    divider = make_axes_locatable(ax)

    cax = divider.append_axes('right', size='5%', pad=0.5)
    cbar = plt.colorbar(caxx, cax=cax, orientation='vertical',spacing = 'proportional')

    # cbar = plt.colorbar(cax)
    cbar.set_label('Completeness',fontsize=20)
    # tick_locator = ticker.MaxNLocator(nbins=len(ticks))
    # cbar.locator = tick_locator
    cbar.ax.tick_params(labelsize=12) 
    cbar.update_ticks()

def plot_gaussian_comparison(DF, unq_id, labels=['MCMC_d','MCMC_ed_d','MCMC_ed_u'],ref=[402,50,50], npoints=1000, verbose=True):
    display(DF.unq_targets_df.loc[DF.unq_targets_df.unq_ids==unq_id])
    m0, std01,std02 = [DF.unq_targets_df.loc[DF.unq_targets_df.unq_ids==unq_id,labels[0]].values[0],DF.unq_targets_df.loc[DF.unq_targets_df.unq_ids==unq_id,labels[1]].values[0],DF.unq_targets_df.loc[DF.unq_targets_df.unq_ids==unq_id,labels[2]].values[0]]
    m1, std1 = [DF.unq_targets_df.loc[DF.unq_targets_df.unq_ids==unq_id,labels[0]].values[0],np.nanmean([DF.unq_targets_df.loc[DF.unq_targets_df.unq_ids==unq_id,labels[1]],DF.unq_targets_df.loc[DF.unq_targets_df.unq_ids==unq_id,labels[1]]])]
    print(m0,std01,std02)
    m2, std21, std22 = ref
    if m1 <= m2:
        color1 = 'b'
        color2 = 'y'
    else:
        color1 = 'y'
        color2 = 'b'

    #Get point of intersect


    x = np.linspace(np.min([m0 - 3 * std01, m2 - 3 * std21]), np.max([m0 + 3 * std01, m2 + 3 * std22]), npoints)
    r = [m2-std21,m2+std22]
    #Get point on surface
    if verbose:
        plt.figure(figsize=(10, 10))
        # print(m0, std01,std02,m1, std1)
        plot1 = plt.plot(x, norm.pdf(x, m1, std1), 'y', label='unq_id %i'%unq_id)
        plot1 = plt.plot(np.append(x[x <= m0],x[x>m1]), np.append(norm.pdf(x[x<=m0], m0, std01),norm.pdf(x[x>m0], m0, std02)), 'g')
        if r[0] <= m0:std_s1=std01
        else: std_s1=std02
        if r[1] <= m0: std_s2=std01
        else: std_s2=std02
        plot3 = plt.plot([r[0],r[1]], np.append(norm.pdf(r[0], m0, std_s1),norm.pdf(r[1], m0, std_s2)), 'o')

        print("Area under curves %.3f" % auc(r, m0, std01, std02))

        plt.xlabel(labels[0])
        # plt.legend()
        plt.show()
    else:
        return(auc(r, m0, std01, std02))

# def plot_kde_comparison(file,pmin=None,pmax=None,sigma=3,verbose=False, bw_method=None,kernel='linear',bandwidth2fit=np.linspace(0.01, 1, 100)):
#     mcmc_dict=read_samples(file)
#     samples=np.array(mcmc_dict['samples'])
#
#     if len(samples) >0:
#         flat_samples=samples.reshape(samples.shape[0]*samples.shape[1],samples.shape[2])
#         filtered_flat_sample=sigma_clip(flat_samples, sigma=sigma, maxiters=5,axis=0)
#         flat_samples=filtered_flat_sample.copy()
#
#         x=np.sort(flat_samples[:,-1][~flat_samples[:,-1].mask])
#
#         # x=np.sort(flat_samples[:,-1])
#         if pmin is None: pmin = np.nanmin(x)
#         if pmax is None: pmax = np.nanmax(x)
#
#         xlinspace=np.linspace(np.nanmin(x),np.nanmax(x),1000)
#         kde=KDE0(np.sort(x), xlinspace, bandwidth=bw_method,kernel=kernel,bandwidth2fit=bandwidth2fit)
#         kde.kde_sklearn()
#         pdf_max=np.max(kde.pdf(xlinspace))
#         w=np.where(kde.pdf(xlinspace)==pdf_max)
#         val=np.nanmedian(xlinspace[w])
#         xlinspace2=xlinspace[(xlinspace>pmin)&(xlinspace<pmax)]
#
#         try:
#             area2 = np.trapz(kde.pdf(xlinspace2), dx=0.01)
#             area = np.trapz(kde.pdf(xlinspace), dx=0.01)
#             area_r=area2/area
#         except:
#             area_r = 0
#
#         if verbose:
#             plt.figure(figsize=(10, 10))
#             plt.plot(xlinspace, kde.pdf(xlinspace), 'y')
#             plt.plot([pmin,pmax], kde.pdf(np.array([pmin,pmax])), 'or')
#             plt.show()
#             print("Area under curves %.3f" % area_r)
#     else: area_r=0
#
#     return(area_r)

