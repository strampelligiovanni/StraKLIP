'''
This module implements the Tile class, which help create/load/save 
tiles cubes centered around each target (data, error, dq, modles, residuals)
'''
from straklip.utils.ancillary import find_max,find_centroid,cosmic_ray_filter,cosmic_ray_filter_la,my_circle_scatter

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

from straklip.utils.utils_plot import mk_arrows
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.visualization import simple_norm
from straklip.stralog import getLogger

class Tile():
    def __init__(self,data=[],delta=0,x=np.nan,y=np.nan,tile_base=1,inst=None,dqdata=[],Python_origin=True,raise_errors=False):
        '''
        Inizialize the Postage Stamp object

        Parameters
        ----------
        data : fits image
            the data input.
        x : float
            x pixel coordinate on the data input to use as a center for the new tile.
        y : float
            y pixel coordinate on the data input to use as a center for the new tile.
        exptime_im : float
            exposure time related to the data input.
        tile_base : TYPE
            side of the square tile. MUST be odd
        # filter : str
        #     filter related to the data input.
        inst : str
            instrument related to the data input. Only used in cosmic ray detection. Default None
        Python_origin : bool
            Choose to specify the origin of the xy input coordinates. For exmaple python array star counting from 0, 
            so a position obtained on a python image will have 0 as first pixel. 
            On the other hand, normal catalogs start counting from 1 (see coordinate on ds9 for example) 
            so we need to subtract 1 to make them compatible when we use those coordinates on python
            The default is True

        Returns
        -------
        None.

        '''
        if raise_errors:
            if (delta % 2) != 0: raise ValueError('delta MUST be even. Your imput is odd: %s'%delta) 
            if (tile_base % 2) == 0: raise ValueError('tile_base MUST be odd. Your imput is even: %s'%tile_base)
        else:
            if (delta % 2) != 0: delta+=1
            if (tile_base % 2) == 0: tile_base+=1
            
        self.data=data
        self.dqdata=dqdata
        self.delta=delta
        self.tile_base=tile_base+delta
        self.inst=inst
        if Python_origin: xy_origin = 0
        else: xy_origin = 1
        self.x=x-xy_origin
        self.y=y-xy_origin
        self.x0=int(tile_base/2)
        self.y0=int(tile_base/2)

    def mk_tile(self,fig=None,ax=None,step=2,title='',cmap='viridis',xy_tile=False,xy_m=False,xy_cen=False,xy_dmax=3,box_size=3,fwhm=2.5,sigma=4,background=0,std=0,showplot=True,cbar=False,lpad=0.5,fx=5,fy=5,legend=False,mk_arrow=False,xa=None,ya=None,theta=0,PAV3=None,L=None,dtx=0.3,dty=0.15,head_width=0.5, head_length=0.5,width=0.15, fc='k', ec='k',tc='k',north=True,east=False,roll=True,simplenorm=None,min_percent=0,max_percent=100,power=1,log=1000,cr_remove=False, la_cr_remove=False,cr_radius=3,verbose=False,kill=False,close=True,vmin=None,vmax=None,pad_data=False,pad=None,keep_size=False,return_tile=False,kill_plots=False,path2savefig=None):
        '''
        Select a smaller portion of an imput image and create a new tile from it.

        Parameters
        ----------
        fig : matplotlib, optional
            outside figure. The default is None
        ax : matplotlib, optional
            outside axis. The default is None.
        step : int, optional
            step of the axis ticks. The default is 1.
        title : str, optional
            title for the plot. The default is ''.
        cmap : str, optional
            color map. The default is 'Greys_r'.
        xy_tile : bool, optional
            choose to show the original coordinates on the tile frame. The default is False.
        xy_cen : bool, optional
            choose to evaluate and show the coordinate of the centroids on the tile frame. The default is False.
        xy_m : bool, optional
            choose to evaluate and show the coordinate of the maximum on the tile frame. The default is False.
        xy_dmax: int, optional
            distance from the center to look for maximum. The default is 2
        showplot : bool, optional
            choose to show the plot. The default is True.
        cbar : bool, optional
            choose to show the colorbar. The default is False.
        lpad : float, optional
            legend border pad. The default is 0.5.
        fx : int, optional
            figure x dimemsion. The default is 5.
        fy : int, optional
            figure y dimension. The default is 5.
        legend : bool, optional
            choose to show the legend. The default is False.
        mk_arrow : bool, optional
            choose to show arrows on tile. The default is False.
        xa : float, optional
            x arrow anchoring point. The default is None.
        ya : float, optional
            y arrow anchoring point. The default is None.
        theta : float, optional
            arrow angle. The default is 0.
        PAV3 : float, optional
            PAV3 of the telescope. The default is None.
        L : float, optional
            length of the arrow. The default is None.
        dtx : float, optional
            add x space between arrow and text. The default is 0.3.
        dty : float, optional
            add y space between arrow and text. The default is 0.3.
        head_width : gloat, optional
            arrow head width. The default is 0.5.
        head_length : float, optional
            arrow head lenght. The default is 0.5.
        width : float, optional
            arrow body width. The default is 0.15.
        fc : str, optional
            arrow face color. The default is 'k'.
        ec : str, optional
            arrow edge color. The default is 'k'.
        tc : str, optional
            arrow text color. The default is 'k'.
        north : bool, optional
            choose to show the north arrow. The default is True.
        east : bool, optional
            choose to show the east arrow. The default is False.
        roll : bool, optional
            choose to show the PA arrow. The default is True.
        simplenorm : TYPE, optional
            use simplenorm normalization. The default is None.
        min_percent: float, optional
            The percentile value used to determine the pixel value of minimum cut level. 
            The default is 0.0. min_percent overrides percent.
        max_percent: float, optional
            The percentile value used to determine the pixel value of maximum cut level. 
            The default is 100.0. max_percent overrides percent.
        power: float, optional
            The power index for stretch='power'. The default is 1.0.
        log: float, optional
            The log index for stretch='log'. The default is 1000.
        cr_remove : bool, optional
            choose to apply cosmic ray removal. The default is False.
        la_cr_remove : bool, optional
            choose to apply L.A. cosmic ray removal. The default is False.
        cr_radius : int, optional
            minimum distance from center where to not apply the cosmic ray filter. The default is 3.
        close : bool, optional
            choose to close plot istances. The default is True.
        kill : bool, optional
            choose to kill bad pixels instead of using the median of the neighbouring pixels. The default is False.
        verbose : bool, optional
            choose to show prints. The default is False.
        vmin : float, optional
             define the data range that the colormap covers. The default is None.
        vmax : float, optional
             define the data range that the colormap covers. The default is None.
        pad_data : bool, optional
            choose to pad the input data. The default is False.
        pad : int, optional
            number of pixels to pad the data. If None use the base of the tile. The default is None.
        return_tile: bool, optional
            choose to return the newly created tile as output of the fuction.
        kill_plots:
            choose to kill all plots created. The default is False.

        Returns
        -------
        im: numpy array. optional
            new created tile.

        '''
        # original_data=self.data
        
        
        if pad_data:
            if pad==None: pad=int(self.tile_base)
            padded_data = np.array(np.pad(self.data,pad,'constant'),dtype='float64')
            self.data=padded_data
            self.xpad=self.x+pad
            self.ypad=self.y+pad
        else:
            self.xpad=self.x
            self.ypad=self.y

        self.xpad_floor=int(round(self.xpad))
        self.ypad_floor=int(round(self.ypad))
        self.xpad_offset=round(self.xpad-self.xpad_floor,3)
        self.ypad_offset=round(self.ypad-self.ypad_floor,3)

        if verbose == True:
            # print('inp xy: ',self.xpad,self.ypad)
            # print('floor padded xy: ',self.xpad_floor,self.ypad_floor)
            getLogger(__name__).debug('inp xy: ',self.xpad,self.ypad)
            getLogger(__name__).debug('floor padded xy: ',self.xpad_floor,self.ypad_floor)

        if keep_size:
            self.xmin0 = 0
            self.ymin0 = 0
            self.xmax0 = self.data.shape[0]
            self.ymax0 = self.data.shape[1]

        else:
            self.xmin0 = int(round(self.xpad_floor - 0.5* (self.tile_base-1)))
            self.ymin0 = int(round(self.ypad_floor - 0.5* (self.tile_base-1)))
            self.xmax0 = int(round(self.xpad_floor + 0.5* (self.tile_base-1)))+1
            self.ymax0 = int(round(self.ypad_floor + 0.5* (self.tile_base-1)))+1
            if self.xmax0 > self.data.shape[1]: self.xmax0=self.data.shape[1]
            if self.ymax0 > self.data.shape[0]: self.ymax0=self.data.shape[0]
            if self.xmin0 <0: self.xmin0=0
            if self.ymin0 <0: self.ymin0=0
            self.data=self.data[self.ymin0:self.ymax0,self.xmin0:self.xmax0]

        self.x_tile=(self.tile_base-1)/2#int(self.xpad-self.xmin0)
        self.y_tile=(self.tile_base-1)/2#int(self.ypad-self.ymin0)
        if cr_remove==True and la_cr_remove==False:
            title+=' CR free'
            if verbose==True:
                # print('\nApplying cosmic ray rejection')
                getLogger(__name__).debug('\nApplying cosmic ray rejection')

            self.data=cosmic_ray_filter(self,cr_radius,delta=3,verbose=False,kill=kill)

        elif la_cr_remove==True and  cr_remove==False:
            title+=' CR free'
            if verbose==True:
                # print('\nApplying LA cosmic ray rejection')
                getLogger(__name__).debug('\nApplying LA cosmic ray rejection')

            self.data=self.data # Must be counts or electrons, NOT e-/s or c/s
            cosmic_ray_filter_la(self,sigclip=4.5,niter=5,verbose=False)
            self.data=self.cr_clean_im
            self.dqdata[self.cr_mask==1]=16384

        if xy_cen==True:
            sources_tab,deltas=find_centroid(self.data,self.x_tile,self.y_tile,xy_dmax,fwhm,sigma,std)
            if  sources_tab!=None:
                sources = sources_tab.to_pandas().sort_values('peak',ascending=False).reset_index(drop=True)
                sources['ycentroid']=sources['ycentroid']+deltas[0]
                sources['xcentroid']=sources['xcentroid']+deltas[1]
                self.sources=sources
                self.x_cen=sources.xcentroid.values[0]
                self.y_cen=sources.ycentroid.values[0]
            else:
                self.sources=None
                self.x_cen=(self.tile_base-1)/2
                self.y_cen=(self.tile_base-1)/2

        else:
            self.sources=None
            self.x_cen=None
            self.y_cen=None
        if xy_m:
            if xy_dmax==None: xy_dmax=int((self.tile_base-1)/2)
            self.x_m,self.y_m=find_max(self.data,int((self.tile_base-1)/2)-xy_dmax,int((self.tile_base-1)/2)+xy_dmax+1,speak=False)
        else:
            self.x_m=None
            self.y_m=None
        if fig == None and ax==None: 
            fig,ax=plt.subplots(1,1,figsize=(fx,fy))
        im=self.plot_tile(fig,ax,title=title,cmap=cmap,xy_tile=xy_tile,xy_cen=xy_cen,xy_m=xy_m,cbar=cbar,lpad=lpad,legend=legend,mk_arrow=mk_arrow,xa=xa,ya=ya,theta=theta,PAV3=PAV3,L=L,dtx=dtx,dty=dty,head_width=head_width, head_length=head_length,width=width, fc=fc, ec=ec,tc=tc,north=north,east=east,roll=roll,showplot=showplot,step=step,simplenorm=simplenorm,min_percent=min_percent,max_percent=max_percent,power=power,log=log,verbose=verbose,kill=kill,close=close,vmin=vmin,vmax=vmax,extent=None,kill_plots=kill_plots,savename=path2savefig)
        if return_tile: return(im)

    def append_tile(self, path2tile,Datacube=None,verbose=False,name='SCI',header=None,return_Datacube=False,write=True):
        if Datacube==None:
            Datacube= fits.HDUList()
            Datacube.append(fits.PrimaryHDU())
        data=self.data[int(self.delta/2):self.tile_base-int(self.delta/2),int(self.delta/2):self.tile_base-int(self.delta/2)].copy()
        Datacube.append(fits.ImageHDU(data=data,name=name,header=header))
        if verbose:
            getLogger(__name__).debug(Datacube.info())

        if return_Datacube: 
            return(Datacube)
        else: 
            if write:Datacube.writeto(path2tile,overwrite=True)
            else:
                Datacube.flush()

    def load_tile(self, path2tile,hdul_max=None,ext=None,verbose=False,return_Datacube=False,mode='readonly',raise_errors=True):
        try:
            Datacube=fits.open(path2tile,memmap=False,mode=mode)
            if hdul_max!=None:
                for n in range(hdul_max,len(Datacube)-1):Datacube.pop(hdul_max+1)
            if verbose:
                getLogger(__name__).debug(Datacube.info())

        except:
            if raise_errors: raise ValueError('%s do not exist'%path2tile)
            else:Datacube=None
        if ext!= None:
            try:
                self.data=Datacube[ext].data.astype(float)
            except:
                if raise_errors: raise ValueError('%s is missing extension %s'%(path2tile,ext))
                else: self.data=np.ones((self.tile_base,self.tile_base))*np.nan

        if return_Datacube: return(Datacube)
        else:
            if raise_errors: Datacube.close()
    
    def plot_tile(self,fig,ax,title=None,cmap='viridis',xy_tile=True,xy_cen=True,xy_m=True,cbar=False,bad_pixel_c='r',lpad=0.5,legend=False,tight=False,mk_arrow=False,xa=None,ya=None,theta=0,PAV3=None,L=None,dtx=0.3,dty=0.15,head_width=0.5,head_length=0.5,width=0.15, fc='k', ec='k',tc='k',north=True,east=False,roll=True,showplot=True,step=2,simplenorm=None,verbose=False,close=True,kill=False,vmin=None,vmax=None,min_percent=0,max_percent=100.0,power=1,log=1000,savename=None,extent=None,kill_plots=False):
        '''
        plot the input tile

        Parameters
        ----------
        fig : matplotlib.pyplot
            plot figure.
        ax : matplotlib.pyplot
            plot axis.
        title : str, optional
            title for the plot. The default is ''.
        cmap : str, optional
            color map. The default is 'Greys_r'.
        xy_tile : bool, optional
            choose to show the original coordinates on the tile frame. The default is False.
        xy_cen : bool, optional (deprecated)
            choose to evaluate and show the coordinate of the centroids on the tile frame. The default is False.
        xy_m : bool, optional
            choose to evaluate and show the coordinate of the maximum on the tile frame. The default is False.
        cbar : bool, optional
            choose to show the colorbar. The default is False.
        bad_pixel_c : str, optional
            NaN pixel color for colorbar. The default is 'r'.
        lpad : float, optional
            legend border pad. The default is 0.5.
        legend : bool, optional
            choose to show the legend. The default is False.
        tight : bool, optional
            choose to apply tight layout to plot. The default is False.
        mk_arrow : bool, optional
            choose to show arrows on tile. The default is False.
        xa : float, optional
            x arrow anchoring point. The default is None.
        ya : float, optional
            y arrow anchoring point. The default is None.
        theta : float, optional
            arrow angle. The default is 0.
        PAV3 : float, optional
            PAV3 of the telescope. The default is None.
        L : float, optional
            length of the arrow. The default is None.
        dtx : float, optional
            add x space between arrow and text. The default is 0.3.
        dty : float, optional
            add y space between arrow and text. The default is 0.3.
        head_width : gloat, optional
            arrow head width. The default is 0.5.
        head_length : float, optional
            arrow head lenght. The default is 0.5.
        width : float, optional
            arrow body width. The default is 0.15.
        fc : str, optional
            arrow face color. The default is 'k'.
        ec : str, optional
            arrow edge color. The default is 'k'.
        tc : str, optional
            arrow text color. The default is 'k'.
        north : bool, optional
            choose to show the north arrow. The default is True.
        east : bool, optional
            choose to show the east arrow. The default is False.
        roll : bool, optional
            choose to show the PA arrow. The default is True.
        showplot : TYPE, optional
            DESCRIPTION. The default is True.
        step : TYPE, optional
            DESCRIPTION. The default is 1.
        simplenorm : TYPE, optional
            use simplenorm normalization. The default is None.
        verbose : bool, optional
            choose to show prints. The default is False.
        close : bool, optional
            choose to close plot istances. The default is True.
        kill : bool, optional
            choose to kill bad pixels instead of using the median of the neighbouring pixels. The default is False.
        vmin : float, optional
             define the data range that the colormap covers. The default is None.
        vmax : float, optional
             define the data range that the colormap covers. The default is None.
        min_percent: float, optional
            The percentile value used to determine the pixel value of minimum cut level. 
            The default is 0.0. min_percent overrides percent.
        max_percent: float, optional
            The percentile value used to determine the pixel value of maximum cut level. 
            The default is 100.0. max_percent overrides percent.
        power: float, optional
            The power index for stretch='power'. The default is 1.0.
        log: float, optional
            The log index for stretch='log'. The default is 1000.
        savename : str, optional
            path and file name to save the plot. The default is None.
        kill_plots:
            choose to kill all plots created. The default is False.

        Returns
        -------
        im: numpy array.
            new created tle.

        '''
        current_cmap = plt.cm.get_cmap()
        current_cmap.set_bad(color=bad_pixel_c)
        if title != None: ax.set_title(title)
        ax.set_xticks(np.arange(0, self.data.shape[1], step))
        ax.set_yticks(np.arange(0, self.data.shape[0], step))

        if simplenorm!=None:
            norm = simple_norm(self.data, simplenorm, min_percent=min_percent,max_percent=max_percent,power=power,log_a=log)
            if vmin!=None and vmax!=None:
                im = ax.imshow(self.data, cmap=cmap, origin='lower', norm=norm, vmin=vmin, vmax=vmax,extent=extent)
            elif vmin!=None and vmax==None:
                im=ax.imshow(self.data,cmap=cmap,origin='lower',norm=norm,vmin=vmin,extent=extent)
            elif vmin==None and vmax!=None:
                im=ax.imshow(self.data,cmap=cmap,origin='lower',norm=norm,vmax=vmax,extent=extent)
            else:
                im=ax.imshow(self.data,cmap=cmap,origin='lower',norm=norm,extent=extent)
        else:
            if vmin!=None and vmax!=None:
                im=ax.imshow(self.data,cmap=cmap,origin='lower',vmin=vmin,vmax=vmax,extent=extent)
            if vmin!=None and vmax==None:
                im=ax.imshow(self.data,cmap=cmap,origin='lower',vmin=vmin,extent=extent)
            if vmin==None and vmax!=None:
                im=ax.imshow(self.data,cmap=cmap,origin='lower',vmax=vmax,extent=extent)
            else:
                im=ax.imshow(self.data,cmap=cmap,origin='lower',extent=extent)

        if mk_arrow==True:mk_arrows(xa,ya,theta,PAV3,L,plt,dtx=dtx,dty=dty,head_width=head_width, head_length=head_length,width=width, fc=fc, ec=ec,tc=tc,north=north,east=east,roll=roll)

        if cbar==True:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im, cax=cax, orientation='vertical')

        if verbose==True:
            # print('tile xy: ',self.x_tile,self.y_tile)
            getLogger(__name__).debug('tile xy: ',self.x_tile,self.y_tile)


        if xy_cen:
            my_circle_scatter(ax, [self.x_cen], [self.y_cen], radius=0.2, alpha=1, color='g')
            if verbose==True:
                # print('centroids xy: ',self.x_cen,self.y_cen)
                getLogger(__name__).debug('centroids xy: ',self.x_cen,self.y_cen)

        if xy_m:
            my_circle_scatter(ax, [self.x_m], [self.y_m], radius=0.2, alpha=1, color='r')
            if verbose==True:
                # print('max xy: ',self.x_m,self.y_m)
                getLogger(__name__).debug('max xy: ',self.x_m,self.y_m)

        my_circle_scatter(ax, [self.x_tile], [self.y_tile], radius=0.2, alpha=1, color='b')
        if legend==True:
            if cbar:ax.legend(loc='center left', bbox_to_anchor=(1.2, 0.5), borderaxespad=lpad)
            else:ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), borderaxespad=lpad)
        if tight==True:plt.tight_layout()
        if savename!=None:
            plt.savefig(savename)
            if verbose:
                # print('Saving ',savename)
                getLogger(__name__).debug('Saving ',savename)

        if showplot: 
            plt.show()
        if kill_plots: 
            fig.clf()
            plt.close()
        return(im)
