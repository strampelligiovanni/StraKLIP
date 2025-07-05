'''
This module implements the DataFrame class, which manages the storage and
manipulation of the data tables (multivisits detections, average visits detections, stamps, photometry, psf subtraction, false positives, etc) 
through  all the pipeline
'''

from straklip.utils.ancillary import keys_list_from_dic
import pandas as pd
from straklip.stralog import getLogger
from glob import glob

class DataFrame():
    # def __getstate__(self):
    #     return {"data": self.values, "columns": self.columns}

    def __init__(self,path2data='',path2out='',path2database='',path2pam='',target='',inst='',pixscale=1,gain=1,PAMdict={},tilebase=15,radec=[],filters=[],xyaxis=[],fitsext='_flt',skipphot=False,dq2mask=[],zpt={},Av={},dist=0,kmodes=[],type='type',maxsep=2,minsep=0,df_ext_in='.csv',df_ext_out='.csv',steps=[]):
        '''
        Create the dataframe object

        Parameters
        ----------
        target : str, optional
            name of the target.
        inst : str, optional
            name of the instrument.
        pixscale : int, optional
            instrument pixscale.
        gain : int, optional
            instrument gain.
        PAMdict : dict, optional
            instrument Pixel Area Map dictionary linking each PAM to the image extenxtion.
        tilebase: int, optipbal
            side of the square tile.
        radec : list, optional
            list of ra/dec labels.
        filters : list, optional
            list of filters labels.
        xyaxis : list, optional
            list of xy axis length of the camera (in pixels).
        skipphot: bool, optional
            If True, skip pipeline aperture photometry and use the input catalog photometry instead. Default is False.
        dq2mask : TYPE, optional
            list of data quality values to mask in data array.
        fitsext: str,optional
            name extension of fits file. The default is flt.
        Av : list, optional
            list of Av=1 extinctions for each filter.
        kmodes : list, optional
            list of kmodes for PSF subtraction.
        dist : float, optional
            distance of the target in parsec. The default is 0.
        load : bool, optional
            automatically load existing Data Frame.
        name : str, optional
            name of the Data Frame.
        df_ext_in: str, optional
            specify the extension of the input Data Frame to be read. Default is '.csv'.
        df_ext_in: str, optional
            specify the extension of the output Data Frame to be saved. Default is '.csv'.
        Returns
        -------
        None.

        '''
        self.df_ext_in=df_ext_in
        self.df_ext_out=df_ext_out
        self.path2out=path2out
        self.path2data=path2data
        self.path2database=path2database
        self.path2pam=path2pam
        self.target=target
        self.inst=inst
        self.filters=filters
        self.radec=radec
        self.gain=gain
        self.PAMdict=PAMdict
        self.xyaxis=xyaxis
        self.fitsext=fitsext
        self.skipphot=skipphot
        self.dq2mask=dq2mask
        self.zpt=zpt
        self.Av=Av
        self.kmodes=kmodes
        self.pixscale=pixscale
        self.tilebase=tilebase
        self.dist=dist
        self.type=type
        self.maxsep=maxsep
        self.minsep=minsep
        self.steps=steps

    ######################
    # Ancillary routines #
    ######################
    def save_dataframes(self,step):
        '''
        Save DataFrame to file

        Parameters
        ----------

        Returns
        -------
        None.

        '''
        if step not in self.steps: self.steps.append(step)
        self.keys=keys_list_from_dic(self.__dict__,'_df')
        getLogger(__name__).info(f'Saving the the following keys in %s to %s files in %s'%(self.keys,self.df_ext_out,str(self.path2out)))
        for elno in range(len(self.keys)):
            key = self.keys[elno]
            filename=str(self.path2out + '/' + key.split('_df')[0] + self.df_ext_out)
            if key == 'crossmatch_ids_df':
                for label in vars(self).keys():
                    if '_df' not in label:
                        getattr(self,key).attrs[label] = vars(self)[label]

            if self.df_ext_out == '.h5':
                getattr(self,key).to_hdf(filename, key=key, mode='w')
            elif self.df_ext_out == '.csv':
                getattr(self,key).to_csv(filename, mode='w', encoding='utf-8-sig', index=False)
            else:
                getLogger(__name__).error(f'DataFrame extension {self.df_ext_out} not supported. Please use .h5 or .csv')

    def load_dataframe(self):
        '''
        Load DataFrame from file

        Returns
        -------
        None.

        '''
        self.list_of_HDF5_keys(self.path2out)
        for key in self.keys:
            filename = self.path2out+'/'+key+self.df_ext_in
            if self.df_ext_in == '.h5':
                df = pd.read_hdf(filename, mode='r')
            elif self.df_ext_in == '.csv':
                df = pd.read_csv(filename, encoding='utf-8-sig')
            else:
                getLogger(__name__).error(f'DataFrame extension {self.df_ext_in} not supported. Please use .h5 or .csv')

            setattr(self, key+'_df', df)


    def list_of_HDF5_keys(self,path, ext=None):
        '''
        generate list of keys in dataframe

        Parameters
        ----------
        verbose : bool, optional
            choose to show prints. The default is False.

        Returns
        -------
        None.

        '''
        if ext is None:
            file = glob(path + f'/*{self.df_ext_in}')
        else:
            file = glob(path + f'/*{ext}')

        self.keys = []
        for name in file:
            self.keys.append(name.split('/')[-1].split('.')[0])

    def remove_HDF5_key(self):
        '''
        remove keys from dataframe

        Returns
        -------
        None.

        '''
        with pd.HDFStore(self.df_path/self.name) as store:
            keys = store.keys()
            print('List of keys:',keys)
            key_pos = input("Enter position of the key to remove (i.e 1,2,3...):")
            if len(key_pos)>=1:
                key_pos=key_pos.split(',')
                for key_pos in key_pos:
                    key_name=keys[int(key_pos)-1]
                    out=input('Removing \'%s\' key from dataframe. Please confirm (y/n):'%key_name)
                    if out=='y': store.remove(str(key_name))
                    else:print('No change made')
            else: print('No change made')
            store.close()
        with pd.HDFStore(self.df_path/self.name) as store:
            keys = store.keys()
            print('List of keys:',keys)
            store.close()
