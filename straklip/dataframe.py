'''
This module implements the DataFrame class, which manages the storage and
manipulation of the data tables (multivisits detections, average visits detections, stamps, photometry, psf subtraction, false positives, etc) 
through  all the pipeline
'''

from astropy.io import fits
from utils.ancillary import keys_list_from_dic
from pathlib import PurePath,Path
import pandas as pd
from stralog import getLogger
from astropy.table import Table
from glob import glob

DEFAULT_FLOW=['buildhdf','tiles']

class DataFrame():
    # def __getstate__(self):
    #     return {"data": self.values, "columns": self.columns}

    def __init__(self,path2data='',path2out='',path2database='',path2pam='',target='',inst='',pixscale=1,gain=1,PAMdict={},tilebase=15,radec=[],filters=[],xyaxis=[],fitsext='_flt',skipphot=False,dq2mask=[],zpt={},Av={},dist=0,kmodes=[],type='type',maxsep=2,minsep=0,kind='dataframe',steps=[]):
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

        Returns
        -------
        None.

        '''
        self.kind=kind
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
        getLogger(__name__).info(f'Saving the the following keys in %s to %s in %s'%(self.keys,self.kind,str(self.path2out)))
        for elno in range(len(self.keys)):
            key = self.keys[elno]
            filename=str(self.path2out + '/' + key.split('_df')[0] + '.h5')
            if key == 'crossmatch_ids_df':
                for label in vars(self).keys():
                    if '_df' not in label:
                        getattr(self,key).attrs[label] = vars(self)[label]

            # try:

            getattr(self,key).to_hdf(filename, key=key, mode='w')

            # with pd.HDFStore(filename) as store:
            #     store.put(key.split('_df')[0], getattr(self,key), format='table')
                    # if key == 'crossmatch_ids_df':
                    #     store.get_storer(key.split('_df')[0]).attrs.metadata = getattr(self,key).attrs

            # except:
            #     getLogger(__name__).critical(f'Saving of %s failed. Abort' % key)
            #     raise ValueError

    def load_dataframe(self):
        '''
        Load DataFrame from file

        Returns
        -------
        None.

        '''
        self.list_of_HDF5_keys(self.path2out)
        for key in self.keys:
            filename = self.path2out+'/'+key+'.h5'
            setattr(self, key+'_df', pd.read_hdf(filename, mode='r'))

            # with pd.HDFStore(self.path2out+'/'+key+'.h5') as store:
            #     # if key == 'crossmatch_ids':
            #     #     metadata = store.get_storer(key).attrs
            #     df = store.get(key)
            #
            #     setattr(self, key+'_df', df)

        # for key in metadata.metadata.keys():
        #     setattr(self, key, metadata.metadata[key])

    def list_of_HDF5_keys(self,path):
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
        # if self.kind == 'table':
        #     ext='.fits'
        # elif self.kind == 'dataframe':
        #     ext='.h5'
        # else:
        #     getLogger(__name__).critical()
        #     raise ValueError()
        file = glob(path+'/*.h5')
        self.keys = []
        for name in file:
            self.keys.append(name.split('/')[-1].split('.')[0])
        # with pd.HDFStore(path) as store:
        #     keys = store.keys()
        #     if verbose:print('List of keys:',keys)
        #     self.keys=keys
        #     store.close()

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
