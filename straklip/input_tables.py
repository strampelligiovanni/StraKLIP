import pandas as pd
from config import objectify, load
from stralog import getLogger
import sys,glob
import numpy as np
from astropy.io import fits

class Tables:
    def __init__(self,data_cfg, pipe_cfg):
        self.pipe_cfg = pipe_cfg
        self.data_cfg = data_cfg
        for table_name in ['avg_table','mvs_table']:
            self.load_table_into_df(table_name)
        self.select_tables()

    def select_tables(self):
        self.mvs_table=self.mvs_table.loc[
            ~self.mvs_table[self.mvs_table.columns[self.mvs_table.columns.str.contains('fits')]].isna().all(
                axis=1)]
        self.avg_table=self.avg_table.loc[self.avg_table.avg_ids.isin(self.mvs_table.avg_ids.unique())]
        self.crossmatch_ids_table=self.mvs_table[['avg_ids','mvs_ids']]
        self.mvs_table.drop('avg_ids',inplace=True,axis=1)

    def canonize(self,label_list=['vis']):
        """ Enforce cannonicity of df str values lowercase"""
        for label in label_list:
            self.mvs_table[label] = self.mvs_table[label].astype(str).str.lower()

    def rename_df(self,table,table_labels='mvs_table',default_labels='default_mvsdf_labels',new_labels_dict={}):
        selected_labels=[]
        for key in [x for x in list(getattr(self.data_cfg,table_labels).keys()) if x in list(self.pipe_cfg.buildhdf[default_labels].keys())]:
            if np.any(key in ['id','x','y']) and table_labels == 'mvs_table':
                if key =='id':
                    for elno in range(len(list(getattr(self.data_cfg, table_labels)[key].keys()))):
                        label = list(getattr(self.data_cfg, table_labels)[key].values())[elno]
                        newlabel = list(getattr(self.data_cfg, table_labels)[key].keys())[elno]
                        new_labels_dict[label] = newlabel
                        selected_labels.append(newlabel)
                else:
                    for elno in range(len(getattr(self.data_cfg,table_labels)[key])):
                        label = self.data_cfg.mvs_table[key][elno]
                        new_labels_dict[label]= self.pipe_cfg.buildhdf['default_mvs_table'][key]+'_%s'%self.data_cfg.filters[elno]
                        selected_labels.append(self.pipe_cfg.buildhdf['default_mvs_table'][key]+'_%s'%self.data_cfg.filters[elno])
            else:
                new_labels_dict[getattr(self.data_cfg,table_labels)[key]] = self.pipe_cfg.buildhdf[default_labels][key]
                selected_labels.append(self.pipe_cfg.buildhdf[default_labels][key])
        getattr(self,table).rename(columns=new_labels_dict,inplace=True)
        setattr(self,table,getattr(self,table)[selected_labels])
        getLogger(__name__).info(f'Renamed df columns to default')

    def load_table_into_df(self,table_name):
        setattr(self, table_name, pd.read_table(getattr(self.pipe_cfg,'paths')['database'] + '/' + getattr(self.data_cfg,table_name)['name'],
                                       sep=getattr(self.data_cfg,table_name)['sep'], skip_blank_lines=True).dropna(
            how='all').reset_index(drop=True))
        getLogger(__name__).info('Loaded "%s" into df' % getattr(self.data_cfg,table_name)['name'])
        self.rename_df(table=table_name, table_labels=table_name, default_labels='default_%s'%table_name,
                       new_labels_dict={})
        if table_name == 'mvs_table':
            self.ancillary_info()

    def check_fits_file_existence(self):
        path = self.pipe_cfg.paths['data']
        fits_list = glob.glob(path + '/*.fits')
        if len(fits_list) == 0:
            getLogger(__name__).critical(f'No fits files found in {path}.')
            raise ValueError(f'No fits files found in {path}.')
        else:
            getLogger(__name__).info(f'{len(fits_list)} fits files found in {path}.')
            return(fits_list)


    def ancillary_info(self):
        self.canonize()
        for label in ['rota','pav3','exptime','fits']:
            for filter in self.data_cfg.filters:
                if label =='fits':
                    d = ''
                else:
                    d = np.nan
                self.mvs_table['%s_%s' % (label,filter.lower())] = np.nan
        fits_list=self.check_fits_file_existence()
        for file in fits_list:
            hdul = fits.open(file)
            filename = hdul[0].header['ROOTNAME']
            vis = filename[4:6].lower()
            try:
                filter1 = hdul[0].header['FILTER1'].lower()
                filter2 = hdul[0].header['FILTER2'].lower()
            except:
                filter1 = hdul[0].header['FILTER'].lower()
                filter2=''
            if np.any([filter in self.data_cfg.filters for filter in [filter1,filter2]]):
                    if filter1[0] == 'f':
                        filter = filter1
                    elif filter2[0] == 'f':
                        filter = filter2

                    EXPTIME = hdul[0].header['EXPTIME']
                    PA_V3 = hdul[0].header['PA_V3']
                    ROTA = hdul[1].header['ORIENTAT']
            #
                    self.mvs_table.loc[(self.mvs_table.vis==str(vis).lower()),['rota_'+filter.lower()]] = round(float(ROTA), 3)
                    self.mvs_table.loc[(self.mvs_table.vis==str(vis).lower()),['pav3_'+filter.lower()]] = round(float(PA_V3), 3)
                    self.mvs_table.loc[(self.mvs_table.vis==str(vis).lower()),['exptime_'+filter.lower()]] = round(float(EXPTIME), 3)
                    self.mvs_table.loc[(self.mvs_table.vis==str(vis).lower()),['fits_'+filter.lower()]] = filename
        getLogger(__name__).info(f'Added ancillary info to df')


