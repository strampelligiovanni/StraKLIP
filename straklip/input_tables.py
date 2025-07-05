import pandas as pd
from straklip.stralog import getLogger
import glob
import numpy as np
from astropy.io import fits

class Tables:
    def __init__(self,data_cfg, pipe_cfg, skip_originals=False):
        self.pipe_cfg = pipe_cfg
        self.data_cfg = data_cfg
        if not skip_originals:
            for table_name in ['mvs_table','unq_table']:
                self.load_table_into_df(table_name)
            self.select_tables()

    def select_tables(self):
        self.mvs_table=self.mvs_table.loc[
            ~self.mvs_table[self.mvs_table.columns[self.mvs_table.columns.str.contains('fits')]].isna().all(
                axis=1)]
        self.unq_table=self.unq_table.loc[self.unq_table.unq_ids.isin(self.mvs_table.unq_ids.unique())]
        self.crossmatch_ids_table=self.mvs_table[['unq_ids','mvs_ids']]
        self.mvs_table.drop('unq_ids',inplace=True,axis=1)

    def canonize(self,label_list=['vis']):
        """ Enforce cannonicity of df str values lowercase"""
        for label in label_list:
            self.mvs_table[label] = self.mvs_table[label].astype(str).str.lower()

    def rename_df(self,table,table_labels='mvs_table',default_labels='default_mvs_table',new_labels_dict={}):
        selected_labels=[]
        for key in [x for x in list(getattr(self.data_cfg,table_labels).keys()) if x in list(self.pipe_cfg.buildhdf[default_labels].keys())]:
            if table_labels == 'mvs_table' and key =='id':
                    for elno in range(len(list(getattr(self.data_cfg, table_labels)[key].keys()))):
                        label = list(getattr(self.data_cfg, table_labels)[key].values())[elno]
                        newlabel = list(getattr(self.data_cfg, table_labels)[key].keys())[elno]
                        new_labels_dict[label] = newlabel
                        selected_labels.append(newlabel)
            elif isinstance(getattr(self.data_cfg, table_labels)[key],list):
                for elno in range(len(getattr(self.data_cfg,table_labels)[key])):
                    label =getattr(self.data_cfg,table_labels)[key][elno]
                    new_labels_dict[label]= self.pipe_cfg.buildhdf[default_labels][key]+'_%s'%self.data_cfg.filters[elno]
                    selected_labels.append(self.pipe_cfg.buildhdf[default_labels][key]+'_%s'%self.data_cfg.filters[elno])
            else:
                new_labels_dict[getattr(self.data_cfg,table_labels)[key]] = self.pipe_cfg.buildhdf[default_labels][key]
                selected_labels.append(self.pipe_cfg.buildhdf[default_labels][key])
        getattr(self,table).rename(columns=new_labels_dict,inplace=True)
        setattr(self,table,getattr(self,table)[selected_labels])
        getLogger(__name__).info(f'Renamed df columns to default')

    def load_table_into_df(self,table_name):
        if table_name == 'mvs_table':
            setattr(self, table_name, pd.read_table(getattr(self.pipe_cfg,'paths')['database'] + '/' +
                                                    getattr(self.data_cfg,table_name)['name'],
                                                    sep=getattr(self.data_cfg,table_name)['sep'], skip_blank_lines=True,
                                                    converters={getattr(self.data_cfg, table_name)['id']['unq_ids']: int,
                                                    getattr(self.data_cfg, table_name)['id']['mvs_ids']: int,
                                                    getattr(self.data_cfg, table_name)['vis']: str}
                                                    ).dropna(
                                                    how='all').reset_index(drop=True))
        else:
            setattr(self, table_name, pd.read_table(
                getattr(self.pipe_cfg, 'paths')['database'] + '/' + getattr(self.data_cfg, table_name)['name'],
                sep=getattr(self.data_cfg, table_name)['sep'], skip_blank_lines=True,
                converters = {getattr(self.data_cfg, table_name)['id']: int,
                              getattr(self.data_cfg, table_name)['type']: int}
                ).dropna(
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
        for label in ['rota','pav3','exptime']:
            for filter in self.data_cfg.filters:
                if not np.any(self.mvs_table.columns.str.contains('%s_%s' % (label,filter.lower()))):
                    self.mvs_table['%s_%s' % (label,filter.lower())] = np.nan
        fits_list=self.check_fits_file_existence()
        for file in fits_list:
            hdul = fits.open(file)
            filename = hdul[0].header['ROOTNAME']
            vis = filename[4:6].lower()
            ext = 1 if hdul[1].header['CCDCHIP'] == 2 else 4
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
                    self.mvs_table.loc[(self.mvs_table.vis==str(vis).lower())&(self.mvs_table.ext==ext)&(self.mvs_table['fits_'+filter.lower()]==filename),['rota_'+filter.lower()]] = round(float(ROTA), 3)
                    self.mvs_table.loc[(self.mvs_table.vis==str(vis).lower())&(self.mvs_table.ext==ext)&(self.mvs_table['fits_'+filter.lower()]==filename),['pav3_'+filter.lower()]] = round(float(PA_V3), 3)
                    self.mvs_table.loc[(self.mvs_table.vis==str(vis).lower())&(self.mvs_table.ext==ext)&(self.mvs_table['fits_'+filter.lower()]==filename),['exptime_'+filter.lower()]] = round(float(EXPTIME), 3)
                    # self.mvs_table.loc[(self.mvs_table.vis==str(vis).lower()),['fits_'+filter.lower()]] = filename
        getLogger(__name__).info(f'Added ancillary info to df')


