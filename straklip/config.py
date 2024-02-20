import inspect
import os,pathlib
import ruamel.yaml
import datetime
import sys
from dataframe import DataFrame
from ancillary import get_Av_dict
from stralog import getLogger
from acstools import acszpt
import numpy as np

try:
    from StringIO import StringIO
    import ConfigParser as configparser
except ImportError:
    import configparser
    from io import StringIO

yaml = ruamel.yaml.YAML()

class objectify(object):
    def __init__(self, d, multilevel=False):
        for k, v in d.items():
            if multilevel:
                if isinstance(k, (list, tuple)):
                    setattr(self, k, [objectify(x) if isinstance(x, dict) else x for x in v])
                else:
                    setattr(self, k, objectify(v) if isinstance(v, dict) else v)
            else:
                setattr(self, k, v)

def dequote(v):
    """Change strings like "'foo'" to "foo"."""
    if (v[0] == v[-1]) and v.startswith(("'", '"')):
        return v[1:-1]
    else:
        return v

def load(file):
    if not isinstance(file, str):
        return file

    if file.lower().endswith(('yaml', 'yml')):
        with open(file, 'r') as f:
            ret = yaml.load(f)
        return ret

def loadoldconfig(cfgfile):
    """Get a configparser instance from an old-style config, including files that were handled by readdict"""
    cp = configparser.ConfigParser()
    try:
        cp.read(cfgfile)
    except configparser.MissingSectionHeaderError:
        #  Some files aren't configparser dicts, pretend they have a DEFUALTS section only
        with open(cfgfile, 'r') as f:
            data = f.readlines()

        for l in (l for l in data if l and l[0]!='#'):
            k, _, v =l.partition('=')
            if not k.strip():
                continue
            cp.set('DEFAULT', k.strip(), v.strip())
    return cp

def configure_pipeline(config_file,pipe_cfg='',data_cfg='',dt_string=''):
    """Load a pipeline config, configuring the pipeline. Any existing configuration will be replaced"""
    opening=(f'\n==========================================================================================================================\n'
                f'StraKLIP pipeline started at date and time: {dt_string}\n'
                f'Pipe_cfg: {pipe_cfg}\n'
                f'Data_cfg: {data_cfg}\n'
                f'==========================================================================================================================\n')
    getLogger(__name__).info(opening)
    config = load(config_file)
    config['name']=config_file
    for key in config.keys():
        try:
            if 'redo' in config[key]:
                if config['redo']:
                    config[key]['redo'] = True
        except: pass
        try:
            if 'debug' in config[key]:
                if config['debug']:
                    config[key]['debug'] = True
        except: pass
    return objectify(config)

def get_paths(config=None):
    """Returns a set of all the required paths from the pipeline config"""
    return list([config.paths['data'],config.paths['database'],config.paths['out'],config.paths['pam'],config.paths['pyklip']])#, config.paths['tmp']])

def verify_paths(config=None, return_missing=False):
    """
    If return_missing=True, returns a list of all the missing paths from the pipeline config. If return_missing=False
    then returns True if there are paths missing, and False if all paths are present.
    """
    paths = get_paths(config=config)
    missing = list(filter(lambda p: p and not os.path.exists(p), paths))
    return missing if return_missing else not bool(missing)


def make_paths(config=None,paths=None):
    """Creates all paths returned from get_paths that do not already exist."""
    if paths is None:
        paths = get_paths(config=config)
    if not isinstance(paths, list):
            paths = list([paths])
    for p in paths:
        if not os.path.exists(p):
            getLogger(__name__).info(f'Creating "{p}"')
            pathlib.Path(p).mkdir(parents=True, exist_ok=False)
        else:
            getLogger(__name__).info(f'"{p}" exists, and will not be created.')

def cannonizekey(k):
    """ Enforce cannonicity of config keys lowercase, no spaces (replace with underscore)"""
    return k.strip().lower().replace(' ', '_')


def cannonizevalue(v):
    """ Make v into a float or int if possible, else remove any extraneous quotation marks from strings """
    if isinstance(v, (float, int)):
        return v
    try:
        v = dequote(v)
    except:
        pass
    try:
        if '.' in v:
            return float(v)
    except:
        pass
    try:
        return int(v)
    except:
        pass
    return v

def configure_data(config_file,pipe_cfg):
    """Load a data config yaml"""
    config = load(config_file)
    config['name']=config_file
    default_data_labels = [k for k in config['mvs_table'].keys()]
    issue_report=validation_summary(pipe_cfg,default_data_labels,null_success=True)
    if issue_report:
        getLogger(__name__).critical(issue_report)
        sys.exit(1)
    else:
        getLogger(__name__).info('Validation of default labels and data successful!')
    return objectify(config)

def validation_summary(pipe_cfg,labels, null_success=False):
    """Nicely formats the errors returned by self.validate"""
    errors = validate(labels,pipe_cfg)
    if not errors:
        return '' if null_success else 'Validation Successful, no issues identified'


    return (f'Validation failed, please these issues fix before continuing.\n'
            f'=============================================================\n'
            f'The following default label are missing from the data.yaml:\n {errors}\n'
            f'=============================================================\n')

def validate(labels,pipe_cfg):
    """
    Ensures that there is no missing or ill-defined data in the data configuration. Returns True if everything is
    good and all is associated. If error=True raise an exception instead of returning False
    """
    errors = []
    for x in pipe_cfg.buildhdf['default_mvs_table'].keys():
        if x not in list(labels):
            name = f'{x} ({repr(x)})' if x in errors else x
            errors.append(name)
    return errors

def closing_statement(DF,pipe_cfg,dataset):
    getLogger(__name__).info(f'===============================================================================================')
    getLogger(__name__).info(f'=============== Pipeline closing summary ======================================================')
    getLogger(__name__).info(f'Closing the pipeline after the following steps:')
    for step in pipe_cfg.flow:
        getLogger(__name__).info(f' - {step}')
    getLogger(__name__).info(f'==============================================================================================')

# def get_Av_dict(dataset):
#     getLogger(__name__).info(f'Fetching extinction dictionary for filters {dataset.data_cfg.filters}.')
#     if isinstance(dataset.data_cfg.target['AVs'],dict):
#         Av1_extinction=list(get_Av_list(dataset.data_cfg.filters,verbose=True,Rv=dataset.data_cfg.target['Rv'],path2saveim=dataset.pipe_cfg.paths['database']).values())
#     elif isinstance(dataset.data_cfg.target['AVs'],list):
#         getLogger(__name__).info(f'Loading Av extinctions at 1 magnitudes from data.yaml')
#         Av1_extinction=dataset.data_cfg.target['AVs']
#     else:
#         getLogger(__name__).critical(f'AVs needs to be either a dictionary for the get_Av_list to work, or a list of values. Please check.')
#         raise ValueError(f'AVs needs to be either a dictionary for the get_Av_list to work, or a list of values. Please check.')
#     return {dataset.data_cfg.filters[i]: Av1_extinction[i] for i in range(len(dataset.data_cfg.filters))}

def get_zpt(dataset):
    getLogger(__name__).info(f'Zero points not provided, fetch them from acstool')

    # # Define zero point list. For example, for ACS filters you can use this code
    # # Create an instance of the Query class
    # q = acszpt.Query(date=dataset.pipe_cfg.zpt['date'], detector=dataset.pipe_cfg.zpt['detector'])
    #
    # # Fetch the results for all filters
    # zpt_table = q.fetch()

    # Create an instance and search for a specific filter
    q_filter = acszpt.Query(date=dataset.data_cfg.target['zpts']['date'],
                            detector=dataset.pipe_cfg.instrument['name'],
                            filt=None)

    filter_zpt = q_filter.fetch().to_pandas()
    filter_zpt['Filter'] = filter_zpt['Filter'].astype(str).apply(str.lower)

    zpt_list = filter_zpt.loc[filter_zpt.Filter.astype(str).isin(dataset.data_cfg.filters)][dataset.data_cfg.target['zpts']['system']].tolist()
    return {dataset.data_cfg.filters[i]: zpt_list[i] for i in range(len(dataset.data_cfg.filters))}

def configure_dataframe(dataset,load=False):
    files_check_list = ['crossmatch_ids','avg_targets','mvs_targets','avg_candidates','mvs_candidates']
    if 'AVs' in dataset.data_cfg.target and isinstance(dataset.data_cfg.target['AVs'],dict):
        Av_dict = dataset.data_cfg.target['AVs']
    else:
        getLogger(__name__).warning('get_Av_dict currently only supports VEGAmag system. Please provide your own set of AVs if in a differest system as AVs : {ext: {mag_filter : value}} in the data.yaml under target')
        Av_dict = get_Av_dict(dataset.data_cfg.filters, verbose=True, Rv=dataset.data_cfg.target['Rv'],
                              path2saveim=dataset.pipe_cfg.paths['database'], band_dict=dataset.pipe_cfg.instrument['AVs'])

    if isinstance(dataset.data_cfg.target['zpts'],dict):
        zpt_dict = get_zpt(dataset)
    elif isinstance(dataset.data_cfg.target['zpts'],list):
        zpt_dict = {dataset.data_cfg.filters[i]: dataset.data_cfg.target['zpts'][i] for i in range(len(dataset.data_cfg.filters))}
    else:
        getLogger(__name__).critical(f'Zero points option from data.yaml as to be either a dictionary with "datetime.date" and "filters" obj or a filterwise list of values')
        raise ValueError

    DF = DataFrame(kind=dataset.data_cfg.target['kind'],
                   path2out=dataset.pipe_cfg.paths['out'],
                   path2data=dataset.pipe_cfg.paths['data'],
                   path2database=dataset.pipe_cfg.paths['database'],
                   path2pam=dataset.pipe_cfg.paths['pam'],
                   target=dataset.data_cfg.target['name'],
                   inst=dataset.pipe_cfg.instrument['name'],
                   pixscale=dataset.pipe_cfg.instrument['pixelscale'],
                   gain=dataset.pipe_cfg.instrument['gain'],
                   PAMdict={key: dataset.pipe_cfg.instrument['pam'][key] for key in list(dataset.pipe_cfg.instrument['pam'].keys())},
                   tilebase=dataset.pipe_cfg.mktiles['tile_base'],
                   radec=[dataset.pipe_cfg.buildhdf['default_avg_table']['ra'],dataset.pipe_cfg.buildhdf['default_avg_table']['dec']],
                   filters=[i for i in dataset.data_cfg.filters],
                   xyaxis={key: dataset.pipe_cfg.instrument['ccd_pix'][key] for key in list(dataset.pipe_cfg.instrument['ccd_pix'].keys())},
                   zpt=zpt_dict,
                   Av=Av_dict,
                   dist=dataset.data_cfg.target['distance'],
                   type=dataset.pipe_cfg.buildhdf['default_avg_table']['type'],
                   maxsep=dataset.pipe_cfg.mktiles['max_separation'],
                   minsep=dataset.pipe_cfg.mktiles['min_separation'],
                   steps=[],
                   fitsext=str(dataset.data_cfg.target['fitsext']),
                   dq2mask=list(dataset.pipe_cfg.buildhdf['dq2mask']),
                   kmodes=dataset.pipe_cfg.psfsubtraction['kmodes'])


    if 'buildhdf' not in dataset.pipe_cfg.flow or (not dataset.pipe_cfg.buildhdf['redo'] and np.all([os.path.exists(dataset.pipe_cfg.paths['out']+'/'+file+'.h5') for file in files_check_list])) or load:
        getLogger(__name__).info(f'Fetching dataframes from %s'%dataset.pipe_cfg.paths['out'])
        DF.load_dataframe()

    return(DF)
