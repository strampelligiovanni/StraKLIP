from straklip import config,input_tables
from datetime import datetime
from ancillary import get_Av_dict
from dataframe import DataFrame

pipe_cfg='/Users/gstrampelli/PycharmProjects/Giovanni/work/pipeline_logs/FFP/pipe.yaml'
data_cfg='/Users/gstrampelli/PycharmProjects/Giovanni/work/pipeline_logs/FFP/data.yaml'
pipe_cfg = config.configure_pipeline(pipe_cfg,pipe_cfg=pipe_cfg,data_cfg=data_cfg,dt_string=datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
data_cfg = config.configure_data(data_cfg,pipe_cfg)
DF = DataFrame(path2out=pipe_cfg.paths['out'])
DF.load_dataframe()
name = pipe_cfg.instrument['name'].lower()
detector = pipe_cfg.instrument['detector'].lower()
band_dict=pipe_cfg.instrument['AVs']

dataset = input_tables.Tables(data_cfg, pipe_cfg)
Av_dict = get_Av_dict(dataset.data_cfg.filters,verbose=True,Rv=dataset.data_cfg.target['Rv'],path2saveim=dataset.pipe_cfg.paths['database'],band_dict=band_dict)
print(Av_dict)