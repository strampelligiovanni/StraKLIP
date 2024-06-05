import sys,os,warnings,glob
sys.path.append('/Users/gstrampelli/PycharmProjects/Giovanni/src/StraKLIP/straklip')
sys.path.append('/Users/gstrampelli/PycharmProjects/Giovanni/src/StraKLIP/straklip/utils')
sys.path.append('/Users/gstrampelli/PycharmProjects/Giovanni/pyklip/pyklip')
sys.path.append('/Users/gstrampelli/PycharmProjects/Giovanni/src/StraKLIP/straklip/steps')
sys.path.append('/Users/gstrampelli/PycharmProjects/Giovanni/src/StraKLIP/straklip/')
from datetime import datetime
import config, input_tables

pipe_cfg='/Users/gstrampelli/PycharmProjects/Giovanni/work/pipeline_logs/FFP/pipe.yaml'
data_cfg='/Users/gstrampelli/PycharmProjects/Giovanni/work/pipeline_logs/FFP/data.yaml'

pipe_cfg = config.configure_pipeline(pipe_cfg,pipe_cfg=pipe_cfg,data_cfg=data_cfg,dt_string=datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
data_cfg = config.configure_data(data_cfg,pipe_cfg)

dataset = input_tables.Tables(data_cfg, pipe_cfg)
DF = config.configure_dataframe(dataset,load=True)