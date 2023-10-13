from straklip import config
from tables import DataTable

pipe_cfg = '/Users/gstrampelli/PycharmProjects/Giovanni/work/pipeline_logs/NGC1976/pipe.yaml'
data_cfg = '/Users/gstrampelli/PycharmProjects/Giovanni/work/pipeline_logs/NGC1976/data.yaml'
table_csv = '/Users/gstrampelli/PycharmProjects/Giovanni/work/database/NGC1976/Full_ONC_ACS_phot.csv'

straklip_cfg = config.configure_pipeline(pipe_cfg)
datatable = DataTable(table_csv, data_cfg)
datatable