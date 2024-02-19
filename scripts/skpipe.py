#!/usr/bin/env python3
import sys,logging
from stralog import getLogger
from straklip import config, input_tables
import pkg_resources as pkg
import argparse
from datetime import datetime
import steps
from steps import buildhdf, mktiles, mkphotometry,fow2cells,psfsubtraction,klipphotometry

def parse():
    # read in command line arguments
    parser = argparse.ArgumentParser(description='StraKLIP Pipeline')
    parser.add_argument('-p', type=str, help='A pipeline config file', default='pipe.yaml', dest='pipe_cfg')
    parser.add_argument('-d', type=str, help='A input data yaml', default='data.yaml', dest='data_cfg')
    parser.add_argument('--vet',  action='store_true', help='Check pipeline configuration and exit', default=False)
    parser.add_argument('--verbose', action='store_true', help='Verbose')
    parser.add_argument('--make-dir', dest='make_paths', help='Create all needed directories', action='store_true')
    parser.add_argument('--logcfg', dest='logcfg', help='Run the pipeline on the outputs', type=str,
                        default=pkg.resource_filename('straklip', './config/logging.yaml'))

    return parser.parse_args()

getLogger('straklip', setup=True, logfile=f'straklip_{datetime.now().strftime("%Y-%m-%d_%H%M")}.log',
                   configfile=pkg.resource_filename('straklip', './config/logging.yaml'))


if __name__ == "__main__":
    args = parse()

    pipe_cfg = config.configure_pipeline(args.pipe_cfg,pipe_cfg=args.pipe_cfg,data_cfg=args.data_cfg,dt_string=datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
    data_cfg = config.configure_data(args.data_cfg,pipe_cfg)

    if args.make_paths:
        config.make_paths(config=pipe_cfg)

    missing_paths = config.verify_paths(config=pipe_cfg, return_missing=True)
    if missing_paths:
        getLogger(__name__).critical(f'Required paths missing:\n\t'+'\n\t'.join(missing_paths))
        sys.exit(1)

    dataset = input_tables.Tables(data_cfg, pipe_cfg)
    DF = config.configure_dataframe(dataset)
    for step in pipe_cfg.flow:
        if step == 'fow2cells':
            getattr(steps, step).run({'DF': DF, 'dataset': dataset})
        else:
            if step in DF.steps:
                if getattr(pipe_cfg, step)['redo']:
                    getattr(steps, step).run({'DF': DF, 'dataset': dataset})
                else:
                    getLogger('straklip').info(f'Skipping step "{step}" because is already been succesfully run for this dataframe and redo is False')
            else:
                getattr(steps, step).run({'DF': DF, 'dataset': dataset})

    config.closing_statement(DF,pipe_cfg,dataset)

