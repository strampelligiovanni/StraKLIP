import concurrent.futures
import logging
# from logger_tt import setup_logging
from stralog import getLogger
from datetime import datetime
import pkg_resources as pkg
import warnings
from logging.handlers import RotatingFileHandler

logging.captureWarnings(True)

logger_file_handler = RotatingFileHandler(f'straklip_{datetime.now().strftime("%Y-%m-%d_%H%M")}.py.warnings.log')
logger_file_handler.setLevel(logging.DEBUG)

getLogger('straklip', setup=True, logfile=f'straklip_{datetime.now().strftime("%Y-%m-%d_%H%M")}.log',
                   configfile=pkg.resource_filename('straklip', './config/logging.yaml'))


# warnings_logger = getLogger(__name__)

warnings_logger = logging.getLogger("py.warnings")
warnings_logger.addHandler(logger_file_handler)

def logger_test(key):
    if key == 1:
        getLogger(__name__).info("yes")
    else:
        getLogger(__name__).error(f"{key} is not = 1")
        warnings.warn(f"{key} is not = 1")

def main():
    key_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 4, 5, 4, 3, 4, 5, 4, 3, 4, 5, 4, 3, 4, 3]
    # error_logger = setup_logging()

    with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
        executor.map(logger_test, key_list)


if __name__ == '__main__':
    main()
    warnings.warn(u'Warning test')
    getLogger(__name__).info(f'=========================================================')
    getLogger(__name__).info(f'Closing the pipeline after finishing the following steps:')
    getLogger(__name__).info(f'==========================================================')