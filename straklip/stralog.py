import logging, yaml, os
import logging.config
import logging.handlers
from multiprocessing_logging import install_mp_handler
import multiprocessing, logging

LOG_FILE_ENV_VAR = 'SHARED_LOG_FILE'
LEVELS = [logging.DEBUG, logging.INFO, logging.WARNING,
          logging.ERROR, logging.CRITICAL]


def getLogger(args,**kwargs):
    setup = kwargs.pop('setup', False)
    logfile = kwargs.pop('logfile', None)
    configfile = kwargs.pop('configfile', None)

    if setup:
        os.environ[LOG_FILE_ENV_VAR] = logfile
        setupLogger(configfile)#,logfile)

    log = logging.getLogger(args)
    if not setup:
        log.addHandler(logging.NullHandler())
    return log

def setupLogger(configfile):#,logfile):
    with open(configfile, 'r') as f:
        config = yaml.safe_load(f.read())
    # if logfile:
    config['handlers']['file']['filename'] = os.environ[LOG_FILE_ENV_VAR]
    logging.config.dictConfig(config)
    logging.captureWarnings(True)
    # install_mp_handler()


class MakeFileHandler(logging.FileHandler):
    def __init__(self, filename, mode='a', encoding=None, delay=0):
        mkdir_p(os.path.dirname(filename))
        logging.FileHandler.__init__(self, filename, mode, encoding, delay)

def mkdir_p(path):
    """http://stackoverflow.com/a/600612/190597 (tzot)"""
    if not path:
        return
    try:
        os.makedirs(path, exist_ok=True)  # Python>3.2
    except TypeError:
        try:
            os.makedirs(path)
        except OSError as exc:  # Python >2.5
            if exc.errno == errno.EEXIST and os.path.isdir(path):
                pass
            else:
                raise


# import logging
# import os
# import datetime
#
# def getLogger(name):
#     pass
#
# LOG_FILE_ENV_VAR = 'SHARED_LOG_FILE'
#
# def setup_logging():
#     log_file = os.getenv(LOG_FILE_ENV_VAR)
#     if not log_file:
#         # Generate a log file name based on the current time
#         log_file = f"straklip_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
#         # Save the log file name to an environment variable
#         os.environ[LOG_FILE_ENV_VAR] = log_file
#
#     logging.basicConfig(
#         filename=log_file,
#         filemode='a',  # Append mode
#         format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#         handlers=[
#             logging.FileHandler(log_file),
#             logging.StreamHandler()],  # Optional: To also log to the console
#         level=logging.DEBUG  # or another level, e.g., INFO, WARNING
#         )

