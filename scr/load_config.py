import yaml
import logging
logger = logging.getLogger(__name__)

def load_config(path='config.yaml'):
    try:
        with open(path ,'r') as read:
            return yaml.safe_load(read)
    except Exception as e:
        logger.exception('error in load_config')
        raise
