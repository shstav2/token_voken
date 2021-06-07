from subprocess import call
import logging
import sys

from common.display_utils import bool_to_symbol

logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s',
                     level=logging.INFO, stream=sys.stdout)

SUCCESS_RETURN_CODE = 0

def run_command(command):
    result = call(command, shell=True)
    success = result == SUCCESS_RETURN_CODE
    status_symbol = bool_to_symbol(success)
    logging.info(f'{status_symbol} {command}')
