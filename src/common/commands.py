from subprocess import call
import logging
import sys

logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s',
                     level=logging.INFO, stream=sys.stdout)

SUCCESS_RETURN_CODE = 0

def run_command(command):
    result = call(command, shell=True)
    success = result == SUCCESS_RETURN_CODE
    status_symbol = '✅ ' if success else '❌'
    logging.info(f'{status_symbol} {command}')
