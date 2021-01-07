import sys
import traceback
import logging
from datetime import datetime
import os

date = datetime.now()
LOG_FILENAME = "Tmp/crash_report.out"
if not os.path.exists('Tmp'):
    os.makedirs('Tmp')
logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG)

logging.debug("Dit is een crash report op " + str(date))


def generic_exception_handler():
    print("------------------------")
    print("Programma heeft een exception")
    traceback.print_exception(*sys.exc_info())
    logging.exception('er is een exception geworpen, kijk hieronder voor de traceback')
    print("------------------------")
