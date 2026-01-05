import json
import time
import sys
import traceback

from src.use_cases.finish_process_notification import FinishProcessNotificationUseCase

sys.stdout.flush()
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from src import Configuration, Queue


def callback(ch, method, properties, body):
    try:
        logger.info(f"Received {str(body)}")

        use_case = FinishProcessNotificationUseCase()

        message = json.loads(body)

        use_case.execute(message)

        ch.basic_ack(delivery_tag=method.delivery_tag)

    except Exception as e:
        logger.error('ERROR ON PROCESS MESSAGE')
        logger.error(traceback.format_exc())

def go():
    while True:
        logger.info('Waiting for messages')

        Queue().consume(config['pipeline_start_queue'], callback)

        time.sleep(1)

config = Configuration().get()
go()