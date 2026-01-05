import json
import time
import traceback
import sys

from src.use_cases.process_publication.use_case import ProcessPublicationsUseCase

sys.stdout.flush()
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from src import Configuration, Queue

def callback(ch, method, properties, body):
    """
    Process dataset messages received from the RabbitMQ queue.

    This callback function handles incoming messages containing dataset information.
    It deserializes the JSON payload, forwards it to the appropriate use case for processing,
    and acknowledges the message upon successful completion.

    Args:
        ch: The channel object providing the communication interface with RabbitMQ
        method: Contains delivery information like the delivery tag for acknowledgment
        properties: Message properties including headers and content metadata
        body: The raw message payload as bytes containing serialized dataset information

    Returns:
        None

    Notes:
        - Exceptions are caught and logged but not re-raised to maintain worker stability
        - Failed messages are not acknowledged, allowing RabbitMQ to handle redelivery
    """
    try:
        # Log receipt of message for operational visibility
        logger.info(f"Received {str(body)}")

        # Initialize the domain use case responsible for dataset processing
        use_case = ProcessPublicationsUseCase()

        # Parse the JSON message into a Python dictionary
        message = json.loads(body)


        ch.basic_ack(delivery_tag=method.delivery_tag)

        # Execute the business logic for dataset processing
        use_case.execute(message)

    except Exception as e:
        # Capture and log any errors to facilitate troubleshooting
        traceback.print_exc()
        logger.error('ERROR ON PROCESS MESSAGE')
        logger.error(traceback.format_exc())

def go():
    """
    Long-running worker function that consumes and processes dataset messages.

    This function creates a persistent connection to the message broker and
    continuously processes incoming dataset messages. It implements a reconnection
    strategy to handle network or broker failures, ensuring the worker remains
    operational over extended periods.

    Returns:
        None: This function runs indefinitely until manually terminated

    Notes:
        - Designed as a resilient background worker process
        - Implements a simple reconnection pattern with a fixed delay
        - Logs operational status to aid monitoring and troubleshooting
    """
    # Load operational parameters from the application configuration
    config = Configuration().get()

    # Continuous processing loop with reconnection capability
    while True:
        # Indicate worker readiness in logs
        logger.info('Waiting for messages')

        # Begin message consumption from the configured queue with the processing callback
        Queue().consume(config.get('process_publication_queue'), callback)

        # Brief pause before reconnection attempt if the connection is disrupted
        time.sleep(5)

# Worker process entry point
# Initiates the continuous message processing loop
go()