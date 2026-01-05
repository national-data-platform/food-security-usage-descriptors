import json
import time
import traceback
import sys

from src.use_cases.process_institution import ProcessInstitutionUseCase

sys.stdout.flush()
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from src import Configuration, Queue


def callback(ch, method, properties, body):
    """
    Process institution data messages consumed from the message queue.
    
    This callback function handles incoming messages containing institution data
    that needs to be processed and stored in the system. It orchestrates the 
    complete processing workflow, from message deserialization to use case execution
    and queue acknowledgment.
    
    Args:
        ch: RabbitMQ channel object for communication with the message broker
        method: Delivery metadata containing the delivery tag for acknowledgment
        properties: Message properties including headers and metadata
        body: Raw message payload as bytes containing JSON-encoded institution data
    
    The function follows a structured workflow:
    1. Log the received message for monitoring and debugging
    2. Initialize the institution processing use case
    3. Deserialize the JSON message payload
    4. Execute the core business logic through the use case
    5. Acknowledge successful processing to the message broker
    
    Exception handling ensures system resilience - any processing errors are
    logged in detail while leaving the message unacknowledged, allowing for
    automatic redelivery according to the broker's retry policies.
    """
    try:
        # Log receipt of message for monitoring and traceability
        logger.info(f"Received {str(body)}")

        # Initialize the domain use case responsible for institution processing
        use_case = ProcessInstitutionUseCase()

        # Parse the message payload from JSON to a Python dictionary
        message = json.loads(body)

        # Execute the core business logic to process the institution data
        use_case.execute(message)

        # Signal successful processing to the message broker
        ch.basic_ack(delivery_tag=method.delivery_tag)

    except Exception as e:
        # Comprehensive error logging for operational monitoring and debugging
        logger.error('ERROR ON PROCESS MESSAGE')
        logger.error(traceback.format_exc())

def go():
    """
    Establish a persistent connection to the message queue and process institution data.
    
    This function serves as the main entry point for the institution processing worker.
    It implements a continuous processing loop that:
    
    1. Connects to the message broker using configuration settings
    2. Subscribes to the institution-specific message queue
    3. Processes incoming messages via the callback function
    4. Handles connection failures with automatic reconnection
    
    The worker is designed for high availability, automatically recovering from
    network disruptions or broker outages with a controlled retry mechanism.
    It implements a fault-tolerant consumer pattern, ensuring no institution
    data is lost even during temporary system failures.
    
    The function runs indefinitely until manually terminated, maintaining
    a continuous institution data processing pipeline within the system.
    """
    # Load application configuration with broker connection details and queue names
    config = Configuration().get()

    # Implement a persistent processing loop with automatic recovery
    while True:
        # Log worker status for operational monitoring
        logger.info('Waiting for messages')

        # Connect to the broker and begin consuming from the institution-specific queue
        Queue().consume(config.get('institution_queue'), callback)

        # Implement controlled reconnection delay to prevent excessive reconnection attempts
        # during broker outages or network disruptions
        time.sleep(5)

# Entry point - initialize the institution processing worker
# This triggers the continuous processing loop for handling institution data messages
go()