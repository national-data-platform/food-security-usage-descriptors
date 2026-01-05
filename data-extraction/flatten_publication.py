import json
import time
import sys
import traceback

from src.use_cases.flatten_publication import FlattenPublicationUseCase

sys.stdout.flush()
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from src import Configuration, Queue

def callback(ch, method, properties, body):
    """
    Process publication data messages for transformation into flattened format.
    
    This callback function handles publication data messages consumed from the message queue,
    transforming hierarchical publication structures into flattened representations suitable
    for search indexing and analytics. It represents a critical step in the ETL pipeline for
    scholarly data processing.
    
    Args:
        ch: RabbitMQ channel object providing communication interface with the broker
        method: Delivery metadata containing routing information and acknowledgment token
        properties: Message properties containing application-specific headers and attributes
        body: Raw message payload as bytes containing JSON-encoded publication data
    
    Processing workflow:
    1. Log the received message for operational monitoring
    2. Initialize the flattening use case for publication transformation
    3. Deserialize the message payload from JSON
    4. Execute the domain-specific flattening logic
    5. Acknowledge successful processing to the message broker
    
    Error handling strategy:
    - Exceptions during processing trigger redirection to a dedicated error queue
    - The original message is preserved intact for diagnostic analysis
    - Detailed error information is logged for operational monitoring
    - No acknowledgment is sent, allowing potential message redelivery based on broker configuration
    """
    try:
        # Record message receipt for operational traceability
        logger.info(f"Received {str(body)}")

        # Initialize the domain service responsible for transforming publication structure
        use_case = FlattenPublicationUseCase()

        # Convert binary payload to structured data object
        message = json.loads(body)

        # Execute the structural transformation of the publication data
        success = use_case.execute(message)

        if success:
            # Confirm successful processing to prevent message redelivery
            ch.basic_ack(delivery_tag=method.delivery_tag)
        else:
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)

    except Exception as e:
        # Implement dead-letter pattern by redirecting failed messages to dedicated error queue
        # This preserves problematic messages for later analysis without blocking main queue
        Queue().publish(
            config['publication_error_queue'],  # Dedicated error queue for publication processing
            config['publication_error_exchange'],  # Exchange for routing to error handlers
            json.loads(body)  # Original message preserved for diagnostic context
        )

        # Comprehensive error logging for operational monitoring and troubleshooting
        logger.error('ERROR ON PROCESS MESSAGE')
        logger.error(traceback.format_exc())

def go():
    """
    Establish and maintain a persistent consumer for publication transformation processing.
    
    This function implements a resilient consumer for the publication processing queue,
    creating a continuous pipeline for transforming publication data from hierarchical
    to flattened structures. It represents the main execution loop of the worker service,
    with built-in fault tolerance for broker connectivity issues.
    
    The function performs these key operations:
    1. Establishes connection to the message broker
    2. Configures consumption from the publication-specific queue
    3. Routes incoming messages to the specialized callback handler
    4. Implements automatic reconnection for fault tolerance
    
    Designed as a long-running service component, this function maintains high availability
    through error isolation and automatic recovery mechanisms, ensuring the publication
    transformation pipeline continues to process data even after temporary infrastructure
    disruptions.
    """
    while True:
        # Indicate worker availability for monitoring and operational status tracking
        logger.info('Waiting for messages')

        # Establish consumer channel for publication messages with specialized handler
        Queue().consume(config['publication_queue'], callback)

        # Implement controlled reconnection strategy to prevent connection storms
        # during broker outages or network instability periods
        time.sleep(1)

# Service entry point - load configuration and initiate the message processing loop
# This bootstraps the publication transformation worker service
config = Configuration().get()
go()