import json

import pika

from src.infra.configuration import Configuration


class Queue:
    """
    RabbitMQ queue manager for handling message publishing and consuming.
    
    This class provides a simplified interface for interacting with RabbitMQ message queues,
    handling connection establishment, message publishing, and consumer setup.
    """
    
    def __init__(self):
        """
        Initialize a connection to the RabbitMQ server using configuration settings.
        
        Establishes a blocking connection to the RabbitMQ server using credentials and
        host information retrieved from the application configuration.
        """
        config = Configuration().get()
        credentials = (
            pika.PlainCredentials(config.get('rabbitmq_user'),
                                  config.get('rabbitmq_pass')))

        self.connection = pika.BlockingConnection(
            pika.ConnectionParameters(
                host=config.get('rabbitmq_host'),
                credentials=credentials))

    def publish(self, queue, exchange, message_content, delay=0):
        """
        Publish a message to the specified queue and exchange.
        
        Serializes the message content to JSON and publishes it to the RabbitMQ server,
        then closes the connection.
        
        Args:
            queue (str): The name of the queue to target (used as routing key)
            exchange (str): The exchange to publish the message to
            message_content (dict): The message payload to be serialized to JSON
            
        Raises:
            Exception: Prints error message if publishing fails
        """
        try:
            channel = self.connection.channel()
            channel.basic_publish(
                exchange=exchange,
                routing_key=queue,
                body=json.dumps(message_content),
                properties=pika.BasicProperties(delivery_mode=2)
            )
            self.close()
        except Exception as e:
            print("ERROR ON PUBLISH MESSAGE")
            print(e)

    def consume(self, queue, callback):
        """
        Set up a consumer to process messages from the specified queue.
        
        Configures a channel to consume messages from the queue with the provided
        callback function for message processing. Quality of service is set to
        prefetch one message at a time.
        
        Args:
            queue (str): The name of the queue to consume messages from
            callback (callable): Function to be called when a message is received
            
        Raises:
            Exception: Prints error message if consuming fails
        """
        try:
            channel = self.connection.channel()
            channel.basic_qos(prefetch_count=1)
            channel.basic_consume(queue, callback, auto_ack=False)

            channel.start_consuming()
            channel.stop_consuming()
            self.close()
        except Exception as e:
            print("ERROR ON CONSUME MESSAGE")
            print(e)

    def close(self):
        """
        Close the connection to the RabbitMQ server.
        
        Properly terminates the connection to free up resources.
        """
        self.connection.close()