import os
from dotenv import load_dotenv, find_dotenv


class Configuration:
    """
    Application configuration manager that loads environment variables.
    
    This class is responsible for loading environment variables from a .env file
    and making them available for use in the application through a structured dictionary.
    It centralizes access to all configuration parameters, facilitating maintenance
    and control of external dependencies.
    """

    def __init__(self):
        """
        Initializes the configuration by loading environment variables.
        
        Uses the dotenv package to automatically locate the closest .env file
        in the directory hierarchy and load its definitions as environment variables.
        """
        load_dotenv(find_dotenv())

    def get(self):
        """
        Returns a dictionary with all application configurations.
        
        Retrieves values from environment variables and organizes them into a structured dictionary,
        logically grouping them according to their purpose:
        - RabbitMQ configurations for messaging system
        - Queues and exchanges for different types of data (publications, datasets, institutions)
        - Credentials and endpoints for external APIs (OpenAlex, Dimensions)
        - MongoDB and Elasticsearch database configurations
        - Authentication and network parameters for the API
        
        Returns:
            dict: Dictionary containing all application settings organized by keys.
        """
        return {
            # RabbitMQ connection settings
            'rabbitmq_host': os.getenv('RABBITMQ_HOST'),
            'rabbitmq_user': os.getenv('RABBITMQ_USER'),
            'rabbitmq_pass': os.getenv('RABBITMQ_PASS'),
            
            # Publication queues and exchanges configuration
            'publication_queue': os.getenv('PUBLICATION_QUEUE'),
            'publication_exchange': os.getenv('PUBLICATION_EXCHANGE'),

            'process_publication_queue': os.getenv('PROCESS_PUBLICATION_QUEUE'),
            'process_publication_exchange': os.getenv('PROCESS_PUBLICATION_EXCHANGE'),

            'publication_error_queue': os.getenv('PUBLICATION_ERROR_QUEUE'),
            'publication_error_exchange': os.getenv('PUBLICATION_ERROR_EXCHANGE'),
            
            # Dataset queues and exchanges configuration
            'dataset_queue': os.getenv('DATASET_QUEUE'),
            'dataset_exchange': os.getenv('DATASET_EXCHANGE'),

            # Pipeline start queues and exchanges configuration
            'pipeline_start_queue': os.getenv('PIPELINE_START_QUEUE'),
            'pipeline_start_exchange': os.getenv('PIPELINE_START_EXCHANGE'),
            
            # Institution queues and exchanges configuration
            'institution_queue': os.getenv('INSTITUTION_QUEUE'),
            'institution_exchange': os.getenv('INSTITUTION_EXCHANGE'),
            
            # OpenAlex API configuration
            'openalex_email': os.getenv('PYALEX_EMAIL'),
            
            # MongoDB configuration
            'mongodb_conn': os.getenv('MONGODB_CONN'),
            'mongodb_publications_table': os.getenv('MONGODB_PUBLICATIONS_TABLE'),
            'mongodb_authors_table': os.getenv('MONGODB_AUTHORS_TABLE'),
            'mongodb_datasets_table': os.getenv('MONGODB_DATASETS_TABLE'),
            'mongodb_institutions_table': os.getenv('MONGODB_INSTITUTIONS_TABLE'),
            'mongodb_topics_table': os.getenv('MONGODB_TOPICS_TABLE'),
            'mongodb_journals_table': os.getenv('MONGODB_JOURNALS_TABLE'),

            'dashboard_url': os.getenv('DASHBOARD_URL'),
            
            # Elasticsearch configuration
            'elastic_url': os.getenv('ELASTIC_URL'),
            'elastic_api_key': os.getenv('ELASTIC_API_KEY'),
            'elastic_ca_certificate': os.getenv('ELASTIC_CA_CERTIFICATE'),
            
            # API basic authentication settings
            'basic_password': os.getenv('BASIC_PASSWORD'),
            'basic_username': os.getenv('BASIC_USERNAME'),
            
            # Network configuration
            'port': os.getenv('PORT'),
            
            # Dimensions API configuration
            'dimensions_api_key': os.getenv('DIMENSIONS_API_KEY'),
            'dimensions_endpoint': os.getenv('DIMENSIONS_ENDPOINT'),

            # General database configuration
            'mongo_general_database': os.getenv('MONGO_GENERAL_DATABASE')
        }