"""
Use case for flattening publication data and indexing to Elasticsearch.

This module transforms hierarchical publication documents from MongoDB into
a denormalized format optimized for search and analytics in Elasticsearch.
"""

import sys
import logging

from src.infra.model.enum import Status

sys.stdout.flush()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import pyalex
from elasticsearch import Elasticsearch, helpers
import urllib3

from src import Configuration, PublicationRepository, DatasetRepository
from src.use_cases.flatten_publication.model import FlatPublication


class FlattenPublicationUseCase:
    """
    Use case for transforming and indexing publication data to Elasticsearch.
    
    This class handles the ETL process of taking publication documents from MongoDB,
    transforming them into a flat structure suitable for search operations, and
    indexing them into Elasticsearch for efficient querying and analytics.
    """
    
    def __init__(self):
        """
        Initialize the use case with Elasticsearch connection and configuration.
        
        Sets up the Elasticsearch client with API key authentication and configures
        the OpenAlex email for API access. SSL warnings are disabled for development.
        """
        config = Configuration().get()
        self.index_name = None
        self.publication_repository = None
        self.elastic = Elasticsearch(config['elastic_url'],
                                     api_key=config['elastic_api_key'],
                                     ca_certs=config['elastic_ca_certificate'],
                                     verify_certs=False)

        pyalex.config.email = config['openalex_email']
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    def execute(self, work) -> bool:
        config = Configuration().get()
        dataset_repository = DatasetRepository(config['mongo_general_database'])

        dataset = dataset_repository.find_by_id(work['task_id'])
        if dataset['status'] != Status.PROCESSING.name:
            return False

        self.index_name = f"dashboard_pediatric"

        self.publication_repository = PublicationRepository(work['origin'])
        publication = self.publication_repository.find_by_id(work['publication_id'])

        if publication is None:
            return True

        if 'sync_elastic'in publication and not publication['sync_elastic']:
            return True

        try:
            if self.elastic.exists(index=self.index_name, id=publication['id']):
                self.elastic.delete(index=self.index_name, id=publication['id'])

            publications = self.flatten_publication(publication)

            helpers.bulk(self.elastic, publications)
            
            publication['sync_elastic'] = True
            self.publication_repository.save(publication)
        except Exception as e:
            publication['sync_elastic'] = False
            self.publication_repository.save(publication)
            raise e

        print('Insert in Elastic is Done')

        return True

    def flatten_publication(self, publication) -> list:
        publication_elastic = FlatPublication().fromDBtoElastic(publication)
        results = [{"_index": self.index_name, "_source": publication_elastic}]
        return results
