"""
Use case for processing datasets and discovering related publications.

This module handles the core business logic for searching academic databases
(OpenAlex, Dimensions) to find publications that reference specific datasets,
and queuing them for further processing in the publication pipeline.
"""

import re

from src import Configuration, Queue
from src.infra.model.enum import Status
from src.infra.repository import PublicationRepository, DatasetRepository, AuthorRepository
from src.use_cases.process_datasets.dimensions import Dimensions
from src.use_cases.process_datasets.model import Publication
from src.use_cases.process_datasets.openalex import OpenAlex


class ProcessDatasetsUseCase:
    """
    Use case for discovering and processing publications related to datasets.
    
    This class orchestrates the search for scholarly publications that reference
    specific datasets across multiple academic databases. It handles query
    construction, result processing, and queuing publications for downstream
    analysis in the ETL pipeline.
    """
    
    def __init__(self):
        """
        Initialize the use case with required repositories and configuration.
        
        Sets up database connections for publications, authors, and datasets,
        loading configuration from environment variables.
        """
        self.config = Configuration().get()

        self.publication_repository = None
        self.author_repository = None
        self.dataset_repository = DatasetRepository(self.config['mongo_general_database'])

    def execute(self, dataset) -> None:
        self.publication_repository = PublicationRepository(dataset['origin'])
        self.author_repository = AuthorRepository(dataset['origin'])
        dataset_from_db = self.dataset_repository.find_by_id(dataset['task_id'])

        dataset_db = {
            "id": dataset_from_db['id'],
            "name": dataset_from_db['name'],
            "is_validated_mention": False,
            "context_quote": "",
            "confidence_score": 0,
            "group_id": dataset_from_db['group_id'],
            "group_name": dataset_from_db['group_name']
        }

        origin = OpenAlex()
        if dataset['origin'] == 'dimensions':
            origin = Dimensions()

        search_results, next_cursor = origin.execute(
            dataset,
            dataset['aliases'],
            dataset['flag_terms'],
            dataset['not_in_dataset'],
            dataset['start_year'],
            dataset['end_year']
        )

        results = []

        dataset_from_db['num_publications'] = len(search_results)
        dataset_from_db['status'] = Status.PROCESSING.name

        self.dataset_repository.save(dataset['task_id'], dataset_from_db)

        for _, row in search_results.iterrows():
            pub_id = row.get('id', '')
            if dataset['origin'] == 'openalex':
                pub_id = re.search(r'W\d+', row.get('id', '')).group()
            
            if dataset['origin'] == 'dimensions':
                publication = Publication().dimensionsToJSON(row)
            else:
                publication = Publication().openalexToJSON(row)

            publication.add_dataset(dataset_db)
            
            Queue().publish(
                self.config['process_publication_queue'],
                self.config['process_publication_exchange'],
                {
                    'task_id': dataset['task_id'],
                    'publication_id': pub_id,
                    'origin': dataset['origin'],
                    'group_id': dataset['group_id'],
                    'dataset': dataset['main_name'],
                    'publication': publication.toJSON()
                }
            )

        (journals_citations_count,
         topics_citations_count,
         authors_citations_count,
         institutions_citations_count,
         category_for_citations_count) = self.publication_repository.get_citations_count(dataset['task_id'])

        publications_count = self.publication_repository.count(dataset['task_id'])

        dataset_from_db['num_publications'] = publications_count
        dataset_from_db['num_publications']= publications_count
        dataset_from_db['num_authors']= len(authors_citations_count)
        dataset_from_db['num_institutions']= len(institutions_citations_count)
        dataset_from_db['num_journals']= len(journals_citations_count)

        dataset_from_db['status'] = Status.PROCESSING.name

        self.dataset_repository.save(dataset['task_id'], dataset_from_db)
