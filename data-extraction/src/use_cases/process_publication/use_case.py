"""
Use case for processing and storing publication data.

This module handles the persistence of publication records and their associated
authors, managing deduplication and dataset association for publications
discovered through the dataset search pipeline.
"""

from src import Configuration, Queue
from src.infra.repository import PublicationRepository, DatasetRepository, AuthorRepository


class ProcessPublicationsUseCase:
    """
    Use case for processing and persisting publication data.
    
    This class manages the storage of publication records, handling both new
    publications and updates to existing ones. It ensures proper deduplication
    of authors and manages the association of datasets with publications.
    """
    
    def __init__(self):
        """
        Initialize the use case with required repositories and configuration.
        
        Sets up database connections for publications, authors, and datasets.
        """
        self.config = Configuration().get()

        self.publication_repository = None
        self.author_repository = None
        self.dataset_repository = DatasetRepository(self.config['mongo_general_database'])

    def execute(self, publication) -> None:
        """
        Process and store a publication with its associated data.
        
        Handles both new publications and updates to existing ones. For new
        publications, stores the publication and its authors. For existing
        publications, appends new dataset references without duplicating.
        
        Args:
            publication: Dictionary containing publication data and metadata
                including publication_id, origin, and the publication content.
        """
        self.publication_repository = PublicationRepository(publication['origin'])
        self.author_repository = AuthorRepository(publication['origin'])
        
        pub_id = publication.get('publication_id', '')
        
        publication_db = self.publication_repository.find_by_id(pub_id)

        if publication_db is None:
            self.publication_repository.save(publication['publication'])

            for author in list(publication['publication']['authors']):
                check_if_author_exists = self.author_repository.find_by_id(author['id'])
                if check_if_author_exists is None:
                    self.author_repository.save(author)
        else:
            dataset_ids = [x['id'] for x in publication_db['mentioned_datasets']]
            for dataset in publication['publication']['mentioned_datasets']:
                if dataset['id'] not in dataset_ids:
                    publication_db['mentioned_datasets'].append(dataset)
            
            self.publication_repository.save(publication_db)

        Queue().publish(
            self.config['institution_queue'],
            self.config['institution_exchange'],
            {
                'task_id': publication['task_id'],
                'publication_id': pub_id,
                'origin': publication['origin'],
                'group_id': publication['group_id'],
                'dataset': publication['dataset']
            }
        )

