from typing import Union, Any
import urllib.parse

from src import Configuration, DatasetRepository, PublicationRepository
from src.infra.model.enum import Status


class GetDatasetStatusUseCase:
    def __init__(self):
        self.config = Configuration().get()
        self.dataset_repository = DatasetRepository(self.config['mongo_general_database'])
        self.publication_repository = None

    def execute(self, task_id) -> dict[str, Union[str, Any]]:
        try:
            result_db = self.dataset_repository.find_by_id(task_id)

            result = {
                'task_id': task_id,
                'status': result_db['status'],
                'summary': {
                    'num_publications': result_db['num_publications'],
                    'num_authors': result_db['num_authors'],
                    'num_institutions': result_db['num_institutions'],
                    'num_journals': result_db['num_journals']
                }
            }

            if result_db['status'] == Status.PROCESSING.name:

                self.publication_repository = PublicationRepository(result_db['origin'])

                processed_publications = self.publication_repository.count_processed(task_id)
                failed_publications = self.publication_repository.count_failed(task_id)

                result_db['processed_publications'] = processed_publications
                result_db['publications_with_errors'] = failed_publications

                result = {
                    'task_id': task_id,
                    'status': result_db['status'],
                    'summary': {
                        'num_publications': result_db['num_publications'],
                        'num_authors': result_db['num_authors'],
                        'num_institutions': result_db['num_institutions'],
                        'num_journals': result_db['num_journals'],
                        'processed_publications': processed_publications,
                        'failed_publications': failed_publications
                    }
                }

                if processed_publications + failed_publications == result_db['num_publications']:
                    result['status'] = Status.DONE.name

                    connection_details = {
                        "mongodb_connection_string": self.config['mongodb_conn'],
                        "mongodb_database": result_db['origin']
                    }

                    encoded_dataset_name = urllib.parse.quote(result_db['name'])

                    dashboard_url = f"{self.config['dashboard_url']}/app/dashboards#/view/35f4488a-e0ba-4263-a325-36a0eda9a877?_a=(filters:!((query:(match_phrase:(dataset.keyword:'{encoded_dataset_name}')))))"

                    result['connection_details'] = connection_details
                    result['dashboard_url'] = dashboard_url

                    result_db['connection_details'] = connection_details
                    result_db['dashboard_url'] = dashboard_url
                    result_db['status'] = Status.DONE.name

                    self.dataset_repository.save(task_id, result_db)

            return result
        except Exception as e:
            raise e
