"""
Use case for initiating dataset search pipelines.

This module handles the creation and initialization of dataset search tasks,
breaking down multi-year searches into individual year-based jobs for
parallel processing through the message queue system.
"""

from src import Configuration, Queue, DatasetRepository
from src.use_cases.pipelines_start.model.model import PipelineStartDTO
from src.use_cases.process_datasets.model import Dataset


class PipelinesStartUseCase:
    """
    Use case for starting dataset publication search pipelines.
    
    This class initializes new dataset search tasks, persisting the dataset
    configuration and queuing individual search jobs for each year in the
    specified date range to enable parallel processing.
    """
    
    def __init__(self):
        """
        Initialize the use case with configuration and dataset repository.
        """
        self.config = Configuration().get()
        self.dataset_repository = DatasetRepository(self.config['mongo_general_database'])

    def execute(self, task_id: str, dataset_dto: PipelineStartDTO) -> None:
        """
        Execute the pipeline initialization for a dataset search.
        
        Creates a dataset record and queues individual search jobs for each
        year in the specified range, enabling parallel processing of the
        publication search across multiple years.
        
        Args:
            task_id: Unique identifier for this pipeline task.
            dataset_dto: Data transfer object containing dataset search parameters.
        """
        try:
            dataset = Dataset(
                task_id,
                dataset_dto.engine,
                dataset_dto.group.id,
                dataset_dto.group.name,
                dataset_dto.main_dataset_name,
                dataset_dto.description,
                dataset_dto.dataset_names,
                dataset_dto.flag_terms,
                dataset_dto.exclude_terms,
                dataset_dto.years_range.start_year,
                dataset_dto.years_range.end_year,
                dataset_dto.home_url,
                dataset_dto.access_type,
                dataset_dto.data_url,
                dataset_dto.schema_url,
                dataset_dto.documentation_url,
                0,
                0,
                0,
                0
            )

            self.dataset_repository.save(task_id, dataset.toJSON())

            for i in range(dataset_dto.years_range.start_year, dataset_dto.years_range.end_year + 1):
                Queue().publish(
                    self.config['dataset_queue'],
                    self.config['dataset_exchange'],
                    {
                        'task_id': task_id,
                        'origin': dataset_dto.engine,
                        'group_id': dataset_dto.group.id,
                        'main_name': dataset_dto.main_dataset_name,
                        'aliases': dataset_dto.dataset_names,
                        'flag_terms': dataset_dto.flag_terms,
                        'not_in_dataset': dataset_dto.exclude_terms,
                        'start_year': i,
                        'end_year': i
                    }
                )
        except Exception as e:
            raise e
