from pyalex import Institutions

from src import Configuration, PublicationRepository, InstitutionRepository, Queue
from src.infra.repository.geo_repository import GeoRepository
from src.use_cases.process_datasets import Dimensions


class ProcessInstitutionUseCase:
    """
    Use case for enriching publication institutions with geographic metadata.

    This class implements the core business logic for processing institution data
    associated with scholarly publications. It retrieves institution information
    from external sources, enriches it with geographical data, and ensures
    consistent storage and propagation of the enhanced metadata through the system.
    """

    def __init__(self):
        """
        Initialize the use case with necessary repositories and configuration.

        Sets up the configuration context and prepares the geo-repository for
        accessing location data. Publication and institution repositories are
        initialized later with origin-specific parameters from the execution data.
        """
        self.config = Configuration().get()

        self.publication_repository = None
        self.institution_repository = None
        self.geocode_repository = GeoRepository()

    def execute(self, data):
        """
        Execute the institution enrichment process for a specific publication.

        This method orchestrates the complete workflow for processing institution data:
        1. Initialize repositories for the specific data origin
        2. Retrieve the target publication from the repository
        3. Enrich each institution with geographic metadata from external sources
        4. Update the publication with the enhanced institution data
        5. Publish a message to trigger subsequent processing stages

        Args:
            data (dict): Message payload containing processing instructions, including:
                - publication_id: Identifier of the publication to process
                - origin: Data source identifier (e.g., 'openalex', 'dimensions')
                - organization: Organization context for the publication
                - dataset: Dataset identifier within the organization

        Raises:
            Exception: If the target publication cannot be found in the repository

        The method performs selective institution enrichment, only querying external
        sources for institutions not already present in the local repository, optimizing
        performance and reducing external API usage.
        """
        # Initialize repositories with the specific origin from the message data
        self.publication_repository = PublicationRepository(data['origin'])
        self.institution_repository = InstitutionRepository(data['origin'])

        # Retrieve the target publication using composite key parameters
        publication = self.publication_repository.find_by_id(data['publication_id'])

        # Validate publication existence to ensure data integrity
        if publication is None:
            raise Exception(
                f"Publication not found for publication_id: {data['publication_id']}")


        # Process each institution to ensure complete geographic metadata
        for authors in publication['authors']:
            institutions_list = []
            for institution in authors['institutions']:
                try:
                    # Check if the institution is already in our repository
                    institution_db = self.institution_repository.find_by_id(institution['id'])
                    if institution_db is None:
                        if data['origin'] == 'openalex':
                            # Institution not found locally - fetch from OpenAlex API
                            institution_oa = Institutions()[institution['id']]

                            # Only process institutions with available city identifiers
                            if institution_oa['geo']['geonames_city_id'] is not None:
                                # Retrieve detailed geographic data from GeoNames
                                geocode = self.geocode_repository.get_geocodes(institution_oa['geo']['geonames_city_id'])

                                # Construct enriched institution record with geographic context
                                institution_db = {
                                    'id': institution['id'],
                                    'name': institution['name'],
                                    'country': geocode['countryName'],
                                    'country_code': geocode['countryCode'],
                                    'city': geocode['asciiName'],
                                    'city_code': institution_oa['geo']['geonames_city_id'],
                                    'state': geocode['adminName1'],
                                    'state_code': geocode['adminCode1'],
                                    'ror': institution_oa['ror']
                                }

                                # Persist the enriched institution data for future reference
                                self.institution_repository.save(institution_db)
                                institutions_list.append(institution_db)
                        elif data['origin'] == 'dimensions':
                            try:
                                dimensions = Dimensions()
                                institution_db = {
                                    'id': institution['id'],
                                    'name': institution['name'],
                                    'country': institution['country'],
                                    'country_code': institution['country_code'],
                                    'city': institution['city'],
                                    'city_code': institution['city_code'],
                                    'state': institution['state'],
                                    'state_code': institution['state_code'],
                                    'ror': ""
                                }

                                if institution['id'] is not None:
                                    organization = dimensions.get_organization(institution['id'])
                                    institution_db = {
                                        'id': institution['id'],
                                        'name': institution['name'],
                                        'country': institution['country'],
                                        'country_code': institution['country_code'],
                                        'city': institution['city'],
                                        'city_code': institution['city_code'],
                                        'state': institution['state'],
                                        'state_code': institution['state_code'],
                                        'ror': organization['ror_ids'][0]
                                    }
                                    # Persist the enriched institution data for future reference
                                    self.institution_repository.save(institution_db)

                                institutions_list.append(institution_db)
                            except Exception as e:
                                print(e)
                        else:
                            # Unable to enrich - retain original institution data
                            institutions_list.append(institution)
                    else:
                        # Institution already exists in repository - use stored data
                        institutions_list.append(institution_db)
                except Exception as e:
                    # Log geocoding errors but continue processing
                    # This ensures partial enrichment rather than complete failure
                    print(f"Error on find geocode: {e}")

            authors['institutions'] = institutions_list

        # Persist the updated publication with enriched institution metadata
        self.publication_repository.save(publication)

        # Signal completion by publishing a message to the next processing stage
        Queue().publish(
            self.config['publication_queue'],  # Target queue for subsequent processing
            self.config['publication_exchange'],  # Message exchange for routing
            data  # Original processing instructions for context continuity
        )