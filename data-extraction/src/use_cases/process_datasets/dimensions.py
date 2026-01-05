import dimcli

from src import Configuration


class Dimensions:
    """
    Interface for querying the Dimensions scholarly database API to discover dataset-related publications.
    
    This class provides a specialized connector for accessing the Dimensions academic research database,
    constructing complex search queries to identify publications that reference specific datasets.
    It handles authentication, query building with appropriate syntax, and result retrieval in a format
    compatible with the application's processing pipeline.
    """

    def __init__(self):
        """
        Initialize the Dimensions API connection with authentication credentials.
        
        Retrieves API access credentials from the application configuration,
        establishes an authenticated session with the Dimensions API service,
        and initializes the DSL (Domain Specific Language) client used for
        constructing and executing semantic search queries.
        """
        # Retrieve configuration parameters
        config = Configuration().get()

        # Establish authenticated connection to Dimensions API
        dimcli.login(key=config['dimensions_api_key'], endpoint=config['dimensions_endpoint'])

        # Initialize the Dimensions DSL query interface
        self.dsl = dimcli.Dsl()

        # Placeholder for dataset context during query execution
        self.dataset = None

    def execute(self, dataset, aliases, flag_terms, not_in_dataset, start_year, end_year):
        """
        Execute a dataset-focused publication search against the Dimensions database.
        
        Constructs and executes a semantically rich search query to identify scholarly 
        publications that reference the specified dataset. The search incorporates 
        multiple dataset identifiers, contextual terms, exclusions, date constraints,
        and filters for US-based research to produce a comprehensive yet precise result set.
        
        Args:
            dataset (dict): Configuration containing dataset metadata and search parameters
            aliases (list): Alternative names and identifiers for the dataset 
                           (searched with OR logic)
            flag_terms (list): Terms that indicate dataset usage in publications
                              (combined with AND logic if provided)
            not_in_dataset (list): Exclusionary terms to filter out false positives
                                  (applied with NOT logic if provided)
            start_year (int): Lower bound for publication year range filter
            end_year (int): Upper bound for publication year range filter
        
        Returns:
            tuple: (
                pandas.DataFrame: Structured results containing publication metadata,
                None: Reserved for future pagination support
            )
        
        Notes:
            The query searches full-text content using the Dimensions DSL syntax and
            retrieves a comprehensive set of publication metadata fields for downstream
            processing, including basic metadata, journal information, conceptual analysis,
            identifiers, and citation metrics.
        """
        # Store dataset context for reference during processing
        self.dataset = dataset

        # Construct primary search expression with quoted dataset identifiers
        aliases_query = " OR ".join([r'\"' + alias + r'\"' for alias in aliases])

        # Incorporate mandatory terms if specified (narrowing results)
        if len(flag_terms) > 0:
            flag_terms_query = " OR ".join([r'\"' + term + r'\"' for term in flag_terms])
            aliases_query = f"({aliases_query}) AND ({flag_terms_query})"

        # Apply exclusion filters if specified (removing false positives)
        if len(not_in_dataset) > 0:
            not_in_dataset_query = " NOT ".join([r'\"' + not_in + r'\"' for not_in in not_in_dataset])
            aliases_query = f"({aliases_query}) NOT ({not_in_dataset_query})"

        # Construct complete Dimensions DSL query with all filters and field selections
        query = f"""
              search publications
              in full_data for "({aliases_query})"
              where year >= {start_year} and 
                    year <= {end_year} and
                    type in ["article", "chapter", "proceeding", "monograph", "preprint"] and
                    research_org_countries = "US"
              return publications[basics + journal_lists + concepts_scores + doi + issn + isbn + linkout + dimensions_url + times_cited + abstract + category_for]
              """

        # Execute query with automatic pagination for large result sets
        response = self.dsl.query_iterative(query)

        # Transform API response to structured DataFrame format
        results = response.as_dataframe()

        # Return results with placeholder for future pagination support
        return results, None

    def get_organization(self, org_id):
        query = f"""
              search organizations
              where id = "{org_id}"
              return organizations[basics + ror_ids]
              """
        # Execute query with automatic pagination for large result sets
        response = self.dsl.query_iterative(query)

        # Transform API response to structured DataFrame format
        results = response.as_dataframe()

        if len(results) > 0:
            return results.iloc[0]

        # Return results with placeholder for future pagination support
        return None
