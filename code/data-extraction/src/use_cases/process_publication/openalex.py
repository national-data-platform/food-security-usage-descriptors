import re

import pandas as pd
import pyalex

from pyalex import Works
from src import Configuration


class OpenAlex:
    """
    Connector for the OpenAlex academic research API to discover dataset-related publications.
    
    This class handles querying the OpenAlex database for scholarly publications that reference
    specific datasets. It constructs appropriate search queries based on dataset identifiers,
    related terms, and publication date ranges.
    """
    def __init__(self):
        """
        Initialize the OpenAlex connector with necessary configuration.
        
        Sets up the PyAlex client with the email address required for API access,
        which helps OpenAlex track usage and provide better service reliability.
        """
        config = Configuration().get()
        pyalex.config.email = config['openalex_email']
        self.dataset = None

    def execute(self, dataset, aliases, flag_terms, not_in_dataset, start_year, end_year):
        """
        Execute a search query against the OpenAlex API to find dataset-related publications.
        
        Constructs a complex search query combining dataset aliases and optional flag terms,
        while applying filters for publication year range, document type, language, and author
        country affiliation. Results are paginated and combined into a single DataFrame.
        
        Args:
            dataset (dict): Dataset configuration with search parameters and metadata
            aliases (list): List of alternative names/identifiers for the dataset
            flag_terms (list): Additional terms that indicate dataset usage in publications
            not_in_dataset (list): Terms that should exclude a publication from results
            start_year (int): Beginning year for publication date range filter
            end_year (int): Ending year for publication date range filter
            
        Returns:
            tuple: (DataFrame of search results, None for cursor compatibility with other connectors)
            
        Notes:
            - The function combines aliases with OR logic and flag terms with AND logic
            - Results are paginated with 200 items per page for efficient API usage
            - Supports resuming searches with next_cursor from the dataset parameter
        """
        self.dataset = dataset

        # Construct query string for dataset aliases, removing punctuation and adding quotation marks
        aliases_query = " OR ".join([r'\"' + re.sub(r'[().,]', '', alias) + r'\"' for alias in aliases])

        # Add flag terms with AND logic if provided to narrow search results
        if len(flag_terms) > 0:
            flag_terms_query = " OR ".join([r'\"' + term + r'\"' for term in flag_terms])
            aliases_query = f"({aliases_query}) AND ({flag_terms_query})"

        # Build the OpenAlex filter query with publication constraints
        query = Works().filter(
            publication_year=f"{start_year}-{end_year}",
            type='article',
            default={"search": aliases_query},
            language="en",
            authorships={"countries": "us"},
        )

        # Initialize or resume pagination with cursor
        next_cursor = "*"  # Default starting cursor
        if 'next_cursor' in dataset:
            next_cursor = dataset['next_cursor']

        # Container for accumulated results
        results = []

        # Paginate through all available results
        while next_cursor is not None:
            # Fetch page of results with metadata for pagination
            openalex_result, metadata = (
                query.get(return_meta=True, per_page=200, cursor=next_cursor))
            next_cursor = metadata['next_cursor']

            # Add each publication to results collection
            for work in openalex_result:
                results.append(work)

        # Convert collected results to DataFrame for consistent return format
        result_df = pd.DataFrame(results)

        # Return results with None for next_cursor to match interface expectations
        return result_df, None