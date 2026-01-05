"""
Data models for flattened publication representation.

This module defines the data structures used to represent publications
in a denormalized format suitable for Elasticsearch indexing and search.
"""

from typing import List
import numpy as np


class Category:
    """Represents a research category or field of study."""
    id: str
    name: str


class Journal:
    """Represents an academic journal where publications are published."""
    name: str
    issn: str


class Institution:
    """
    Represents a research institution or organization.
    
    Contains geographic information including country, state, and city
    along with the ROR (Research Organization Registry) identifier.
    """
    id: str
    name: str
    country: str
    country_code: str
    state: str
    state_code: str
    city: str
    city_id: str
    ror_id: str


class Author:
    """
    Represents an author of a publication.
    
    Includes ORCID identifier and affiliated institutions.
    """
    id: str
    name: str
    orcid: str
    institutions: List[Institution]


class Topic:
    """
    Represents a research topic associated with a publication.
    
    The score indicates the relevance of the topic to the publication.
    """
    id: str
    name: str
    score: float


class Dataset:
    """
    Represents a dataset mentioned in a publication.
    
    Tracks whether the mention has been validated and includes
    the context quote where the dataset was referenced.
    """
    id: str
    name: str
    is_validated_mention: bool
    context_quote: str


class FlatPublication:
    publication_id: str
    name: str
    doi: str
    publication_year: int
    citation_count: int
    all_search_text: str
    categories: List[Category] = []
    journals: List[Journal] = []
    authors: List[Author] = []
    topics: List[Topic] = []
    datasets: List[Dataset] = []
    dataset_joins: List[str] = []

    def fromDBtoElastic(self, publication):
        all_search_list = []
        journals = [
            {
                'name': journal['name'],
                'issn': journal['issn']
            }
            for journal in list(filter(lambda x: str(x['name']) != 'nan', publication['journals']))
        ]
        authors = [
            {
                'id': author['id'],
                'name': author['name'],
                'orcid': author['orcid'],
                'institutions': [
                    {
                        'id': institution['id'],
                        'name': institution['name'],
                        'country': institution['country'],
                        'country_code': institution['country_code'],
                        'city': institution['city'],
                        'city_id': institution['city_id'] if 'city_id' in institution else institution['city_code'],
                        'state': institution['state'],
                        'state_code': institution['state_code'],
                        'ror_id': institution['ror']
                    }
                    for institution in author['institutions']
                ]
            }
            for author in publication['authors']
        ]
        topics = [
            {
                'id': topic['id'],
                'name': topic['name'],
                'score': topic['score']
            }
            for topic in publication['topics']
        ]

        if len(topics) > 0:
            sorted_topics = sorted(
                topics,
                key=lambda x: x['score'] if isinstance(x, dict) else 0,
                reverse=True
            )
            if sorted_topics[0]['score'] > 0.7:
                topics = [topic for topic in sorted_topics if topic['score'] > 0.7]
            else:
                topics = [sorted_topics[0]]

        datasets = [
            {
                'id': dataset['id'],
                'name': dataset['name'],
                'is_validated_mention': dataset['is_validated_mention'],
                'context_quote': dataset['context_quote'],
                'group_id': dataset['group_id'] if 'group_id' in dataset else '',
                'group_name': dataset['group_name'] if 'group_name' in dataset else '',
            }
            for dataset in publication['mentioned_datasets']
        ]
        categories_for = [
            {
                'id': category['id'],
                'name': category['name']
            }
            for category in publication['category_for']
        ]
        all_search_list.append(publication['title'])
        [all_search_list.append(journal['name']) for journal in journals]
        [all_search_list.append(author['name']) for author in authors]
        [[all_search_list.append(institution['name'])
         for institution in author['institutions']]
         for author in authors]
        [all_search_list.append(topic['name']) for topic in topics]
        [all_search_list.append(category['name']) for category in categories_for]
        [all_search_list.append(dataset['name']) for dataset in datasets]
        return {
            'publication_id': publication['id'],
            'name': publication['title'],
            'doi': publication['doi'],
            'publication_year': publication['publication_year'],
            'citation_count': publication['citation_count'],
            'publication_type': publication['type'],
            'journals': journals,
            'authors': authors,
            'topics': topics,
            'categories': categories_for,
            'datasets': datasets,
            'dataset_joins': [],
            'all_search_text': ', '.join(list(filter(lambda x: str(x) != 'nan', all_search_list)))
        }
