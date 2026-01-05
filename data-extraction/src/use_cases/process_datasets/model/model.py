"""
Data models for the dataset processing pipeline.

This module defines the core data structures used throughout the publication
extraction system, including models for datasets, publications, authors,
institutions, journals, and topics. These models handle data transformation
between different API formats (OpenAlex, Dimensions) and the internal format.
"""

from typing import List
import re
from datetime import datetime

from pyalex import Institution

from src.infra.model.enum import Status


class Dataset:
    id: str
    origin: str
    group_id: str
    group_name: str
    name: str
    description: str
    aliases: List[str]
    flag_terms: List[str]
    exclude_terms: List[str]
    status: Status
    start_year: int
    end_year: int
    home_url: str
    access_type: str
    data_url: str
    schema_url: str
    documentation_url: str
    inserted_at: str
    updated_at: str
    mentioned_datasets: [] = []
    is_validated: bool
    num_publications: int
    num_authors: int
    num_institutions: int
    num_journals: int

    def __init__(self, id, origin, group_id, group_name, name, description,
                 aliases, flag_terms, exclude_terms, start_year, end_year,
                 home_url, access_type, data_url, schema_url, documentation_url,
                 num_publications, num_authors, num_institutions, num_journals):
        current_date = datetime.now().isoformat()

        self.id = id
        self.origin = origin
        self.group_id = group_id
        self.group_name = group_name
        self.name = name
        self.description = description
        self.aliases = aliases
        self.flag_terms = flag_terms
        self.exclude_terms = exclude_terms
        self.status = Status.WAITING.name
        self.start_year = start_year
        self.end_year = end_year
        self.home_url = home_url
        self.access_type = access_type
        self.data_url = data_url
        self.schema_url = schema_url
        self.documentation_url = documentation_url
        self.inserted_at = current_date
        self.updated_at = current_date
        self.mentioned_datasets = []
        self.is_validated = False
        self.num_publications = num_publications
        self.num_authors = num_authors
        self.num_institutions = num_institutions
        self.num_journals = num_journals

    def toJSON(self):
        return {
            'id': self.id,
            'origin': self.origin,
            'group_id': self.group_id,
            'group_name': self.group_name,
            'name': self.name,
            'description': self.description,
            'aliases': self.aliases,
            'flag_terms': self.flag_terms,
            'exclude_terms': self.exclude_terms,
            'status': self.status,
            'start_year': self.start_year,
            'end_year': self.end_year,
            'home_url': self.home_url,
            'access_type': self.access_type,
            'data_url': self.data_url,
            'schema_url': self.schema_url,
            'documentation_url': self.documentation_url,
            'inserted_at': self.inserted_at,
            'updated_at': self.updated_at,
            'mentioned_datasets': self.mentioned_datasets,
            'is_validated': self.is_validated,
            'num_publications': self.num_publications,
            'num_authors': self.num_authors,
            'num_institutions': self.num_institutions,
            'num_journals': self.num_journals
        }

class DatasetShort:
    id: str
    name: str
    is_validated_mention: bool
    context_quote: str
    confidence_score: str

    def __init__(self, **kwargs):
        self.id = kwargs.get('id', '')
        self.name = kwargs.get('name', '')
        self.is_validated_mention = kwargs.get('is_validated_mention', False)
        self.context_quote = kwargs.get('context_quote', '')
        self.confidence_score = kwargs.get('confidence_score', '')

    def toJSON(self):
        return {
            'id': self.id,
            'name': self.name,
            'is_validated_mention': self.is_validated_mention,
            'context_quote': self.context_quote,
            'confidence_score': self.confidence_score
        }


class Institution:
    id: str
    name: str
    city: str
    city_code: str
    state: str
    state_code: str
    country: str
    country_code: str
    ror_id: str

    def __init__(self, **kwargs):
        self.id = kwargs.get('id', '') 
        self.name = kwargs.get('name', '')
        self.city = kwargs.get('city', '')
        self.city_code = kwargs.get('city_code', '')
        self.state = kwargs.get('state', '')
        self.state_code = kwargs.get('state_code', '')
        self.country = kwargs.get('country', '')
        self.country_code = kwargs.get('country_code', '')
        self.ror_id = kwargs.get('ror_id', '')

    def toJSON(self):
        return {
            'id': self.id,
            'name': self.name,
            'city': self.city,
            'city_code': self.city_code,
            'state': self.state,
            'state_code': self.state_code,
            'country': self.country,
            'country_code': self.country_code,
            'ror_id': self.ror_id
        }


class Author:
    id: str
    name: str
    orcid: str
    institutions: List[Institution]

    def __init__(self, id, name, orcid):
        self.id = id
        self.name = name
        self.orcid = orcid
        self.institutions = []

    def toJSON(self):
        institution_list = [institution.toJSON() for institution in self.institutions]
        return {
            'id': self.id,
            'name': self.name,
            'orcid': self.orcid,
            'institutions': institution_list
        }

    def add_institution(self, institution):
        self.institutions.append(institution)


class Journal:
    id: str
    name: str
    issn: str

    def __init__(self, id, name, issn):
        self.id = id
        self.name = name
        self.issn = issn

    def toJSON(self):
        return {
            'id': self.id,
            'name': self.name,
            'issn': self.issn
        }


class Topic:
    id: str
    type: str
    name: str
    score: float

    def __init__(self, name, score, id=None, type=None):
        self.id = id
        self.type = type
        self.name = name
        self.score = score

    def toJSON(self):
        return {
            'id': self.id,
            'type': self.type,
            'name': self.name,
            'score': self.score
        }


class CategoryFor:
    id: str
    name: str

    def __init__(self, id, name):
        self.id = id
        self.name = name

    def toJSON(self):
        return {
            'id': self.id,
            'name': self.name
        }


class Publication:
    id: str
    title: str
    doi: str
    type: str
    publication_year: int
    citation_count: int
    journals: List[Journal]
    authors: List[Author]
    topics: List[Topic]
    category_for: List[CategoryFor]
    mentioned_datasets: List[DatasetShort]
    llm_analysis_ref_id: List[str]

    def toJSON(self):
        journal_list = [journal.toJSON() for journal in self.journals]
        author_list = [author.toJSON() for author in self.authors]
        topic_list = [topic.toJSON() for topic in self.topics]
        category_for_list = [category_for.toJSON() for category_for in self.category_for]
        dataset_list = [dataset.toJSON() for dataset in self.mentioned_datasets]

        return {
            'id': self.id,
            'title': self.title,
            'doi': self.doi,
            'type': self.type,
            'publication_year': self.publication_year,
            'citation_count': self.citation_count,
            'journals': journal_list,
            'authors': author_list,
            'topics': topic_list,
            'category_for': category_for_list,
            'mentioned_datasets': dataset_list,
            'llm_analysis_ref_id': self.llm_analysis_ref_id
        }

    def dimensionsToJSON(self, row):
        self.id = row.get('id', '')
        self.title = row.get('title', '')
        self.doi = row.get('doi', '')
        self.type = row.get('type', '')
        self.publication_year = row.get('year', '')
        self.citation_count = row.get('times_cited', 0)
        self.journals = []
        self.authors = []
        self.topics = []
        self.category_for = []
        self.mentioned_datasets = []
        self.llm_analysis_ref_id = []

        if isinstance(row.get('authors'), list):
            for i, author in enumerate(row.get('authors', [])):
                orcid_str = ''
                if author.get('orcid') is not None:
                    if isinstance(author['orcid'], list):
                        orcid_str = ', '.join([str(o) for o in author['orcid'] if o is not None])
                    else:
                        orcid_str = str(author['orcid'])

                author_row = {
                    'id': author.get('researcher_id', ''),
                    'name': f"{author.get('first_name', '')} {author.get('last_name', '')}".strip(),
                    'orcid': orcid_str
                }

                author_row = Author(**author_row)

                if isinstance(author.get('affiliations'), list):
                    for _, aff in enumerate(author.get('affiliations', [])):
                        if isinstance(aff, dict):
                            aff_row = {
                                'id': aff.get('id', ''),
                                'name': aff.get('name', ''),
                                'country': aff.get('country', ''),
                                'country_code': aff.get('country_code', ''),
                                'city': aff.get('city', ''),
                                'city_code': aff.get('city_id', ''),
                                'state': aff.get('state', ''),
                                'state_code': aff.get('state_code', ''),
                                'ror_id': ''
                            }
                            author_row.add_institution(Institution(**aff_row))
                self.authors.append(author_row)

        if isinstance(row.get('concepts_scores'), list):
            sorted_concepts = sorted(
                row.get('concepts_scores', []),
                key=lambda x: x.get('relevance', 0) if isinstance(x, dict) else 0,
                reverse=True
            )

            for concept_data in sorted_concepts:
                if isinstance(concept_data, dict):
                    concept_row = {
                        'name': concept_data.get('concept', ''),
                        'score': concept_data.get('relevance', 0)
                    }
                    self.topics.append(Topic(**concept_row))

        if isinstance(row.get('category_for', []), list):
            for category in row.get('category_for', []):
                if isinstance(category, dict):
                    category_row = {
                        'id': category.get('id', ''),
                        'name': category.get('name', 0)
                    }
                    self.category_for.append(CategoryFor(**category_row))

        issn_str = ''
        if isinstance(row.get('issn'), list):
            issn_str = ', '.join(row.get('issn', []))

        self.journals.append(Journal(0, row.get('journal.title', ''), issn_str))

        return self

    def openalexToJSON(self, row):
        self.id = re.search(r'W\d+', row.get('id', '')).group()
        self.title = row.get('display_name', '')
        self.doi = row.get('doi', '')
        self.type = row.get('type', '')
        self.publication_year = row.get('publication_year', '')
        self.citation_count = row.get('cited_by_count', 0)
        self.journals = []
        self.authors = []
        self.topics = []
        self.category_for = []
        self.mentioned_datasets = []
        self.llm_analysis_ref_id = []

        if isinstance(row.get('authorships'), list):
            for author in row.get('authorships', []):
                try:
                    author_name = author['author']['display_name'] if author['author'].get('display_name') else ''
                except Exception:
                    author_name = ''
                try:
                    orcid = re.search(r'\d{4}-\d{4}-\d{4}-\d{4}', author['author']['orcid']).group() if author[
                        'author'].get('orcid') else ''
                except Exception:
                    orcid = ''
                author_row = {
                    'id': re.search(r'A\d+', author['author']['id']).group(),
                    'name': author_name,
                    'orcid': orcid
                }
                author_row = Author(**author_row)
                if isinstance(author.get('institutions'), list):
                    for aff in author.get('institutions', []):
                        if isinstance(aff, dict):
                            aff_id = re.search(r'I\d+', aff['id']).group()
                            aff_row = {
                                'id': aff_id,
                                'name': aff.get('display_name', ''),
                                'country': aff.get('country', ''),
                                'country_code': aff.get('country_code', ''),
                                'city': aff.get('city', ''),
                                'city_code': aff.get('city_id', ''),
                                'state': aff.get('state', ''),
                                'state_code': aff.get('state_code', ''),
                                'ror_id': aff.get('ror', '')
                            }
                            author_row.add_institution(Institution(**aff_row))
                self.authors.append(author_row)

        journals = list(map(lambda journal: journal['source'],
                            list(row['locations'])))

        for journal_input in journals:
            try:
                id = re.search(r'S\d+', journal_input['id']).group()
            except Exception:
                id = ''
            try:
                name = journal_input['display_name']
            except Exception:
                name = ''
            try:
                issn = journal_input['issn_l']
            except Exception:
                issn = ''
            journal = {
                'id': id,
                'name': name,
                'issn': issn
            }
            self.journals.append(Journal(**journal))

        if isinstance(row.get('topics'), list):
            for _, topic in enumerate(row.get('topics', [])):
                concept_row = {
                    'name': topic.get('display_name', ''),
                    'score': topic.get('score', 0)
                }
                self.topics.append(Topic(**concept_row))

        return self

    def add_author(self, author):
        self.authors.append(author)

    def add_journal(self, journal):
        self.journals.append(journal)

    def add_topic(self, topic):
        self.topics.append(topic)

    def add_dataset(self, dataset):
        self.mentioned_datasets.append(DatasetShort(**dataset))
