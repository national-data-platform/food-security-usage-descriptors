from typing import Optional

from pydantic import BaseModel


class YearsRangeDTO(BaseModel):
    start_year: int
    end_year: int

class GroupDTO(BaseModel):
    id: Optional[str] = None
    name: str

class PipelineStartDTO(BaseModel):
    engine: Optional[str]
    group: GroupDTO
    main_dataset_name: str
    description: str
    dataset_names: list[str] = []
    flag_terms: list[str] = []
    exclude_terms: list[str] = []
    webhook_url: Optional[str]
    years_range: YearsRangeDTO
    filter_us_affiliation: bool
    publication_types: list[str] = []
    home_url: Optional[str]
    access_type: Optional[str]
    data_url: Optional[str]
    schema_url: Optional[str]
    documentation_url: Optional[str]