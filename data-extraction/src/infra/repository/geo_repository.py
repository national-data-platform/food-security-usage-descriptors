import urllib

import requests


class GeoRepository:
    """
    Repository for accessing geographical data from GeoNames web service.
    
    This class provides a standardized interface for retrieving geographical
    information related to institutional affiliations, supporting geospatial
    analysis of scholarly data across different regions and locations.
    """
    
    def __init__(self):
        """
        Initialize the GeoRepository with the GeoNames web service endpoint.
        
        Sets up the base URL for the GeoNames JSON API that will be used for
        all geographic data retrieval operations.
        """
        self.base_url = "https://www.geonames.org/getJSON"


    def get_geocodes(self, city_code):
        """
        Retrieve detailed geographic information for a location by its GeoNames ID.
        
        Queries the GeoNames web service to obtain comprehensive geographical data
        for a city or location, including coordinates, administrative hierarchies,
        population, and other relevant metadata for spatial analysis.
        
        Args:
            city_code (str): GeoNames identifier for the city or location
            
        Returns:
            dict: Complete geographical metadata for the location if found,
                  including latitude, longitude, country, administrative divisions,
                  and hierarchical relationships, or None if retrieval fails
        """
        params = {
            'id': city_code,
            'style': "gui"
        }
        query_string = urllib.parse.urlencode(params)
        url = f"{self.base_url}?{query_string}"

        response = requests.get(url)
        if response.status_code == 200:
            json = response.json()
            return json
        else:
            print(f"City code {city_code} not found")