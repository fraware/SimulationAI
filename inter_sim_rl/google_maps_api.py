"""
Google Maps API client: geocoding, nearby places, directions.
"""

import random
from datetime import datetime

import googlemaps

from inter_sim_rl.config import get_api_key

_API_KEY = get_api_key()
gmaps = googlemaps.Client(key=_API_KEY)


def get_coordinates(location):
    """
    Use Geocoding API to convert a location to latitude and longitude.

    Parameters:
    - location (str): Address or location name.

    Returns:
    tuple: (lat, lng) or None if geocoding fails.
    """
    geocode_result = gmaps.geocode(location)
    if geocode_result:
        lat = geocode_result[0]["geometry"]["location"]["lat"]
        lng = geocode_result[0]["geometry"]["location"]["lng"]
        return lat, lng
    return None


def get_random_nearby_place(current_address):
    """
    Get a random nearby place from the current address using Places API.

    Parameters:
    - current_address (str): Address for which nearby places are sought.

    Returns:
    str: Vicinity of a randomly selected nearby place, or None if not found.
    """
    geocode_result = gmaps.geocode(current_address)
    if not geocode_result:
        print(f"Failed to geocode address: {current_address}")
        return None
    lat = geocode_result[0]["geometry"]["location"]["lat"]
    lng = geocode_result[0]["geometry"]["location"]["lng"]
    location = (lat, lng)
    places_result = gmaps.places_nearby(location=location, radius=500)
    nearby_places = places_result["results"]
    if nearby_places:
        return random.choice(nearby_places)["vicinity"]
    return None


def get_directions(origin, destination, mode="driving"):
    """
    Get driving directions between two locations using the Directions API.

    Parameters:
    - origin (str): Starting location or address.
    - destination (str): Destination location or address.
    - mode (str): Travel mode (default: 'driving').

    Returns:
    list: List of steps, or None if no directions found.
    """
    directions_result = gmaps.directions(
        origin, destination, mode=mode, departure_time=datetime.now()
    )
    if directions_result:
        return directions_result[0]["legs"][0]["steps"]
    return None
