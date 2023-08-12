import googlemaps
from datetime import datetime
import random

API_KEY = "YourAPIKey"

# Create a Google Maps client
gmaps = googlemaps.Client(key=API_KEY)

def get_coordinates(location):
    """
    Use Geocoding API to convert a location to latitude and longitude.

    Parameters:
    - location (str): Address or location name.

    Returns:
    tuple: Latitude and longitude as a tuple (lat, lng), or None if geocoding fails.
    """
    # Use Geocoding API to convert location to latitude and longitude
    geocode_result = gmaps.geocode(location)
    if geocode_result:
        lat = geocode_result[0]['geometry']['location']['lat']
        lng = geocode_result[0]['geometry']['location']['lng']
        return lat, lng
    else:
        return None

def get_random_nearby_place(current_address):
    """
    Get a random nearby place or street from the current address using Geocoding and Places APIs.

    Parameters:
    - current_address (str): Current address for which nearby places are sought.

    Returns:
    str: Vicinity (address) of a randomly selected nearby place, or None if not found.
    """
    # Convert the current address to coordinates using Geocoding API
    geocode_result = gmaps.geocode(current_address)
    if geocode_result:
        lat = geocode_result[0]['geometry']['location']['lat']
        lng = geocode_result[0]['geometry']['location']['lng']
        location = (lat, lng)

        # Use Places API to get a list of nearby places or streets from the current address coordinates
        places_result = gmaps.places_nearby(location=location, radius=500)
        nearby_places = places_result['results']
        if nearby_places:
            return random.choice(nearby_places)['vicinity']
        else:
            return None
    else:
        print(f"Failed to geocode address: {current_address}")
        return None

def get_directions(origin, destination, mode='driving'):
    """
    Get driving directions between two locations using the Directions API.

    Parameters:
    - origin (str): Starting location or address.
    - destination (str): Destination location or address.
    - mode (str): Travel mode for directions (default: 'driving').

    Returns:
    list: List of steps comprising the driving directions, or None if no directions found.
    """
    # Use Directions API to get driving directions
    directions_result = gmaps.directions(origin, destination, mode=mode, departure_time=datetime.now())
    if directions_result:
        return directions_result[0]['legs'][0]['steps']
    else:
        return None