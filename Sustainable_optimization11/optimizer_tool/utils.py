import numpy as np
from tqdm.auto import tqdm
from geopy.geocoders import Nominatim
from geopy import distance

np.random.seed(42)

# Reference for approx. road distance calculation: https://pubmed.ncbi.nlm.nih.gov/12609652/#:~:text=There%20was%20a%20strong%20linear,%3D%201.3%20(air%20miles).
def distance_calculator(source: np.ndarray, destination: np.ndarray, mode: str) -> list:
    """Calculate geographical euclidean distance (air) and approx. road distance.

    Args:
        source (np.ndarray): numpy array containing source locations
        destination (np.ndarray): numpy array containing source locations
        mode (str): mode of transportation [required for finding distance in that mode]

    Returns:
        (list): list of calculated distances
    """

    geocoder = Nominatim(user_agent="geobot-splchn")

    dist_list = []

    loc1 = source.tolist()
    loc2 = destination.tolist()

    for i in tqdm(range(len(loc1))):
        geo1 = geocoder.geocode(loc1[i])
        geo2 = geocoder.geocode(loc2[i])

        lat1, lon1 = geo1.latitude, geo1.longitude
        lat2, lon2 = geo2.latitude, geo2.longitude

        coord1 = (lat1, lon1)
        coord2 = (lat2, lon2)

        if mode.lower() == "air":
            dist_in_km = float(str(distance.distance(coord1, coord2)).replace('km', ''))
            dist_list.append(round(dist_in_km, 2))

        elif mode.lower() == "road":
            air_dist_in_mile = 0.621371*float(str(distance.distance(coord1, coord2)).replace('km', ''))
            ground_dist_in_mile = 0.94 + 1.25*(air_dist_in_mile)
            ground_dist_in_km = 1.60934*ground_dist_in_mile
            dist_list.append(round(ground_dist_in_km, 2))

        elif mode.lower() == "rail":
            air_dist_in_mile = 0.621371*float(str(distance.distance(coord1, coord2)).replace('km', ''))
            ground_dist_in_mile = 0.63 + 1.04*(air_dist_in_mile)
            ground_dist_in_km = 1.60934*ground_dist_in_mile
            dist_list.append(round(ground_dist_in_km, 2))

    return dist_list
