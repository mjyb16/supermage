import numpy as np
from astropy.coordinates import Distance
from astropy import constants as c
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM as lCDM

def arcsec_to_parsec(distance_mpc, angular_size_arcsec):
    """
    Convert angular sizes in arcseconds to distances in parsecs.
    
    Parameters:
    distance_mpc (numpy.ndarray): Distance to galaxy
    angular_size_arcsec (numpy.ndarray): Array of angular sizes in arcseconds.
    
    Returns:
    numpy.ndarray: Array of distances in parsecs.
    """
    # Convert distance from Mpc to parsecs
    distance_pc = distance_mpc * 1e6
    
    # Convert angular size from arcseconds to radians
    angular_size_rad = np.radians(angular_size_arcsec / 3600)
    
    # Calculate physical size using the small angle approximation
    physical_size_pc = distance_pc * angular_size_rad
    
    return physical_size_pc

def parsec_to_arcsec(distance_mpc, physical_size_pc):
    """
    Convert physical sizes in parsecs to angular sizes in arcseconds.
    
    Parameters:
    -----------
    distance_mpc : float or np.ndarray
        Distance to the galaxy in megaparsecs (Mpc).
    physical_size_pc : float or np.ndarray
        Physical sizes in parsecs.

    Returns:
    --------
    np.ndarray
        Angular sizes in arcseconds.
    """
    # Convert distance from Mpc to parsecs
    distance_pc = distance_mpc * 1e6
    
    # Using the small-angle approximation:
    # physical_size_pc = distance_pc * angular_size_rad
    # => angular_size_rad = physical_size_pc / distance_pc
    angular_size_rad = physical_size_pc / distance_pc
    
    # Convert from radians to arcseconds
    # 1 radian = (180/pi) degrees, and 1 degree = 3600 arcseconds
    # so 1 radian = 206265 arcseconds
    angular_size_arcsec = np.degrees(angular_size_rad) * 3600
    
    return angular_size_arcsec