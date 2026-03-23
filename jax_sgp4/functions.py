from .model import Satellite
from .propagation import sgp4
import jax.numpy as jnp

def sgp4_jdfr(sat: Satellite, jd, fr):
    
    """
    SGP4 propagation algorithm using Julian Date and Fractional Day.
    
    Inputs:
      sat     : Satellite object containing orbital elements and parameters
      jd      : Julian Date (integer part)
      fr      : Fractional part of the day

    Returns:
      rv      : concatenated array of position and velocity vectors (km and km/s)
    """

    # Calculate epoch in Julian Date and Fractional Day
    year = sat.epochyr
    days, fraction = jnp.divmod(sat.epochdays, 1.0)
    jd_epoch = year * 365 + (year - 1) // 4 + days + 1721044.5
    fr_epoch = jnp.round(fraction, 8) # round to match TLE precision

    tsince = (jd - jd_epoch) * 1440.0 + (fr - fr_epoch) * 1440.0
    rv = sgp4(sat, tsince)

    return rv
