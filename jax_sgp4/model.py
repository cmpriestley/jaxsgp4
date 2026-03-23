# Define the Satellite class

from typing import NamedTuple

class Satellite(NamedTuple):
    n0: float      # Mean motion (revs/day)
    e0: float      # Eccentricity
    i0: float      # Inclination (degrees)
    w0: float      # Argument of perigee (degrees)
    Omega0: float  # Right ascension of the ascending node (degrees)
    M0: float      # Mean anomaly (degrees)
    Bstar: float   # Drag coefficient
    epochdays: float      # Epoch in days of year FIX this description later
    epochyr: float      # Epoch year FIX this description later
                        # (Note: standard sgp4 library gives epochyr as a two digit year but we use a four digit year here)