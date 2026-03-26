"""SGP4 near-Earth orbit propagation algorithm (WGS-72)."""

import jax.numpy as jnp
from jax import lax
import numpy as np

from .model import Satellite

# ==============================================================================
# SGP4 CONSTANTS (WGS-72 / Appendix B)
# ==============================================================================

# Gravitational constants (WGS-72)
J2 = 1.082616e-3                # Unnormalised zonal harmonic coefficients
J3 = -0.253881e-5
J4 = -1.65597e-6
GM = 3.986008e5                 # Earth gravitational parameter km^3/s^2
aE = 6378.135                   # Earth equatorial radius in km 
ke = 60.0 / np.sqrt(aE**3 / GM) # sqrt(GM) in units (Earth Radii)^1.5 / min)

# Derived constants (note: in normalised units as implied by paper i.e. aE = 1) 
k2 = 0.5 * J2                   # (Earth Radii)^2 
A30 = -J3                       # Normalized
k4 = -3/8 * J4                  # Normalized

# Density constants
q0_val = 120.0                  # km
s_val = 78.0                    # km

def sgp4(sat: Satellite, tsince):
    """SGP4 propagation algorithm.

    Inputs:
      sat        : Satellite object containing orbital elements and parameters
      tsince     : Time since epoch (minutes)

    Returns:
      rv         : Concatenated position and velocity array [x, y, z, vx, vy, vz]
                   in km and km/s (TEME frame)
      error_code : 0 if no error, otherwise:
                   1 = mean eccentricity out of range
                   2 = mean motion <= 0
                   4 = semi-latus rectum < 0
                   6 = satellite radius below Earth's surface
    """
    
    # --------------------------------------------------------------------------
    # A. INITIALIZATION
    # --------------------------------------------------------------------------
    
    # Internal units: Earth Radii for distance, minutes for time, radians for angles.
    # Output converted to km and km/s at end.

    # Unpack satellite parameters
    n0 = sat.n0
    e0 = sat.e0
    i0 = sat.i0
    w0 = sat.w0
    Omega0 = sat.Omega0
    M0 = sat.M0
    Bstar = sat.Bstar

    # Convert inputs to radians
    i0 = jnp.radians(i0)
    w0 = jnp.radians(w0)
    Omega0 = jnp.radians(Omega0)
    M0 = jnp.radians(M0)
    
    # Convert Mean Motion from revs/day to rad/min
    n0_rad_min = n0 * (2 * jnp.pi) / 1440.0

    # Recover Brouwer mean motion from Kozai mean motion
    a1 = (ke / n0_rad_min) ** (2 / 3)
    
    cos_i0 = jnp.cos(i0)
    sin_i0 = jnp.sin(i0)
    theta = cos_i0
    theta2 = theta ** 2
    theta4 = theta ** 4
    
    delta1 = (1.5 * k2 * (3 * theta2 - 1)) / ((1 - e0 ** 2) ** 1.5 * a1 ** 2)
    a2 = a1 * (1 - delta1 / 3 - delta1 ** 2 - 134 / 81 * delta1 ** 3) 
    delta0 = (1.5 * k2 * (3 * theta2 - 1)) / ((1 - e0 ** 2) ** 1.5 * a2 ** 2)

    n0_brouwer = n0_rad_min / (1 + delta0)
    a0_brouwer = (ke / n0_brouwer) ** (2 / 3)

    # 1. Initialization for Secular Effects of Atmospheric Drag
    # ----------------------------------------------------------

    # Epoch perigee height (km)
    rp = (a0_brouwer * (1 - e0) - 1.0) * aE 
    
    # Logic for 's' parameter (Earth Radii) based on perigee height
    s = lax.cond(
        rp >= 156,
        lambda rp: (s_val / aE) + 1.0,
        lambda rp: lax.cond(
            rp >= 98,
            lambda rp: (rp - s_val) / aE + 1.0,
            lambda rp: (20.0 / aE) + 1.0,
            rp
        ), 
        rp
    )
    
    # q0 parameter (Earth Radii)
    q0 = (q0_val / aE) + 1.0

    xi = 1.0 / (a0_brouwer - s)
    beta0 = jnp.sqrt(1 - e0 ** 2)
    eta = a0_brouwer * e0 * xi

    # C2 Calculation
    term_c2_1 = a0_brouwer * (1 + 1.5 * eta ** 2 + 4 * e0 * eta + e0 * eta ** 3)
    term_c2_2 = (1.5 * k2 * xi / (1 - eta ** 2)) * (-0.5 + 1.5 * theta2) * (8 + 24 * eta ** 2 + 3 * eta ** 4)
    C2 = (q0 - s) ** 4 * xi ** 4 * n0_brouwer * (1 - eta ** 2) ** (-3.5) * (term_c2_1 + term_c2_2)

    C1 = Bstar * C2
   
    C3 = jnp.where(e0 > 1e-4, 
                   (q0 - s) ** 4 * xi ** 5 * A30 * n0_brouwer * sin_i0 / (k2 * e0), 
                   0.0) # Avoid divide by zero for circular orbits

    # C4 Calculation
    term_c4_1 = 2 * eta * (1 + e0 * eta) + 0.5 * e0 + 0.5 * eta ** 3
    term_c4_2 = (2 * k2 * xi / (a0_brouwer * (1 - eta ** 2))) * \
                (3 * (1 - 3 * theta2) * (1 + 1.5 * eta ** 2 - 2 * e0 * eta - 0.5 * e0 * eta ** 3) + 
                 0.75 * (1 - theta2) * (2 * eta ** 2 - e0 * eta - e0 * eta ** 3) * jnp.cos(2 * w0))
    
    C4 = 2 * n0_brouwer * (q0 - s) ** 4 * xi ** 4 * a0_brouwer * beta0 ** 2 * \
         (1 - eta ** 2) ** (-3.5) * (term_c4_1 - term_c4_2)

    C5 = 2 * (q0 - s) ** 4 * xi ** 4 * a0_brouwer * beta0 ** 2 * \
         (1 - eta ** 2) ** (-3.5) * (1 + 2.75 * eta * (eta + e0) + e0 * eta ** 3)
         
    D2 = 4 * a0_brouwer * xi * C1 ** 2
    D3 = (4 / 3) * a0_brouwer * xi ** 2 * (17 * a0_brouwer + s) * C1 ** 3
    D4 = (2 / 3) * a0_brouwer ** 2 * xi ** 3 * (221 * a0_brouwer + 31 * s) * C1 ** 4

    # 2. Initialization for Secular Effects of Earth Zonal Harmonics
    # --------------------------------------------------------------

    Mdot = n0_brouwer * (3 * k2 * (-1 + 3 * theta2) / (2 * a0_brouwer ** 2 * beta0 ** 3) + 
                         3 * k2 ** 2 * (13 - 78 * theta2 + 137 * theta4) / (16 * a0_brouwer ** 4 * beta0 ** 7))
    
    wdot = n0_brouwer * (-3 * k2 * (1 - 5 * theta2) / (2 * a0_brouwer ** 2 * beta0 ** 4) + 
                         3 * k2 ** 2 * (7 - 114 * theta2 + 395 * theta4) / (16 * a0_brouwer ** 4 * beta0 ** 8) + 
                         5 * k4 * (3 - 36 * theta2 + 49 * theta4) / (4 * a0_brouwer ** 4 * beta0 ** 8))
                         
    Omegadot = n0_brouwer * (-3 * k2 * theta / (a0_brouwer ** 2 * beta0 ** 4) + 
                             3 * k2 ** 2 * (4 * theta - 19 * theta ** 3) / (2 * a0_brouwer ** 4 * beta0 ** 8) + 
                             5 * k4 * theta * (3 - 7 * theta2) / (2 * a0_brouwer ** 4 * beta0 ** 8))

    # 3. Initialization for Deep Space 
    # --------------------------------------------------------------

    # Check for Deep Space (Period >= 225 minutes)
    period_min = 2 * jnp.pi / n0_rad_min
    is_deep_space = period_min >= 225.0
    
    # NOTE: Deep-space initialisation not yet implemented.
    # When added, this will populate the following variables with the secular rates derived from the deep space analysis (Appendix A)

    dot_M_LS = 0.0
    dot_w_LS = 0.0
    dot_Omega_LS = 0.0
    dot_e_LS = 0.0
    dot_i_LS = 0.0

    # --------------------------------------------------------------------------
    # B. UPDATE
    # --------------------------------------------------------------------------
    
    # 1. Secular Update for Earth Zonal Gravity and Partial Atmospheric Drag Effects
    # ------------------------------------------------------------------------------

    MDF = M0 + (n0_brouwer + Mdot) * tsince
    wDF = w0 + wdot * tsince
    OmegaDF = Omega0 + Omegadot * tsince
    
    # Check for perigee height < 220 km for drag terms (also used for some of the deep space logic when implemented)
    is_high_perigee = rp >= 220.0
    
    deltaw = lax.cond(
        is_high_perigee, 
        lambda _: Bstar * C3 * jnp.cos(w0) * tsince, 
        lambda _: 0.0,
        operand=None   
    )

    # deltaM update with added eccentricity guard
    deltaM = lax.cond(
      is_high_perigee,
      lambda _: jnp.where(e0 > 1e-4,
                          -2/3 * (q0 - s) ** 4 * Bstar * xi ** 4 / (e0 * eta) *
                          ((1 + eta * jnp.cos(MDF)) ** 3 - (1 + eta *
                            jnp.cos(M0)) ** 3),
                          0.0),
      lambda _: 0.0,
      operand=None
  )

    M_secular = MDF + deltaw + deltaM
    w_secular = wDF - deltaw - deltaM
    Omega_secular = OmegaDF - 21/2 * (n0_brouwer * k2 * theta / (a0_brouwer ** 2 * beta0 ** 2)) * C1 * tsince ** 2

    # 2. Secular Updates for Lunar and Solar Gravity (Deep Space)
    # --------------------------------------------------------------

    # NOTE: Placeholder for deep-space (not yet implemented)
    # When added, this will apply the secular rates derived from Deep Space analysis 
    # Note: These are 0.0 if period < 225 min
    M_secular += dot_M_LS * tsince
    w_secular += dot_w_LS * tsince
    Omega_secular += dot_Omega_LS * tsince
    e_secular = e0 + dot_e_LS * tsince
    i_secular = i0 + dot_i_LS * tsince

    # 3. Secular Updates for Resonance Effects of Earth Gravity
    # --------------------------------------------------------------
    # NOTE: Placeholder for resonance logic 

    # 4. Secular Update for Remaining Atmospheric Drag Effects
    # --------------------------------------------------------------
    
    t2 = tsince ** 2
    t3 = tsince ** 3
    t4 = tsince ** 4

    # Branch 1: High Perigee (>= 220km)
    def e_high(_):
        e = e_secular - Bstar * C4 * tsince - Bstar * C5 * (jnp.sin(M_secular) - jnp.sin(M0))
        a = (ke / n0_brouwer) ** (2/3) * (1 - C1 * tsince - D2 * t2 - D3 * t3 - D4 * t4) ** 2
        IL = M_secular + w_secular + Omega_secular + n0_brouwer * \
                 (1.5 * C1 * t2 + (D2 + 2 * C1 ** 2) * t3 + 
                  0.25 * (3 * D3 + 12 * C1 * D2 + 10 * C1 ** 3) * t4 + 
                  0.2 * (3 * D4 + 12 * C1 * D3 + 6 * D2 ** 2 + 30 * C1 ** 2 * D2 + 15 * C1 ** 4) * tsince ** 5)
        return e, a, IL
    
    # Branch 2: Low Perigee (< 220km)
    def e_low(_):
        e = e0 - Bstar * C4 * tsince
        a = (ke / n0_brouwer) ** (2/3) * (1 - C1 * tsince) ** 2
        IL = M_secular + w_secular + Omega_secular + n0_brouwer * (1.5 * C1 * t2)
        return e, a, IL
    
    # Select based on perigee condition
    e_final_sec, a_final_sec, IL = lax.cond(
        is_high_perigee, 
        e_high, 
        e_low,
        operand=None
    )

    # Error if mean motion less than zero (unphysical)
    # NOTE: once deep-space implemented this should check the variable: n0_brouwer + deep space resonance
    error_code = jnp.where(n0_brouwer <= 0.0, 2, 0)

    # Error if eccentricity out of valid range
    error_code = jnp.where((e_final_sec < -0.001) | (e_final_sec >= 1), 1, error_code)

    # Enforce eccentricity limit to avoid a divide by zero
    e_final_sec = jnp.clip(e_final_sec, 1e-6, 1.0 - 1e-6)

    # Wrap secular angles to [0, 2π)
    twopi = 2 * jnp.pi
    Omega_secular = Omega_secular % twopi
    w_secular = w_secular % twopi
    IL = IL % twopi
    
    # Calculate Mean Motion 'n' at time t
    n = ke / (a_final_sec ** 1.5)
    beta = jnp.sqrt(1 - e_final_sec ** 2)

    # 5. Update for Long-Period Periodic Effects of Lunar and Solar Gravity
    # ---------------------------------------------------------------------

    # NOTE: Long-period lunar/solar periodics would be applied here once deep-space is implemented.
    # would also add error check 3 for perturbed eccentricity, variable: e_final_sec + lunar-solar periodics applied

    # 6. Update for Long-Period Periodic Effects of Earth Gravity
    # -----------------------------------------------------------

    axN = e_final_sec * jnp.cos(w_secular)
    
    term_ill = (A30 * jnp.sin(i_secular)) / (8 * k2 * a_final_sec * beta ** 2)
    ILL = term_ill * axN * (3 + 5 * jnp.cos(i_secular)) / (1 + jnp.cos(i_secular))
    ayNL = A30 * jnp.sin(i_secular) / (4 * k2 * a_final_sec * beta ** 2)
    
    ILT = IL + ILL
    ayN = e_final_sec * jnp.sin(w_secular) + ayNL

    # 7. Update for Short-Period Periodic Effects of Earth Gravity
    # ------------------------------------------------------------
    
    U = ILT - Omega_secular
    
    # Fixed-Point Iteration Solver for Kepler's Equation
    def kepler_body(i, Ew_curr):
        numerator = U - ayN * jnp.cos(Ew_curr) + axN * jnp.sin(Ew_curr) - Ew_curr
        denominator = 1 - ayN * jnp.sin(Ew_curr) - axN * jnp.cos(Ew_curr)
        return Ew_curr + numerator / denominator
    
    Ew_initial = U
    
    # Run 10 iterations as per standard SGP4 implementation 
    # (note: standard implementation also includes a cut off if correction becomes negligibly small and a limit on step size - if this is implemented, use lax.while_loop instead)
    Ew = lax.fori_loop(0, 10, kepler_body, Ew_initial)

    # Preliminary quantities for short-period periodics
    ecosE = axN * jnp.cos(Ew) + ayN * jnp.sin(Ew)
    esinE = axN * jnp.sin(Ew) - ayN * jnp.cos(Ew)
    
    e_osc = jnp.sqrt(ecosE ** 2 + esinE ** 2) 
    # e_osc = jnp.clip(e_osc, 1e-6, 1.0 - 1e-6) # Safety clip 
    
    pL = a_final_sec * (1 - e_osc ** 2)

    # Error if semi-latus rectum less than zero (unphysical)
    error_code = jnp.where(pL <= 0.0, 4, error_code)

    r = a_final_sec * (1 - ecosE)
    
    rdot = ke * jnp.sqrt(a_final_sec) / r * esinE
    rfdot = ke * jnp.sqrt(pL) / r
    
    # Arguments of Latitude (u)
    cosu = (a_final_sec / r) * (jnp.cos(Ew) - axN + ayN * esinE / (1 + jnp.sqrt(1 - e_osc ** 2)))
    sinu = (a_final_sec / r) * (jnp.sin(Ew) - ayN - axN * esinE / (1 + jnp.sqrt(1 - e_osc ** 2)))
    u = jnp.arctan2(sinu, cosu)

    # Short-Period Corrections
    sin2u = jnp.sin(2 * u)
    cos2u = jnp.cos(2 * u)
    
    # Corrections (Deltas)
    # Using i_secular (mean inclination) for these factors
    sin_i = jnp.sin(i_secular)
    cos_i = jnp.cos(i_secular)
    
    Deltar = (k2 / (2 * pL)) * (1 - cos_i ** 2) * cos2u
    Deltau = (-k2 / (4 * pL ** 2)) * (7 * cos_i ** 2 - 1) * sin2u
    DeltaOmega = (3 * k2 * cos_i / (2 * pL ** 2)) * sin2u
    Deltai = (3 * k2 * cos_i / (2 * pL ** 2)) * sin_i * cos2u
    Deltardot = (-k2 * n / pL) * (1 - cos_i ** 2) * sin2u
    Deltarfdot = (k2 * n / pL) * ((1 - cos_i ** 2) * cos2u - 1.5 * (1 - 3 * cos_i ** 2))

    # Osculating Elements
    rk = r * (1 - 1.5 * k2 * jnp.sqrt(1 - e_osc ** 2) / (pL ** 2) * (3 * cos_i ** 2 - 1)) + Deltar
    uk = u + Deltau
    Omegak = Omega_secular + DeltaOmega
    ik = i_secular + Deltai
    rdotk = rdot + Deltardot
    rfdotk = rfdot + Deltarfdot

    # --------------------------------------------------------------------------
    # C. VECTORS (Position and Velocity in TEME frame)
    # --------------------------------------------------------------------------
    
    # Orientation Vectors (M and N)
    M = jnp.array([
        -jnp.sin(Omegak) * jnp.cos(ik), 
        jnp.cos(Omegak) * jnp.cos(ik), 
        jnp.sin(ik)
    ])

    N = jnp.array([
        jnp.cos(Omegak), 
        jnp.sin(Omegak), 
        0.0
    ])

    sin_uk = jnp.sin(uk)
    cos_uk = jnp.cos(uk)

    U = M * sin_uk + N * cos_uk
    V = M * cos_uk - N * sin_uk

    # Position and Velocity in TEME frame (Earth Radii and Earth Radii/min)
    # converted to output unit Distance = km, Velocity = km/s in one step
    r_vec = rk * U * aE
    v_vec = (rdotk * U + rfdotk * V) * aE / 60.0

    # Error if radius below Earth's surface (unphysical)
    error_code = jnp.where(rk < 1.0, 6, error_code)

    return jnp.concatenate((r_vec, v_vec)), error_code