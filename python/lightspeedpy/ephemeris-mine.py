"""
EPHEMERIS FORMAT

The ephemeris files should be stored like this:
KEY1 VALUE1
# Comment
KEY2 VALUE2
...

where the valid keys are PEPOCH, NU, NUDOT, and NUDDOT.
"""

import numpy as np

VALID_KEYS = ["pepoch", "nu", "nudot", "nuddot", "asini", "pb", "tasc"]

class Ephemeris():
    def __init__(self, pepoch, nu, nudot=0, nuddot=0, asini=0, pb=1, tasc=0):
        # asini has to be in light seconds
        self.pepoch = pepoch * 3600 * 24 # Convert from mjd to seconds
        self.nu = nu
        self.nudot = nudot
        self.nuddot = nuddot

        self.asini = asini
        self.orbit_freq = 2*np.pi/pb
        self.tasc = tasc * 3600 * 24 - self.pepoch # Convert to seconds difference from pepoch

    def from_file(filename):
        values = {}
        with open(filename, 'r') as f:
            for line in f.readlines():
                if line == "": continue
                if line == "\n": continue
                if line.startswith("#"): continue
                key, value = line.split(" ")
                key = key.lower()
                if key not in VALID_KEYS:
                    # Skip the line
                    continue
                value = float(value)
                values[key] = value
        return Ephemeris(**values)
    
    def get_phase(self, time):
        delta_t = (time - self.pepoch)
        phase = self.nu*delta_t
        phase += self.nudot*delta_t**2 / 2
        phase += self.nuddot*delta_t**3 / 6
        if self.asini != 0:
            # Binary orbit
            orbit_phase = self.orbit_freq*(delta_t - self.tasc)
            orbit_phase0 = self.orbit_freq*(0 - self.tasc)
            sin_phase = self.asini * np.sin(orbit_phase)
            cos_phase = self.asini * np.cos(orbit_phase)
            sin_phase0 = self.asini * np.sin(orbit_phase0)
            cos_phase0 = self.asini * np.cos(orbit_phase0)
            phase -= self.nu * sin_phase/self.orbit_freq
            phase -= self.nudot * (
                delta_t*sin_phase/self.orbit_freq + 
                (cos_phase-cos_phase0)/self.orbit_freq**2
            )
            phase -= self.nuddot * (
                delta_t**2*sin_phase/(2*self.orbit_freq) + 
                delta_t*cos_phase/self.orbit_freq**2 + 
                - (sin_phase - sin_phase0)/self.orbit_freq**3
            )

        phase -= np.floor(phase)
        return phase