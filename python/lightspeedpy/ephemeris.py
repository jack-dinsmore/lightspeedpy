import numpy as np
from astropy import units as u
from astropy.time import Time
import pint
from pint.models import model_builder

VALID_KEYS = ["pepoch", "nu", "nudot", "nuddot", "asini", "pb", "tasc"]

class Ephemeris():
    def __init__(self, parfile):
        self.model = model_builder.get_model(parfile)
        print(self.model.get_params_dict())
        self.nu = self.model.F0.value

    def from_file(parfile):
        return Ephemeris(parfile)

    def get_phase(self, times):
        times_mjd = times / (3600 * 24)
        toas = pint.toa.TOA(
            Time(times_mjd, format="mjd", scale="tt"), # TODO
            4.5 * u.us, # TODO
            # freq=300 * u.THz, # TODO
            # obs="LCO",
            # backend="GUPPI",
        )
        phase = self.model.phase(toas)
        print(phase)
        return phase