import numpy as np
from scipy.interpolate import interp1d
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import EarthLocation
import pint
from pint.models import model_builder
pint.logging.setup(level="WARNING")

class Ephemeris():
    def __init__(self, parfile, timestamps, observatory):
        pint.observatory.topo_obs.TopoObs(observatory, location=EarthLocation.of_site(observatory))

        self.model = model_builder.get_model(parfile)
        self.nu = self.model["F0"].value
        ephem = self.model["EPHEM"].value

        self.timestamps = np.copy(timestamps)
        times_mjd = timestamps / (3600 * 24)
        toas = pint.toa.get_TOAs_array(
            Time(times_mjd, format="mjd", scale="utc"),
            freqs=np.ones(len(times_mjd)) * 500 * u.THz,  # Dummy frequency
            errors=np.ones(len(times_mjd)) * 1 * u.us,  # Dummy errors
            ephem=ephem,
            obs=observatory,
        )
        phases = self.model.phase(toas)
        delta_phase_int = phases.int - np.min(phases.int)
        self.phases = delta_phase_int.astype(np.float64) + phases.frac.astype(np.float64)
        self.interpolator = interp1d(self.timestamps, self.phases, bounds_error=False, fill_value="extrapolate") # Extrapolate. This will extrapolate the calculated phases throughout the last frame, which is out of bounds of the interpolator.

    def get_phase(self, time):
        phase = self.interpolator(time)
        return phase - np.floor(phase)