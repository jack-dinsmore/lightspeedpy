import numpy as np
from scipy.interpolate import interp1d
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import EarthLocation
import pint, os, pickle, datetime
from pint.models import model_builder
pint.logging.setup(level="WARNING")

TMP_LOCATION = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "tmp"))
SAVE_TIME = 30 # days

def models_are_equal(m1, m2):
    for key, val in m1.items():
        if key not in m2:
            return False

        saved_val = val.value
        model_val = m2[key].value
        if type(saved_val) is float or type(saved_val) is np.longdouble or type(saved_val) is np.float64:
            if float(saved_val) != float(model_val) and not np.isnan(saved_val):
                return False
        elif type(saved_val) is str:
            if saved_val != str(model_val):
                return False
        elif type(saved_val) is bool:
            if saved_val != bool(model_val):
                return False
        elif type(saved_val) is int:
            if saved_val != int(model_val):
                return False
    
    return True

class EphemerisLibrary():
    def __init__(self):
        if not os.path.exists(TMP_LOCATION):
            os.mkdir(TMP_LOCATION)

        self.ephemerides = {}
        for filename in os.listdir(TMP_LOCATION):
            if not filename.endswith(".dat"): continue
            stem = filename.split('.')[0]

            with open(f"{TMP_LOCATION}/{filename}", 'r') as f:
                time_since_used = datetime.datetime.now() - datetime.datetime.fromisoformat(f.readline())
                if time_since_used.days > SAVE_TIME:
                    # Delete the file for being out of date
                    os.remove(f"{TMP_LOCATION}/{stem}.pkl")
                    os.remove(f"{TMP_LOCATION}/{stem}.npy")
                    os.remove(f"{TMP_LOCATION}/{stem}.dat")
                    continue

            psr, index = stem.split("_")
            if psr in self.ephemerides:
                self.ephemerides[psr].append(int(index))
            else:
                self.ephemerides[psr] = [int(index)]

    def get(self, model, these_timestamps):
        psr = model["PSR"].value
        if psr not in self.ephemerides:
            return None
        
        for index in self.ephemerides[psr]:
            with open(f"{TMP_LOCATION}/{psr}_{index}.pkl", 'rb') as f:
                saved_model = pickle.load(f)
                if not models_are_equal(saved_model, model): continue

            # This model matches the provided ephemeris.
            timestamps, phases = np.load(f"{TMP_LOCATION}/{psr}_{index}.npy")

            # Update the time file
            with open(f"{TMP_LOCATION}/{psr}_{index}.dat", 'w') as f:
                f.write(str(datetime.datetime.now()))

            if len(timestamps) != len(these_timestamps):
                # The timestamps are not the same, meaning different files are being loaded with the same ephem
                return None
            max_delta = np.max(np.abs(timestamps - these_timestamps))
            if max_delta > 1e-8:
                # The timestamps are not the same, meaning different files are being loaded with the same ephem
                return None

            return phases
        
        # The model was not found
        return None

    def push(self, model, timestamps, phases):
        psr = model['PSR'].value
        if psr not in self.ephemerides:
            index = 0
        else:
            for index in range(max(self.ephemerides[psr])+2):
                if index not in self.ephemerides[psr]:
                    break
        save_name = f"{psr}_{index}"

        if os.path.exists(f"{TMP_LOCATION}/{save_name}.pkl"):
            print("WARNING: Overwriting existing time data")

        with open(f"{TMP_LOCATION}/{save_name}.pkl", 'wb') as f:
            pickle.dump(model, f)
        np.save(f"{TMP_LOCATION}/{save_name}.npy", [timestamps, phases])
        with open(f"{TMP_LOCATION}/{save_name}.dat", 'w') as f:
            f.write(str(datetime.datetime.now()))

class Ephemeris():
    """
    Class to contain a PINT ephemeris and assign phases to lightspeed data. Ephemerides are saved in the lightspeedpy/tmp directory so that they don't have to be recomputed.

    Parameters
    ----------
    parfile : str
        File name of the PINT ephemeris file
    data_set : DataSet
        The data set to create timestamps for
    observatory : str, optional
        Observatory where Lightspeed was. Default: LCO
    """
    def __init__(self, parfile, data_set, observatory="LCO"):
        timestamps = data_set._get_timestamps()
        pint.observatory.topo_obs.TopoObs(observatory, location=EarthLocation.of_site(observatory))

        self.model = model_builder.get_model(parfile)
        self.nu = self.model["F0"].value

        library = EphemerisLibrary()
        result = library.get(self.model, timestamps)
        self.timestamps = np.copy(timestamps)

        if result is None:
            print("Calculating phases")
            ephem = self.model["EPHEM"].value

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
            library.push(self.model, self.timestamps, self.phases)
            print("Finished calculating phases")

        else:
            self.phases = result

        self.interpolator = interp1d(self.timestamps, self.phases, bounds_error=False, fill_value="extrapolate") # Extrapolate. This will extrapolate the calculated phases throughout the last frame, which is out of bounds of the interpolator.

    def get_phase(self, time):
        """
        Returns the phase (0 to 1) corresponding to a specific time
        """
        phase = self.interpolator(time)
        return phase - np.floor(phase)