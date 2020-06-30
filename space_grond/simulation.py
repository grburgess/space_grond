import pandas as pd
import astropy.time as astro_time
import astropy.units as u

from space_grond.orbit import Orbit
from space_grond.grb import GRB, Observation


class GRBFactory(object):
    def __init__(self, file_name, t0):

        self._day_string = t0.to_value("isot").split("T")[0]

        self._grbs = {}

        grb_list = pd.read_table(file_name, names=["grb", "time", "ra", "dec"])

        for g in grb_list.iterrows():
            utc = f"{self._day_string}T{g[1]['time']}"
            time = astro_time.Time(utc)

            grb = GRB(g[1]["ra"], g[1]["dec"], time)

            self._grbs[g[1]["grb"]] = grb

    @property
    def grbs(self):

        return self._grbs


class Simulation(object):

    def __init__(self, grbs, altitude, t0, u0, delay_time):
        """
        Create and run a simulation with a given set of initial conditions
        Be sure to specify units via astropy

        :param grbs: 
        :param altitude: 
        :param t0: 
        :param u0: 
        :param delay_time: 
        :returns: 
        :rtype: 

        """

        self._observations = {}

        orbit = Orbit(a=altitude, t0=t0, u0=u0 * u.deg)

        for grb_name, grb in grbs.items():

            obs = Observation(grb, orbit)

            self._observations[grb_name] = obs

    @property
    def observations(self):

        return self._observations

    @classmethod
    def from_observation_file(cls, file_name, altitude, t0, u0, delay_time):

        factory = GRBFactory(file_name, t0)

        return cls(factory.grbs, altitude, t0, u0, delay_time)
