import numpy as np
import astropy.units as u
import astropy.time as astro_time
from astropy.coordinates import get_sun



GM = 398.6004415e12 * u.Unit("m3 /s2")
a_e = 6378137 * u.m
J_2 = 1.0826359e-3


class Orbit(object):

    def __init__(self, a, t0, u0):
        """FIXME! briefly describe function

        :param a: 
        :param t0: 
        :param u0: 
        :returns: 
        :rtype: 

        """

        
        self.a = a
        self.t0 = t0
        self.u0 = u0
        self.n_factor = (GM**(1/2) * self.a**(-3/2)).to('1/s') * u.rad
        self.OmegaSun = -0.5 * np.pi

        omega_sun = 2*np.pi / 365.2422/86400 * u.Unit("rad/s")
        arg = -(omega_sun * np.power(self.a/a_e, 7./2.) /
                (1.5 * np.sqrt(GM/(a_e**3)) * J_2)).value

        self.inclination = np.arccos(arg)

        self.pre_matrix = self.R3(-self.OmegaSun).dot(self.R1(-self.inclination))

    def n(self, t):
        """FIXME! briefly describe function

        :param t: 
        :returns: 
        :rtype: 

        """

        return (self.n_factor * t).to(u.Unit("rad"))

    def u(self, t):
        """FIXME! briefly describe function

        :param t: 
        :returns: 
        :rtype: 

        """

        return self.n(t-self.t0) + self.u0

    def r_orb(self, t):
        """FIXME! briefly describe function

        :param t: 
        :returns: 
        :rtype: 

        """

        ut = self.u(t)

        return self.a * np.array([np.cos(ut), np.sin(ut), [0] * len(ut)])

    def r_term(self, t):

        return (np.array([np.dot(self.pre_matrix, r) for r in self.r_orb(t).T]) * u.m).T

    def r_eci(self, time):


        a_sun = get_sun(time).ra.rad

        rt = self.r_term(time)

        return (np.array([np.dot(self.R3(-a), r) for a, r in zip(a_sun, rt.T)]) * u.m).T

    def r_ecef(self, time):

        #a_sun = np.deg2rad(self.ra_sun(time))
        a_sun = self.lon_sun(time)

        rt = self.r_term(time)

        return (np.array([np.dot(self.R3(-a), r) for a, r in zip(a_sun, rt.T)]) * u.m).T

    def R3(self, alpha):

        cosa = np.cos(alpha)
        sina = np.sin(alpha)

        return np.array([[cosa, sina, 0.],
                         [-sina, cosa, 0.],
                         [0., 0., 1.]
                         ])

    def R1(self, alpha):

        cosa = np.cos(alpha)
        sina = np.sin(alpha)

        return np.array([[1., 0., 0.],
                         [0., cosa, sina],
                         [0, -sina, cosa]
                         ])

    def lon_sun(self, time):

        UT = np.mod(time.mjd, 1)
        return (UT + 0.5) * 2 * np.pi

    def ra_sun(self, time):

        # self.theta0(time)
        return self.lon_sun(time) + time.sidereal_time("mean", "greenwich").rad

    def theta0(self, time):
        mjd0 = np.floor(time.mjd)
        UT = np.mod(time.mjd, 1)
        T = (mjd0-51544.5)/36525
        theta0 = 24110.54841+8640184.812866*T + 0.093104*(T**2) - 6.1e-6*(T**3)
        theta0 /= 86400.
        return theta0 * 2 * np.pi  # + 1.0027379093*UT
