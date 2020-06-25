import astropy.time as astro_time
import astropy.units as u
import numpy as np
from astropy.coordinates import get_sun

# essential constans
GM = 398.6004415e12 * u.Unit("m3 /s2")
a_e = 6378137 * u.m
J_2 = 1.0826359e-3


class Orbit(object):

    def __init__(self, a, t0, u0):
        """

        :param a: orbital altitude ABOVE Earth
        :param t0: MJD of orbit start
        :param u0: latitude of orbit start
        :returns: 
        :rtype: 

        """

        self.a = a + a_e
        self.t0 = t0
        self.u0 = u0
        self.n_factor = (GM**(1/2) * self.a**(-3/2)).to('1/s') * u.rad

        # set to dawn dusk
        OmegaSun = -0.5 * np.pi

        # compute the inclination of the orbit
        omega_sun = 2*np.pi / 365.2422/86400 * u.Unit("rad/s")

        arg = -(omega_sun * np.power(self.a/a_e, 7./2.) /
                (1.5 * np.sqrt(GM/(a_e**3)) * J_2)).value

        self.inclination = np.arccos(arg)

        # pre compute the roatation out of the
        # raw coordiate frame into the terminator frame
        self.pre_matrix = self.R3(-OmegaSun).dot(self.R1(-self.inclination))

    def n(self, t):
        """
        raw orbital position at time t

        :param t: 
        :returns: 
        :rtype: 

        """

        return (self.n_factor * t).to(u.Unit("rad"))

    def u(self, t):
        """
        orbital position at time t

        :param t: 
        :returns: 
        :rtype: 

        """

        return self.n(t-self.t0) + self.u0

    def r_orb(self, t):
        """
        3D vector in raw Earth coordinates

        :param t: 
        :returns: 
        :rtype: 

        """

        ut = self.u(t)

        return self.a * np.array([np.cos(ut), np.sin(ut), [0] * len(ut)])

    def r_term(self, t):
        """
        3D vector in Earth terminator coordinates

        :param t: 
        :returns: 
        :rtype: 

        """

        return (np.array([np.dot(self.pre_matrix, r) for r in self.r_orb(t).T]) * u.m).T

    def r_eci(self, time):
        """
        3D vector in the earth interial frame


        :param time: 
        :returns: 
        :rtype: 

        """

        # get the Sun's RA from astropy
        a_sun = get_sun(time).ra.rad

        # get the 3D vector in the terminator frame
        rt = self.r_term(time)

        # shift the coordinate system to the Sun moving frame

        return (np.array([np.dot(self.R3(-a), r) for a, r in zip(a_sun, rt.T)]) * u.m).T

    def r_ecef(self, time):

        raise NotImplementedError()

        #a_sun = np.deg2rad(self.ra_sun(time))
        a_sun = self.lon_sun(time)

        rt = self.r_term(time)

        return (np.array([np.dot(self.R3(-a), r) for a, r in zip(a_sun, rt.T)]) * u.m).T

    def R3(self, alpha):
        """
        rotation matrix

        :param alpha: 
        :returns: 
        :rtype: 

        """

        cosa = np.cos(alpha)
        sina = np.sin(alpha)

        return np.array([[cosa, sina, 0.],
                         [-sina, cosa, 0.],
                         [0., 0., 1.]
                         ])

    def R1(self, alpha):
        """
        rotation matrix


        :param alpha: 
        :returns: 
        :rtype: 

        """

        cosa = np.cos(alpha)
        sina = np.sin(alpha)

        return np.array([[1., 0., 0.],
                         [0., cosa, sina],
                         [0, -sina, cosa]
                         ])

    def lon_sun(self, time):
        """
        This is supposed to compute the
        longitude of the sun

        :param time: 
        :returns: 
        :rtype: 

        """

        UT = np.mod(time.mjd, 1)
        return (UT + 0.5) * 2 * np.pi

    def ra_sun(self, time):
        """
        this is supposed to get the RA of the 
        Sun. It seems to not work

        :param time: 
        :returns: 
        :rtype: 

        """

        # self.theta0(time)
        return self.lon_sun(time) + time.sidereal_time("mean", "greenwich").rad

    def theta0(self, time):
        """

        This computes teh mean sideal and of
        the Sun

        :param time: 
        :returns: 
        :rtype: 

        """

        mjd0 = np.floor(time.mjd)
        UT = np.mod(time.mjd, 1)
        T = (mjd0-51544.5)/36525
        theta0 = 24110.54841+8640184.812866*T + 0.093104*(T**2) - 6.1e-6*(T**3)
        theta0 /= 86400.
        return theta0 * 2 * np.pi  # + 1.0027379093*UT
