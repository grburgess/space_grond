import astropy.coordinates as coord
import astropy.time as astro_time
import astropy.units as u
from astropy.coordinates import get_sun, get_moon

import numpy as np
import numba as nb

a_e = 6378137 * u.m


class GRB(object):
    def __init__(self, ra, dec, time):
        """
        Simple container class for GRB information

        :param ra: 
        :param dec: 
        :param time: 
        :returns: 
        :rtype: 

        """

        self.ra = ra
        self.dec = dec
        self.time = time
        self.coord = coord.SkyCoord(ra=ra, dec=dec, unit="deg", frame="icrs")

        self.cart_position = ang2cart(ra, dec)

    def is_visible(self, orbit, time, limb_angle=None, sun_angle=None, moon_angle=None):
        """
        checks if the GRB is visible above the Earth limb
        for the given times

        :param orbit: 
        :param time: 
        :returns: 
        :rtype: 

        """

        horizon_angle = 90 - np.rad2deg(
            np.arccos((a_e / orbit.a).to(u.dimensionless_unscaled).value)
        )

        # add Earth limb restraint

        if limb_angle is not None:

            horizon_angle += limb_angle  # deg

        pos = orbit.r_eci(time).to("km")

        sun_pos = None

        if sun_angle is not None:

            sun_pos = get_moon(time).cartesian.xyz.to("km").value.T

        moon_pos = None

        if moon_angle is not None:

            moon_pos = get_moon(time).cartesian.xyz.to("km").value.T

        return _is_visibile(
            pos.T,
            self.cart_position,
            horizon_angle,
            sun_angle,
            sun_pos,
            moon_angle,
            moon_pos,
            len(time),
        )


@nb.njit(fastmath=True)
def _is_visibile(
    sc_pos, grb_position, horizon_angle, sun_angle, sun_pos, moon_angle, moon_pos, N
):

    is_visible = np.zeros(N)

    for n in range(N):
        ang_sep = np.rad2deg(get_ang(grb_position, -sc_pos[n]))

        sun_block = False
        earth_block = False
        moon_block = False

        if sun_angle is not None:

            sun_sep = np.rad2deg(get_ang(grb_position, sun_pos[n]))

            if sun_sep <= sun_angle:

                sun_block = True

        if moon_angle is not None:

            moon_sep = np.rad2deg(get_ang(grb_position, moon_pos[n]))

            if moon_sep <= moon_angle:

                moon_block = True

        if ang_sep <= horizon_angle:

            earth_block = True

        if (not earth_block) and (not sun_block) and (not moon_block):

            is_visible[n] = 1

    return is_visible


class Observation(object):
    def __init__(
        self,
        grb,
        orbit,
        delay_time=15 * u.min,
        max_waiting_time=200 * u.min,
        limb_angle=28,
        sun_angle=None,
        moon_angle=None,
    ):
        """
        computes the observational parameters

        :param grb: 
        :param orbit: 
        :param delay_time:
        :param max_waiting_time: maximum time to wait before giving up observation
        :param limb_angle: distance from earth limb in deg
        :param sun_angle: distance from sun in deg
        :param moon_angle: distance from moon in deg
        :returns: 
        :rtype: 

        """

        self._grb = grb
        self._orbit = orbit
        self._delay_time = delay_time
        self._max_waiting_time = max_waiting_time
        self._limb_angle = limb_angle
        self._sun_angle = sun_angle
        self._moon_angle = moon_angle

        self._observed_at_start = False
        self._will_never_be_seen = False
        self._observe()

    def _observe(self):

        # set up at least two orbits
        obs_time = self._grb.time + self._delay_time

        n_times = 500

        time = (
            obs_time
            + np.linspace(0.0, self._max_waiting_time.to("minute").value, n_times)
            * u.minute
        )

        # find the visible times
        viz = self._grb.is_visible(
            self._orbit, time, self._limb_angle, self._sun_angle, self._moon_angle
        )

        # return the indices containing visible times

        if viz[0]:

            self._observed_at_start = True

        tmp = np.where(viz)[0]

        if len(tmp) == 0:

            self._will_never_be_seen = True

        else:

            on_times = slice_disjoint(np.where(viz)[0])

            self._obs_is_too_short = False

            if self._observed_at_start:

                # if the GRB is visible at the observing time

                end_of_current_obs = time[on_times[0][1]]

                self._time_left_to_observe = (end_of_current_obs - obs_time).to("min")

                if self._time_left_to_observe.value < 10.0:

                    self._obs_is_too_short = True

                    begin_of_next_obs = time[on_times[1][0]]
                    end_of_next_obs = time[on_times[1][1]]

                    self._next_visible_time_from_now = (
                        begin_of_next_obs - end_of_current_obs
                    ).to("min")

                    self._remaining_time = (end_of_next_obs - begin_of_next_obs).to(
                        "min"
                    )

            else:

                begin_of_next_obs = time[on_times[0][0]]

                end_of_next_obs = time[on_times[0][1]]

                self._next_visible_time_from_now = (begin_of_next_obs - obs_time).to(
                    "min"
                )

                self._remaining_time = (end_of_next_obs - begin_of_next_obs).to("min")

    @property
    def observed_at_start(self):
        return self._observed_at_start

    @property
    def will_never_be_seen(self):
        return self._will_never_be_seen

    @property
    def obs_is_too_short(self):
        return self._obs_is_too_short

    @property
    def time_left_to_observe(self):
        return self._time_left_to_observe

    @property
    def next_visible_time_from_now(self):
        return self._next_visible_time_from_now

    @property
    def remaining_time(self):
        return self._remaining_time

    def __repr__(self):

        if self._will_never_be_seen:

            output = f"This GRB is blocked for all of the max waiting time of {self._max_waiting_time.to('minute')}"

        elif self._observed_at_start:
            output = f"Observation started at {self._grb.time + self._delay_time}"
            output = (
                f"{output} \nand will be observable for {self._time_left_to_observe}"
            )

            if self._obs_is_too_short:

                output = f"{output}\nAs the observation is not long enough"
                output = f"{output}\nwe will be able to observe again in {self._next_visible_time_from_now} for {self._remaining_time}"

        else:

            output = (
                f"Observation was NOT possible at {self._grb.time + self._delay_time}"
            )
            output = f"{output}\nThus, we wait for {self._next_visible_time_from_now} and can observe for {self._remaining_time}"

        return output


@nb.njit(fastmath=True)
def ang2cart(ra, dec):
    """
    :param ra:
    :param dec:
    :return:
    """
    pos = np.zeros(3)
    ra = np.deg2rad(ra)
    dec = np.deg2rad(dec)

    pos[0] = np.cos(dec) * np.cos(ra)
    pos[1] = np.cos(dec) * np.sin(ra)
    pos[2] = np.sin(dec)

    return pos


@nb.njit(fastmath=True)
def get_ang(X1, X2):
    """
    :param X1:
    :param X2:
    :return:
    """
    norm1 = np.sqrt(X1.dot(X1))
    norm2 = np.sqrt(X2.dot(X2))
    # tmp = np.clip(np.dot(X1 / norm1, X2 / norm2), -1, 1)
    tmp = np.dot(X1 / norm1, X2 / norm2)
    return np.arccos(tmp)


def slice_disjoint(arr):
    """
    Returns an array of disjoint indices from a bool array
    :param arr: and array of bools
    """

    slices = []
    start_slice = arr[0]
    counter = 0
    for i in range(len(arr) - 1):
        if arr[i + 1] > arr[i] + 1:
            end_slice = arr[i]
            slices.append([start_slice, end_slice])
            start_slice = arr[i + 1]
            counter += 1
    if counter == 0:
        return [[arr[0], arr[-1]]]
    if end_slice != arr[-1]:
        slices.append([start_slice, arr[-1]])
    return slices
