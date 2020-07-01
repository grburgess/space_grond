import astropy.coordinates as coord
import astropy.time as astro_time
import astropy.units as u
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

        self.cart_position = ang2cart(ra,dec)

    def is_visible(self, orbit, time):
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

        horizon_angle += 28  # deg

        pos = orbit.r_eci(time).to("km")
        return _is_visibile(pos.T, self.cart_position, horizon_angle, len(time))

@nb.njit(fastmath=True)
def _is_visibile(sc_pos, grb_position, horizon_angle, N):

    is_visible = np.zeros(N)

    for n in range(N):
        ang_sep = np.rad2deg(get_ang(grb_position, -sc_pos[n]))

        if ang_sep > horizon_angle:

            is_visible[n] = 1

    return is_visible
    
    

class Observation(object):
    def __init__(self, grb, orbit, delay_time=15 * u.min):
        """
        computes the observational parameters

        :param grb: 
        :param orbit: 
        :param delay_time: 
        :returns: 
        :rtype: 

        """

        self._grb = grb
        self._orbit = orbit
        self._delay_time = delay_time

        self._observed_at_start = False
        self._observe()

    def _observe(self):

        # set up at least two orbits
        obs_time = self._grb.time + self._delay_time

        n_times = 500

        time = obs_time + np.linspace(0.0, 200, n_times) * u.minute

        # find the visible times
        viz = self._grb.is_visible(self._orbit, time)

        # return the indices containing visible times

        if viz[0]:

            self._observed_at_start = True

        on_times = slice_disjoint(np.where(viz)[0])

        self._obs_is_too_short = False

        if self._observed_at_start:

            # if the GRB is visible at the observing time

            end_of_current_obs = time[on_times[0][1]]

            self._time_left_to_observe = (
                end_of_current_obs - obs_time).to("min")

            if self._time_left_to_observe.value < 10.0:

                self._obs_is_too_short = True

                begin_of_next_obs = time[on_times[1][0]]
                end_of_next_obs = time[on_times[1][1]]

                self._next_visible_time_from_now = (
                    begin_of_next_obs - end_of_current_obs
                ).to("min")

                self._remaining_time = (
                    end_of_next_obs - begin_of_next_obs).to("min")

        else:

            begin_of_next_obs = time[on_times[0][0]]

            end_of_next_obs = time[on_times[0][1]]

            self._next_visible_time_from_now = (
                begin_of_next_obs - obs_time).to("min")

            self._remaining_time = (
                end_of_next_obs - begin_of_next_obs).to("min")

    @property
    def observed_at_start(self):
        return self._observed_at_start

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



        if self._observed_at_start:
            output = f"Observation started at {self._grb.time + self._delay_time}"
            output = f"{output} \nand will be observable for {self._time_left_to_observe}"

            if self._obs_is_too_short:

                output = f"{output}\nAs the observation is not long enough"
                output = f"{output}\nwe will be able to observe again in {self._next_visible_time_from_now} for {self._remaining_time}"

        else:

            output = f"Observation was NOT possible at {self._grb.time + self._delay_time}"
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
    #tmp = np.clip(np.dot(X1 / norm1, X2 / norm2), -1, 1)
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
