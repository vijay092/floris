import numpy as np
import bisect

class WindFieldBuffer():
    def __init__(self, combination_function, num_turbines):
        self.test = 0

        self._future_wind_speeds = []
        self._current_wind_speed = None

        self._future_wind_dirs = []
        self._current_wind_dir = None
        self._current_coord = None

        self._future_wake_deficits = [ [] for _ in range(num_turbines) ]
        self._current_wake_deficits = [ None for _ in range(num_turbines) ]

        self._combination_function = combination_function

    def add_wind_direction(self, old_wind_direction, new_wind_direction, sim_time, old_coord):
        self._future_wind_dirs.append((old_wind_direction, new_wind_direction, sim_time, old_coord))

    def add_wind_speed(self, new_wind_speed, sim_time):

        # NOTE: I think bisect_left is the correct choice, maybe bisect_right
        slice_index = bisect.bisect_left([wind_speed[1] for wind_speed in self._future_wind_speeds], sim_time)

        self._future_wind_speeds.insert(slice_index, (new_wind_speed, sim_time))

        self._future_wind_speeds = self._future_wind_speeds[:slice_index+1]

        #self._future_wind_speeds.append((new_wind_speed, sim_time))

    def initialize_wind_speed(self, old_wind_speed, new_wind_speed, overwrite=False):
        """
        This method is intended to set the initial wind speed if it is not already set.

        Args:
            wind_speed: Wind speed in degrees relative to 270 to be initialized (float).

            overwrite: Whether or not the current wind speed should be overwritten (boolean).
        """
        if self._current_wind_speed is None and not overwrite:
            self._current_wind_speed = old_wind_speed
        elif overwrite:
            self._current_wind_speed = new_wind_speed
        #print("Called initialize_wind_speed, current wind speed is", self._current_wind_speed)
        return

    def initialize_wind_direction(self, wind_direction, coord, overwrite=False):
        """
        This method is intended to set the initial wind direction if it is not already set.

        Args:
            wind_direction: Wind direction in degrees relative to 270 to be initialized (float).

            coord: Turbine coordinate relative to wind direction.

            overwrite: Whether or not the current wind direction should be overwritten (boolean).
        """

        if self._current_wind_dir is None:
            self._current_wind_dir = wind_direction
            self._current_coord = coord
            #print("Current wind direction None, setting to", wind_direction)
        elif overwrite:
            self._current_wind_dir = wind_direction
            self._current_coord = coord
            #print("Overwrite is True, setting wind direction to", wind_direction)
        return

    def get_wind_direction(self, wind_direction, coord, send_wake, sim_time):
        """
        Method to determine what wind direction the flow field should be set to.

        Args:
            wind_direction: Wind direction in degrees relative to 270 that the farm should be set to
                if there are no wind directions stored in the internal buffer (float).

            coord: Turbine coord that should be set if there are no wind directions stored in the
                internal buffer.

            send_wake: Variable specifying whether or not the turbine currently should 
                propagate its wake downstream.

            sim_time: Current simulation time (int).

        Returns:
            Tuple of wind direction setpoint, coordinate setpoint, and send_wake, a boolean that
                signifies whether or not a turbine needs to propagate its wake downstream.
        """

        if len(self._future_wind_dirs) > 0 and self._future_wind_dirs[0][1] == sim_time:
            send_wake_temp = True
            self._current_wind_dir = self._future_wind_dirs[0][0]
            self._current_coord = self._future_wind_dirs[0][3]
            self._future_wind_dirs.pop(0)
        else:
            send_wake_temp = False

        if self._current_wind_dir is not None and self._current_coord is not None:
            wind_direction_set = self._current_wind_dir
            coord_set = self._current_coord
        else:
            wind_direction_set = wind_direction
            coord_set = coord

        return (wind_direction_set, coord_set, send_wake or send_wake_temp)

    def get_wind_speed(self, wind_speed, send_wake, sim_time):
        """
        Method to determine what wind speed the flow field should be set to.

        Args:
            wind_speed: Wind speed that the farm should be set to
                if there are no wind speeds stored in the internal buffer (float).

            send_wake: Variable specifying whether or not the turbine currently should 
                propagate its wake downstream.

            sim_time: Current simulation time (int).

        Returns:
            Tuple of wind direction setpoint, coordinate setpoint, and send_wake, a boolean that
                signifies whether or not a turbine needs to propagate its wake downstream.
        """

        if len(self._future_wind_speeds) > 0 and self._future_wind_speeds[0][1] == sim_time:
            send_wake_temp = True
            self._current_wind_speed = self._future_wind_speeds[0][0]
            self._future_wind_speeds.pop(0)
        else:
            send_wake_temp = False
        
        if self._current_wind_speed is not None:
            wind_speed_set = self._current_wind_speed
        else:
            wind_speed_set = wind_speed

        return (wind_speed_set, send_wake or send_wake_temp)

    def _search_and_combine_u_wakes(self, wake_deficit, wake_dims, index, sim_time):
        """
        Finds all wake deficits at a given simulation time in the buffer and averages them.

        Args:
            wake_deficit: Wake deficit that should be used if there are no wake deficits in 
                the buffer.
            
            wake_dims: Dimensions of the wake deficit matrix (tuple).

            index: What index of the wake deficit buffer to look at (int).

            sim_time: The current simulation time (int).
        """
        current_effects = []
 
        for wake_effect in self._future_wake_deficits[index]:
            if wake_effect[1] == sim_time:
                current_effects.append(wake_effect[0])

        #if len(current_effects) >= 1: print("Well, this should not happen.")
        if len(current_effects) == 0:
            return wake_deficit
        else:
            return np.mean(current_effects, axis=0)

    def get_u_wake(self, wake_dims, send_wake, sim_time):
        """
        This method returns the filled u_wake matrix for the buffer.

        Args:
            wake_dims: Dimensions of the wake deficit matrix (tuple, ints)

            send_wake: Variable specifying whether or not the turbine currently should 
                propagate its wake downstream.

            sim_time: Current simulation time (int).
        """

        send_wake_temp = False
        #print(self._current_wake_deficits)
        # iterate through the turbine.wake_effects list, which contains a wake_effect entry for every turbine in the farm
        for k in range(len(self._future_wake_deficits)):
            # the third element of the wake_effect entry is the index of the turbine (in the sorted_map) that the wake 
            # corresponds to 
            new_u_wake = self._search_and_combine_u_wakes(self._current_wake_deficits[k], wake_dims, k, sim_time)
            old_u_wake = self._current_wake_deficits[k]
            #print(new_u_wake is None and old_u_wake is None)
            send_wake_temp = send_wake_temp or not(new_u_wake is None and old_u_wake is None and (np.array([new_u_wake == old_u_wake])).all())
            self._current_wake_deficits[k] = new_u_wake

        # combine the effects of all the turbines together
        u_wake = np.zeros(wake_dims)
        for i in range(len(self._current_wake_deficits)):
            if self._current_wake_deficits[i] is not None: 
                # the first element of the turb_u_wakes entry is the actual wake deficit contribution from the 
                # turbine at that index
                u_wake = self._combination_function(u_wake, self._current_wake_deficits[i])

        return (u_wake, send_wake or send_wake_temp)

    def initialize_wake_deficit(self, wake_deficit, index):
        """
        This method is intended to set an initial wake deficit after the most upstream turbine has determined its
        wake deficit.
        """

        # for el in self._current_wake_deficits:
        #     if el is None:
        #         self._current_wake_deficits[index] = (wake_deficit, None)

        if self._current_wake_deficits[index] is None:
            
            self._current_wake_deficits[index] = wake_deficit

        #print(self._current_wake_deficits)

    def add_wake_deficit(self, new_wake_deficit, index, sim_time):
        """
        This method is intended to add wake deficit matrices into the buffer. 

        Args:
            wake_effect: The wake effect to be added (np array).

            index: The index of the wake deficit buffer the wake effect should be added at (this
                corresponds to which turbine number caused the wake) (int).

            sim_time: The simulation time that the wake should come into effect at (int).
        """

        # NOTE: I think bisect_left is the correct choice, maybe bisect_right
        slice_index = bisect.bisect_left([wake_deficit[1] for wake_deficit in self._future_wake_deficits[index]], sim_time)

        self._future_wake_deficits[index].insert(slice_index, (new_wake_deficit, sim_time))

        self._future_wake_deficits[index] = self._future_wake_deficits[index][:slice_index+1]

        #self._future_wake_deficits[index].append((new_wake_deficit, sim_time))