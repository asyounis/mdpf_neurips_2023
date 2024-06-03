# Standard Imports
import numpy as np
from tqdm import tqdm
import torch
import os

# Project imports
from utils import *
from datasets.base_dataset import *


class Car:
    def __init__(self, car_velocity, max_angle_pertubation, XY_RANGE=20.0, max_steps_for_target=1000):

        # Parameters of how the car moves
        self.car_velocity = float(car_velocity)
        self.max_angle_pertubation = float(max_angle_pertubation)
        self.XY_RANGE = float(XY_RANGE)
        
        # Car starting location
        self.car_state = np.asarray([0.0, 0.0, 0.0], dtype="float32") 

        # Randomly init car
        self.car_state[0] = np.random.uniform(-self.XY_RANGE/2.0, self.XY_RANGE/2.0)
        self.car_state[1] = np.random.uniform(-self.XY_RANGE/2.0, self.XY_RANGE/2.0)
        self.car_state[2] = np.random.uniform(-np.pi, np.pi)

        # The current target location
        new_x = np.random.uniform(-self.XY_RANGE/2.0, self.XY_RANGE/2.0)
        new_y = np.random.uniform(-self.XY_RANGE/2.0, self.XY_RANGE/2.0)
        self.current_target_location = np.asarray([new_x, new_y])

        # Some things to keep track of
        self.max_steps_for_target = max_steps_for_target
        self.steps_since_direction_change = 0

        # The total number of steps taken
        self.total_number_of_steps = 0


    def step_car(self, car_state, dt=0.25):
        current_location = car_state[0:2]
        current_angle = car_state[2]

        # If we arrived at our target location then we have made it
        if(self.check_if_car_is_within_range_of_point(current_location, self.current_target_location, 0.5) or (self.steps_since_direction_change > self.max_steps_for_target) or (self.total_number_of_steps == 0)):
            new_x = np.random.uniform(-self.XY_RANGE/2.0, self.XY_RANGE/2.0)
            new_y = np.random.uniform(-self.XY_RANGE/2.0, self.XY_RANGE/2.0)
            self.current_target_location = np.asarray([new_x, new_y])

            self.steps_since_direction_change = 0
            
        # Update the heading of the car to point towards the target
        current_angle += self.get_heading_adjustment_for_location(current_location, current_angle, self.current_target_location)
        current_angle = self.wrap_angle(current_angle)

        # Update the car position based on the new angle
        dx = self.car_velocity * np.cos(current_angle)
        dy = self.car_velocity * np.sin(current_angle)

        # Compute the new x and y positions of the car
        new_x = (dx*dt) + current_location[0]
        new_y = (dy*dt) + current_location[1]

        # Create the new state
        new_state = np.asarray([new_x, new_y, current_angle])


        self.total_number_of_steps += 1
        self.steps_since_direction_change += 1

        return new_state



    def simulate_car_for_n_steps(self, num_steps, skip=None):

        # If we are going to skip a step we still want to return the correct number of steps
        if(skip is not None):
            num_steps = num_steps * skip

        car_states = []
        car_states.append(self.car_state)

        last_state = car_states[-1]
        # print("Simulating Car")
        # for i in tqdm(range(num_steps)):
        for i in range(num_steps):
            state = self.step_car(last_state)
            last_state = state

            if(skip is None):
                car_states.append(state)
            else:
                if((i % skip) == 0):
                    car_states.append(state)



        return np.asarray(car_states)


    def get_heading_adjustment_for_location(self, current_location, current_angle, target_location):

        # Get the angle between the car and the
        angle_to_origin = np.arctan2(target_location[1]-current_location[1], target_location[0]-current_location[0])

        # Find the optimal way to adjust the car angle to head back to the origin
        # Note this only does the sign so we need to multiply by the magnitude later
        # Taken from:
        #       https://stackoverflow.com/questions/25506470/how-to-get-to-an-angle-choosing-the-shortest-rotation-direction
        if(current_angle < angle_to_origin):
            if(np.abs(current_angle - angle_to_origin) < np.pi):
                angle_adjust = 1
            else:
                angle_adjust =-1
        else:
            if(np.abs(current_angle - angle_to_origin) < np.pi):
                angle_adjust = -1
            else:
                angle_adjust = 1


        # Get the magnitude of the change since we only computed the sign so far
        angle_adjust *= min(np.abs(current_angle - angle_to_origin), self.max_angle_pertubation)

        return angle_adjust

    def check_if_car_is_within_range_of_point(self, current_location, point, range_distance):
        distance = np.sum((current_location-point)**2)
        distance = np.sqrt(distance)
        return (distance <= range_distance)

    def wrap_angle(self, angle):
        # return angle
        return ((angle - np.pi) % (2.0 * np.pi)) - np.pi



class Sensor:
    def __init__(self, sensor_name, sensor_params):

        # In case we want to reference the sensor at a later point in time
        self.sensor_name = sensor_name

        # Extract the params
        x_location = get_parameter_safely("x", sensor_params, "sensor_params")
        y_location = get_parameter_safely("y", sensor_params, "sensor_params")
        self.sensor_bias_concentration = get_parameter_safely("sensor_bias_concentration", sensor_params, "sensor_params")        
        self.sensor_concentration = get_parameter_safely("sensor_concentration", sensor_params, "sensor_params")
        
        # Check if we should mix with a uniform dist
        self.mix_with_uniform = get_parameter_safely("mix_with_uniform", sensor_params, "sensor_params")
        if(self.mix_with_uniform):
            self.mix_with_uniform_alpha = get_parameter_safely("mix_with_uniform_alpha", sensor_params, "sensor_params")

        # Create the sensor location
        self.sensor_location = np.asarray([x_location, y_location])

        # Get a bias value for the sensor
        # self.sensor_bias = np.random.vonmises(0, self.sensor_bias_concentration)
        self.sensor_bias = 0.0

    def create_sensor_observation_for_states(self, car_states):

        # Get the true bearing between the car and the sensor
        diff = car_states - self.sensor_location
        true_bearings = np.arctan2(diff[:, 1], diff[:, 0])

        # Create the von mises
        von_mises = np.random.vonmises(true_bearings, self.sensor_concentration)
        von_mises += self.sensor_bias

        if(self.mix_with_uniform):

            # Get the uniform 
            uniform = np.random.uniform(-np.pi, np.pi, size=von_mises.shape)

            # Select from von mise or uniform
            selections = np.random.uniform(0, 1.0, size=von_mises.shape)
            von_mises[selections > self.mix_with_uniform_alpha] = 0.0
            uniform[selections <= self.mix_with_uniform_alpha] = 0.0
            sensor_output = von_mises + uniform
        else:
            sensor_output = von_mises

        sensor_output = np.reshape(sensor_output, (-1, 1))

        return sensor_output

    def get_position(self):
        return self.sensor_location


    def __eq__(self, other):
        if(not isinstance(other, Sensor)):
            return False

        if(self.sensor_name != other.sensor_name):
            return False

        if(self.sensor_concentration != other.sensor_concentration):
            return False

        if(self.sensor_bias_concentration != other.sensor_bias_concentration):
            return False

        if(np.any(self.sensor_location != other.sensor_location)):
            return False

        return True

class BearingsOnlyDatasetVelocity(BaseDataset):
    def __init__(self, dataset_params, dataset_type):
        
        # Save the inputs in case we need them later
        self.dataset_type = dataset_type
        self.dataset_params = dataset_params

        # Extract the params        
        self.subsequence_length = get_parameter_safely("subsequence_length", dataset_params, "dataset_params")
        self.dynamics_skip_amount = get_parameter_safely("dynamics_skip_amount", dataset_params, "dataset_params")

        # See if we should use sparse ground truths
        if("sparse_ground_truth_keep_modulo" in dataset_params):
            self.sparse_ground_truth_keep_modulo = dataset_params["sparse_ground_truth_keep_modulo"]
        else:
            self.sparse_ground_truth_keep_modulo = None


        if("scale_xy" in dataset_params):
            self.scale_xy = dataset_params["scale_xy"]
        else:
            self.scale_xy = 1.0



        # Extract the size of the dataset for this dataset
        dataset_sizes = get_parameter_safely("dataset_sizes", dataset_params, "dataset_params")
        self.dataset_size = get_parameter_safely(dataset_type, dataset_sizes, "dataset_sizes")

        # Generate the car
        car_params = get_parameter_safely("car", dataset_params, "dataset_params")
        self.car_velocity = get_parameter_safely("velocity", car_params, "car_params")
        self.car_max_angle_pertubation_degrees = get_parameter_safely("max_angle_pertubation_degrees", car_params, "car_params")

        # Get the sensors that we will be using
        self.sensors = list()
        sensors_params = get_parameter_safely("sensors", dataset_params, "dataset_params")
        for sensor_name in sensors_params:
            sensor_params = get_parameter_safely(sensor_name, sensors_params, "sensors_params")
            sensor = Sensor(sensor_name, sensor_params)
            self.sensors.append(sensor)
        
        # check if we can load a dataset
        did_load_dataset = self.load_from_save(dataset_params, dataset_type)

        # If we did not load a dataset then lets generate and save one (if we can save it)
        if(not did_load_dataset):

            # Generate the dataset
            self.generate_dataset()

            # Save our newly generated dataset
            self.save_dataset()

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):

        states = self.car_states[idx]

        states[..., 0] = states[..., 0] * self.scale_xy
        states[..., 1] = states[..., 1] * self.scale_xy


        # Compute the return dictionary
        return_dict = {}
        return_dict["states"] = states
        return_dict["observations"] = self.sensor_obs[idx]
        return_dict["dataset_index"] = idx


        if(self.sparse_ground_truth_keep_modulo is not None):
            ground_truth_mask = torch.full(size=(self.car_states[idx].shape[0],), fill_value=False)

            for i in range(ground_truth_mask.shape[0]):
                if((i % self.sparse_ground_truth_keep_modulo) == 0):
                    ground_truth_mask[i] = True

            return_dict["ground_truth_mask"] = ground_truth_mask

        # # If we dont have actions then we dont have anything to return
        # if((self.actions is not None) and (len(self.actions) > 0)):
        #     return_dict["actions"] = self.actions[idx]


        return return_dict


    def get_subsequence_length(self):
        return self.subsequence_length


    def load_from_save(self, dataset_params, dataset_type):

        # The dafault save location is none (aka dont save)
        self.save_location = None

        # Check if we have the save params, if not then we cant load from save
        if("dataset_saves" not in dataset_params):
            return False

        # Extract this dataset save location 
        dataset_saves = get_parameter_safely("dataset_saves", dataset_params, "dataset_params")

        # If this dataset type is not in the list of saved datasets then we cannot save or load this specific dataset
        if(dataset_type not in dataset_saves):
            return False

        # Extract the save location
        self.save_location = get_parameter_safely(dataset_type, dataset_saves, "dataset_saves")

        # If the directory does not exist then we want to create it!
        data_dir, _ = os.path.split(self.save_location)
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        # Check if the file exists, if not then we cant load 
        if(not os.path.exists(self.save_location)):
            return False

        # Load and unpack data
        all_data = torch.load(self.save_location)
        car_states = all_data["car_states"]
        sensor_obs = all_data["sensor_obs"]
        sensors = all_data["sensors"]
        dynamics_skip_amount = all_data["dynamics_skip_amount"]

        # Make sure the sensors are the same
        for i in range(len(self.sensors)):
            if(sensors[i] != self.sensors[i]):
                return False


        # Check the data to make sure that it is correct
        if(car_states.shape[0] != self.dataset_size):
            return False
        if(car_states.shape[1] != self.subsequence_length):
            return False
        if(dynamics_skip_amount != self.dynamics_skip_amount):
            return False




        # The data looks good so we should use it
        self.car_states = car_states
        self.sensor_obs = sensor_obs
        self.sensors = sensors

        # We have successfully loaded the dataset
        return True



    def save_dataset(self):

        # If we dont have a save location then we cant save
        if(self.save_location is None):
            return

        # Pack everything into a single dict that we can save
        save_dict = dict()
        save_dict["car_states"] = self.car_states
        save_dict["sensor_obs"] = self.sensor_obs
        save_dict["sensors"] = self.sensors
        save_dict["dynamics_skip_amount"] = self.dynamics_skip_amount

        # Save that dict
        torch.save(save_dict, self.save_location)



    def generate_dataset(self):

        # Compute the long sequence length that we wish to compute
        # long_sequence_length = self.dataset_size * self.subsequence_length

        car_states = []
        for i in tqdm(range(self.dataset_size)):
            car = Car(self.car_velocity, np.radians(self.car_max_angle_pertubation_degrees))
            states = car.simulate_car_for_n_steps(self.subsequence_length, skip=self.dynamics_skip_amount)
            states = states[1:]
            car_states.append(states)

        car_states = np.concatenate(car_states)


        # Convert the last dim from from angle to x, y velocities
        car_states_velocity = np.zeros((car_states.shape[0], 4))
        car_states_velocity[:, 0] = car_states[:, 0]
        car_states_velocity[:, 1] = car_states[:, 1]
        car_states_velocity[:, 2] = np.cos(car_states[:, 2]) * self.car_velocity
        car_states_velocity[:, 3] = np.sin(car_states[:, 2]) * self.car_velocity
        car_states = car_states_velocity

        # # Compute a long sequence for the car
        # car_states = self.car.simulate_car_for_n_steps(long_sequence_length, skip=3)

        # # Simulation includes the starting state.  We dont need that state so lets just remove it
        # car_states = car_states[1:]

        # Car state is 3D (x, y, theta) but we only want 2D (x, y)
        # car_states = car_states[..., :2]

        # Compute all the sensor observations we will be using for the car
        all_sensor_obs = []
        for sensor in self.sensors:
            obs = sensor.create_sensor_observation_for_states(car_states[:, :2])
            all_sensor_obs.append(obs)
        all_sensor_obs = np.concatenate(all_sensor_obs, axis=-1)


        # Slice the data into smaller sequences
        sliced_car_states = np.zeros(shape=(self.dataset_size, self.subsequence_length, car_states.shape[-1]))
        sliced_sensor_obs = np.zeros(shape=(self.dataset_size, self.subsequence_length, all_sensor_obs.shape[-1]))

        for i in range(self.dataset_size):
            s = i * self.subsequence_length
            e = s + self.subsequence_length

            sliced_car_states[i] = car_states[s:e]
            sliced_sensor_obs[i] = all_sensor_obs[s:e]


        # Save the data 
        self.car_states = sliced_car_states
        self.sensor_obs = sliced_sensor_obs
        self.actions = None

        # Everything needs to be a float
        self.car_states = self.car_states.astype("float32")
        self.sensor_obs = self.sensor_obs.astype("float32")
        # self.actions = self.actions.float()        

        # Convert to pytorch 
        self.car_states = torch.from_numpy(self.car_states)
        self.sensor_obs = torch.from_numpy(self.sensor_obs)
        # self.actions = torch.from_numpy(self.actions)


    def get_x_range(self):
        return (-15*self.scale_xy, 15*self.scale_xy)

    def get_y_range(self):
        return (-15*self.scale_xy, 15*self.scale_xy)

    def get_sensors(self):
        return self.sensors

    def get_subsequence_length(self):
        return self.subsequence_length