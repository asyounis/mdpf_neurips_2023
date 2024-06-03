


# Project Imports
from loss_functions.loss_functions import *

from problems.bearings_only.bearings_only_problem import *
from problems.bearings_only_velocity.bearings_only_velocity_problem import *
from problems.bearings_only_vector_angle.bearings_only_vector_angle_problem import *
from problems.toy_problem.toy_problem import *
from problems.deepmind_maze.deepmind_maze_problem import *
from problems.synthetic_disk_tracking.synthetic_disk_tracking_problem import *
from problems.lasot_problem.lasot_problem import *
from problems.uav123_problem.uav123_problem import *

from data_generators.specific_data_generators.particle_set_data_generator import *

def create_problem_data_generator(experiment, application_type, model, save_dir, device):

    # The types of datasets we want the the problem to create so we can get them
    dataset_types = ["training", "validation"]

    # Create the problem
    if(application_type == "bearings_only"):
        problem = BearingsOnlyProblem(experiment, save_dir, device, dataset_types)
    elif(application_type == "bearings_only_velocity"):
        problem = BearingsOnlyVelocityProblem(experiment, save_dir, device, dataset_types)
    elif(application_type == "bearings_only_vector_angle"):
        problem = BearingsOnlyVectorAngleProblem(experiment, save_dir, device, dataset_types)
    elif(application_type == "toy_problem"):
        problem = ToyProblem(experiment, save_dir, device, dataset_types)
    elif(application_type == "deepmind_maze"):
        problem = DeepMindMazeProblem(experiment, save_dir, device, dataset_types)
    elif(application_type == "synthetic_disk_tracking"):
        problem = SyntheticDiskTrackingProblem(experiment, save_dir, device, dataset_types)
    elif(application_type == "lasot"):
        problem = LasotProblem(experiment, save_dir, device, dataset_types)
    elif(application_type == "uav123"):
        problem = UAV123Problem(experiment, save_dir, device, dataset_types)
    else:
        print("Unknown application_type: {}".format(application_type))
        assert(False)

    return problem


def create_data_generator(experiment, model, problem, save_dir, device):

    # Extract the training type
    data_generator_type = experiment["data_generator_type"] 

    # Create the trainer
    if(data_generator_type == "particle_set_data_generator"):
        trainer = ParticleSetDataGenerator(experiment, model, problem, save_dir, device)
    else:
        print("Unknown data_generator_type: {}".format(data_generator_type))
        assert(False)

    return trainer


def setup_data_generator(experiment, model, save_dir, device):

    # Get the application: 
    application_type = experiment["application"]
    print("Application Type: {}".format(application_type))

    # Create the problem statement
    problem = create_problem_data_generator(experiment, application_type, model, save_dir, device)

    # Create The data generator
    data_genrator = create_data_generator(experiment, model, problem, save_dir, device)

    # Generate Dat Data!
    data_genrator.generate_data()