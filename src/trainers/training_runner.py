


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
from problems.house3d.house3d_problem import *
from problems.simple_brownian_motion_tracking.simple_brownian_motion_tracking_problem import *

from trainers.specific_trainers.full_sequence_trainer import *
from trainers.specific_trainers.initializer_model_trainer import *
from trainers.specific_trainers.proposal_trainer import *
from trainers.specific_trainers.bandwidth_model_trainer import *
from trainers.specific_trainers.weight_model_trainer import *


def create_problem_training(experiment, application_type, model, save_dir, device):

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
    elif(application_type == "house3d"):
        problem = House3DProblem(experiment, save_dir, device, dataset_types)
    elif(application_type == "simple_brownian_motion_tracking"):
        problem = SimpleBrownianMotionTrackingProblem(experiment, save_dir, device, dataset_types)
    else:
        print("Unknown application_type: {}".format(application_type))
        assert(False)

    return problem


def create_trainer(experiment, model, problem, save_dir, device):

    # Extract the training type
    training_type = experiment["training_type"] 

    # Create the trainer
    if(training_type == "full"):
        trainer = FullSequenceTrainer(model, problem, experiment, save_dir, device)
    elif(training_type == "initilizer"):
        trainer = InitializerModelTrainer(model, problem, experiment, save_dir, device)
    elif(training_type == "proposal"):
        trainer = ProposalModelTrainer(model, problem, experiment, save_dir, device)
    elif(training_type == "bandwidth"):
        trainer = BandwidthModelTrainer(model, problem, experiment, save_dir, device)
    elif(training_type == "weights"):
        trainer = WeightModelTrainer(model, problem, experiment, save_dir, device)
    else:
        print("Unknown training_type: {}".format(training_type))
        assert(False)

    return trainer


def setup_training(experiment, model, save_dir, device):

    # Get the application: 
    application_type = experiment["application"]
    print("Application Type: {}".format(application_type))

    # Create the problem statement
    problem = create_problem_training(experiment, application_type, model, save_dir, device)

    # ds = problem.get_training_dataset()

    # print(len(ds))

    # for i in tqdm(range(len(ds))):
    #     data = ds[i]

    # exit()

    # Create The trainer
    trainer = create_trainer(experiment, model, problem, save_dir, device)

    # Create and add the optimizers to the trainer
    model.create_and_add_optimizers(trainer.get_training_params(), trainer, trainer.get_training_type())

    # Add the models to the trainer
    model.add_models(trainer, trainer.get_training_type())

    # Extract the loss parameters and create the loss function
    if("loss_params" in trainer.get_training_params()):
        loss_params = trainer.get_training_params()["loss_params"]
        loss_function = create_loss_function(loss_params, model)
        trainer.add_loss_function(loss_function)

    # train!
    trainer.train()