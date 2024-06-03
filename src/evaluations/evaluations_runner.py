


# Project Imports
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

# Evaluation imports
from evaluations.initilizer_evaluations.bearings_only_initilizer_evaluation import *
from evaluations.initilizer_evaluations.toy_problem_initilizer_evaluation import *
from evaluations.initilizer_evaluations.deepmind_initilizer_evaluation import *
from evaluations.proposal_evaluations.bearings_only_proposal_evaluation import *
from evaluations.proposal_evaluations.toy_problem_proposal_evaluation import *
from evaluations.proposal_evaluations.synthetic_disk_tracking_proposal_evaluation import *
from evaluations.proposal_evaluations.deepmind_proposal_evaluation import *
from evaluations.proposal_evaluations.house3d_proposal_evaluation import *
from evaluations.full_sequence_evaluations.bearngs_only_full_sequence_evaluation import *
from evaluations.full_sequence_evaluations.bearings_only_velocity_full_sequence_evaluation import *
from evaluations.full_sequence_evaluations.toy_problem_full_sequence_evaluation import *
from evaluations.full_sequence_evaluations.deepmind_maze_full_sequence_evaluation import *
from evaluations.full_sequence_evaluations.synthetic_disk_tracking_full_sequence_evaluation import *
from evaluations.full_sequence_evaluations.uav123_full_sequence_evaluation import *
from evaluations.full_sequence_evaluations.lasot_full_sequence_evaluation import *
from evaluations.full_sequence_evaluations.house3d_full_sequence_evaluation import *
from evaluations.full_sequence_evaluations.simple_brownian_motion_tracking_full_sequence_evaluation import *


from evaluations.weight_evaluations.deepmind_weight_evaluations import *



def create_problem_evaluation(experiment, application_type, model, save_dir, device):

    # The types of datasets we want the the problem to create so we can get them
    dataset_types = ["evaluation"]
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

def create_evaluation(experiment, application_type, problem, model, save_dir, device):

    # Extract the training type
    evaluation_type = experiment["evaluation_type"] 

    # Create the trainer


    if(evaluation_type == "initilizer"):

        if(application_type == "bearings_only"):
            evaluation = BearingsOnlyInitilizerEvaluations(experiment, problem, model, save_dir, device)
        elif(application_type == "bearings_only_velocity"):
            evaluation = BearingsOnlyInitilizerEvaluations(experiment, problem, model, save_dir, device)
        elif(application_type == "bearings_only_vector_angle"):
            evaluation = BearingsOnlyInitilizerEvaluations(experiment, problem, model, save_dir, device)
        elif(application_type == "deepmind_maze"):
            evaluation = DeepMindInitilizerEvaluations(experiment, problem, model, save_dir, device)
        elif(application_type == "toy_problem"):
            evaluation = ToyProblemInitilizerEvaluations(experiment, problem, model, save_dir, device)
        else:
            print("Unknown application_type: {}".format(application_type))
            assert(False)
    
    elif(evaluation_type == "proposal"):

        if(application_type == "bearings_only"):
            evaluation = BearingsOnlyProposalEvaluations(experiment, problem, model, save_dir, device)
        elif(application_type == "bearings_only_velocity"):
            evaluation = BearingsOnlyProposalEvaluations(experiment, problem, model, save_dir, device)
        elif(application_type == "bearings_only_vector_angle"):
            evaluation = BearingsOnlyProposalEvaluations(experiment, problem, model, save_dir, device)
        elif(application_type == "toy_problem"):
            evaluation = ToyProblemProposalEvaluations(experiment, problem, model, save_dir, device)
        elif(application_type == "synthetic_disk_tracking"):
            evaluation = SyntheticDiskTrackingProposalEvaluations(experiment, problem, model, save_dir, device)
        elif(application_type == "deepmind_maze"):
            evaluation = DeepMindMazeProposalEvaluations(experiment, problem, model, save_dir, device)
        elif(application_type == "house3d"):
            evaluation = House3DProposalEvaluations(experiment, problem, model, save_dir, device)
        else:
            print("Unknown application_type: {}".format(application_type))
            assert(False)

    elif(evaluation_type == "full"):

        if(application_type == "bearings_only"):
            evaluation = BearingsOnlyFullSequenceEvaluation(experiment, problem, model, save_dir, device)
        elif(application_type == "bearings_only_velocity"):
            evaluation = BearingsOnlyVelocityFullSequenceEvaluation(experiment, problem, model, save_dir, device)
        elif(application_type == "bearings_only_vector_angle"):
            evaluation = BearingsOnlyVelocityFullSequenceEvaluation(experiment, problem, model, save_dir, device)
        elif(application_type == "toy_problem"):
            evaluation = ToyProblemFullSequenceEvaluation(experiment, problem, model, save_dir, device)
        elif(application_type == "deepmind_maze"):
            evaluation = DeepmindMazeFullSequenceEvaluation(experiment, problem, model, save_dir, device)
        elif(application_type == "synthetic_disk_tracking"):
            evaluation = SyntheticDiskTrackingFullSequenceEvaluation(experiment, problem, model, save_dir, device)
        elif(application_type == "uav123"):
            evaluation = UAV123FullSequenceEvaluation(experiment, problem, model, save_dir, device)
        elif(application_type == "lasot"):
            evaluation = LasotFullSequenceEvaluation(experiment, problem, model, save_dir, device)
        elif(application_type == "house3d"):
            evaluation = House3DFullSequenceEvaluation(experiment, problem, model, save_dir, device)
        elif(application_type == "simple_brownian_motion_tracking"):
            evaluation = SimpleBrownianMotionTrackingFullSequenceEvaluation(experiment, problem, model, save_dir, device)
        else:
            print("Unknown application_type: {}".format(application_type))
            assert(False)


    elif(evaluation_type == "weight"):
        if(application_type == "deepmind_maze"):
            evaluation = DeepmindMazeWeightEvaluation(experiment, problem, model, save_dir, device)
        else:
            print("Unknown application_type: {}".format(application_type))
            assert(False)


    else:
        print("Unknown evaluation_type: {}".format(evaluation_type))
        assert(False)

    return evaluation


def setup_evaluation(experiment, model, save_dir, device):

    # Get the application: 
    application_type = experiment["application"]
    print("Application Type: {}".format(application_type))

    # Create the problem statement
    problem = create_problem_evaluation(experiment, application_type, model, save_dir, device)

    # Create the evaluation
    evaluation = create_evaluation(experiment,application_type, problem, model, save_dir, device)

    # Run the evaluation
    evaluation.run_evaluation()
