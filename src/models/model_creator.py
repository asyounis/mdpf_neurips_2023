

# Project Imports
from models.kde_particle_filter.kde_particle_filter import *
from models.kde_particle_filter.mse_set_transformer import *
from models.differentiable_particle_filter.differentiable_particle_filter import *
from models.optimal_transport_df.optimal_transport_particle_filter import *
from models.soft_resampling.soft_resampling_particle_filter import *
from models.importance_sampling_particle_filter.importance_sampling_particle_filter import *
from models.discrete_concrete.discrete_concrete import *
from models.rnns.lstm_rnn import *


def create_model(model_params, model_architecture_params):


	# Extract the main model.  We will use this to determine which 
	assert("main_model" in model_params)
	main_model_name = model_params["main_model"]

	# Extract the model type
	assert(main_model_name in model_architecture_params)
	model_arch_info = model_architecture_params[main_model_name]
	assert("type" in model_arch_info)
	model_type = model_arch_info["type"]


	# Create the model based on the model type
	if(model_type == "KDEParticleFilter"):
		model = KDEParticleFilter(model_params, model_architecture_params)
	
	elif(model_type == "KDEParticleFilterMSESolution"):
		model = KDEParticleFilterMSESolution(model_params, model_architecture_params)		
	# elif(model_type == "DifferentiableParticleFilter"):
		# model = DifferentiableParticleFilter(model_params, model_architecture_params)
	elif(model_type == "DifferentiableParticleFilterLearnedBand"):
		model = DifferentiableParticleFilterLearnedBand(model_params, model_architecture_params)
	# elif(model_type == "DifferentiableParticleFilterROTBandwidth"):
		# model = DifferentiableParticleFilterROTBandwidth(model_params, model_architecture_params)
	
	# elif(model_type == "OptimalTransportParticleFilter"):
		# model = OptimalTransportParticleFilter(model_params, model_architecture_params)
	elif(model_type == "OptimalTransportParticleFilterLearnedBand"):
		model = OptimalTransportParticleFilterLearnedBand(model_params, model_architecture_params)
	# elif(model_type == "OptimalTransportParticleFilterROTBandwidth"):
		# model = OptimalTransportParticleFilterROTBandwidth(model_params, model_architecture_params)
	
	# elif(model_type == "SoftResamplingParticleFilter"):
		# model = SoftResamplingParticleFilter(model_params, model_architecture_params)
	elif(model_type == "SoftResamplingParticleFilterLearnedBand"):
		model = SoftResamplingParticleFilterLearnedBand(model_params, model_architecture_params)
	# elif(model_type == "SoftResamplingParticleFilterROTBandwidth"):
		# model = SoftResamplingParticleFilterROTBandwidth(model_params, model_architecture_params)
	
	# elif(model_type == "ImportanceSamplingParticleFilter"):
		# model = ImportanceSamplingParticleFilter(model_params, model_architecture_params)
	elif(model_type == "ImportanceSamplingParticleFilterLearnedBand"):
		model = ImportanceSamplingParticleFilterLearnedBand(model_params, model_architecture_params)

	elif(model_type == "ConcreteParticleFilterLearnedBand"):
		model = ConcreteParticleFilterLearnedBand(model_params, model_architecture_params)

	elif(model_type == "LSTMRnn"):
		model = LSTMRnn(model_params, model_architecture_params)
	else:
		print("Unknown Model Type: {}".format(type))
		assert(False)

	return model

	OptimalTransportParticleFilterROTBandwidth