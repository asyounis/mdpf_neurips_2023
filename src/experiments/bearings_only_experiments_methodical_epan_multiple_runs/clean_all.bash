
echo "Are you sure?: y or n"

read answer


if [[ "$answer" != "y" ]]
then
	echo "Answered No.  Exiting"
	exit
fi


# Delete
cd diffy_particle_filter/ && rm -rf saves && cd ..
cd optimal_transport_pf/ && rm -rf saves && cd ..
cd soft_resampling_particle_filter/ && rm -rf saves && cd ..
cd importance_sampling_pf/ && rm -rf saves && cd ..
cd diffy_particle_filter_learned_band/ && rm -rf saves && cd ..
cd optimal_transport_pf_learned_band/ && rm -rf saves && cd ..
cd soft_resampling_particle_filter_learned_band/ && rm -rf saves && cd ..
cd importance_sampling_pf_learned_band/ && rm -rf saves && cd ..
cd experiment0001/ && rm -rf saves && cd ..
cd experiment0002_importance/ && rm -rf saves && cd ..
cd experiment0002_implicit/ && rm -rf saves && cd ..
cd experiment0002_concrete/ && rm -rf saves && cd ..
cd experiment0003_importance/ && rm -rf saves && cd ..
cd experiment0003_implicit/ && rm -rf saves && cd ..
cd experiment0003_concrete/ && rm -rf saves && cd ..
cd discrete_concrete/ && rm -rf saves && cd ..




# cd individual_training/ && rm -rf saves && cd ..

