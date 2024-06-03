

echo "diffy_particle_filter_learned_band"
cd diffy_particle_filter_learned_band/
./run.bash
cd ..

echo "discrete_concrete"
cd discrete_concrete/
./run.bash
cd ..

echo "experiment0001"
cd experiment0001/
./run.bash
cd ..

# echo "experiment0002_implicit"
# cd experiment0002_implicit/
# ./run.bash
# cd ..

echo "experiment0002_importance"
cd experiment0002_importance/
./run.bash
cd ..

# echo "experiment0003_importance"
# cd experiment0003_importance/
# ./run.bash
# # cd ..

echo "experiment0003_importance_init"
cd experiment0003_importance_init/
./run.bash
cd ..

echo "importance_sampling_pf_learned_band"
cd importance_sampling_pf_learned_band/
./run.bash
cd ..

echo "lstm_rnn"
cd lstm_rnn/
./run.bash
cd ..



echo "soft_resampling_particle_filter_learned_band"
cd soft_resampling_particle_filter_learned_band/
./run.bash
cd ..


# echo "optimal_transport_pf_learned_band"
# cd optimal_transport_pf_learned_band/
# ./run.bash
# cd ..

echo "optimal_transport_pf_learned_band_always_on_override"
cd optimal_transport_pf_learned_band_always_on_override/
./run.bash
cd ..