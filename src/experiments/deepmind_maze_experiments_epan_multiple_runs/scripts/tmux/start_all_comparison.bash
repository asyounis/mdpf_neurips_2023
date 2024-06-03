#!/bin/bash


echo "Diffy Learned Bandwidth"
tmux new -d -s deepmind_multiple_epan_diffy_lb
tmux send-keys -t deepmind_multiple_epan_diffy_lb "cd ../../diffy_particle_filter_learned_band" C-m
tmux send-keys -t deepmind_multiple_epan_diffy_lb "./run.bash" C-m
sleep 2

echo "OT Learned Bandwidth"
tmux new -d -s deepmind_multiple_epan_ot_lb
tmux send-keys -t deepmind_multiple_epan_ot_lb "cd ../../optimal_transport_pf_learned_band" C-m
tmux send-keys -t deepmind_multiple_epan_ot_lb "./run.bash" C-m
sleep 2

echo "Soft Learned Bandwidth"
tmux new -d -s deepmind_multiple_epan_sr_lb
tmux send-keys -t deepmind_multiple_epan_sr_lb "cd ../../soft_resampling_particle_filter_learned_band" C-m
tmux send-keys -t deepmind_multiple_epan_sr_lb "./run.bash" C-m
sleep 2

echo "DIS Learned Bandwidth"
tmux new -d -s deepmind_multiple_epan_dis_lb
tmux send-keys -t deepmind_multiple_epan_dis_lb "cd ../../importance_sampling_pf_learned_band" C-m
tmux send-keys -t deepmind_multiple_epan_dis_lb "./run.bash" C-m
sleep 2


echo "DC Learned Bandwidth"
tmux new -d -s deepmind_multiple_epan_dc_lb
tmux send-keys -t deepmind_multiple_epan_dc_lb "cd ../../discrete_concrete" C-m
tmux send-keys -t deepmind_multiple_epan_dc_lb "./run.bash" C-m
sleep 2

