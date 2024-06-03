#!/bin/bash



echo "Experiment 1"
tmux new -d -s bearings_epan_multi_exp1
tmux send-keys -t bearings_epan_multi_exp1 "cd ../../experiment0001" C-m
tmux send-keys -t bearings_epan_multi_exp1 "./run.bash" C-m
sleep 2

echo "Experiment 2 Importance"
tmux new -d -s bearings_epan_multi_exp2_importance
tmux send-keys -t bearings_epan_multi_exp2_importance "cd ../../experiment0002_importance" C-m
tmux send-keys -t bearings_epan_multi_exp2_importance "./run.bash" C-m
sleep 2


echo "Experiment 2 Implicit"
tmux new -d -s bearings_epan_multi_exp2_implicit
tmux send-keys -t bearings_epan_multi_exp2_implicit "cd ../../experiment0002_implicit" C-m
tmux send-keys -t bearings_epan_multi_exp2_implicit "./run.bash" C-m
sleep 2

# echo "Experiment 3 Importance"
# tmux new -d -s bearings_25_multi_exp3_importance
# tmux send-keys -t bearings_25_multi_exp3_importance "cd ../../experiment0003_importance" C-m
# tmux send-keys -t bearings_25_multi_exp3_importance "./run.bash" C-m
