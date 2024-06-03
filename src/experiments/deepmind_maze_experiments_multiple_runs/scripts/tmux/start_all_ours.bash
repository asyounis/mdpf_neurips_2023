#!/bin/bash



echo "Experiment 1"
tmux new -d -s deepmind_multiple_exp1
tmux send-keys -t deepmind_multiple_exp1 "cd ../../experiment0001" C-m
tmux send-keys -t deepmind_multiple_exp1 "./run.bash" C-m
sleep 2

echo "Experiment 2 Importance"
tmux new -d -s deepmind_multiple_exp2_importance
tmux send-keys -t deepmind_multiple_exp2_importance "cd ../../experiment0002_importance" C-m
tmux send-keys -t deepmind_multiple_exp2_importance "./run.bash" C-m
sleep 2

echo "Experiment 3 Importance"
tmux new -d -s deepmind_multiple_exp3_importance_dis
tmux send-keys -t deepmind_multiple_exp3_importance_dis "cd ../../experiment0003_importance_dis" C-m
tmux send-keys -t deepmind_multiple_exp3_importance_dis "./run.bash" C-m
