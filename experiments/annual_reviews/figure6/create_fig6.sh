#!/bin/bash

# GP-MPC (with 150 points kernel) for quadrotor environment with diagonal constraint.
python3.11 ./gp_mpc_experiment.py --task quadrotor --algo gp_mpc --overrides ./config_overrides/gp_mpc_quad.yaml --image_dir ./media/media_gp_mpc_h3/
