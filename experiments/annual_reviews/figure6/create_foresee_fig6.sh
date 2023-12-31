#!/bin/bash

# GP-MPC (with 150 points kernel) for quadrotor environment with diagonal constraint.
python3.11 ./gp_mpc_experiment.py --task quadrotor --algo foresee_gp_mpc2 --overrides ./config_overrides/gp_mpc_quad.yaml --image_dir ./media/media_foresee_gp_mpc2/
