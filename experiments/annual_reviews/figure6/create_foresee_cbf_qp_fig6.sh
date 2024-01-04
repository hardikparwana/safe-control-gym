#!/bin/bash

# GP-MPC (with 150 points kernel) for quadrotor environment with diagonal constraint.
python3.11 ./gp_mpc_experiment.py --task quadrotor --algo foresee_cbf_qp --overrides ./config_overrides/gp_mpc_quad.yaml --image_dir ./media/media_foresee_cbf_qp/
