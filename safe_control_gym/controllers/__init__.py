"""Register controllers.

"""
from safe_control_gym.utils.registration import register

# register(id="cbf",
#          entry_point="safe_control_gym.controllers.cbf.cbf_qp:CBF_QP",
#          config_entry_point="safe_control_gym.controllers.cbf:cbf_qp.yaml")

# register(id="lqr",
#          entry_point="safe_control_gym.controllers.lqr.lqr:LQR",
#          config_entry_point="safe_control_gym.controllers.lqr:lqr.yaml")

# register(id="ilqr",
#          entry_point="safe_control_gym.controllers.lqr.ilqr:iLQR",
#          config_entry_point="safe_control_gym.controllers.lqr:ilqr.yaml")

register(id="mpc",
         entry_point="safe_control_gym.controllers.mpc.mpc:MPC",
         config_entry_point="safe_control_gym.controllers.mpc:mpc.yaml")

register(id="linear_mpc",
         entry_point="safe_control_gym.controllers.mpc.linear_mpc:LinearMPC",
         config_entry_point="safe_control_gym.controllers.mpc:linear_mpc.yaml")

register(id="gp_mpc",
         entry_point="safe_control_gym.controllers.mpc.gp_mpc:GPMPC",
         config_entry_point="safe_control_gym.controllers.mpc:gp_mpc.yaml")

register(id="foresee_gp_mpc",
         entry_point="safe_control_gym.controllers.mpc.foresee_gp_mpc:FORESEE_GPMPC",
         config_entry_point="safe_control_gym.controllers.mpc:gp_mpc.yaml")

register(id="foresee_gp_mpc2",
         entry_point="safe_control_gym.controllers.mpc.foresee_gp_mpc2:FORESEE_GPMPC2",
         config_entry_point="safe_control_gym.controllers.mpc:gp_mpc.yaml")

register(id="foresee_gp_mpc3",
         entry_point="safe_control_gym.controllers.mpc.foresee_gp_mpc3:FORESEE_GPMPC3",
         config_entry_point="safe_control_gym.controllers.mpc:gp_mpc.yaml")

register(id="foresee_cbf_qp",
         entry_point="safe_control_gym.controllers.mpc.foresee_cbf_qp:FORESEE_CBF_QP",
         config_entry_point="safe_control_gym.controllers.mpc:gp_mpc.yaml")

register(id="foresee_cbf_qp_cov",
         entry_point="safe_control_gym.controllers.mpc.foresee_cbf_qp_cov:FORESEE_CBF_QP_COV",
         config_entry_point="safe_control_gym.controllers.mpc:gp_mpc.yaml")

register(id="mpsc",
         entry_point="safe_control_gym.controllers.mpsc.mpsc:MPSC",
         config_entry_point="safe_control_gym.controllers.mpsc:mpsc.yaml")

register(id="pid",
         entry_point="safe_control_gym.controllers.pid.pid:PID",
         config_entry_point="safe_control_gym.controllers.pid:pid.yaml")

register(id="ppo",
         entry_point="safe_control_gym.controllers.ppo.ppo:PPO",
         config_entry_point="safe_control_gym.controllers.ppo:ppo.yaml")

register(id="sac",
         entry_point="safe_control_gym.controllers.sac.sac:SAC",
         config_entry_point="safe_control_gym.controllers.sac:sac.yaml")

register(id="safe_explorer_ppo",
         entry_point="safe_control_gym.controllers.safe_explorer.safe_ppo:SafeExplorerPPO",
         config_entry_point="safe_control_gym.controllers.safe_explorer:safe_ppo.yaml")

register(id="rarl",
         entry_point="safe_control_gym.controllers.rarl.rarl:RARL",
         config_entry_point="safe_control_gym.controllers.rarl:rarl.yaml")

register(id="rap",
         entry_point="safe_control_gym.controllers.rarl.rap:RAP",
         config_entry_point="safe_control_gym.controllers.rarl:rap.yaml")
