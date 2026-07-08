import ray
import os
import re
import subprocess
import wandb
from ray.air.integrations.wandb import WandbLoggerCallback, setup_wandb
import time
os.environ["RAY_TMPDIR"] = "/work/cmcc/jg24923/tmp/ray"
ray.init(_temp_dir="/work/cmcc/jg24923/tmp/ray")
import shutil
import math
from shared_critic_encoder_bottleneck import SharedCriticTorchRLModule, AGENT_TYPE_PREFIXES
from ray import tune
import numpy as np
from ray.tune.registry import register_env
from CM_CfD_EU_ETS import CM_EoM
from action_mask_rlm_new import TorchActionMaskRLM
from ray.rllib.examples.algorithms.mappo.mappo import MAPPO, MAPPOConfig
from ray.rllib.examples.algorithms.mappo.torch.shared_critic_torch_rl_module import SharedCriticTorchRLModule
from ray.rllib.examples.algorithms.mappo.shared_critic_catalog import SharedCriticCatalog

# ── RLModuleSpec imports: forward-compatible with 2.35+ ──────────────────────
try:
    from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
    from ray.rllib.core.rl_module.rl_module import RLModuleSpec as SingleAgentRLModuleSpec
except ImportError:
    # Fallback for older 2.x builds
    from ray.rllib.core.rl_module.marl_module import MultiAgentRLModuleSpec as MultiRLModuleSpec
    from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
# ─────────────────────────────────────────────────────────────────────────────

from action_mask_lstm_custom_actions import CustomPPOCatalog
import pandas as pd
from ray.rllib.policy.policy import PolicySpec


def find_latest_checkpoint(base_path):
    latest_checkpoint = None
    latest_time = 0
    
    print(f"Searching in: {base_path}")  # Debug
    
    # Look for checkpoint directories with 6-digit padding
    for root, dirs, files in os.walk(base_path):
        for dir_name in dirs:
            # Match RLLib checkpoint pattern: checkpoint_ followed by exactly 6 digits
            if re.match(r"^checkpoint_\d{6}$", dir_name):
                full_path = os.path.join(root, dir_name)
                print(f"Found checkpoint: {full_path}")  # Debug
                
                try:
                    dir_time = os.path.getmtime(full_path)
                    if dir_time > latest_time:
                        latest_checkpoint = full_path
                        latest_time = dir_time
                except OSError as e:
                    print(f"Error accessing {full_path}: {e}")
    
    print(f"Latest checkpoint: {latest_checkpoint}")  # Debug
    return latest_checkpoint

# Define the path to the count file
count_file_path = 'run_count_tax_ss_661.txt'

def read_run_count(file_path):
    """Read the run count from the file."""
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            count = file.read().strip()
            try:
                # Attempt to convert to a float
                return float(count)
            except ValueError:
                # If conversion fails, return 0
                return 0
    return 0

def write_run_count(file_path, count):
    """Write the run count to the file."""
    with open(file_path, 'w') as file:
        file.write(str(count))

# Generate M×N matrix with different lambda per column
def generate_delay_matrix(m_rows, lambdas):
    n_cols = len(lambdas)
    delay_matrix = np.zeros((m_rows, n_cols), dtype=int)
    
    for col, lam in enumerate(lambdas):
        delay_matrix[:, col] = np.random.poisson(lam, m_rows)
    
    return delay_matrix

if __name__ == "__main__":

    demand_growth_type = "Deterministic"

    penalty_type = 2
    
    os.environ['WANDB_DISABLE_SERVICE']= "True"
    wandb.init(project="Test_WB", mode='offline', dir='/work/cmcc/jg24923/wandb_logs')
    # def wandb_callback(result):
        # Log the metrics to W&B
    #    wandb.log(result)# Define the directory where you want to start searching
    
    root_directory = "/users_home/cmcc/jg24923/ray_results/MAPPO"

    # Define the prefix of the directories you want to delete
    prefix = "PPO_CM_EoM_ss_battery_661"

    # Walk through the directory tree
    for dirpath, dirnames, filenames in os.walk(root_directory):
        for dirname in dirnames:
            if dirname.startswith(prefix):
                full_path = os.path.join(dirpath, dirname)
                print(f"Deleting directory: {full_path}")
                shutil.rmtree(full_path)
                print(f"Deleted {full_path}")
    
    run_count = read_run_count(count_file_path)

    base_path = "/work/cmcc/jg24923/CM_EoM_ss_battery_661/MAPPO"

    latest_checkpoint = find_latest_checkpoint(base_path)
    
    if latest_checkpoint:
        print(f"Restoring from checkpoint: {latest_checkpoint}")
        restore_string = latest_checkpoint
        print("Checkpoint restored successfully.")
    else:
        print("No checkpoint found.")  

    # ray.init()

    agent_g = 16

    agent_p = 1

    curriculum_step = 1

    dir_input_data = '050526 Entry data - Italy 2024 Alice - 16 Agents - Pypsa - 10 years - Super Tax - V2.xlsx'

    demand_growth_type = "Deterministic"

    penalty_factor = 0.0

    max_t = 144 * 30

    years_slack_termination = 8
    
    years_slack_profits = 12

    iters_outer_loop = 5

    short_t = 24
    
    VoLL_max = 4000

    period_inv = 12   

    step_g_bids = 7

    step_SoC_control = 7

    step_g_inv = 6   

    env_config={"start_level": 1}

    n_tech = 6

    n_tech_RES = 3

    n_tech_battery = 2

    n_tech_decom = 2

    decom_time = [2,2]
    
    decom_cost = [9000, 18000]
    
    decom_cost = np.tile(decom_cost, (30, 1))

    decom_activation_year = 2
    
    decom_activation = True

    random_g = np.random.random([503, agent_g * (n_tech + 3)])

    random_g_cm = np.random.random([503, agent_g * (n_tech + 1)])

    random_g_CfD = np.random.random([503, agent_g * (n_tech_RES)])
    
    v_c_g = pd.read_excel(dir_input_data, sheet_name='Variable_Costs')
    
    v_c_g = np.array(v_c_g.iloc[:,:])
    
    inv_init_g = pd.read_excel(dir_input_data, sheet_name='Agents')
    
    inv_init_g = np.array(inv_init_g.iloc[:,:])
                
    demand_growth = pd.read_excel(dir_input_data, sheet_name='Demand_Growth')

    demand_growth = np.array(demand_growth.iloc[0,0])
    
    inv_data = pd.read_excel(dir_input_data, sheet_name='Tech_Characteristics')
    
    inv_max_original = pd.read_excel(dir_input_data, sheet_name='Investment_steps')

    inv_max = np.array(inv_max_original.iloc[:,:])/curriculum_step

    life_time = np.array(inv_data.iloc[1,:])

    construction_time = np.array(inv_data.iloc[2,:])

    inv_cost = pd.read_excel(dir_input_data, sheet_name='Inv_Tech')
    
    inv_cost = np.array(inv_cost.iloc[:,:])

    fixed_cost = pd.read_excel(dir_input_data, sheet_name='Fixed_Cost_Tech')
    
    fixed_cost = np.array(fixed_cost.iloc[:,:])

    aging = pd.read_excel(dir_input_data, sheet_name='Aging')

    aging = np.array(aging.iloc[:,:])

    CO2_tax_scenario = pd.read_excel(dir_input_data, sheet_name='CO2_tax')

    CO2_tax_scenario = np.array(CO2_tax_scenario.iloc[:,:])

    CO2_social_cost = CO2_tax_scenario

    CO2_tax_original = 176.0

    CO2_tax_max = 300.0

    CO2_tax_min = 1.0

    CO2_growth_rate_max = 0.25

    CO2_tax_tech = pd.read_excel(dir_input_data, sheet_name='CO2_tax_tech')

    CO2_tax_tech = np.array(CO2_tax_tech.iloc[:,:])

    CO2_tech = np.array([0, 0, 0, 1.01, 0.56, 0.41, 0, 0.25])

    policy_ids = pd.read_excel(dir_input_data, sheet_name='Policy_IDs')

    policy_ids = np.array(policy_ids.iloc[:,:])

    cm_design = pd.read_excel(dir_input_data, sheet_name='CM_Design')

    cm_price_cap =  np.array(cm_design.iloc[0,0])

    cm_option_strike = np.array(cm_design.iloc[0,1])
    
    cm_tech_value = np.ones(n_tech) * 50
    
    cm_tech_value = np.ones(n_tech) * 50

    cm_design = pd.read_excel(dir_input_data, sheet_name='CM_Design')
    
    inv_data_battery = pd.read_excel(dir_input_data, sheet_name='Tech_Characteristics_Battery')
    
    inv_max_battery_original = pd.read_excel(dir_input_data, sheet_name='Investment_steps_battery')

    inv_max_battery = np.array(inv_max_battery_original.iloc[:,:])/curriculum_step

    life_time_battery = np.array(inv_data_battery.iloc[1,:])

    construction_time_battery = np.array(inv_data_battery.iloc[2,:])

    inv_cost_battery = pd.read_excel(dir_input_data, sheet_name='Inv_Tech_Battery')
    
    inv_cost_battery = np.array(inv_cost_battery.iloc[:,:])

    fixed_cost_battery = pd.read_excel(dir_input_data, sheet_name='Fixed_Cost_Tech_Battery')
    
    fixed_cost_battery = np.array(fixed_cost_battery.iloc[:,:])

    opportunity_cost = 0.08

    opportunity_cost_planner = 0.05

    cm_excess_demand = 0
    cm_target_growth_rate_max = 0.1

    cm_price_cap_max = 30
    cm_price_cap_min = 10
    cm_price_growth_rate = 0.25
    cm_price_cap_original = 30

    cm_strike_max = 300 
    cm_strike_min = 80
    cm_strike_growth_rate = 0.25
    cm_strike_original = 300

    CfD_price_cap = pd.read_excel(dir_input_data, sheet_name='CfD_Design')
    CfD_price_cap = np.array(CfD_price_cap.iloc[:,:])

    CfD_target = pd.read_excel(dir_input_data, sheet_name='CfD_target')
    CfD_target = np.array(CfD_target.iloc[:,:])
    CfD_target = 0
    CfD_growth_rate_max = 10

    CfD_price_cap_max = 150
    CfD_price_cap_min = 40
    CfD_price_cap_original = 150
    CfD_price_growth_rate = 0.25

    scenario_failures = pd.read_excel(dir_input_data, sheet_name='Scenarios_Failures')

    scenario_failures = np.array(scenario_failures.iloc[:,:])

    random_number_demand = np.random.randint(0, 20, size=(499))

    random_number_availability = np.random.randint(0, 20, size=(509, n_tech))

    random_number_failures = np.random.randint(0, 20, size=(52100))

    inv_init_battery = pd.read_excel(dir_input_data, sheet_name='Storage_short-term')

    inv_init_battery = np.array(inv_init_battery.iloc[:,:])

    print("inv_init_battery")

    print(inv_init_battery)

    storage_lt_info = pd.read_excel(dir_input_data, sheet_name='Storage_long-term')

    inv_init_storage_lt = np.array(storage_lt_info.iloc[:agent_g,0])

    SoC_max_merchant_storage_lt = np.array(storage_lt_info.iloc[:agent_g,1])

    fixed_cost_storage_lt = pd.read_excel(dir_input_data, sheet_name='Fixed_Cost_Tech_Storage_lt')

    fixed_cost_storage_lt =  np.array(fixed_cost_storage_lt.iloc[:,:])

    investments_enabled = pd.read_excel(dir_input_data, sheet_name='Investments_enabled')

    investments_enabled =  np.array(investments_enabled.iloc[:,:])

    average_failures = pd.read_excel(dir_input_data, sheet_name='Average_failures')

    average_failures =  np.array(average_failures.iloc[:,:])

    time_series = pd.read_excel(dir_input_data, sheet_name='Time_Series')

    time_series =  np.array(time_series.iloc[:,:])

    time_series = time_series[:,2:]

    demand = time_series[:,8]

    percentile_demand = np.percentile(demand,90) 

    print('Time Series')

    print(time_series[0,:])

    print('Percentile demand')

    print(percentile_demand)

    print('demand')

    print(demand)

    average_bimester_series = pd.read_excel(dir_input_data, sheet_name='Average_bimester_series')

    average_bimester_series =  np.array(average_bimester_series.iloc[:,:])

    average_bimester_series = average_bimester_series[:,4:]

    average_yearly_series = pd.read_excel(dir_input_data, sheet_name='Average_yearly_series')

    average_yearly_series =  np.array(average_yearly_series.iloc[:,:])

    average_yearly_series = average_yearly_series[:,1:]

    print('investments_enabled')

    print(investments_enabled)

    step_planner = 6
    
    flexibility_multiplier_max = 4

    coporate_tax_activation = True

    CO2_activation = False 

    flexibility_activation = True

    CfD_activation = True

    cm_activation = True

    policy_scenarios = pd.read_excel(dir_input_data, sheet_name='Policy_Scenarios')

    policy_scenarios = np.array(policy_scenarios.iloc[:,:])

    entrants = pd.read_excel(dir_input_data, sheet_name='Entrants')

    entrants = np.array(entrants.iloc[:,:])

    print("Policy_scenarios")

    print(policy_scenarios)

    flexibility_target = 0

    flexibility_target_growth_rate_max = 10

    random_g_flexibility = np.random.random([503, agent_g * 1])

    flexibility_price_cap = 30000

    policy_deterministic_activation = True
    
    policy_deterministic = 6

    carbon_deterministic_activation = True
    
    carbon_deterministic = 10

    demand_deterministic_activation = True

    demand_deterministic = 0

    lambdas = [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5]  # Different risk for each project
    construction_delays = generate_delay_matrix(1000, lambdas)

    lambdas_failure = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
    failure_flag = generate_delay_matrix(1000, lambdas_failure)

    type_action_regulator = 2

    CO2_tax_scenario_activation = True

    corporate_tax_rate_growth_max = 0.3

    agent_g_public = [0]

    public_generators = False

    agent_g_lobby = [0]

    lobby = False

    # activation markets 

    CfD_activation_year = 0
    cm_activation_year = 0
    flexibility_activation_year = 0

    # maximum targets in markets

    cm_maximum_target = 2
    CfD_maximum_target = 80
    flexibility_maximum_target = 80

    # Terminal reward flag

    terminal_reward_flag = False

    # Scenario flag

    scenario_tax_decree = 0

    shock_flag = True
    
    # CM targets
    
    cm_reliability_target_values = [0.000001, 0.00001, 0.0001, 0.0005, 0.01]
    
    # CfD targets
    
    CfD_target_increments_steps = [0, 1, 2, 3, 4]
    
    # Flexibility targets
    
    flexibility_target_increments_steps = [0, 1, 2, 3, 4]

    scenario_tax_decree_deactivation_random_flag = False

    env = CM_EoM(max_t, short_t, VoLL_max,
			  agent_g, step_g_bids, step_g_inv, n_tech, n_tech_battery, n_tech_decom, step_SoC_control, step_planner,
			  v_c_g, inv_init_g, inv_max, inv_cost, fixed_cost, construction_time, life_time, decom_time, decom_cost, decom_activation_year, decom_activation,
			  aging, CO2_tech, CO2_tax_original, CO2_tax_min, CO2_tax_max, CO2_growth_rate_max, CO2_social_cost,  CO2_tax_scenario, CO2_tax_scenario_activation,
			  random_g, random_g_cm, random_g_CfD, demand_growth,
			  cm_price_cap_max, cm_price_cap_min, cm_price_growth_rate, cm_price_cap_original, 
			  cm_strike_max, cm_strike_min, cm_strike_growth_rate, cm_strike_original, cm_target_growth_rate_max,
			  opportunity_cost, opportunity_cost_planner,
			  n_tech_RES, CfD_target, CfD_growth_rate_max, penalty_type, demand_growth_type, CfD_price_cap_max, CfD_price_cap_min, CfD_price_cap_original, CfD_price_growth_rate,
			  average_failures,
              scenario_failures,
			  random_number_failures, 
			  penalty_factor, 
			  inv_init_battery, inv_max_battery, inv_cost_battery, fixed_cost_battery, construction_time_battery, life_time_battery,
			  inv_init_storage_lt, SoC_max_merchant_storage_lt, fixed_cost_storage_lt,
			  investments_enabled,
			  time_series, average_bimester_series, average_yearly_series,
			  coporate_tax_activation, cm_activation, CfD_activation, flexibility_activation, cm_reliability_target_values, CfD_target_increments_steps, flexibility_target_increments_steps,
			  policy_scenarios, entrants, policy_deterministic_activation, policy_deterministic, carbon_deterministic_activation, carbon_deterministic, demand_deterministic_activation, demand_deterministic,
			  flexibility_target, flexibility_target_growth_rate_max, random_g_flexibility, flexibility_price_cap,
			  corporate_tax_rate_growth_max,
			  construction_delays, failure_flag,
			  type_action_regulator,
			  agent_g_public, public_generators, 
			  agent_g_lobby, lobby,
			  CfD_activation_year, cm_activation_year, flexibility_activation_year,
			  cm_maximum_target, CfD_maximum_target, flexibility_maximum_target,
			  terminal_reward_flag, 
			  years_slack_termination, years_slack_profits,
			  scenario_tax_decree, shock_flag, scenario_tax_decree_deactivation_random_flag)

    temp_action_space = env.action_space.sample()

    print("Action Space")
    print(temp_action_space)

    def env_creator(args):
        
        temp_env = CM_EoM(max_t, short_t, VoLL_max,
			  agent_g, step_g_bids, step_g_inv, n_tech, n_tech_battery, n_tech_decom, step_SoC_control, step_planner,
			  v_c_g, inv_init_g, inv_max, inv_cost, fixed_cost, construction_time, life_time, decom_time, decom_cost, decom_activation_year, decom_activation,
			  aging, CO2_tech, CO2_tax_original, CO2_tax_min, CO2_tax_max, CO2_growth_rate_max, CO2_social_cost,  CO2_tax_scenario, CO2_tax_scenario_activation,
			  random_g, random_g_cm, random_g_CfD, demand_growth,
			  cm_price_cap_max, cm_price_cap_min, cm_price_growth_rate, cm_price_cap_original, 
			  cm_strike_max, cm_strike_min, cm_strike_growth_rate, cm_strike_original, cm_target_growth_rate_max,
			  opportunity_cost, opportunity_cost_planner,
			  n_tech_RES, CfD_target, CfD_growth_rate_max, penalty_type, demand_growth_type, CfD_price_cap_max, CfD_price_cap_min, CfD_price_cap_original, CfD_price_growth_rate,
			  average_failures,
              scenario_failures,
			  random_number_failures, 
			  penalty_factor, 
			  inv_init_battery, inv_max_battery, inv_cost_battery, fixed_cost_battery, construction_time_battery, life_time_battery,
			  inv_init_storage_lt, SoC_max_merchant_storage_lt, fixed_cost_storage_lt,
			  investments_enabled,
			  time_series, average_bimester_series, average_yearly_series,
			  coporate_tax_activation, cm_activation, CfD_activation, flexibility_activation, cm_reliability_target_values, CfD_target_increments_steps, flexibility_target_increments_steps,
			  policy_scenarios, entrants, policy_deterministic_activation, policy_deterministic, carbon_deterministic_activation, carbon_deterministic, demand_deterministic_activation, demand_deterministic,
			  flexibility_target, flexibility_target_growth_rate_max, random_g_flexibility, flexibility_price_cap,
			  corporate_tax_rate_growth_max,
			  construction_delays, failure_flag,
			  type_action_regulator,
			  agent_g_public, public_generators, 
			  agent_g_lobby, lobby,
			  CfD_activation_year, cm_activation_year, flexibility_activation_year,
			  cm_maximum_target, CfD_maximum_target, flexibility_maximum_target,
			  terminal_reward_flag, 
			  years_slack_termination, years_slack_profits,
			  scenario_tax_decree, shock_flag, scenario_tax_decree_deactivation_random_flag)

        return temp_env

    register_env("CM_EoM_ss_battery_661", env_creator)
    print(policy_ids)

    ## Assigning policies to GENCOs
    possible_agents_g = [f"Agent_g_{i}" for i in range(agent_g)]
    possible_policies_g = [f"Policy_g_{i}" for i in range(int(max(policy_ids) + 1))]

    policies = {}
    policies_g = {}
    policies_p = {}
    policies_name_g = []

    # Define all policies for g-agents
    for agent_id in env.possible_agents_g:
        policy_name = f"policy_{agent_id}"
        policies_name_g.append(policy_name)

        policy_spec = PolicySpec(
            observation_space=env.observation_spaces_g[agent_id],
            action_space=env.action_spaces_g[agent_id],
            config={}
        )

        policies[policy_name] = policy_spec
        policies_g[policy_name] = policy_spec

    # Define all policies for p-agents
    for agent_id in env.possible_agents_p:
        policy_name = f"policy_{agent_id}"

        policy_spec = PolicySpec(
            observation_space=env.observation_spaces_p[agent_id],
            action_space=env.action_spaces_p[agent_id],
            config={}
        )

        policies[policy_name] = policy_spec
        policies_p[policy_name] = policy_spec

    # Create RL module specs — each agent gets its own obs/action spaces
    # explicitly, since g-agents and p-agents have different spaces.
    RL_module_g = {
        f"policy_{agent_id}": SingleAgentRLModuleSpec(
            module_class=TorchActionMaskRLM,
            catalog_class=CustomPPOCatalog,
            observation_space=env.observation_spaces_g[agent_id],
            action_space=env.action_spaces_g[agent_id],
        )
        for agent_id in env.possible_agents_g
    }

    RL_module_p = {
        f"policy_{agent_id}": SingleAgentRLModuleSpec(
            module_class=TorchActionMaskRLM,
            catalog_class=CustomPPOCatalog,
            observation_space=env.observation_spaces_p[agent_id],
            action_space=env.action_spaces_p[agent_id],
        )
        for agent_id in env.possible_agents_p
    }

    RL_module_dict = {**RL_module_g, **RL_module_p}

    # Add shared critic module required by MAPPO.
    # The SharedCriticRLModule receives the concatenated observations of all
    # agent modules and produces one value estimate per agent.
    import gymnasium as gym
    import numpy as _np
    
    # ── Critic hyper-parameters — edit here to reconfigure the network ────────────
    CRITIC_CONFIG = {
        "bottleneck_dim": 8,         
        "encoder_hiddens": [256,32],        
        "mixer_hiddens":   [256,256],
        "activation":      "relu",
    }

    # ── Joint obs space for the shared critic ─────────────────────────────────────
    # Agent ordering must match AGENT_TYPE_PREFIXES in shared_critic.py:
    #   [Agent_g_*, Agent_p_*]  (g-agents first, then p-agents, each sorted)
    def _unwrap_obs_space(sp):
        """Strip action-mask Dict wrapper if present."""
        if isinstance(sp, gym.spaces.Dict) and "observations" in sp.spaces:
            return sp.spaces["observations"]
        return sp

    _obs_spaces_unwrapped = {
        agent_id: _unwrap_obs_space(sp)
        for agent_id, sp in (
            list(env.observation_spaces_g.items()) +
            list(env.observation_spaces_p.items())
        )
    }

    # Build combined obs space in the same order the critic will split it
    _agent_order = [
        aid
        for prefix in AGENT_TYPE_PREFIXES
        for aid in sorted([a for a in _obs_spaces_unwrapped if a.startswith(prefix)])
    ]

    _combined_obs_dim = sum(_obs_spaces_unwrapped[aid].shape[0] for aid in _agent_order)
    _combined_obs_space = gym.spaces.Box(
        low=-_np.inf, high=_np.inf, shape=(_combined_obs_dim,), dtype=_np.float32
    )

    _n_agents        = len(_agent_order)
    _dummy_action_space = list(env.action_spaces_g.values())[0]

    RL_module_dict["shared_critic"] = SingleAgentRLModuleSpec(
        module_class=SharedCriticTorchRLModule,
        catalog_class=SharedCriticCatalog,
        observation_space=_combined_obs_space,
        action_space=_dummy_action_space,
        inference_only=False,
        model_config={
            **CRITIC_CONFIG,                        # network hyper-parameters
            "n_agents":          _n_agents,
            "observation_spaces": _obs_spaces_unwrapped,
        },
    )

    def policy_mapping_fn(agent_id, episode, worker=None, **kwargs):
        return f"policy_{agent_id}"

    # ── Build config with new API ─────────────────────────────────────────────
    config = (
        MAPPOConfig()
        .environment(
            "CM_EoM_ss_battery_661",
            # Provide per-policy Dict spaces so the AgentToModuleMapping
            # connector can map agent IDs to their module's obs/action space
            # without querying the env runner group before it is ready.
            observation_space=gym.spaces.Dict({
                **{f"policy_{agent_id}": env.observation_spaces_g[agent_id]
                   for agent_id in env.possible_agents_g},
                **{f"policy_{agent_id}": env.observation_spaces_p[agent_id]
                   for agent_id in env.possible_agents_p},
            }),
            action_space=gym.spaces.Dict({
                **{f"policy_{agent_id}": env.action_spaces_g[agent_id]
                   for agent_id in env.possible_agents_g},
                **{f"policy_{agent_id}": env.action_spaces_p[agent_id]
                   for agent_id in env.possible_agents_p},
            }),
        )
        # New API stack flags (replaces .experimental(_enable_new_api_stack=True))
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )
        .rl_module(
            rl_module_spec=MultiRLModuleSpec(
                rl_module_specs=RL_module_dict,
            ),
        )
        .framework("torch")
        # .learners() replaces .resources(num_gpus_per_learner_worker=...)
        .learners(
            num_gpus_per_learner=1,
        )
        # .env_runners() replaces .rollouts()
        .env_runners(
            num_env_runners=60,
            num_envs_per_env_runner=1,
        )
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
            policy_states_are_swappable=False,
            policies_to_train=list(policies.keys()) + ["shared_critic"], 

            algorithm_config_overrides_per_module={
                **{f"policy_{agent_id}": MAPPOConfig.overrides(
                    clip_param=0.1,
                ) for agent_id in env.possible_agents_g},

                **{f"policy_{agent_id}": MAPPOConfig.overrides(
                    clip_param=0.1,
                ) for agent_id in env.possible_agents_p},
            },
        )
        .training(
            train_batch_size=10800,
            minibatch_size=1800,
            use_value_normalization=True,
            vf_clip_param=5.0,
            num_sgd_iter=10,
            gamma=1,
            lambda_=0.995,
            entropy_coeff=0.01,
            kl_coeff=0.2,
            kl_target=0.05,
            model={
                "fcnet_hiddens": [512, 512],     
                "fcnet_activation": "relu", 
                "head_hidden":      64, 
            },
        )
    )
    # ─────────────────────────────────────────────────────────────────────────

    wandb.config.update(config.to_dict())

    start_time = time.time()

    run_count = read_run_count(count_file_path)

    if run_count == 0:

        base_path = "/work/cmcc/jg24923/CM_EoM_ss_battery_661/MAPPO"
        latest_checkpoint = find_latest_checkpoint(base_path)

        if latest_checkpoint:
            print(f"Restoring from checkpoint: {latest_checkpoint}")
            restore_string = latest_checkpoint
            print("Checkpoint restored successfully.")
        else:
            print("No checkpoint found.")

        # ── tune.run() is deprecated; use tune.Tuner().fit() ─────────────────
        tuner = tune.Tuner(
            MAPPO,
            run_config=tune.RunConfig(
                name="MAPPO",
                stop={"time_total_s": 112000},
                checkpoint_config=tune.CheckpointConfig(
                    checkpoint_frequency=50,
                    checkpoint_at_end=True,
                ),
                storage_path="/work/cmcc/jg24923/CM_EoM_ss_battery_661",
                callbacks=[WandbLoggerCallback(project="Test_WC")],
            ),
            param_space=config.to_dict(),
        )
        tuner.fit()
        # ─────────────────────────────────────────────────────────────────────
