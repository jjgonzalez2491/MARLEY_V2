import pandas as pd
import os
import numpy as np
import torch
from ray.rllib.core.columns import Columns
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.utils.numpy import convert_to_numpy, softmax
import time
import sys
import numpy_financial as npf
import glob

n = int(sys.argv[1]) + 10

# Generate M×N matrix with different lambda per column
def generate_delay_matrix(m_rows, lambdas):
    n_cols = len(lambdas)
    delay_matrix = np.zeros((m_rows, n_cols),  dtype=int)
    
    for col, lam in enumerate(lambdas):
        delay_matrix[:, col] = np.random.poisson(lam, m_rows)
    
    return delay_matrix

def find_latest_checkpoint(input_string, CVaR):
    if CVaR == 1:
        base_path = f"/work/cmcc/jg24923/{input_string}/PPO_cvar"
    elif CVaR == 2:
        base_path = f"/work/cmcc/jg24923/{input_string}/MAPPO"
    else:
        base_path = f"/work/cmcc/jg24923/{input_string}/PPO"

    if not os.path.exists(base_path):
        raise FileNotFoundError(f"The path {base_path} does not exist.")

    # Find all checkpoint files under the base path, regardless of folder name
    checkpoint_paths = glob.glob(os.path.join(base_path, "*/checkpoint_*"))

    if not checkpoint_paths:
        raise FileNotFoundError("No checkpoints found for the given input_string.")

    # Extract the checkpoint number from the path and find the latest one
    latest_checkpoint = max(checkpoint_paths, key=os.path.getmtime)

    return latest_checkpoint

test_idx = n - 1

# Define scenarios ## TODO Change Some of the runs to 568 instead of 567

scenarios = {           
    10: (6,  636, 'CM_CfD_', 1, 0, 100, 80, 0.50, False, 0, True),           
    11: (6,  637, 'CM_CfD_', 1, 1, 100, 80, 0.50, False, 0, True),           
    12: (1,  638, 'CM_CfD_', 1, 0, 100, 80, 0.50, False, 0, True),           
    13: (1,  639, 'CM_CfD_', 1, 1, 100, 80, 0.50, False, 0, True),           
    14: (6,  640, 'CM_CfD_', 1, 0, 100, 80, 0.50, False, 0, True),           
    15: (6,  641, 'CM_CfD_', 1, 1, 100, 80, 0.50, False, 0, True),            
    16: (1,  642, 'CM_CfD_', 1, 1, 4, 80, 0.50, False, 0, True),             
    17: (1,  643, 'CM_CfD_', 2, 0, 100, 80, 0.50, False, 0, True),             
    18: (6,  644, 'CM_CfD_', 1, 1, 4, 80, 0.50, False, 0, True),           
    19: (6,  645, 'CM_CfD_', 1, 1, 4, 120, 1.00, False, 0, True),           
    20: (1,  646, 'CM_CfD_', 2, 1, 100, 80, 0.50, False, 0, True),           
    21: (1,  647, 'CM_CfD_', 3, 0, 100, 80, 0.50, False, 0, True),           
    22: (1,  648, 'CM_CfD_', 3, 1, 100, 80, 0.50, False, 0, True),           
    23: (6,  649, 'CM_CfD_', 2, 0, 100, 80, 0.50, False, 0, True),           
    24: (6,  650, 'CM_CfD_', 2, 1, 100, 80, 0.50, False, 0, True),           
    25: (6,  651, 'CM_CfD_', 3, 0, 100, 80, 0.50, False, 0, True),           
    26: (6,  652, 'CM_CfD_', 3, 1, 100, 80, 0.50, False, 0, True),           
    27: (6,  653, 'CM_CfD_', 1, 0, 100, 120, 1.00, False, 0, True),           
    28: (6,  654, 'CM_CfD_', 1, 1, 100, 120, 1.00, False, 0, True),           
    29: (6,  655, 'CM_CfD_', 1, 0, 100, 60, 0.375, False, 0, True),           
    30: (6,  656, 'CM_CfD_', 1, 1, 100, 60, 0.375, False, 0, True),            
    31: (6,  615, 'CM_CfD_', 1, 0, 100, 80, 0.5, False, 2, True),           
    32: (6,  616, 'CM_CfD_', 1, 1, 100, 80, 0.5, False, 2, True),           
    33: (1,  617, 'CM_CfD_', 1, 0, 100, 80, 0.5, False, 2, True),           
    34: (1,  618, 'CM_CfD_', 1, 1, 100, 80, 0.5, False, 2, True),            
    35: (6,  657, 'CM_CfD_', 1, 1, 4, 60, 0.375, False, 0, True),                                                                                
}

policy_deterministic, test_init, scenario_name, excel, scenario_tax_decree, scenario_tax_decree_start, mechanism_target, multiplier_long_term_markets, scenario_tax_decree_deactivation_random_flag, MAPPO, shock_flag = scenarios[test_idx]

from CM_CfD_EU_ETS_sampling import CM_EoM

if excel == 1:
    
    dir_input_data = '050526 Entry data - Italy 2025 - 16 Agents - Pypsa - 10 years - Super Tax - V2.xlsx'

checkpoint = find_latest_checkpoint(f'CM_EoM_ss_battery_{test_init}', MAPPO)

rl_module_base_path = os.path.join(
    checkpoint,          # your existing checkpoint path variable
    "learner_group",
    "learner",
    "rl_module",
)

string = f'{n}_{scenario_name}{n}'
 
if __name__ == "__main__":

    start_time = time.time()

    penalty_type = 2

    agent_g = 16

    type_action_regulator = 2

    agent_p = 1

    curriculum_step = 1

    demand_growth_type = "Deterministic"

    penalty_factor = 0.0

    max_t = 144 * 30

    years_slack_termination = 8
    
    years_slack_profits = 12

    iters_outer_loop = 1000

    short_t = 24
    
    VoLL_max = 4000

    period_inv = 10   

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

    average_bimester_series = pd.read_excel(dir_input_data, sheet_name='Average_bimester_series')

    average_bimester_series =  np.array(average_bimester_series.iloc[:,:])

    average_bimester_series = average_bimester_series[:,4:]

    average_yearly_series = pd.read_excel(dir_input_data, sheet_name='Average_yearly_series')

    average_yearly_series =  np.array(average_yearly_series.iloc[:,:])

    average_yearly_series = average_yearly_series[:,1:]

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

    flexibility_target = 0

    flexibility_target_growth_rate_max = 10

    random_g_flexibility = np.random.random([503, agent_g * 1])

    flexibility_price_cap = 30000

    policy_deterministic_activation = True

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

    agent_g_public = [0]

    public_generators = False

    agent_g_lobby = [0]

    lobby = False

    # activation markets 

    cm_activation_year = 0

    CfD_activation_year = 0
    
    flexibility_activation_year = 0

    # maximum targets in markets

    cm_maximum_target = 2
    CfD_maximum_target = mechanism_target
    flexibility_maximum_target = mechanism_target 

    # Terminal reward flag

    terminal_reward_flag = False
    
    # CM targets
    
    cm_reliability_target_values = [0.000001, 0.00001, 0.0001, 0.0005, 0.01]
    
    # CfD targets

    CfD_target_increments_steps = np.array([0, 2, 4, 6, 8]) * multiplier_long_term_markets
    
    # Flexibility targets
    
    flexibility_target_increments_steps = np.array([0, 2, 4, 6, 8]) * multiplier_long_term_markets

    corporate_tax_rate_growth_max = 0.3

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
              scenario_tax_decree,shock_flag,scenario_tax_decree_deactivation_random_flag,scenario_tax_decree_start)
    

    iters_max = int(max_t/short_t)

    agent_ids_g = env.possible_agents_g

    agent_ids_p = env.possible_agents_p

    # ── 1. Build agent_id -> module_id mapping ────────────────────────────────────
    policy_ids_dict = {}

    policies_g = [f"policy_{agent_id}" for agent_id in env.possible_agents_g]
    policies_p = [f"policy_{agent_id}" for agent_id in env.possible_agents_p]

    for idx_g in range(agent_g):
        policy_ids_dict[env.possible_agents_g[idx_g]] = policies_g[idx_g]

    for idx_p in range(agent_p):
        policy_ids_dict[env.possible_agents_p[idx_p]] = policies_p[idx_p]

    modules = {}

    for agent_id in agent_ids_g + agent_ids_p:
        module_id = policy_ids_dict[agent_id]
        print(f"Restoring module {module_id} ...", end="")
        modules[agent_id] = RLModule.from_checkpoint(
            os.path.join(rl_module_base_path, module_id)
        )
        print(" ok")

    # ── 2. Pre-compute nvec per agent for logit splitting ────────────────────────
    # For MultiDiscrete spaces, RLlib flattens logits as:
    #   [logits_sub0 | logits_sub1 | ... | logits_subN]
    # where each block has size nvec[i]. We need nvec to split them back.
    # nvec is stored on the module's action_space after restoration.
    nvecs = {}
    for agent_id in agent_ids_g + agent_ids_p:
        action_space = modules[agent_id].action_space
        if hasattr(action_space, "nvec"):
            nvecs[agent_id] = action_space.nvec.tolist()
        else:
            # Discrete: single sub-action
            nvecs[agent_id] = [action_space.n]

    # ── 3. Helper: sample a MultiDiscrete action from flat logits ─────────────────
    def sample_multidiscrete_action(logits_flat, nvec, agent_id=None):
        action = []
        offset = 0
        for idx, n in enumerate(nvec):
            sub_logits = logits_flat[offset: offset + n]

            # Shift logits for numerical stability before softmax
            sub_logits = sub_logits - np.max(sub_logits)

            probs = np.exp(sub_logits)
            probs_sum = probs.sum()

            if probs_sum == 0 or np.isnan(probs_sum):
                print(f"WARNING: degenerate sub-action block {idx} for agent {agent_id}, falling back to uniform")
                probs = np.ones(n) / n
            else:
                probs = probs / probs_sum
                probs = np.clip(probs, 0.0, 1.0)
                probs /= probs.sum()

            action.append(np.random.choice(n, p=probs))
            offset += n
        return np.array(action)

    weighted_prices = np.zeros([iters_max, iters_outer_loop])
    
    total_costs = np.zeros([iters_max, iters_outer_loop])

    weighted_short_term_prices = np.zeros([iters_max * 24, iters_outer_loop])

    weighted_short_term_prices_net = np.zeros([iters_max * 24, iters_outer_loop])

    weighted_prices_reliability = np.zeros([iters_max, iters_outer_loop])

    weighted_prices_mid_day = np.zeros([iters_max, iters_outer_loop])

    weighted_prices_afternoon = np.zeros([iters_max, iters_outer_loop])

    cm_premium_price = np.zeros([iters_max, iters_outer_loop])

    res_share = np.zeros([iters_max, iters_outer_loop])

    fossil_fuel_capacity_factor = np.zeros([iters_max, iters_outer_loop])

    storage_share = np.zeros([iters_max, iters_outer_loop])

    cm_balance_total = np.zeros([iters_max, iters_outer_loop])

    CfD_premium_price = np.zeros([iters_max, iters_outer_loop])

    flexibility_premium_price = np.zeros([iters_max, iters_outer_loop])

    CfD_balance_total = np.zeros([iters_max, iters_outer_loop])

    flexibility_balance_total = np.zeros([iters_max, iters_outer_loop])

    capacity_2030 = np.zeros([n_tech + n_tech_battery, iters_outer_loop])

    capacity_2030_merchant = np.zeros([n_tech + n_tech_battery, iters_outer_loop])

    capacity_2030_cm = np.zeros([n_tech + n_tech_battery, iters_outer_loop])

    capacity_2030_CfD = np.zeros([n_tech + n_tech_battery, iters_outer_loop])

    capacity_2030_flexibility = np.zeros([n_tech + n_tech_battery, iters_outer_loop])

    capacity_2030_decom = np.zeros([n_tech + n_tech_battery, iters_outer_loop])



    capacity_2035 = np.zeros([n_tech + n_tech_battery, iters_outer_loop])

    capacity_2035_merchant = np.zeros([n_tech + n_tech_battery, iters_outer_loop])

    capacity_2035_cm = np.zeros([n_tech + n_tech_battery, iters_outer_loop])

    capacity_2035_CfD = np.zeros([n_tech + n_tech_battery, iters_outer_loop])

    capacity_2035_flexibility = np.zeros([n_tech + n_tech_battery, iters_outer_loop])

    capacity_2035_decom = np.zeros([n_tech + n_tech_battery, iters_outer_loop])

    

    capacity_2040 = np.zeros([n_tech + n_tech_battery, iters_outer_loop])

    capacity_2040_merchant = np.zeros([n_tech + n_tech_battery, iters_outer_loop])

    capacity_2040_cm = np.zeros([n_tech + n_tech_battery, iters_outer_loop])

    capacity_2040_CfD = np.zeros([n_tech + n_tech_battery, iters_outer_loop])

    capacity_2040_flexibility = np.zeros([n_tech + n_tech_battery, iters_outer_loop])

    capacity_2040_decom = np.zeros([n_tech + n_tech_battery, iters_outer_loop])



    capacity_2030_agent = np.zeros([agent_g, iters_outer_loop])

    capacity_2035_agent = np.zeros([agent_g, iters_outer_loop])

    capacity_2035_agent_existing = np.zeros([agent_g, iters_outer_loop])

    capacity_2040_agent = np.zeros([agent_g, iters_outer_loop])

    capacity_2040_agent_existing = np.zeros([agent_g, iters_outer_loop])

    capacity_2030_agent_t1 = np.zeros([agent_g, iters_outer_loop])

    capacity_2035_agent_t1 = np.zeros([agent_g, iters_outer_loop])

    capacity_2040_agent_t1 = np.zeros([agent_g, iters_outer_loop])

    capacity_2030_agent_t2 = np.zeros([agent_g, iters_outer_loop])

    capacity_2035_agent_t2 = np.zeros([agent_g, iters_outer_loop])

    capacity_2040_agent_t2 = np.zeros([agent_g, iters_outer_loop])

    capacity_2030_agent_t3 = np.zeros([agent_g, iters_outer_loop])

    capacity_2035_agent_t3 = np.zeros([agent_g, iters_outer_loop])

    capacity_2040_agent_t3 = np.zeros([agent_g, iters_outer_loop])

    capacity_2030_agent_t4 = np.zeros([agent_g, iters_outer_loop])

    capacity_2035_agent_t4 = np.zeros([agent_g, iters_outer_loop])

    capacity_2040_agent_t4 = np.zeros([agent_g, iters_outer_loop])

    capacity_2030_agent_t5 = np.zeros([agent_g, iters_outer_loop])

    capacity_2035_agent_t5 = np.zeros([agent_g, iters_outer_loop])

    capacity_2040_agent_t5 = np.zeros([agent_g, iters_outer_loop])

    capacity_2030_agent_t6 = np.zeros([agent_g, iters_outer_loop])

    capacity_2035_agent_t6 = np.zeros([agent_g, iters_outer_loop])

    capacity_2040_agent_t6 = np.zeros([agent_g, iters_outer_loop])

    capacity_2030_agent_t7 = np.zeros([agent_g, iters_outer_loop])

    capacity_2035_agent_t7 = np.zeros([agent_g, iters_outer_loop])

    capacity_2040_agent_t7 = np.zeros([agent_g, iters_outer_loop])

    capacity_2030_agent_t8 = np.zeros([agent_g, iters_outer_loop])

    capacity_2035_agent_t8 = np.zeros([agent_g, iters_outer_loop])

    capacity_2040_agent_t8 = np.zeros([agent_g, iters_outer_loop])

    reward_tech = np.zeros([n_tech + n_tech_battery, iters_outer_loop])

    total_reward_cm = np.zeros([iters_outer_loop])

    demand_growth = np.zeros([iters_outer_loop])

    total_reward_merchant = np.zeros([iters_outer_loop])

    CO2_emissions = np.zeros([iters_max, iters_outer_loop])

    mark_up_2021 = np.zeros([n_tech, iters_outer_loop])

    mark_up_2030 = np.zeros([n_tech, iters_outer_loop])

    mark_up_2035 = np.zeros([n_tech, iters_outer_loop])

    prices_hour_2025 = np.zeros([iters_outer_loop, int(short_t*6)])

    prices_hour_2030 = np.zeros([iters_outer_loop, int(short_t*6)])

    prices_hour_2035 = np.zeros([iters_outer_loop, int(short_t*6)])

    prices_hour_2040 = np.zeros([iters_outer_loop, int(short_t*6)])

    prices_hour_2025_net = np.zeros([iters_outer_loop, int(short_t*6)])
    prices_hour_2030_net = np.zeros([iters_outer_loop, int(short_t*6)])
    prices_hour_2035_net = np.zeros([iters_outer_loop, int(short_t*6)])
    prices_hour_2040_net = np.zeros([iters_outer_loop, int(short_t*6)])

    mark_up_2021_temp = np.zeros([n_tech])

    mark_up_2030_temp = np.zeros([n_tech])

    mark_up_2035_temp = np.zeros([n_tech])

    total_production = np.zeros([iters_max, n_tech])

    capacity_factor = np.zeros([iters_max, n_tech])

    capacity_cm = np.zeros([iters_max, n_tech + n_tech_battery])

    cc_cm_tech = np.zeros([iters_max, n_tech + n_tech_battery])

    capacity_cm_battery = np.zeros([iters_max, n_tech_battery])

    average_demand = np.zeros([iters_max, 1])

    capacity_cm_uc = np.zeros([iters_max, n_tech])

    capacity_CfD = np.zeros([iters_max, n_tech])

    capacity_merchant_agent = np.zeros([iters_max, agent_g])

    reward_tech_cm = np.zeros([iters_max, n_tech + n_tech_battery])

    reward_tech_CfD = np.zeros([iters_max, n_tech + n_tech_battery])

    reward_tech_merchant = np.zeros([iters_max, n_tech + n_tech_battery])

    reward_tech_existing = np.zeros([iters_max, n_tech + n_tech_battery])

    reward_tech_flexibility = np.zeros([iters_max, n_tech + n_tech_battery])

    price_hour = np.zeros([iters_max,24])

    price_hour_flexibility = np.zeros([iters_max,24])

    SoC_3_hours = np.zeros([iters_max,24])
    
    SoC_8_hours = np.zeros([iters_max,24])

    discharge_3_hours = np.zeros([iters_max,24])
    charge_3_hours = np.zeros([iters_max,24])
        
    SoC_long = np.zeros(iters_max)
    SoC_long_individual = np.zeros([agent_g, iters_max])

    # Variables short-term storage

    storage_level_short_aggregated = np.zeros([1, agent_g])
    storage_strategy_short_aggregated = np.zeros([1, agent_g])

    # Variables long-term storage

    storage_level_long_aggregated = np.zeros([1, agent_g])
    storage_strategy_long_aggregated = np.zeros([1, agent_g])

    capacity_merchant = np.zeros([iters_max, n_tech + n_tech_battery])

    capacity_decom = np.zeros([iters_max, n_tech])
    inv_g_aging = np.zeros([iters_max, n_tech])

    capacity_merchant_uc = np.zeros([iters_max, n_tech])

    capacity_flexibility = np.zeros([iters_max, n_tech + n_tech_battery])

    capacity_cm_agent = np.zeros([iters_max, agent_g])

    capacity_CfD_agent = np.zeros([iters_max, agent_g])

    total_installed_capacity = np.zeros([iters_max,n_tech + n_tech_battery])

    capacity_CfD_uc = np.zeros([iters_max, n_tech])

    penalty_term_tech = np.zeros([n_tech + 2, iters_outer_loop])

    penalty_term_agent = np.zeros([agent_g, iters_outer_loop])

    reward_penalty_agent = np.zeros([agent_g, iters_outer_loop])

    entry_enabled = np.zeros([agent_g, iters_outer_loop])

    strategy_3_hours = np.zeros([iters_max, env.short_t])

    strategy_4_hours = np.zeros([iters_max, env.short_t])

    strategy_8_hours = np.zeros([iters_max, env.short_t])

    SoC_state = np.zeros([iters_max, agent_g])

    SoC_system = np.zeros([iters_max, iters_outer_loop])

    cm_scarcity = np.zeros([iters_max, iters_outer_loop])

    CfD_scarcity = np.zeros([iters_max, iters_outer_loop])

    cm_income_existing = np.zeros([iters_max, iters_outer_loop])

    flexibility_scarcity = np.zeros([iters_max, iters_outer_loop])

    CfD_price_cap = np.zeros([iters_max, iters_outer_loop])

    short_term_scarcity = np.zeros([iters_max, iters_outer_loop])

    reward_agents = np.zeros([iters_max, agent_g])

    CO2_tax = np.zeros([iters_max, iters_outer_loop])

    carbon_tax_return = np.zeros([iters_max, iters_outer_loop])

    CfD_target = np.zeros([iters_max, iters_outer_loop])

    CfD_target_action = np.zeros([iters_max, iters_outer_loop])

    cm_target = np.zeros([iters_max, iters_outer_loop])

    cm_price_cap = np.zeros([iters_max, iters_outer_loop])

    cm_strike = np.zeros([iters_max, iters_outer_loop])

    cm_income_total = np.zeros([iters_max, iters_outer_loop])

    flexibility_target = np.zeros([iters_max, iters_outer_loop])

    flexibility_target_action = np.zeros([iters_max, iters_outer_loop])

    corporate_tax_rate = np.zeros([iters_max, iters_outer_loop])

    rewards_planner = np.zeros([8, iters_outer_loop])

    policy_scenarios_selected = np.zeros([iters_outer_loop])

    cost_markets = np.zeros([4,iters_outer_loop])

    RES_curtailed = np.zeros([iters_max, iters_outer_loop])

    shock_multiplier = np.zeros([iters_max, iters_outer_loop])

    energy_not_served = np.zeros([iters_max, iters_outer_loop])

    cost_spot_other_markets = np.zeros([iters_max, 8])

    cost_spot_other_markets_0 = np.zeros([iters_max, iters_outer_loop])

    cost_spot_other_markets_1 = np.zeros([iters_max, iters_outer_loop])

    cost_spot_other_markets_2 = np.zeros([iters_max, iters_outer_loop])

    cost_spot_other_markets_3 = np.zeros([iters_max, iters_outer_loop])

    cost_spot_other_markets_4 = np.zeros([iters_max, iters_outer_loop])

    cost_spot_other_markets_5 = np.zeros([iters_max, iters_outer_loop])

    cost_spot_other_markets_6 = np.zeros([iters_max, iters_outer_loop])

    cost_spot_other_markets_7 = np.zeros([iters_max, iters_outer_loop])

    for capacity_iteration in range(iters_outer_loop):

        short_term_prices_net_tmp = [] 

        short_term_prices_tmp = []

        capacity_2030_agent_temp_tech = np.zeros([agent_g, n_tech + n_tech_battery])

        capacity_2035_agent_temp_tech = np.zeros([agent_g, n_tech + n_tech_battery])

        capacity_2040_agent_temp_tech = np.zeros([agent_g, n_tech + n_tech_battery])

        capacity_2030_temp = np.zeros([n_tech + n_tech_battery])

        capacity_2030_agent_temp = np.zeros([agent_g])

        capacity_2035_temp = np.zeros([n_tech + n_tech_battery])

        capacity_2035_agent_temp = np.zeros([agent_g])

        capacity_2035_agent_existing_temp = np.zeros([agent_g])

        capacity_2040_temp = np.zeros([n_tech + n_tech_battery])

        capacity_2040_agent_temp = np.zeros([agent_g])

        capacity_2040_agent_existing_temp = np.zeros([agent_g])

        action = {}

        obs, info = env.reset()

        print("Demand and Carbon scenario pre-reset")

        print(f'demand_{env.demand_deterministic}_carbon_{env.carbon_deterministic}')

        entry_enabled[:,capacity_iteration] = env.entry_enabled

        demand_growth[capacity_iteration] = env.demand_growth

        i = 0

        price = np.zeros([iters_max])
        short_term_price = np.zeros([iters_max])
        CO2_emissions_temp = np.zeros([iters_max])
        LL_marker = np.zeros([iters_max])
        d_Total = np.zeros([iters_max])

        cm_price = np.zeros([iters_max])
        cm_balance = np.zeros([iters_max])
        flexibility_balance = np.zeros([iters_max])

        CfD_price = np.zeros([iters_max])
        CfD_balance = np.zeros([iters_max])

        flexibility_price = np.zeros([iters_max])

        # Variables short-term storage

        storage_level_short = np.zeros([agent_g, iters_max])
        storage_strategy_short = np.zeros([agent_g, iters_max])

        # Variables long-term storage

        storage_level_long = np.zeros([agent_g, iters_max])
        storage_strategy_long = np.zeros([agent_g, iters_max])

        total_generation = np.zeros([n_tech, iters_max])

        # Rewards

        reward_n = np.zeros([agent_g, iters_max])

        # Prices

        hourly_prices = np.zeros([short_t, iters_max])

        flexibility_prices = np.zeros([short_t, iters_max])

        hourly_prices_reliablity = np.zeros([iters_max])

        prices_mid_day = np.zeros([iters_max])

        prices_afternoon = np.zeros([iters_max])

        low = 0
        high = 1

        i = 0

        delta = 0.001

        t_2025 = 0

        t_2030 = 0

        t_2035 = 0

        t_2040 = 0    

        # ── 5. Initialise RNN states (empty dict for feedforward modules) ─────────────
        states = {}
        for agent_id in agent_ids_g + agent_ids_p:
            states[agent_id] = modules[agent_id].get_initial_state()

        # ── 6. Evaluation loop ────────────────────────────────────────────────────────
        while i < iters_max:

            actions = {}
            agents_c = 0
            t = env.time

            for agent_id in agent_ids_g + agent_ids_p:

                module    = modules[agent_id]
                agent_obs = obs[agent_id]

                action_mask = agent_obs["action_mask"]
                inner_obs   = agent_obs["observations"]

                if max(inner_obs) > 1 or min(inner_obs) < -1:
                    pass  # your existing obs-range guard

                # Build input batch with leading batch dimension of 1
                input_batch = {
                    SampleBatch.OBS: {
                        "action_mask":  torch.tensor(action_mask, dtype=torch.float32).unsqueeze(0),
                        "observations": torch.tensor(inner_obs,   dtype=torch.float32).unsqueeze(0),
                    }
                }

                if states[agent_id]:
                    input_batch[Columns.STATE_IN] = states[agent_id]

                with torch.no_grad():
                    out = module.forward_inference(input_batch)

                # out[SampleBatch.ACTION_DIST_INPUTS] shape: (1, sum(nvec))
                # The action mask has already been applied inside mask_forward_fn_torch,
                # so invalid actions have logit = -inf and will never be sampled.
                logits_flat = convert_to_numpy(out[SampleBatch.ACTION_DIST_INPUTS])[0]

                action = sample_multidiscrete_action(logits_flat, nvecs[agent_id])

                # If your env expects a tuple instead of numpy array, uncomment:
                # action = tuple(action.tolist())

                actions[agent_id] = action
                states[agent_id]  = out.get(Columns.STATE_OUT, {})

                agents_c += 1

            # Step the environment
            obs, rewards, terminateds, truncateds, infos = env.step(actions)
            
            CO2_tax[i,capacity_iteration] = env.CO2_tax

            CfD_target[i,capacity_iteration] = env.CfD_target_penetration

            CfD_target_action[i,capacity_iteration] = env.CfD_increment_target

            CfD_price_cap[i, capacity_iteration] = env.CfD_price_cap

            cm_target[i,capacity_iteration] = env.cm_demand_target

            cm_strike[i,capacity_iteration] = env.cm_strike

            cm_income_total[i,capacity_iteration] = (np.sum(env.cm_income) + np.sum(env.cm_income_existing)) * env.normalization_factor/(env.short_t * env.hour_month)

            res_share[i,capacity_iteration] = np.sum(env.total_production_tech[0:3]/np.sum(env.total_production_tech))

            fossil_fuel_capacity_factor[i, capacity_iteration] = np.sum(env.total_production_tech[3:6])/((np.sum(env.inv_g[:,3]) + np.sum(env.inv_g[:,4]) + np.sum(env.inv_g[:,5])) * env.short_t * env.hour_month)

            projected_flexibility_capacity_3_hours = np.sum(
                env.capacity_merchant_battery_uc[:,0] 
                + env.capacity_cm_battery_uc[:,0] 
                + env.capacity_flexibility_battery_uc[:,0] 
                + env.inv_init_battery[:,0]
            ) * 3

            projected_flexibility_capacity_8_hours = np.sum(
                env.capacity_merchant_battery_uc[:,1] 
                + env.capacity_cm_battery_uc[:,1] 
                + env.capacity_flexibility_battery_uc[:,1] 
                + env.inv_init_battery[:,1]
            ) * 8

            storage_share[i,capacity_iteration] = (np.sum(projected_flexibility_capacity_3_hours + projected_flexibility_capacity_8_hours) * env.hour_month * env.short_t)/np.sum(env.total_production_tech)

            cm_price_cap[i,capacity_iteration] = env.cm_price_cap

            flexibility_target[i, capacity_iteration] = env.flexibility_target_penetration

            flexibility_target_action[i, capacity_iteration] = env.flexibility_increment_target

            corporate_tax_rate[i, capacity_iteration] = env.corporate_tax_rate

            cm_price[i] = ((np.sum(env.cm_income) + np.sum(env.cm_income_battery)) /(env.short_t * env.hour_month))/(np.sum(env.cm_inv_options) + np.sum(env.cm_inv_options_battery) + delta) * env.normalization_factor

            cm_balance[i] = env.cm_balance_real/env.percentile_demand

            flexibility_balance[i] = env.flexibility_balance_real/env.percentile_demand

            CfD_income = 0

            print(f"Shock_variable{env.year_int}")
            print(env.penalty_term)

            flexibility_income = 0

            CfD_aggregated_capacity = 0

            flexibility_aggregated_capacity = 0

            agents_c = 0

            for agent_id in agent_ids_g:

                reward_agents[i,agents_c] += (np.sum(env.reward_tech_merchant_step[agents_c,:] + env.reward_tech_cm_step[agents_c,:] + env.reward_tech_CfD_step[agents_c,:]) + np.sum(env.reward_tech_merchant_battery_step[agents_c,:] + env.reward_tech_cm_battery_step[agents_c,:] + env.reward_tech_flexibility_battery_step[agents_c,:]) + env.reward_tech_existing_storage_lt_step[agents_c] - env.taxes_agents[agents_c])

                for tech in range(env.n_tech):

                    CfD_income += env.CfD_price_agents[agents_c,tech] * env.capacity_CfD[agents_c,tech] * env.availability_tech_average_yearly[env.year_int, tech]

                    CfD_aggregated_capacity +=  env.capacity_CfD[agents_c,tech]  * env.availability_tech_average_yearly[env.year_int, tech]

                for tech in range(env.n_tech_battery):

                    flexibility_income += env.flexibility_price_agents[agents_c,tech] * env.capacity_flexibility_battery[agents_c,tech]

                    flexibility_aggregated_capacity +=  env.capacity_flexibility_battery[agents_c,tech]

                agents_c += 1

            CfD_price[i] = CfD_income/(CfD_aggregated_capacity + delta)

            CfD_balance[i] = env.CfD_balance_real/env.percentile_demand

            flexibility_price[i] = flexibility_income/(flexibility_aggregated_capacity + delta)

            price[i] = env.price_net

            carbon_tax_return[i, capacity_iteration] = env.carbon_tax_returns_total

            short_term_price[i] = np.average(env.hourly_prices)

            short_term_prices_net_tmp.append(env.hourly_prices_net)

            short_term_prices_tmp.append(env.hourly_prices)

            capacity_cm[i,:n_tech] += np.sum(env.capacity_cm, axis=0)/iters_outer_loop

            capacity_cm[i,n_tech:n_tech + n_tech_battery] += np.sum(env.capacity_cm_battery, axis=0)/iters_outer_loop

            total_installed_capacity[i, :n_tech] += np.sum(env.inv_g, axis=0)/iters_outer_loop

            total_installed_capacity[i, n_tech:n_tech + n_tech_battery] += np.sum(env.inv_init_battery + env.capacity_merchant_battery + env.capacity_cm_battery + env.capacity_flexibility_battery, axis=0)/iters_outer_loop

            for j in range(env.n_tech + env.n_tech_battery): 
                cc_cm_tech[i,j] += env.cm_tech_cc[0,j]/iters_outer_loop

            capacity_cm_battery[i,:n_tech_battery] += np.sum(env.capacity_cm_battery)/iters_outer_loop

            capacity_merchant[i,:n_tech] += np.sum(env.capacity_merchant, axis=0)/iters_outer_loop 

            capacity_merchant[i,n_tech:n_tech + n_tech_battery] += np.sum(env.capacity_merchant_battery, axis=0)/iters_outer_loop 

            capacity_decom[i,:n:tech] += np.sum(env.capacity_decom, axis=0)/iters_outer_loop 

            inv_g_aging[i,:n:tech] += np.sum(env.inv_g_aging, axis=0)/iters_outer_loop 

            capacity_cm_uc[i,:] += np.sum(env.capacity_cm_uc, axis=0)/iters_outer_loop

            capacity_merchant_uc[i,:] += np.sum(env.capacity_merchant_uc, axis=0)/iters_outer_loop 

            capacity_CfD_uc[i,:] += np.sum(env.capacity_CfD_uc, axis=0)/iters_outer_loop

            capacity_cm_agent[i,:] += np.sum(env.capacity_cm, axis=1)/iters_outer_loop

            capacity_merchant_agent[i,:] += np.sum(env.capacity_merchant, axis=1)/iters_outer_loop

            capacity_CfD[i,:] += np.sum(env.capacity_CfD, axis=0)/iters_outer_loop

            capacity_flexibility[i,0] += np.sum(env.capacity_flexibility_battery)/iters_outer_loop

            capacity_CfD_agent[i,:] += np.sum(env.capacity_CfD, axis=1)/iters_outer_loop

            for t in range(24):

                price_hour[i,t] += env.hourly_prices[t]/iters_outer_loop
                if env.hourly_prices[t] > 1000:
                    short_term_scarcity[i,capacity_iteration] += 1

                price_hour_flexibility[i,t] += env.price_flexibility[t]/iters_outer_loop

                SoC_3_hours[i,t] += env.SoC_short_term_storage[t,0]/iters_outer_loop
                SoC_8_hours[i,t] += env.SoC_short_term_storage[t,1]/iters_outer_loop

                discharge_3_hours[i,t] += env.discharge_short_term_battery[t,0]/iters_outer_loop
                charge_3_hours[i,t] += env.charge_short_term_battery[t,0]/iters_outer_loop
                
            if env.year == 1:
                pass

            elif env.year_int == 1:

                for t in range(24):
                    prices_hour_2025[capacity_iteration,t_2025] = env.hourly_prices[t]
                    prices_hour_2025_net[capacity_iteration,t_2025] = env.hourly_prices_net[t]
                    t_2025 += 1

            elif env.year_int == 6:
                capacity_2030_temp[:env.n_tech] += np.sum(env.inv_g, axis = 0)/6
                capacity_2030_agent_temp += np.sum(env.inv_g, axis = 1)/6 
                capacity_2030_agent_temp += np.sum(env.capacity_merchant_battery + env.capacity_cm_battery + env.capacity_flexibility_battery, axis = 1)/6
                
                capacity_2030_temp[env.n_tech:env.n_tech + env.n_tech_battery] += np.sum(env.capacity_merchant_battery + env.capacity_cm_battery + env.capacity_flexibility_battery + env.inv_init_battery, axis = 0)/6
                
                capacity_2030_agent_temp_tech[:,:(env.n_tech)] += env.inv_g/6
                capacity_2030_agent_temp_tech[:,env.n_tech:env.n_tech + env.n_tech_battery] += (env.capacity_merchant_battery + env.capacity_cm_battery + env.capacity_flexibility_battery + env.inv_init_battery)/6

                # Capacity merchant

                capacity_2030_merchant[:env.n_tech, capacity_iteration] += np.sum(env.capacity_merchant, axis = 0)/6
                capacity_2030_merchant[env.n_tech:env.n_tech + env.n_tech_battery, capacity_iteration] += np.sum(env.capacity_merchant_battery, axis = 0)/6

                # Capacity cm

                capacity_2030_cm[:env.n_tech, capacity_iteration] += np.sum(env.capacity_cm, axis = 0)/6
                capacity_2030_cm[env.n_tech:env.n_tech + env.n_tech_battery, capacity_iteration] += np.sum(env.capacity_cm_battery, axis = 0)/6

                # Capacity CfD

                capacity_2030_CfD[:env.n_tech, capacity_iteration] += np.sum(env.capacity_CfD, axis = 0)/6

                # Capacity Flexibility

                capacity_2030_flexibility[env.n_tech:env.n_tech + env.n_tech_battery, capacity_iteration] += np.sum(env.capacity_flexibility_battery, axis = 0)/6

                # Capacity decom

                capacity_2030_decom[:env.n_tech, capacity_iteration] += np.sum(env.capacity_decom, axis = 0)/6

                for t in range(24):
                    prices_hour_2030[capacity_iteration,t_2030] = env.hourly_prices[t]
                    prices_hour_2030_net[capacity_iteration,t_2030] = env.hourly_prices_net[t]
                    t_2030 += 1


            elif env.year_int == 11:
                capacity_2035_temp[:env.n_tech] += np.sum(env.inv_g, axis = 0)/6
                capacity_2035_agent_temp += np.sum(env.inv_g, axis = 1)/6 
                capacity_2035_agent_temp += np.sum(env.capacity_merchant_battery + env.capacity_cm_battery + env.capacity_flexibility_battery, axis = 1)/6

                capacity_2035_agent_existing_temp += np.sum(env.inv_g_aging, axis = 1)/6 

                capacity_2035_temp[env.n_tech:env.n_tech + env.n_tech_battery] += np.sum(env.capacity_merchant_battery + env.capacity_cm_battery + env.capacity_flexibility_battery + env.inv_init_battery, axis = 0)/6

                capacity_2035_agent_temp_tech[:,:(env.n_tech)] += env.inv_g/6
                capacity_2035_agent_temp_tech[:,env.n_tech:env.n_tech + env.n_tech_battery] += (env.capacity_merchant_battery + env.capacity_cm_battery + env.capacity_flexibility_battery + env.inv_init_battery)/6
                
                for t in range(24):
                    prices_hour_2035[capacity_iteration,t_2035] = env.hourly_prices[t]
                    prices_hour_2035_net[capacity_iteration,t_2035] = env.hourly_prices_net[t]
                    t_2035 += 1

                # Capacity merchant

                capacity_2035_merchant[:env.n_tech, capacity_iteration] += np.sum(env.capacity_merchant, axis = 0)/6
                capacity_2035_merchant[env.n_tech:env.n_tech + env.n_tech_battery, capacity_iteration] += np.sum(env.capacity_merchant_battery, axis = 0)/6

                # Capacity cm

                capacity_2035_cm[:env.n_tech, capacity_iteration] += np.sum(env.capacity_cm, axis = 0)/6
                capacity_2035_cm[env.n_tech:env.n_tech + env.n_tech_battery, capacity_iteration] += np.sum(env.capacity_cm_battery, axis = 0)/6

                # Capacity CfD

                capacity_2035_CfD[:env.n_tech, capacity_iteration] += np.sum(env.capacity_CfD, axis = 0)/6

                # Capacity Flexibility

                capacity_2035_flexibility[env.n_tech:env.n_tech + env.n_tech_battery, capacity_iteration] += np.sum(env.capacity_flexibility_battery, axis = 0)/6

                # Capacity decom

                capacity_2035_decom[:env.n_tech, capacity_iteration] += np.sum(env.capacity_decom, axis = 0)/6


            elif env.year_int == 16:
                capacity_2040_temp[:env.n_tech] += np.sum(env.inv_g, axis = 0)/6
                capacity_2040_agent_temp += np.sum(env.inv_g, axis = 1)/6 
                capacity_2040_agent_temp += np.sum(env.capacity_merchant_battery + env.capacity_cm_battery + env.capacity_flexibility_battery, axis = 1)/6

                capacity_2040_agent_existing_temp += np.sum(env.inv_g_aging, axis = 1)/6 

                capacity_2040_temp[env.n_tech:env.n_tech + env.n_tech_battery] += np.sum(env.capacity_merchant_battery + env.capacity_cm_battery + env.capacity_flexibility_battery + env.inv_init_battery, axis = 0)/6
                capacity_2040_agent_temp_tech[:,:(env.n_tech)] += env.inv_g/6
                capacity_2040_agent_temp_tech[:,env.n_tech:env.n_tech + env.n_tech_battery] += (env.capacity_merchant_battery + env.capacity_cm_battery + env.capacity_flexibility_battery + env.inv_init_battery)/6
                
                for t in range(24):
                    prices_hour_2040[capacity_iteration,t_2040] = env.hourly_prices[t]
                    prices_hour_2040_net[capacity_iteration,t_2040] = env.hourly_prices_net[t]
                    t_2040 += 1

                # Capacity merchant

                capacity_2040_merchant[:env.n_tech, capacity_iteration] += np.sum(env.capacity_merchant, axis = 0)/6
                capacity_2040_merchant[env.n_tech:env.n_tech + env.n_tech_battery, capacity_iteration] += np.sum(env.capacity_merchant_battery, axis = 0)/6
                # Capacity cm

                capacity_2040_cm[:env.n_tech, capacity_iteration] += np.sum(env.capacity_cm, axis = 0)/6
                capacity_2040_cm[env.n_tech:env.n_tech + env.n_tech_battery, capacity_iteration] += np.sum(env.capacity_cm_battery, axis = 0)/6

                # Capacity CfD

                capacity_2040_CfD[:env.n_tech, capacity_iteration] += np.sum(env.capacity_CfD, axis = 0)/6

                # Capacity Flexibility

                capacity_2040_flexibility[env.n_tech:env.n_tech + env.n_tech_battery, capacity_iteration] += np.sum(env.capacity_flexibility_battery, axis = 0)/6

                # Capacity decom

                capacity_2040_decom[:env.n_tech, capacity_iteration] += np.sum(env.capacity_decom, axis = 0)/6

            for tech in range(env.n_tech):
                reward_tech_merchant[i, tech] += (np.sum(env.reward_tech_merchant_step[:,tech] - env.taxes_tech_merchant_step[:,tech])/iters_outer_loop)
                reward_tech_cm[i,tech] += (np.sum(env.reward_tech_cm_step[:,tech] - env.taxes_tech_cm_step[:,tech])/iters_outer_loop)
                reward_tech_existing[i, tech] += (np.sum(env.reward_tech_existing_step[:,tech] - env.taxes_tech_existing_step[:,tech])/iters_outer_loop)
                reward_tech_CfD[i,tech] += (np.sum(env.reward_tech_CfD_step[:,tech] - env.taxes_tech_CfD_step[:,tech])/iters_outer_loop)
                
            for tech_battery in range(env.n_tech_battery):

                reward_tech_merchant[i, tech + tech_battery] += (np.sum(env.reward_tech_merchant_battery_step[:,tech_battery] - env.taxes_tech_merchant_battery_step[:,tech_battery])/iters_outer_loop)
                reward_tech_cm[i,tech + tech_battery] += (np.sum(env.reward_tech_cm_battery_step[:,tech_battery] - env.taxes_tech_cm_battery_step[:,tech_battery])/iters_outer_loop)
                reward_tech_flexibility[i, tech_battery] += (np.sum(env.reward_tech_flexibility_battery_step[:,tech_battery] - env.taxes_tech_flexibility_battery_step[:,tech_battery])/iters_outer_loop)

            CO2_emissions_temp[i] = env.CO2_emissions_step

            total_production[i,:] += (env.total_production_tech)/iters_outer_loop  

            capacity_factor[i,:] += (env.total_production_tech/(np.sum(env.inv_g, axis = 0) * env.hour_month * env.short_t + delta))/iters_outer_loop  


            SoC_state[i,:] += (env.SoC_merchant_storage_lt)/(env.SoC_max_merchant_storage_lt + delta)/iters_outer_loop

            SoC_system[i, capacity_iteration] = np.sum(env.SoC_merchant_storage_lt)/(np.sum(env.SoC_max_merchant_storage_lt) + delta)

            strategy_3_hours[i,:] += (env.strategy_discharge_3_hours - env.strategy_charge_3_hours)/iters_outer_loop

            strategy_4_hours[i,:] += (env.strategy_discharge_4_hours - env.strategy_charge_4_hours)/iters_outer_loop

            strategy_8_hours[i,:] += (env.strategy_discharge_8_hours - env.strategy_charge_8_hours)/iters_outer_loop

            cm_scarcity[i,capacity_iteration] = env.cm_scarcity

            CfD_scarcity[i,capacity_iteration] = env.CfD_scarcity

            cm_income_existing[i,capacity_iteration] = np.sum(env.cm_income_existing)  * env.normalization_factor/(env.short_t * env.hour_month)

            flexibility_scarcity[i, capacity_iteration] = env.flexibility_scarcity

            total_costs[i, capacity_iteration] += (env.cost_total_step)

            cost_spot_other_markets[i, 0] += (env.cost_total_step)/iters_outer_loop
            cost_spot_other_markets_0[i, capacity_iteration] = env.cost_total_step

            cost_spot_other_markets[i, 1] += ((env.cost_cm_step))/iters_outer_loop
            cost_spot_other_markets_1[i, capacity_iteration] = env.cost_cm_step

            cost_spot_other_markets[i, 2] += ((env.cost_CfD_step))/iters_outer_loop
            cost_spot_other_markets_2[i, capacity_iteration] = env.cost_CfD_step

            cost_spot_other_markets[i, 3] += ((env.cost_Flexibility_step))/iters_outer_loop
            cost_spot_other_markets_3[i, capacity_iteration] = env.cost_Flexibility_step

            cost_spot_other_markets[i, 4] += ((env.cost_scarcity_step))/iters_outer_loop
            cost_spot_other_markets_4[i, capacity_iteration] = env.cost_scarcity_step

            cost_spot_other_markets[i, 5] += ((env.cost_carbon_tax_return_step))/iters_outer_loop
            cost_spot_other_markets_5[i, capacity_iteration] = env.cost_carbon_tax_return_step

            cost_spot_other_markets[i, 6] += ((env.cost_merchant_step))/iters_outer_loop
            cost_spot_other_markets_6[i, capacity_iteration] = env.cost_merchant_step

            cost_spot_other_markets[i, 7] += ((env.cost_existing_step))/iters_outer_loop
            cost_spot_other_markets_7[i, capacity_iteration] = env.cost_existing_step

            RES_curtailed[i,capacity_iteration] = env.RES_curtailed_step

            shock_multiplier[i,capacity_iteration] = env.shock.multiplier

            energy_not_served[i,capacity_iteration] = env.energy_not_served_step

            average_demand[i, 0] += env.demand_trimester_average/iters_outer_loop

            i += 1

        rewards_planner[0,capacity_iteration] = env.reward_planner_system_cost/(10**6)

        rewards_planner[1,capacity_iteration] = env.reward_planner_adequacy/(10**6)

        rewards_planner[2,capacity_iteration] = env.reward_planner_emissions/(10**6)

        rewards_planner[3,capacity_iteration] = env.reward_planner_depreciation/(10**6)

        rewards_planner[4,capacity_iteration] = env.reward_planner_flexibility/(10**6)

        rewards_planner[5,capacity_iteration] = env.reward_planner_taxes/(10**6)

        rewards_planner[6,capacity_iteration] = env.reward_planner_rent/(10**6)

        rewards_planner[7,capacity_iteration] = env.reward_planner_emissions_not_discounted/(10**6)

        ## Data from iterations 

        weighted_prices[:, capacity_iteration] = price

        ## Concatenating hourly prices net

        hours_per_year = 24 * 6 

        prices = np.concatenate(short_term_prices_net_tmp)

        prices_reshaped = prices.reshape(-1, hours_per_year)

        yearly_means = prices_reshaped.mean(axis=1)

        prices_normalized = prices_reshaped / yearly_means[:, np.newaxis]

        prices_normalized_flat = prices_normalized.flatten()

        weighted_short_term_prices_net[:, capacity_iteration] = prices_normalized_flat

         ## Concatenating hourly prices

        hours_per_year = 24 * 6 

        prices = np.concatenate(short_term_prices_tmp)

        prices_reshaped = prices.reshape(-1, hours_per_year)

        yearly_means = prices_reshaped.mean(axis=1)

        prices_normalized = prices_reshaped / yearly_means[:, np.newaxis]

        prices_normalized_flat = prices_normalized.flatten()

        weighted_short_term_prices[:, capacity_iteration] = prices_normalized_flat

        ##

        weighted_prices_reliability[:, capacity_iteration] = hourly_prices_reliablity

        weighted_prices_mid_day[:, capacity_iteration] = prices_mid_day

        weighted_prices_afternoon[:, capacity_iteration] = prices_afternoon

        cm_premium_price[:, capacity_iteration] = cm_price

        cm_balance_total[:, capacity_iteration] = cm_balance

        CfD_premium_price[:, capacity_iteration] = CfD_price

        CfD_balance_total[:, capacity_iteration] = CfD_balance

        flexibility_premium_price[:, capacity_iteration] = flexibility_price

        flexibility_balance_total[:, capacity_iteration] = flexibility_balance

        capacity_2030[:, capacity_iteration] = capacity_2030_temp

        capacity_2035[:, capacity_iteration] = capacity_2035_temp

        capacity_2040[:, capacity_iteration] = capacity_2040_temp

        capacity_2030_agent[:, capacity_iteration] = capacity_2030_agent_temp

        capacity_2035_agent[:, capacity_iteration] = capacity_2035_agent_temp

        capacity_2040_agent[:, capacity_iteration] = capacity_2040_agent_temp

        capacity_2035_agent_existing[:, capacity_iteration] = capacity_2035_agent_existing_temp

        capacity_2040_agent_existing[:, capacity_iteration] = capacity_2040_agent_existing_temp

        CO2_emissions[:, capacity_iteration] = CO2_emissions_temp

        penalty_term_tech[:n_tech + 1,capacity_iteration] = np.sum(env.penalty_term, axis = 0)

        penalty_term_tech[n_tech + 1,capacity_iteration] = np.sum(env.penalty_term)

        reward_tech[:n_tech, capacity_iteration] = np.sum(env.reward_tech_merchant, axis = 0) + np.sum(env.reward_tech_cm, axis = 0) + np.sum(env.reward_tech_CfD, axis = 0) 

        reward_tech[n_tech:n_tech + n_tech_battery, capacity_iteration] =  np.sum(env.reward_tech_merchant_battery, axis = 0) + np.sum(env.reward_tech_cm_battery, axis = 0) + np.sum(env.reward_tech_flexibility_battery, axis = 0)

        penalty_term_agent[:,capacity_iteration] = np.sum(env.penalty_term, axis = 1)

        reward_penalty_agent[:,capacity_iteration] = np.sum(env.reward_tech_existing, axis = 1) + np.sum(env.reward_tech_merchant, axis = 1) + np.sum(env.reward_tech_cm, axis = 1) + np.sum(env.reward_tech_CfD, axis = 1) + np.sum(env.reward_tech_merchant_battery + env.reward_tech_cm_battery + env.reward_tech_flexibility_battery, axis = 1) + env.reward_tech_existing_storage_lt - env.taxes_agents

        capacity_2030_agent_t1[:, capacity_iteration] = capacity_2030_agent_temp_tech[:, 0]

        capacity_2035_agent_t1[:, capacity_iteration] = capacity_2035_agent_temp_tech[:, 0]

        capacity_2040_agent_t1[:, capacity_iteration] = capacity_2040_agent_temp_tech[:, 0]

        capacity_2030_agent_t2[:, capacity_iteration] = capacity_2030_agent_temp_tech[:, 1]

        capacity_2035_agent_t2[:, capacity_iteration] = capacity_2035_agent_temp_tech[:, 1]

        capacity_2040_agent_t2[:, capacity_iteration] = capacity_2040_agent_temp_tech[:, 1]

        capacity_2030_agent_t3[:, capacity_iteration] = capacity_2030_agent_temp_tech[:, 2]

        capacity_2035_agent_t3[:, capacity_iteration] = capacity_2035_agent_temp_tech[:, 2]

        capacity_2040_agent_t3[:, capacity_iteration] = capacity_2040_agent_temp_tech[:, 2]

        capacity_2030_agent_t4[:, capacity_iteration] = capacity_2030_agent_temp_tech[:, 3]

        capacity_2035_agent_t4[:, capacity_iteration] = capacity_2035_agent_temp_tech[:, 3]

        capacity_2040_agent_t4[:, capacity_iteration] = capacity_2040_agent_temp_tech[:, 3]

        capacity_2030_agent_t5[:, capacity_iteration] = capacity_2030_agent_temp_tech[:, 4]

        capacity_2035_agent_t5[:, capacity_iteration] = capacity_2035_agent_temp_tech[:, 4]

        capacity_2040_agent_t5[:, capacity_iteration] = capacity_2040_agent_temp_tech[:, 4]

        capacity_2030_agent_t6[:, capacity_iteration] = capacity_2030_agent_temp_tech[:, 5]

        capacity_2035_agent_t6[:, capacity_iteration] = capacity_2035_agent_temp_tech[:, 5]

        capacity_2040_agent_t6[:, capacity_iteration] = capacity_2040_agent_temp_tech[:, 5]

        capacity_2030_agent_t7[:, capacity_iteration] = capacity_2030_agent_temp_tech[:, 6]

        capacity_2035_agent_t7[:, capacity_iteration] = capacity_2035_agent_temp_tech[:, 6]

        capacity_2040_agent_t7[:, capacity_iteration] = capacity_2040_agent_temp_tech[:, 6]

        capacity_2030_agent_t8[:, capacity_iteration] = capacity_2030_agent_temp_tech[:, 7]

        capacity_2035_agent_t8[:, capacity_iteration] = capacity_2035_agent_temp_tech[:, 7]

        capacity_2040_agent_t8[:, capacity_iteration] = capacity_2040_agent_temp_tech[:, 7]

        ## costs of markets

        cost_markets[0, capacity_iteration] = env.cost_total/env.normalization_factor_planner
        cost_markets[1, capacity_iteration] = env.cost_cm/env.normalization_factor_planner
        cost_markets[2, capacity_iteration] = env.cost_CfD/env.normalization_factor_planner
        cost_markets[3, capacity_iteration] = env.cost_Flexibility/env.normalization_factor_planner


    # ── IRR calculations ──────────────────────────────────────────────────────
    IRR_tech_merchant = np.zeros([env.n_tech + 1])
    IRR_tech_cm = np.zeros([env.n_tech + 1])
    IRR_tech_CfD = np.zeros([env.n_tech + 1])
    IRR_tech_flexibility = np.zeros([env.n_tech + 1])

    for tech in range(env.n_tech + 1):

        IRR_tech_merchant[tech] = (1 + npf.irr(reward_tech_merchant[:,tech])) ** 6 - 1
        IRR_tech_merchant[tech] = IRR_tech_merchant[tech] if np.isfinite(IRR_tech_merchant[tech]) else 100
        IRR_tech_merchant[tech] = IRR_tech_merchant[tech] if np.sum(capacity_merchant[:,tech]) > 0 else 200

        IRR_tech_cm[tech] = (1 + npf.irr(reward_tech_cm[:,tech])) ** 6 - 1
        IRR_tech_cm[tech] = IRR_tech_cm[tech] if np.isfinite(IRR_tech_cm[tech]) else 100
        IRR_tech_cm[tech] = IRR_tech_cm[tech] if np.sum(capacity_cm[:,tech]) > 0 else 200

        IRR_tech_CfD[tech] = (1 + npf.irr(reward_tech_CfD[:,tech])) ** 6 - 1
        IRR_tech_CfD[tech] = IRR_tech_CfD[tech] if np.isfinite(IRR_tech_CfD[tech]) else 100

        capacity_CfD_temp = np.sum(capacity_CfD[:,tech]) if tech < env.n_tech else 0
        IRR_tech_CfD[tech] = IRR_tech_CfD[tech] if capacity_CfD_temp > 0 else 200

        IRR_tech_flexibility[tech] = (1 + npf.irr(reward_tech_flexibility[:,tech])) ** 6 - 1
        IRR_tech_flexibility[tech] = IRR_tech_flexibility[tech] if np.isfinite(IRR_tech_flexibility[tech]) else 100

    IRR_agents = np.zeros([agent_g])
    for agents_c, agent_id in enumerate(agent_ids_g):
        IRR_agents[agents_c] = (1 + npf.irr(reward_agents[:,agents_c])) ** 6 - 1

    # ── Save all outputs ──────────────────────────────────────────────────────
    # Check if the directory exists, if not, create it
    if not os.path.exists(f'Folder_58/{string}'):
        os.makedirs(f'Folder_58/{string}')

    outputs = {
        'energy_not_served.csv': energy_not_served,
        'shock_multiplier.csv': shock_multiplier,
        'RES_curtailed.csv': RES_curtailed,
        'cost_spot_other_markets.csv': cost_spot_other_markets,
        'cost_spot_other_markets_0.csv': cost_spot_other_markets_0,
        'cost_spot_other_markets_1.csv': cost_spot_other_markets_1,
        'cost_spot_other_markets_2.csv': cost_spot_other_markets_2,
        'cost_spot_other_markets_3.csv': cost_spot_other_markets_3,
        'cost_spot_other_markets_4.csv': cost_spot_other_markets_4,
        'cost_spot_other_markets_5.csv': cost_spot_other_markets_5,
        'cost_spot_other_markets_6.csv': cost_spot_other_markets_6,
        'cost_spot_other_markets_7.csv': cost_spot_other_markets_7,
        'cost_markets.csv': cost_markets,
        'SoC_system.csv': SoC_system,
        'SoC_agent.csv': SoC_state,
        'total_cost.csv': total_costs,
        'prices.csv': weighted_prices,
        'prices_short_term.csv': weighted_short_term_prices,
        'prices_short_term_net.csv': weighted_short_term_prices_net,
        'prices_hour_2025.csv': prices_hour_2025,
        'prices_hour_2030.csv': prices_hour_2030,
        'average_demand.csv': average_demand,
        'prices_hour_2035.csv': prices_hour_2035,
        'prices_hour_2040.csv': prices_hour_2040,
        'prices_hour_2025_net.csv': prices_hour_2025_net,
        'prices_hour_2030_net.csv': prices_hour_2030_net,
        'prices_hour_2035_net.csv': prices_hour_2035_net,
        'prices_hour_2040_net.csv': prices_hour_2040_net,
        'prices_hour.csv': price_hour,
        'prices_hour_flexibility.csv': price_hour_flexibility,
        'SoC_3_hours.csv': SoC_3_hours,
        'carbon_tax_return.csv': carbon_tax_return,
        'SoC_8_hours.csv': SoC_8_hours,
        'discharge_3_hours.csv': discharge_3_hours,
        'charge_3_hours.csv': charge_3_hours,
        'cm_premium_price.csv': cm_premium_price,
        'cm_balance_total.csv': cm_balance_total,
        'CfD_premium_price.csv': CfD_premium_price,
        'flexibility_premium_price.csv': flexibility_premium_price,
        'CfD_balance_total.csv': CfD_balance_total,
        'flexibility_balance_total.csv': flexibility_balance_total,
        'capacity_2030.csv': capacity_2030,
        'capacity_2030_merchant.csv': capacity_2030_merchant,
        'capacity_2030_cm.csv': capacity_2030_cm,
        'capacity_2030_CfD.csv': capacity_2030_CfD,
        'capacity_2030_flexibility.csv': capacity_2030_flexibility,
        'capacity_2030_decom.csv': capacity_2030_decom,
        'capacity_2035.csv': capacity_2035,
        'capacity_2035_merchant.csv': capacity_2035_merchant,
        'capacity_2035_cm.csv': capacity_2035_cm,
        'capacity_2035_CfD.csv': capacity_2035_CfD,
        'capacity_2035_flexibility.csv': capacity_2035_flexibility,
        'capacity_2035_decom.csv': capacity_2035_decom,
        'capacity_2030_agent.csv': capacity_2030_agent,
        'capacity_2035_agent.csv': capacity_2035_agent,
        'capacity_2035_agent_existing.csv': capacity_2035_agent_existing,
        'capacity_2040.csv': capacity_2040,
        'capacity_2040_merchant.csv': capacity_2040_merchant,
        'capacity_2040_cm.csv': capacity_2040_cm,
        'capacity_2040_CfD.csv': capacity_2040_CfD,
        'capacity_2040_flexibility.csv': capacity_2040_flexibility,
        'capacity_2040_decom.csv': capacity_2040_decom,
        'capacity_2040_agent.csv': capacity_2040_agent,
        'capacity_2040_agent_existing.csv': capacity_2040_agent_existing,
        'capacity_merchant.csv': capacity_merchant,
        'capacity_merchant_uc.csv': capacity_merchant_uc,
        'capacity_decom.csv': capacity_decom,
        'inv_g_aging.csv': inv_g_aging,
        'capacity_cm.csv': capacity_cm,
        'capacity_cm_battery.csv': capacity_cm_battery,
        'cm_cc_tech.csv': cc_cm_tech,
        'capacity_CfD.csv': capacity_CfD,
        'capacity_flexibility.csv': capacity_flexibility,
        'CO2_emissions.csv': CO2_emissions,
        'EoM_E_TIC.csv': total_installed_capacity,
        'Reward_cm.csv': reward_tech_cm,
        'Reward_CfD.csv': reward_tech_CfD,
        'Reward_flexibility.csv': reward_tech_flexibility,
        'Reward_merchant.csv': reward_tech_merchant,
        'Reward_existing.csv': reward_tech_existing,
        'Demand_Growth.csv': demand_growth,
        'Penalty_term.csv': penalty_term_tech,
        'Penalty_term_agent.csv': penalty_term_agent,
        'Reward_Penalty_agent.csv': reward_penalty_agent,
        'Total_production_tech.csv': total_production,
        'Capacity_factor_tech.csv': capacity_factor,
        'cm_scarcity.csv': cm_scarcity,
        'CfD_scarcity.csv': CfD_scarcity,
        'cm_income_existing.csv': cm_income_existing,
        'flexibility_scarcity.csv': flexibility_scarcity,
        'capacity_2030_agent_t1.csv': capacity_2030_agent_t1,
        'capacity_2035_agent_t1.csv': capacity_2035_agent_t1,
        'capacity_2040_agent_t1.csv': capacity_2040_agent_t1,
        'capacity_2030_agent_t2.csv': capacity_2030_agent_t2,
        'capacity_2035_agent_t2.csv': capacity_2035_agent_t2,
        'capacity_2040_agent_t2.csv': capacity_2040_agent_t2,
        'capacity_2030_agent_t3.csv': capacity_2030_agent_t3,
        'capacity_2035_agent_t3.csv': capacity_2035_agent_t3,
        'capacity_2040_agent_t3.csv': capacity_2040_agent_t3,
        'capacity_2030_agent_t4.csv': capacity_2030_agent_t4,
        'capacity_2035_agent_t4.csv': capacity_2035_agent_t4,
        'capacity_2040_agent_t4.csv': capacity_2040_agent_t4,
        'capacity_2030_agent_t5.csv': capacity_2030_agent_t5,
        'capacity_2035_agent_t5.csv': capacity_2035_agent_t5,
        'capacity_2040_agent_t5.csv': capacity_2040_agent_t5,
        'capacity_2030_agent_t6.csv': capacity_2030_agent_t6,
        'capacity_2035_agent_t6.csv': capacity_2035_agent_t6,
        'capacity_2040_agent_t6.csv': capacity_2040_agent_t6,
        'capacity_2030_agent_t7.csv': capacity_2030_agent_t7,
        'capacity_2035_agent_t7.csv': capacity_2035_agent_t7,
        'capacity_2040_agent_t7.csv': capacity_2040_agent_t7,
        'capacity_2030_agent_t8.csv': capacity_2030_agent_t8,
        'capacity_2035_agent_t8.csv': capacity_2035_agent_t8,
        'capacity_2040_agent_t8.csv': capacity_2040_agent_t8,
        'short_term_scarcity.csv': short_term_scarcity,
        'CO2_tax.csv': CO2_tax,
        'CfD_target.csv': CfD_target,
        'CfD_target_action.csv': CfD_target_action,
        'CfD_price_cap.csv': CfD_price_cap,
        'cm_target.csv': cm_target,
        'cm_price_cap.csv': cm_price_cap,
        'cm_strike.csv': cm_strike,
        'cm_income_total.csv': cm_income_total,
        'res_share.csv': res_share,
        'fossil_fuel_capacity_factor.csv': fossil_fuel_capacity_factor,
        'storage_share.csv': storage_share,
        'flexibility_target.csv': flexibility_target,
        'flexibility_target_action.csv': flexibility_target_action,
        'corporate_tax_rate.csv': corporate_tax_rate,
        'rewards_planner.csv': rewards_planner,
        'IRR_cm.csv': IRR_tech_cm,
        'IRR_CfD.csv': IRR_tech_CfD,
        'IRR_merchant.csv': IRR_tech_merchant,
        'IRR_tech_flexibility.csv': IRR_tech_flexibility,
        'IRR_agents.csv': IRR_agents,
        'Policy_scenarios.csv': policy_scenarios_selected,
        'entry_enabled.csv': entry_enabled,
        'reward_tech.csv': reward_tech,
    }

    for filename, array in outputs.items():
        pd.DataFrame(array).to_csv(f'Folder_58/{string}/{filename}')

    last_time = time.time()
