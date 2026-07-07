import numpy as np
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gymnasium.spaces import Dict, Box, MultiDiscrete
from collections import OrderedDict
import copy
from linopy import Model
from dispatch_linopy_V19 import DispatchHighsRL
from elcc_calculator_V9 import ELCCCalculator
from gas_price_module import GasPriceShockModel

"""
Energy only Market with n_g GENCOs, and n_u Utilities. 

"""

class CM_EoM(MultiAgentEnv):
	def __init__(self, max_t, short_t, VoLL_max,
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
			  scenario_tax_decree, shock_flag, scenario_tax_decree_deactivation_random_flag):

		self.shock_flag = shock_flag

		## 0 - do nothing
		## 1 - decree
		## 2 - no incentive to decarbonize
		## 3 - decree and no incentive to decarbonize

		self.scenario_tax_decree = scenario_tax_decree
		self.scenario_tax_decree_deactivation_random_flag = scenario_tax_decree_deactivation_random_flag

		# Final slack for termination conditions

		self.years_slack_termination = years_slack_termination
		self.years_slack_profits = years_slack_profits

		# Prueba - CM targets

		self.cm_reliability_target_values = cm_reliability_target_values

		# Prueba - CfD targets

		self.CfD_target_increments_steps = CfD_target_increments_steps

		# Prueba - Flexibility targets

		self.flexibility_target_increments_steps = flexibility_target_increments_steps

		# Terminal reward flag

		self.terminal_reward_flag = terminal_reward_flag

		# Activation years

		self.CfD_activation_year = CfD_activation_year
		self.cm_activation_year = cm_activation_year
		self.flexibility_activation_year = flexibility_activation_year

		# Maximum targets in markets

		self.cm_maximum_target = cm_maximum_target
		self.CfD_maximum_target = CfD_maximum_target
		self.flexibility_maximum_target = flexibility_maximum_target

		# Lobby key

		self.lobby = lobby
		self.agent_g_lobby = agent_g_lobby

		# GENCO public key

		self.public_generators = public_generators
		self.agent_g_public = agent_g_public
		
		# Type of action regulator

		self.type_action_regulator = type_action_regulator
		
		## entrant vector

		self.entrants = entrants

		## Policy scenarios

		self.policy_scenarios = policy_scenarios
		self.policy_deterministic_activation = policy_deterministic_activation
		self.policy_deterministic = policy_deterministic
		self.carbon_deterministic_activation = carbon_deterministic_activation
		self.carbon_deterministic = carbon_deterministic
		self.demand_deterministic_activation = demand_deterministic_activation
		self.demand_deterministic = demand_deterministic

		## Taxes

		self.corporate_tax_rate_growth_max = corporate_tax_rate_growth_max
		self.corporate_tax_rate = self.corporate_tax_rate_growth_max

		## Market activation signals (for both agents and planner)

		self.coporate_tax_activation = coporate_tax_activation
		self.cm_activation = cm_activation
		self.CfD_activation = CfD_activation
		self.flexibility_activation = flexibility_activation

		# Average failures

		self.average_failures = average_failures

		# Investments enabled

		self.investments_enabled = investments_enabled

		# Short term time

		self.short_t = short_t

		self.yearly_resolution = 6
		
		# Battery

		self.inv_init_battery = inv_init_battery

		self.inv_max_battery = inv_max_battery

		self.inv_cost_battery = inv_cost_battery

		self.fixed_cost_battery = fixed_cost_battery
		self.fixed_cost_storage_lt = fixed_cost_storage_lt

		self.construction_time_battery = construction_time_battery 

		self.life_time_battery = life_time_battery

		self.strategy_charge_3_hours = np.zeros([self.short_t])
		self.strategy_discharge_3_hours = np.zeros([self.short_t])

		self.strategy_charge_4_hours = np.zeros([self.short_t])
		self.strategy_discharge_4_hours = np.zeros([self.short_t])

		self.strategy_charge_8_hours = np.zeros([self.short_t])
		self.strategy_discharge_8_hours = np.zeros([self.short_t])

		self.strategy_charge_cm = np.zeros([self.short_t])
		self.strategy_discharge_cm = np.zeros([self.short_t])

		self.inv_init_battery = inv_init_battery
		
		# Termination and truncation
		
		self.terminateds = set()
		self.truncateds = set()

		# Penalty and Demand Growth type 

		self.penalty_type = penalty_type
		self.penalty_factor = penalty_factor
		self.demand_growth_type = demand_growth_type

		## Time conditions

		# Init time

		self.max_t_o = max_t
		self.t_init = 0

		self.planning_horizon = 4

		# Number of agents

		self.agents_g = agent_g
		self.agents_n = self.agents_g 
		
		# Number of technologies (and the quantity and price actions for storage)

		self.n_tech = n_tech

		# Number of storage technologies

		self.n_tech_battery = n_tech_battery

		# Number of decom technologies

		self.n_tech_decom = n_tech_decom
		self.n_tech_decom_offset = 4

		# Initial time and maximum horizon

		self.time = 0

		self.time_random = 0 
		
		self.max_t = self.max_t_o

		self.count_last = np.zeros([self.n_tech])

		self.count_last_battery = np.zeros([self.n_tech_battery])

		# Simulation year and month

		self.year = 0
		self.year_int = 0
		self.month = 0
		self.hour_year = 0
		self.hour_month = 62.4

		# Maximum year

		self.max_year = self.max_t/(self.short_t * self.yearly_resolution)

		# Market informastion

		self.VoLL_max = VoLL_max
		self.VoLL_norm = 500
		self.VoLL_norm_prices = 500
		self.VoLL_planner = self.VoLL_max * 3
		self.price_avg = self.VoLL_max * np.random.random()/50

		self.cm_price = 0
		self.cm_auction_indicator = False
		self.cm_demand_target = 0
		self.CfD_price = 0
		self.CfD_price_agents = np.zeros([self.agents_g, self.n_tech])
		self.CfD_auction_indicator = False
		
		# GENCOs' parameters (Variable costs, initial investment, max investment, and investment costs)

		self.v_c_g = v_c_g
		self.availability_tech = time_series[:,:9]
		self.availability_tech_average_bimester = average_bimester_series[:,:9]
		self.availability_tech_average_yearly = average_yearly_series[:,:9]

		# Utilities' parameters (maximum demand and inflexible demand)

		self.demand_growth = demand_growth

		# Adjusting demand to the starting year
		self.demand_scenario = time_series[:,10:]
		self.demand_average_bimester_scenario = average_bimester_series[:,10:]
		self.demand_average_year_scenario = average_yearly_series[:,10:]

		self.max_demand = np.max(self.demand_scenario)
		self.percentile_demand = np.percentile(self.demand_scenario,90) 

		self.demand = self.demand_scenario[:,0]
		self.demand_average_bimester = self.demand_average_bimester_scenario [:,0]
		self.demand_average_year = self.demand_average_year_scenario[:,0]
		
		# Discretization steps for MultiDiscrete action spaces (step_g_p should be 11 for strategic bidding)

		self.step_g_bids = step_g_bids
		self.step_g_inv = step_g_inv

		self.step_SoC_control = step_SoC_control

		self.step_planner = step_planner

		# Planner number

		self.agents_p = 1

		# Agent ids

		self.possible_agents_g = [f"Agent_g_{i}" for i in range (self.agents_g)]

		# Agent planner

		self.possible_agents_p = [f"Agent_p_{i}" for i in range (self.agents_p)]

		# Agents

		self.agents = self.possible_agents_g + self.possible_agents_p
		self.possible_agents = self.agents
		self._agent_ids = set(self.possible_agents)

		# Setting investment to initial investment 

		self.inv_init_g = inv_init_g
		self.inv_g = copy.deepcopy(self.inv_init_g)
		self.inv_max = inv_max
		self.inv_new = np.zeros([self.agents_g, self.n_tech])
		self.inv_age = np.zeros([self.agents_g, self.n_tech])
		self.inv_age_merchant = np.zeros([self.agents_g, self.n_tech])
		self.capacity_merchant = np.zeros([self.agents_g, self.n_tech]) 
		self.inv_g_aging = copy.deepcopy(self.inv_init_g)

		self.inv_age_cm = np.zeros([self.agents_g, self.n_tech])
		self.inv_age_CfD = np.zeros([self.agents_g, self.n_tech])
		self.capacity_cm = np.zeros([self.agents_g, self.n_tech])
		self.capacity_merchant_uc = np.zeros([self.agents_g, self.n_tech])
		self.capacity_cm_uc = np.zeros([self.agents_g, self.n_tech])
		
		self.inv_cost_accum_merchant = np.zeros([self.agents_g, self.n_tech])
		self.inv_cost_accum = np.zeros([self.agents_g, self.n_tech])
		self.inv_cost_accum_cm = np.zeros([self.agents_g, self.n_tech])
		self.inv_cost_accum_CfD = np.zeros([self.agents_g, self.n_tech])

		# Battery investment variables 

		self.capacity_merchant_battery = np.zeros([self.agents_g, self.n_tech_battery]) 
		self.capacity_merchant_battery_uc = np.zeros([self.agents_g, self.n_tech_battery])  

		self.inv_init_storage_lt = inv_init_storage_lt

		self.capacity_existing_storage_lt = self.inv_init_storage_lt
		self.capacity_existing_storage_lt_uc = np.zeros([self.agents_g]) 
		self.SoC_max_merchant_storage_lt = SoC_max_merchant_storage_lt

		self.SoC_merchant_storage_lt = self.SoC_max_merchant_storage_lt * 0.5

		self.inv_age_merchant_battery = np.zeros([self.agents_g, self.n_tech_battery])
		self.inv_cost_accum_merchant_battery = np.zeros([self.agents_g, self.n_tech_battery])

		self.capacity_cm_battery = np.zeros([self.agents_g, self.n_tech_battery]) 
		self.capacity_cm_battery_uc = np.zeros([self.agents_g, self.n_tech_battery]) 
		self.inv_age_cm_battery = np.zeros([self.agents_g, self.n_tech_battery])
		self.inv_cost_accum_cm_battery = np.zeros([self.agents_g, self.n_tech_battery])

		self.capacity_flexibility_battery = np.zeros([self.agents_g, self.n_tech_battery]) 
		self.capacity_flexibility_battery_uc = np.zeros([self.agents_g, self.n_tech_battery]) 
		self.inv_age_flexibility_battery = np.zeros([self.agents_g, self.n_tech_battery])
		self.inv_cost_accum_flexibility_battery = np.zeros([self.agents_g, self.n_tech_battery])

		# Depreciation and opportunity cost

		self.opportunity_cost = opportunity_cost
		self.opportunity_cost_merchant = opportunity_cost
		self.opportunity_cost_cm = opportunity_cost
		self.opportunity_cost_CfD = opportunity_cost
		self.opportunity_cost_flexibility = opportunity_cost

		# Opportunity cost planner

		self.opportunity_cost_planner = opportunity_cost_planner

		# Capacity valuation, capacity credtis for market and firm capactiy

		self.cm_tech_cc = np.ones([self.n_tech]) * 0.9
		self.cm_fc_under_construction = np.zeros([self.n_tech])
		self.cm_fc_under_construction_battery = 0

		# Income from Capacity Market

		self.cm_income = np.zeros([self.agents_g, self.n_tech])

		# Capacity with options coming from the capacity market

		self.cm_inv_options = np.zeros([self.agents_g, self.n_tech])

		self.cm_price_cap_max = cm_price_cap_max
		self.cm_price_growth_rate = cm_price_growth_rate
		self.cm_price_cap_original = cm_price_cap_original
		self.cm_price_cap_min = cm_price_cap_min
		self.cm_price_cap = cm_price_cap_original

		self.cm_strike_max = cm_strike_max
		self.cm_strike_growth_rate = cm_strike_growth_rate
		self.cm_strike_original = cm_strike_original
		self.cm_strike_min = cm_strike_min
		self.cm_strike = cm_strike_original

		self.cm_target_growth_rate_max = cm_target_growth_rate_max

		## TODO initial price for the cm market
		self.cm_price_existing = self.cm_price_cap/(self.step_g_bids)

		# CfD auctions

		self.CfD_price_cap_max = CfD_price_cap_max
		self.CfD_price_growth_rate = CfD_price_growth_rate
		self.CfD_price_cap_original = CfD_price_cap_original
		self.CfD_price_cap_min = CfD_price_cap_min
		self.CfD_price_cap = CfD_price_cap_original

		self.CfD_target_original = CfD_target
		self.CfD_target = CfD_target
		self.CfD_growth_rate_max = CfD_growth_rate_max
		
		# Flexibility market

		self.flexibility_target_original = flexibility_target
		self.flexibility_target = flexibility_target
		self.flexibility_target_growth_rate_max = flexibility_target_growth_rate_max
		self.flexibility_price_cap = flexibility_price_cap
		self.flexibility_price_cap_original = flexibility_price_cap

		## Reward per technology

		self.reward_tech_merchant = np.zeros([self.agents_g, self.n_tech])
		self.reward_tech_cm = np.zeros([self.agents_g, self.n_tech])
		self.reward_tech_CfD = np.zeros([self.agents_g, self.n_tech])
		self.reward_tech_existing = np.zeros([self.agents_g, self.n_tech])

		## Resetting accumulated profit

		self.accumulated_profit_cm = np.zeros([self.n_tech])
		self.accumulated_profit_merchant = np.zeros([self.n_tech])
		self.accumulated_profit_CfD= np.zeros([self.n_tech])

		# Reward and accumulated profit - Battery

		self.reward_tech_merchant_battery = np.zeros([self.agents_g, self.n_tech_battery])
		self.reward_tech_merchant_battery_step = np.zeros([self.agents_g, self.n_tech_battery])

		self.reward_tech_existing_storage_lt = np.zeros([self.agents_g])
		self.reward_tech_existing_storage_lt_step = np.zeros([self.agents_g])

		self.reward_tech_cm_battery = np.zeros([self.agents_g, self.n_tech_battery])
		self.reward_tech_cm_battery_step = np.zeros([self.agents_g, self.n_tech_battery])

		self.reward_tech_flexibility_battery = np.zeros([self.agents_g, self.n_tech_battery])
		self.reward_tech_flexibility_battery_step = np.zeros([self.agents_g, self.n_tech_battery])

		self.accumulated_profit_merchant_battery = np.zeros([self.n_tech_battery])
		self.accumulated_profit_cm_battery = np.zeros([self.n_tech_battery])
		self.accumulated_profit_flexibility_battery = np.zeros([self.n_tech_battery])

		# Regret penalty - merchant only

		self.penalty_merchant = np.zeros([self.agents_g, self.n_tech])
		self.penalty_merchant_battery = np.zeros([self.agents_g, self.n_tech_battery])

		# Reward planner

		self.reward_planner = 0

		self.reward_planner_adequacy = 0
		self.reward_planner_system_cost = 0
		self.reward_planner_emissions = 0
		self.reward_planner_depreciation = 0
		self.reward_planner_flexibility = 0

		self.reward_planner_adequacy_step = 0
		self.reward_planner_system_cost_step = 0
		self.reward_planner_emissions_step = 0
		self.reward_planner_depreciation_step = 0
		self.reward_planner_flexibility_step = 0

		# Scarcity signals

		self.cm_scarcity = 0
		self.CfD_scarcity = 0
		self.flexibility_scarcity = 0

		# Aging 
  
		self.aging = aging

		# Monthly installments corresponding to investment costs

		self.inv_cost = inv_cost

		# Fixed costs

		self.fixed_cost = fixed_cost

		# Decom costs

		self.decom_cost = decom_cost

		# Decom activation

		self.decom_activation_year = decom_activation_year
		self.decom_activation = decom_activation

		# Construction time for technologies and construction vector

		self.construction_time = construction_time

		self.construction_delays = construction_delays
		self.counter_construction_delays = 0
		self.max_row_construction_delays = 1000
		
		self.failure_flag = failure_flag
		self.max_row_failure_flag = 1000
		self.counter_failure_flag = 0

		# Decom

		self.decom_time = decom_time

		# Project Lifetime

		self.life_time = life_time

		# CO2 tax per technology and CO2 tax

		self.CO2_tax_original = CO2_tax_original
		self.CO2_tech_original = CO2_tech

		self.CO2_tax = CO2_tax_original
		self.CO2_tax_max = CO2_tax_max
		self.CO2_tech = CO2_tech
		self.CO2_growth_rate_max = CO2_growth_rate_max
		self.CO2_emissions_step = 0
		self.CO2_social_cost_original = CO2_social_cost
		self.CO2_social_cost = self.CO2_social_cost_original[:,11]
		self.CO2_tax_min = CO2_tax_min

		self.CO2_tax_scenario = CO2_tax_scenario
		self.CO2_tax_scenario_activation = CO2_tax_scenario_activation

		self.under_construction = {}

		for a in self.possible_agents_g:
			# Notation: tech, capacity, time, merchant/Capacity market, capacity price
			self.under_construction.update({a:np.zeros([1,6])})

		# Random values for bids 

		self.random_g_p = random_g		
		self.random_g_cm = random_g_cm		
		self.random_g_CfD = random_g_CfD
		self.random_g_flexibility = random_g_flexibility
		
		# Number of investments bid plus investment actions
		
		self.act_inv = self.n_tech + self.n_tech_battery

		# Actions for capacity market auctions

		self.act_bids_cm = self.n_tech + self.n_tech_battery
		self.act_inv_cm = self.n_tech + self.n_tech_battery

		# Actions for CfD auctions

		self.n_tech_RES = n_tech_RES

		self.act_bids_CfD = self.n_tech_RES
		self.act_inv_CfD = self.n_tech_RES

		# Actions for flexibility market

		self.act_bids_flexibility = self.n_tech_battery
		self.act_inv_flexibility = self.n_tech_battery

		# Actions for decom of existing OCGT and CCGT

		self.act_decom = self.n_tech_decom

		# Actions for SoC lt

		self.act_SoC = 1

		# Number of total bids

		self.n_act_g = self.act_inv + self.act_bids_cm + self.act_inv_cm + self.act_bids_CfD + self.act_inv_CfD + self.act_bids_flexibility + self.act_inv_flexibility + self.act_decom + self.act_SoC

		self._action_space_in_preferred_format = True

		# Action spaces - GENCOs

		action_matrix_g = np.concatenate((np.ones([self.act_inv])*self.step_g_inv, 
								  np.ones([self.act_bids_cm]) * self.step_g_bids, np.ones([self.act_inv_cm])*self.step_g_inv,
								  np.ones([self.act_bids_CfD]) * self.step_g_bids, np.ones([self.act_inv_CfD])*self.step_g_inv,
								  np.ones([self.act_bids_flexibility]) * self.step_g_bids, np.ones([self.act_inv_flexibility])*self.step_g_inv,
								  np.ones([self.act_decom]) * self.step_g_inv,
								   np.ones([self.act_SoC]) * self.step_SoC_control,
								  ),axis=0)

		self.action_spaces_g = {i: MultiDiscrete(action_matrix_g) for i in self.possible_agents_g}

		# Action spaces - planner

		self.act_p_CfD_market = 1

		self.act_p_flexibility = 1

		self.n_act_p = self.act_p_CfD_market + self.act_p_flexibility

		action_matrix_p = np.concatenate((np.ones([self.act_p_CfD_market]) * self.step_planner,
								  np.ones([self.act_p_flexibility]) * self.step_planner
								  ),axis=0)

		self.action_spaces_p = {i: MultiDiscrete(action_matrix_p) for i in self.possible_agents_p}

		# Unifying dictionaries

		self.action_space = Dict(dict(self.action_spaces_g, **self.action_spaces_p))

		# Observation spaces (General and Specific observations)

		self.n_obs_short_term_prices = 4

		self.n_obs_flexibility_prices = 4

		self.n_obs_resources = 0

		self.n_obs_cap_ind = 8

		self.n_obs_cap_total = 5

		self.n_obs_battery_cap_ind = 5

		self.n_obs_battery_cap_total = 4

		# Time observation includes Long-term storage

		self.n_observation_time = 7		

		self.n_observation_cm_CfD = 21

		self.n_observation_reward_tech = 4

		self.n_observation_reward_tech_CfD = 2

		self.n_observation_reward_tech_battery = 6
		
		self.n_observation_reward_total = 1

		self.n_obs_g = self.n_obs_short_term_prices + self.n_obs_flexibility_prices + self.n_obs_resources + (self.n_tech * self.n_obs_cap_ind) + (self.n_tech * self.n_obs_cap_total) + (self.n_obs_battery_cap_ind * self.n_tech_battery) + (self.n_obs_battery_cap_total * self.n_tech_battery) + self.n_observation_time + self.n_observation_cm_CfD + (self.n_observation_reward_tech * self.n_tech) + (self.n_observation_reward_tech_CfD * self.n_tech_RES) + (self.n_observation_reward_tech_battery * self.n_tech_battery) + self.n_observation_reward_total
		
		self._obs_space_in_preferred_format = True

		# Observation spaces - GENCOs

		self.observation_spaces_g = {
            i: Dict({"observations": Box(low = np.ones(self.n_obs_g) * -1, high = np.ones(self.n_obs_g), dtype=np.float32),
					 "action_mask": Box(0.0, 1.0, shape=(int(self.step_g_inv * self.act_inv + self.step_g_bids * self.act_bids_cm + self.step_g_inv * self.act_inv_cm
											  + self.step_g_bids * self.act_bids_CfD + self.step_g_inv * self.act_inv_CfD
											  + self.step_g_bids * self.act_bids_flexibility + self.step_g_inv * self.act_inv_flexibility
											  + self.step_g_inv * self.act_decom
											  + self.step_SoC_control * self.act_SoC),), 
										 dtype=np.float32),})
            for i in self.possible_agents_g
        }

		# Observation spaces - Planner

		self.n_obs_short_term_prices_p = 4

		self.n_obs_flexibility_prices_p = 4

		self.n_obs_resources_p = 0

		self.n_obs_cap_ind_p = 0

		self.n_obs_cap_total_p = 5

		self.n_obs_battery_cap_ind_p = 0

		self.n_obs_battery_cap_total_p = 4

		self.n_observation_time_p = 7		

		self.n_observation_cm_CfD_p = 21

		self.n_observation_reward_p = 8

		self.n_obs_p = self.n_obs_short_term_prices_p + self.n_obs_flexibility_prices_p + self.n_obs_resources_p + (self.n_tech * self.n_obs_cap_ind_p) + (self.n_tech * self.n_obs_cap_total_p) + (self.n_obs_battery_cap_ind_p * self.n_tech_battery) + (self.n_obs_battery_cap_total_p * self.n_tech_battery) + self.n_observation_time_p + self.n_observation_cm_CfD_p + self.n_observation_reward_p
		
		self.observation_spaces_p = {
            i: Dict({"observations": Box(low = np.ones(self.n_obs_p) * -1, high = np.ones(self.n_obs_p), dtype=np.float32),
					 "action_mask": Box(0.0, 1.0, shape=(int(self.step_planner * self.act_p_CfD_market + self.step_planner * self.act_p_flexibility
											  ),), 
										 dtype=np.float32),})
            for i in self.possible_agents_p
        }

		# Unifying agents

		self.observation_space = Dict(dict(self.observation_spaces_g, **self.observation_spaces_p))

		# Demand growth (yearly rate)

		# Scenarios

		self.scenario_failures = scenario_failures

		self.random_number_failures = random_number_failures

		self.time_random_demand = 0

		self.time_random_availability = 0

		self.time_random_failures = 0

		## Initializaing dispatch model

		i = 0

		self.index_agents_long_term_storage = []

		self.agents_long_term_storage = 0

		for a in self.possible_agents_g:

			if self.SoC_max_merchant_storage_lt[i] > 0.1:

				self.agents_long_term_storage += 1

				self.index_agents_long_term_storage.append(i)

			i += 1

		#  Initializing dispatch

		self.coal_tech_idx = 3

		self.model = DispatchHighsRL(self.agents_long_term_storage, short_t, n_tech + 2, 0.95, 0.95, 0.95, 0.95, self.coal_tech_idx)

		# Initializing capacity estimation

		self.elcc_calculator = ELCCCalculator(self)

		self.year_shock = 6

		self.shock = GasPriceShockModel()

		# Init method

		super().__init__()
		
	def reset(self, *, seed=None, options=None):

		self.shock.reset()

		# Reward

		self.reward_agents_g = np.zeros([self.agents_g])
		self.reward_agents_p = np.zeros([self.agents_p])
		
		# Corporate tax rate

		self.corporate_tax_rate = self.corporate_tax_rate_growth_max
		
		# Mechanism costs
		
		self.cost_total = 0
		self.cost_spot_existing = 0
		self.cost_cm = 0
		self.cost_CfD = 0
		self.cost_Flexibility = 0
		
		self.cost_total_step = 0
		self.cost_merchant_step = 0
		self.cost_carbon_tax_return_step = 0
		self.cost_scarcity_step = 0
		self.cost_spot_existing_step = 0
		self.cost_cm_step = 0
		self.cost_CfD_step = 0
		self.cost_Flexibility_step = 0
		self.cost_existing_step = 0

		self.projects_dead = 0

		## Decree random activation flag (set very high if no deactivation needed)

		if self.scenario_tax_decree_deactivation_random_flag == True:

			self.scenario_tax_decree_deactivation = np.random.choice([4, 8, 12, 100])

		else:
			
			self.scenario_tax_decree_deactivation = 100

		## Entrants enabled

		self.entry_enabled = np.zeros([self.agents_g])
			
		## 

		## Market scenario 

		# Assigning conditions to policy activation masks

		if self.policy_deterministic_activation == True:

			self.random_policy_scenario = self.policy_deterministic

		else:
			
			self.random_policy_scenario = np.random.randint(0, 6)

		policy_tmp = self.policy_scenarios[self.random_policy_scenario, :]

		# Merchant Activation

		self.merchant_activation = True if policy_tmp[0] == 1 else False

		# Corporate tax activation

		self.coporate_tax_activation = True if policy_tmp[1] == 1 else False

		# Flexibility market activation

		self.flexibility_activation = True if policy_tmp[2] == 1 else False

		# CfD activation

		self.CfD_activation = True if policy_tmp[3] == 1 else False

		# Capacity Market activation

		self.cm_activation = True if policy_tmp[4] == 1 else False

		## Carbon Scenario

		if self.carbon_deterministic_activation == True:

			self.random_carbon_scenario = self.carbon_deterministic

		else:
			
			self.random_carbon_scenario = np.random.randint(0, 11)

		self.CO2_social_cost = self.CO2_social_cost_original[:, 11]
			
		self.CO2_tax_scenario = self.CO2_social_cost_original[:, self.random_carbon_scenario]

		## Demand scenario

		if self.demand_deterministic_activation == True:

			self.random_demand_scenario = self.demand_deterministic

		else:
			
			self.random_demand_scenario = np.random.randint(0, 7)  

		self.demand = self.demand_scenario[:,self.random_demand_scenario]
		self.demand_average_bimester = self.demand_average_bimester_scenario [:,self.random_demand_scenario]
		self.demand_average_year = self.demand_average_year_scenario[:,self.random_demand_scenario]

		## 
		
		self.delta_auction = 0
		
		self.supply = 0

		self.demand_CfD = 0

		# Resetting Terminateds and Truncateds
		# super().reset(seed=seed)

		# self.resetted = True

		self.normalization_factor = 4000 * 4000 * self.max_t_o

		self.normalization_factor_planner = 4000 * 4000 * self.max_t_o * 20

		self.CO2_emissions_step = 0

		delta = 0.001 
		
		if self.demand_growth_type == "Stochastic":

			self.demand_growth = np.random.uniform(0.01, 0.03)

		self.terminateds = set()
		self.truncateds = set() 
		
		# Empty info vector

		infos = {a: {} for a in self.agents}

		# Discount Factor

		self.discount_factor = 1/((1 + self.opportunity_cost) ** self.year)
		self.discount_factor_merchant =  1/((1 + (self.opportunity_cost_merchant)) ** self.year)
		self.discount_factor_cm = self.discount_factor
		self.discount_factor_CfD = self.discount_factor
		self.discount_factor_flexibility = self.discount_factor

		# Discount factor planner

		self.discount_factor_planner = 1/((1 + self.opportunity_cost_planner) ** self.year)

		# Init time

		self.t_init = 0

		# Initial time and maximum horizon

		self.time = 0
		
		self.max_t = self.max_t_o

		self.count_last = np.zeros([self.n_tech])

		self.count_last_battery = np.zeros([self.n_tech_battery])

		self.year_int = 0

		# Reseting investments to initial conditions

		self.inv_g = copy.deepcopy(self.inv_init_g)
		self.inv_g_aging = copy.deepcopy(self.inv_init_g)
		
		self.under_construction = {}

		for a in self.possible_agents_g:
			
			self.under_construction.update({a:np.zeros([1,6])})

		self.inv_new = np.zeros([self.agents_g, self.n_tech])
		self.inv_age = np.zeros([self.agents_g, self.n_tech])
		self.inv_age_cm = np.zeros([self.agents_g, self.n_tech])
		self.inv_age_CfD = np.zeros([self.agents_g, self.n_tech])
		self.inv_age_merchant = np.zeros([self.agents_g, self.n_tech])

		self.capacity_merchant = np.zeros([self.agents_g, self.n_tech])
		self.capacity_cm = np.zeros([self.agents_g, self.n_tech])
		self.capacity_CfD = np.zeros([self.agents_g, self.n_tech])
		self.capacity_decom = np.zeros([self.agents_g, self.n_tech])

		self.capacity_merchant_uc = np.zeros([self.agents_g, self.n_tech])
		self.capacity_cm_uc = np.zeros([self.agents_g, self.n_tech])
		self.capacity_CfD_uc = np.zeros([self.agents_g, self.n_tech])
		self.capacity_decom_uc = np.zeros([self.agents_g, self.n_tech])

		self.inv_cost_accum = np.zeros([self.agents_g, self.n_tech])
		self.inv_cost_accum_merchant = np.zeros([self.agents_g, self.n_tech])
		self.inv_cost_accum_cm = np.zeros([self.agents_g, self.n_tech])
		self.inv_cost_accum_CfD = np.zeros([self.agents_g, self.n_tech])
		self.inv_cost_accum_decom = np.zeros([self.agents_g, self.n_tech])

		self.cm_fc_under_construction = np.zeros([self.n_tech])
		self.cm_fc_under_construction_battery_merchant = np.zeros([self.n_tech_battery])
		self.cm_fc_under_construction_battery_cm = np.zeros([self.n_tech_battery])
		self.cm_fc_under_construction_battery_flexibility = np.zeros([self.n_tech_battery])
		
		self.cm_income = np.zeros([self.agents_g, self.n_tech])
		self.cm_inv_options = np.zeros([self.agents_g, self.n_tech])

		self.cm_income_battery = np.zeros([self.agents_g, self.n_tech_battery])
		self.cm_inv_options_battery = np.zeros([self.agents_g, self.n_tech_battery])

		self.cm_income_existing = np.zeros([self.agents_g, self.n_tech])
		self.cm_inv_options_existing = np.zeros([self.agents_g, self.n_tech])

		self.CfD_fc_under_construction = np.zeros([self.n_tech])
		self.CfD_income = np.zeros([self.agents_g, self.n_tech])
		self.CfD_price_pond = np.zeros([self.agents_g, self.n_tech])

		self.flexibility_price_pond = np.zeros([self.agents_g, self.n_tech_battery])

		## Reward per technology

		self.reward_tech_merchant = np.zeros([self.agents_g, self.n_tech])
		self.reward_tech_cm = np.zeros([self.agents_g, self.n_tech])
		self.reward_tech_existing = np.zeros([self.agents_g, self.n_tech])
		self.reward_tech_CfD = np.zeros([self.agents_g, self.n_tech])

		# Reward planner

		self.reward_planner = 0
		self.reward_planner_step = 0

		self.reward_planner_adequacy = 0
		self.reward_planner_system_cost = 0
		self.reward_planner_emissions = 0
		self.reward_planner_depreciation = 0
		self.reward_planner_flexibility = 0
		self.reward_planner_taxes = 0
		self.reward_planner_rent = 0 

		self.reward_planner_adequacy_not_discounted = 0
		self.reward_planner_system_cost_not_discounted = 0
		self.reward_planner_emissions_not_discounted = 0
		self.reward_planner_depreciation_not_discounted = 0
		self.reward_planner_flexibility_not_discounted = 0
		self.reward_planner_taxes_not_discounted = 0
		self.reward_planner_rent_not_discounted = 0 

		self.reward_planner_adequacy_step = 0
		self.reward_planner_system_cost_step = 0
		self.reward_planner_emissions_step = 0
		self.reward_planner_depreciation_step = 0
		self.reward_planner_flexibility_step = 0
		self.reward_planner_taxes_step = 0

		self.energy_not_served_step = 0

		## Resetting accumulated profit

		self.accumulated_profit_cm = np.zeros([self.n_tech])
		self.accumulated_profit_merchant = np.zeros([self.n_tech])
		self.accumulated_profit_CfD = np.zeros([self.n_tech])

		# Reset battery

		self.reward_tech_merchant_battery = np.zeros([self.agents_g, self.n_tech_battery])
		self.reward_tech_merchant_battery_step = np.zeros([self.agents_g, self.n_tech_battery])
		self.accumulated_profit_merchant_battery = np.zeros([self.n_tech_battery])

		self.reward_tech_existing_storage_lt = np.zeros([self.agents_g])
		self.reward_tech_existing_storage_lt_step = np.zeros([self.agents_g])

		self.reward_tech_cm_battery = np.zeros([self.agents_g, self.n_tech_battery])
		self.reward_tech_cm_battery_step = np.zeros([self.agents_g, self.n_tech_battery])
		self.accumulated_profit_cm_battery = np.zeros([self.n_tech_battery])

		self.reward_tech_flexibility_battery = np.zeros([self.agents_g, self.n_tech_battery])
		self.reward_tech_flexibility_battery_step = np.zeros([self.agents_g, self.n_tech_battery])
		self.accumulated_profit_flexibility_battery = np.zeros([self.n_tech_battery])

		self.capacity_merchant_battery = np.zeros([self.agents_g, self.n_tech_battery])
		self.capacity_merchant_battery_uc = np.zeros([self.agents_g, self.n_tech_battery])

		self.inv_age_merchant_battery = np.zeros([self.agents_g, self.n_tech_battery])
		self.inv_cost_accum_merchant_battery = np.zeros([self.agents_g, self.n_tech_battery])

		self.capacity_cm_battery = np.zeros([self.agents_g, self.n_tech_battery])
		self.capacity_cm_battery_uc = np.zeros([self.agents_g, self.n_tech_battery]) 

		self.inv_age_cm_battery = np.zeros([self.agents_g, self.n_tech_battery])
		self.inv_cost_accum_cm_battery = np.zeros([self.agents_g, self.n_tech_battery])

		self.capacity_flexibility_battery = np.zeros([self.agents_g, self.n_tech_battery])
		self.capacity_flexibility_battery_uc = np.zeros([self.agents_g, self.n_tech_battery])

		self.inv_age_flexibility_battery = np.zeros([self.agents_g, self.n_tech_battery])
		self.inv_cost_accum_flexibility_battery = np.zeros([self.agents_g, self.n_tech_battery])

		self.capacity_existing_storage_lt = self.inv_init_storage_lt
		self.capacity_existing_storage_lt_uc = np.zeros([self.agents_g]) 

		self.SoC_merchant_storage_lt = self.SoC_max_merchant_storage_lt * 0.5

		# Scarcity signals

		self.cm_scarcity = 0
		self.CfD_scarcity = 0

		# Resetting Carbon Tax, CfD target, and adqequacy target

		self.CO2_tax = self.CO2_tax_original
		self.CfD_target = self.CfD_target_original
		self.CfD_price_cap = self.CfD_price_cap_original
		self.CfD_increment_target = 0
		self.CfD_target_penetration = 0
		
		self.cm_excess_demand = 0
		self.cm_price_cap = self.cm_price_cap_original
		self.cm_strike = self.cm_strike_original
		self.cm_demand_target = 0.01

		self.flexibility_target = self.flexibility_target_original
		self.flexibility_price_cap = self.flexibility_price_cap_original
		self.flexibility_increment_target = 0
		self.flexibility_target_penetration = 0
		
		# Penalty term

		self.penalty_term = np.zeros([self.agents_g, self.n_tech + 1])

		# Regret penalty - merchant only

		self.penalty_merchant = np.zeros([self.agents_g, self.n_tech])
		self.penalty_merchant_battery = np.zeros([self.agents_g, self.n_tech_battery])

		# Simulation year and month

		self.year = 0

		self.month = 0

		self.hour_year = 0

		# random values for bids 

		# Availability and demand per scenario

		self.scenario = 0

		index_init = (self.year_int) * self.short_t * self.yearly_resolution * 5 + self.month * self.short_t * 5 + self.scenario * self.short_t

		index_final = (self.year_int) * self.short_t * self.yearly_resolution * 5 + self.month* self.short_t * 5 + (self.scenario + 1) * self.short_t


		self.availability_tech_step = self.availability_tech[int(index_init): int(index_final),:]
 
		self.demand_step = self.demand[int(index_init): int(index_final)]

		observations = self.observation_space.sample()

		i = 0

		## Feeling up relevant observations after reset

		self.cm_price = 0
		self.cm_auction_indicator = False
		self.cm_balance_real = 0

		## TODO initial price for the cm market
		self.cm_price_existing = self.cm_price_cap/(self.step_g_bids)

		self.cm_balance, self.cm_auction_indicator, self.cm_tech_cc, self.hourly_balance, self.cm_balance_real = self.elcc_calculator.cm_balance_estimation()
		## TODO
		self.cm_tech_cc = np.nan_to_num(self.cm_tech_cc, nan=0.0, posinf=0.0, neginf=0.0)

		self.CfD_price = 0
		self.CfD_auction_indicator = False
		self.CfD_balance_real = 0 

		self.CfD_balance, self.CfD_auction_indicator, self.CfD_balance_real, self.CfD_target_penetration = self.CfD_balance_estimation()

		self.CfD_price_agents = np.zeros([self.agents_g, self.n_tech])

		self.flexibility_price = 0
		self.flexibility_auction_indicator = False
		self.flexibility_balance_real = 0

		self.flexibility_balance, self.flexibility_auction_indicator, self.flexibility_balance_real, self.flexibility_target_penetration  = self.flexibility_balance_estimation()
		self.flexibility_price_agents = np.zeros([self.agents_g, self.act_bids_flexibility])

		obs_temp_g = np.zeros([self.agents_g, self.n_obs_g])
						
		# Demand projection

		average_demand_projection_short = self.demand_average_bimester[int(self.year_int * self.yearly_resolution + self.month)]
		average_demand_projection_long = self.demand_average_year[self.year_int + self.planning_horizon]
		
		# Resource availability (short)

		average_solar_availability_short = self.availability_tech_average_bimester[int(self.year_int * self.yearly_resolution + self.month), 0]

		average_wind_availability_short = self.availability_tech_average_bimester[int(self.year_int * self.yearly_resolution + self.month), 1]

		average_hydro_availability_short = self.availability_tech_average_bimester[int(self.year_int * self.yearly_resolution + self.month), self.n_tech]

		average_hydro_ror_availability_short = self.availability_tech_average_bimester[int(self.year_int * self.yearly_resolution + self.month), self.n_tech + 1]

		average_solar_availability_long = self.availability_tech_average_yearly[self.year_int, 0]

		average_wind_availability_long = self.availability_tech_average_yearly[self.year_int, 1]

		average_hydro_availability_long = self.availability_tech_average_yearly[self.year_int, self.n_tech]

		average_hydro_ror_availability_long = self.availability_tech_average_yearly[self.year_int, self.n_tech + 1]

		i = 0

		for a in self.possible_agents_g:

			## Prices 
			
			## Normal price observations

			obs_temp_g[i, 0] = np.clip((50 * 2 / self.VoLL_norm - 1), -1, 1)
			obs_temp_g[i, 1] = np.clip((50 / self.VoLL_norm), 0, 1)
			obs_temp_g[i, 2] = np.clip((50 * 2 / self.VoLL_norm - 1), -1, 1)
			obs_temp_g[i, 3] = np.clip((50 * 2 / self.VoLL_norm - 1), -1, 1)

			## Flexibility price observations

			obs_temp_g[i, self.n_obs_short_term_prices + 0] = np.clip((0 * 2 / self.VoLL_norm - 1), -1, 1)
			obs_temp_g[i, self.n_obs_short_term_prices + 1] = np.clip((0 / self.VoLL_norm), 0, 1)
			obs_temp_g[i, self.n_obs_short_term_prices + 2] = np.clip((0* 2 / self.VoLL_norm - 1), -1, 1)
			obs_temp_g[i, self.n_obs_short_term_prices + 3] = np.clip((0 * 2 / self.VoLL_norm - 1), -1, 1)
			
			## Installed capacity (individual and total)

			for j in range(self.n_tech):

				ind_installed_capacity = self.inv_g[i,j]

				ind_total_installed_capacity = np.sum(self.inv_g[i,:])

				total_installed_capacity = np.sum(self.inv_g[:,j])

				ind_installed_capacity_existing = self.inv_g_aging[i,j]

				total_installed_capacity_existing = np.sum(self.inv_g_aging[:,j])

				ind_installed_capacity_merchant = self.capacity_merchant_uc[i,j]

				total_installed_capacity_merchant = np.sum(self.capacity_merchant_uc[:,j])

				ind_installed_capacity_cm = self.capacity_cm_uc[i,j]

				total_installed_capacity_cm = np.sum(self.capacity_cm_uc[:,j])

				ind_installed_capacity_CfD = self.capacity_CfD_uc[i,j]

				total_installed_capacity_CfD = np.sum(self.capacity_CfD_uc[:,j])

				## Demand projections with respect to install technologies

				# Individual	
						
				obs_temp_g[i, self.n_obs_short_term_prices + self.n_obs_flexibility_prices  + self.n_obs_resources + (self.n_obs_cap_ind * j) + 0] = np.clip((average_demand_projection_long/self.agents_g - ind_installed_capacity)/(average_demand_projection_long/self.agents_g + ind_installed_capacity), -1, 1)

				# Individual	
						
				obs_temp_g[i, self.n_obs_short_term_prices + self.n_obs_flexibility_prices  + self.n_obs_resources + (self.n_obs_cap_ind * j) + 1] = np.clip((average_demand_projection_long - total_installed_capacity)/(average_demand_projection_long + total_installed_capacity), -1, 1)
				
				# Individual	
						
				obs_temp_g[i, self.n_obs_short_term_prices + self.n_obs_flexibility_prices  + self.n_obs_resources + (self.n_obs_cap_ind * j) + 2] = np.clip((ind_installed_capacity)/(ind_total_installed_capacity + delta) * 2 - 1, -1, 1)

				# ind - Existing

				obs_temp_g[i, self.n_obs_short_term_prices + self.n_obs_flexibility_prices  + self.n_obs_resources + (self.n_obs_cap_ind * j) + 3] = np.clip(ind_installed_capacity_existing / (ind_installed_capacity + delta) * 2 - 1, -1, 1)
				
				# ind - Merchant

				obs_temp_g[i, self.n_obs_short_term_prices + self.n_obs_flexibility_prices  + self.n_obs_resources + (self.n_obs_cap_ind * j) + 4] = np.clip(ind_installed_capacity_merchant / (ind_installed_capacity + delta) * 2 - 1, -1, 1)
				
				# ind - CM

				obs_temp_g[i, self.n_obs_short_term_prices + self.n_obs_flexibility_prices  + self.n_obs_resources + (self.n_obs_cap_ind * j) + 5] = np.clip(ind_installed_capacity_cm / (ind_installed_capacity + delta) * 2 - 1, -1, 1)

				# ind - CfD

				obs_temp_g[i, self.n_obs_short_term_prices + self.n_obs_flexibility_prices  + self.n_obs_resources + (self.n_obs_cap_ind * j) + 6] = np.clip(ind_installed_capacity_CfD / (ind_installed_capacity + delta) * 2 - 1, -1, 1)

				# Individual

				obs_temp_g[i, self.n_obs_short_term_prices + self.n_obs_flexibility_prices  + self.n_obs_resources + (self.n_obs_cap_ind * j) + 7] = np.clip((ind_installed_capacity)/(total_installed_capacity + delta) * 2 - 1, -1, 1)

				# total - existing

				obs_temp_g[i, self.n_obs_short_term_prices + self.n_obs_flexibility_prices  + self.n_obs_resources + (self.n_obs_cap_ind * self.n_tech) + (self.n_obs_cap_total * j) + 0] = np.clip((total_installed_capacity_existing)/(total_installed_capacity + delta) * 2 - 1, -1, 1)

				# total - Merchant

				obs_temp_g[i, self.n_obs_short_term_prices + self.n_obs_flexibility_prices  + self.n_obs_resources + (self.n_obs_cap_ind * self.n_tech) + (self.n_obs_cap_total * j) + 1] = np.clip((total_installed_capacity_merchant)/(total_installed_capacity + delta) * 2 - 1, -1, 1)

				# total - CM

				obs_temp_g[i, self.n_obs_short_term_prices + self.n_obs_flexibility_prices  + self.n_obs_resources + (self.n_obs_cap_ind * self.n_tech) + (self.n_obs_cap_total * j) + 2] = np.clip((total_installed_capacity_cm)/(total_installed_capacity + delta) * 2 - 1, -1, 1)

				# total - CfD

				obs_temp_g[i, self.n_obs_short_term_prices + self.n_obs_flexibility_prices  + self.n_obs_resources + (self.n_obs_cap_ind * self.n_tech) + (self.n_obs_cap_total * j) + 3] = np.clip((total_installed_capacity_CfD)/(total_installed_capacity + delta) * 2 - 1, -1, 1)

				# Capacity credits - CM

				obs_temp_g[i, self.n_obs_short_term_prices + self.n_obs_flexibility_prices  + self.n_obs_resources + (self.n_obs_cap_ind * self.n_tech) + (self.n_obs_cap_total * j) + 4] = np.clip((self.cm_tech_cc[0,j]) * 2 - 1, -1, 1)

			## Adjustment of rewards due to investment costs (Batteries)
			
			for j in range(self.n_tech_battery):

				## Battery and long-term storage observations (TODO)

				ind_installed_capacity_merchant_battery = self.capacity_merchant_battery[i,j] +  self.inv_init_battery[i,j]
				total_installed_capacity_merchant_battery = np.sum(self.capacity_merchant_battery[:, j] + self.inv_init_battery[:, j])

				ind_installed_capacity_cm_battery = self.capacity_cm_battery[i,j]
				total_installed_capacity_cm_battery = np.sum(self.capacity_cm_battery[:,j])

				ind_installed_capacity_flexibility_battery = self.capacity_flexibility_battery[i,j]
				total_installed_capacity_flexibility_battery = np.sum(self.capacity_flexibility_battery[:,j])

				ind_installed_capacity_existing_storage_lt = self.capacity_existing_storage_lt[i]
				total_installed_capacity_existing_storage_lt = np.sum(self.capacity_existing_storage_lt)

				ind_SoC_existing_storage_lt = self.SoC_merchant_storage_lt[i]/(self.SoC_max_merchant_storage_lt[i] + delta)
				total_SoC_existing_storage_lt = np.sum(self.SoC_merchant_storage_lt)/(np.sum(self.SoC_max_merchant_storage_lt) + delta)

				# Individual - merchant
							
				obs_temp_g[i, self.n_obs_short_term_prices + self.n_obs_flexibility_prices + self.n_obs_resources + (self.n_obs_cap_ind * self.n_tech) + (self.n_obs_cap_total * self.n_tech) + (self.n_obs_battery_cap_ind * j) + 0] = np.clip((average_demand_projection_long/self.agents_g - ind_installed_capacity_merchant_battery)/(average_demand_projection_long/self.agents_g + ind_installed_capacity_merchant_battery), -1, 1)

				# Individual - cm
							
				obs_temp_g[i, self.n_obs_short_term_prices + self.n_obs_flexibility_prices + self.n_obs_resources + (self.n_obs_cap_ind * self.n_tech) + (self.n_obs_cap_total * self.n_tech) + (self.n_obs_battery_cap_ind * j) + 1] = np.clip((average_demand_projection_long/self.agents_g - ind_installed_capacity_cm_battery)/(average_demand_projection_long/self.agents_g + ind_installed_capacity_cm_battery), -1, 1)

				# Individual - flexibility
							
				obs_temp_g[i, self.n_obs_short_term_prices + self.n_obs_flexibility_prices + self.n_obs_resources + (self.n_obs_cap_ind * self.n_tech) + (self.n_obs_cap_total * self.n_tech) + (self.n_obs_battery_cap_ind * j) + 2] = np.clip((average_demand_projection_long/self.agents_g - ind_installed_capacity_flexibility_battery)/(average_demand_projection_long/self.agents_g + ind_installed_capacity_flexibility_battery), -1, 1)

				# Individual - Existing Merchant Long-term
							
				obs_temp_g[i, self.n_obs_short_term_prices + self.n_obs_flexibility_prices + self.n_obs_resources + (self.n_obs_cap_ind * self.n_tech) + (self.n_obs_cap_total * self.n_tech) + (self.n_obs_battery_cap_ind * j) + 3] = np.clip((average_demand_projection_long/self.agents_g - ind_installed_capacity_existing_storage_lt)/(average_demand_projection_long/self.agents_g + ind_installed_capacity_existing_storage_lt), -1, 1)

				# Individual - SoC Existing Merchant Long-term
							
				obs_temp_g[i, self.n_obs_short_term_prices + self.n_obs_flexibility_prices + self.n_obs_resources + (self.n_obs_cap_ind * self.n_tech) + (self.n_obs_cap_total * self.n_tech) + (self.n_obs_battery_cap_ind * j) + 4] = np.clip(ind_SoC_existing_storage_lt * 2 - 1, -1, 1)

				# Total	- merchant
							
				obs_temp_g[i, self.n_obs_short_term_prices + self.n_obs_flexibility_prices + self.n_obs_resources + (self.n_obs_cap_ind * self.n_tech) + (self.n_obs_cap_total * self.n_tech) + (self.n_obs_battery_cap_ind * self.n_tech_battery) + (self.n_obs_battery_cap_total * j) + 0]= np.clip((average_demand_projection_long - total_installed_capacity_merchant_battery)/(average_demand_projection_long + total_installed_capacity_merchant_battery), -1, 1)

				# Total	- cm
							
				obs_temp_g[i, self.n_obs_short_term_prices + self.n_obs_flexibility_prices + self.n_obs_resources + (self.n_obs_cap_ind * self.n_tech) + (self.n_obs_cap_total * self.n_tech) + (self.n_obs_battery_cap_ind * self.n_tech_battery) + (self.n_obs_battery_cap_total * j) + 1]= np.clip((average_demand_projection_long - total_installed_capacity_cm_battery)/(average_demand_projection_long + total_installed_capacity_cm_battery), -1, 1)

				# Total	- flexibility
							
				obs_temp_g[i, self.n_obs_short_term_prices + self.n_obs_flexibility_prices + self.n_obs_resources + (self.n_obs_cap_ind * self.n_tech) + (self.n_obs_cap_total * self.n_tech) + (self.n_obs_battery_cap_ind * self.n_tech_battery) + (self.n_obs_battery_cap_total * j) + 2]= np.clip((average_demand_projection_long - total_installed_capacity_flexibility_battery)/(average_demand_projection_long + total_installed_capacity_flexibility_battery), -1, 1)

				# Capacity factor battery  

				obs_temp_g[i, self.n_obs_short_term_prices + self.n_obs_flexibility_prices + self.n_obs_resources + (self.n_obs_cap_ind * self.n_tech) + (self.n_obs_cap_total * self.n_tech) + (self.n_obs_battery_cap_ind * self.n_tech_battery) + (self.n_obs_battery_cap_total * j) + 3]= np.clip((self.cm_tech_cc[0, self.n_tech + j]) * 2 - 1, -1, 1)

			# Total	- Existing Merchant Long-term
						
			obs_temp_g[i, self.n_obs_short_term_prices + self.n_obs_flexibility_prices + self.n_obs_resources + (self.n_obs_cap_ind * self.n_tech) + (self.n_obs_cap_total * self.n_tech) + (self.n_obs_battery_cap_ind * self.n_tech_battery) + (self.n_obs_battery_cap_total * self.n_tech_battery) + 0]= np.clip((average_demand_projection_long - total_installed_capacity_existing_storage_lt)/(average_demand_projection_long + total_installed_capacity_existing_storage_lt), -1, 1)

			# Total	- SoC Existing Merchant Long-term
						
			obs_temp_g[i, self.n_obs_short_term_prices + self.n_obs_flexibility_prices + self.n_obs_resources + (self.n_obs_cap_ind * self.n_tech) + (self.n_obs_cap_total * self.n_tech) + (self.n_obs_battery_cap_ind * self.n_tech_battery) + (self.n_obs_battery_cap_total * self.n_tech_battery) + 1]= np.clip(total_SoC_existing_storage_lt * 2 - 1, -1, 1)

			## Time observations

			# Month

			obs_temp_g[i, self.n_obs_short_term_prices + self.n_obs_flexibility_prices + self.n_obs_resources + (self.n_obs_cap_ind * self.n_tech) + (self.n_obs_cap_total * self.n_tech) + (self.n_obs_battery_cap_ind * self.n_tech_battery) + (self.n_obs_battery_cap_total * self.n_tech_battery) + 2] = np.clip((self.month/self.yearly_resolution) * 2 - 1, -1, 1)
   
			# Year
   
			obs_temp_g[i, self.n_obs_short_term_prices + self.n_obs_flexibility_prices + self.n_obs_resources + (self.n_obs_cap_ind * self.n_tech) + (self.n_obs_cap_total * self.n_tech) + (self.n_obs_battery_cap_ind * self.n_tech_battery) + (self.n_obs_battery_cap_total * self.n_tech_battery) + 3] = np.clip((self.year/self.max_year) * 2 - 1, -1, 1)
   
			# Time
   
			obs_temp_g[i, self.n_obs_short_term_prices + self.n_obs_flexibility_prices + self.n_obs_resources + (self.n_obs_cap_ind * self.n_tech) + (self.n_obs_cap_total * self.n_tech) + (self.n_obs_battery_cap_ind * self.n_tech_battery) + (self.n_obs_battery_cap_total * self.n_tech_battery) + 4] = np.clip(self.hour_year/(self.max_year * self.yearly_resolution) * 2 - 1, -1, 1)

			# CO2 tax
   
			obs_temp_g[i, self.n_obs_short_term_prices + self.n_obs_flexibility_prices + self.n_obs_resources + (self.n_obs_cap_ind * self.n_tech) + (self.n_obs_cap_total * self.n_tech) + (self.n_obs_battery_cap_ind * self.n_tech_battery) + (self.n_obs_battery_cap_total * self.n_tech_battery) + 5] = np.clip((self.CO2_tax_scenario[self.year_int + 4]/300) * 2 - 1, -1, 1)

			# Corporate tax rate
   
			obs_temp_g[i, self.n_obs_short_term_prices + self.n_obs_flexibility_prices + self.n_obs_resources + (self.n_obs_cap_ind * self.n_tech) + (self.n_obs_cap_total * self.n_tech) + (self.n_obs_battery_cap_ind * self.n_tech_battery) + (self.n_obs_battery_cap_total * self.n_tech_battery) + 6] = np.clip((self.corporate_tax_rate) * 2 - 1, -1, 1)
			
			## Capacity Market and CfD observations

			# CM Balance 

			obs_temp_g[i, self.n_obs_short_term_prices + self.n_obs_flexibility_prices + self.n_obs_resources + (self.n_obs_cap_ind * self.n_tech) + (self.n_obs_cap_total * self.n_tech) + (self.n_obs_battery_cap_ind * self.n_tech_battery) + (self.n_obs_battery_cap_total * self.n_tech_battery) + self.n_observation_time + 0] = np.clip(self.cm_balance/self.percentile_demand,-1,1)

			# CM Balance Real

			obs_temp_g[i, self.n_obs_short_term_prices + self.n_obs_flexibility_prices + self.n_obs_resources + (self.n_obs_cap_ind * self.n_tech) + (self.n_obs_cap_total * self.n_tech) + (self.n_obs_battery_cap_ind * self.n_tech_battery) + (self.n_obs_battery_cap_total * self.n_tech_battery) + self.n_observation_time + 1] = np.clip(self.cm_balance_real/self.percentile_demand,-1,1)

			# CM price

			self.cm_price_aggregated = np.clip(((((np.sum(self.cm_income) + np.sum(self.cm_income_battery)) * self.normalization_factor /(self.short_t * self.hour_month))/(np.sum(self.cm_inv_options) + np.sum(self.cm_inv_options_battery) + delta))), 0, self.cm_price_cap_max)

			obs_temp_g[i, self.n_obs_short_term_prices + self.n_obs_flexibility_prices + self.n_obs_resources + (self.n_obs_cap_ind * self.n_tech) + (self.n_obs_cap_total * self.n_tech) + (self.n_obs_battery_cap_ind * self.n_tech_battery) + (self.n_obs_battery_cap_total * self.n_tech_battery) + self.n_observation_time + 2] = np.clip(self.cm_price_aggregated/self.cm_price_cap_max * 2 - 1, -1, 1)

			# CM scarcity

			obs_temp_g[i, self.n_obs_short_term_prices + self.n_obs_flexibility_prices + self.n_obs_resources + (self.n_obs_cap_ind * self.n_tech) + (self.n_obs_cap_total * self.n_tech) + (self.n_obs_battery_cap_ind * self.n_tech_battery) + (self.n_obs_battery_cap_total * self.n_tech_battery) + self.n_observation_time + 3] = np.clip((self.cm_scarcity * 2) - 1, -1, 1)

			# CM target

			obs_temp_g[i, self.n_obs_short_term_prices + self.n_obs_flexibility_prices + self.n_obs_resources + (self.n_obs_cap_ind * self.n_tech) + (self.n_obs_cap_total * self.n_tech) + (self.n_obs_battery_cap_ind * self.n_tech_battery) + (self.n_obs_battery_cap_total * self.n_tech_battery) + self.n_observation_time + 4] = np.clip((self.cm_demand_target / np.max(self.cm_reliability_target_values))*2 - 1, -1, 1)

			# CfD Balance 

			obs_temp_g[i, self.n_obs_short_term_prices + self.n_obs_flexibility_prices + self.n_obs_resources + (self.n_obs_cap_ind * self.n_tech) + (self.n_obs_cap_total * self.n_tech) + (self.n_obs_battery_cap_ind * self.n_tech_battery) + (self.n_obs_battery_cap_total * self.n_tech_battery) + self.n_observation_time + 5] = np.clip(self.CfD_balance/self.percentile_demand,-1,1)

			# CfD Balance Real

			obs_temp_g[i, self.n_obs_short_term_prices + self.n_obs_flexibility_prices + self.n_obs_resources + (self.n_obs_cap_ind * self.n_tech) + (self.n_obs_cap_total * self.n_tech) + (self.n_obs_battery_cap_ind * self.n_tech_battery) + (self.n_obs_battery_cap_total * self.n_tech_battery) + self.n_observation_time + 6] = np.clip(self.CfD_balance_real/self.percentile_demand,-1,1)

			# CfD Price 

			self.CfD_price_aggregated = np.clip((np.sum(self.CfD_price_pond)/(np.sum(self.capacity_CfD) + delta)), 0, self.CfD_price_cap_max)

			obs_temp_g[i, self.n_obs_short_term_prices + self.n_obs_flexibility_prices + self.n_obs_resources + (self.n_obs_cap_ind * self.n_tech) + (self.n_obs_cap_total * self.n_tech) + (self.n_obs_battery_cap_ind * self.n_tech_battery) + (self.n_obs_battery_cap_total * self.n_tech_battery) + self.n_observation_time + 7] = np.clip(self.CfD_price_aggregated/self.CfD_price_cap_max * 2 - 1, -1, 1)

			# CfD scarcity

			obs_temp_g[i, self.n_obs_short_term_prices + self.n_obs_flexibility_prices + self.n_obs_resources + (self.n_obs_cap_ind * self.n_tech) + (self.n_obs_cap_total * self.n_tech) + (self.n_obs_battery_cap_ind * self.n_tech_battery) + (self.n_obs_battery_cap_total * self.n_tech_battery) + self.n_observation_time + 8] = np.clip((self.CfD_scarcity * 2) - 1, -1, 1)

			# CfD target

			obs_temp_g[i, self.n_obs_short_term_prices + self.n_obs_flexibility_prices + self.n_obs_resources + (self.n_obs_cap_ind * self.n_tech) + (self.n_obs_cap_total * self.n_tech) + (self.n_obs_battery_cap_ind * self.n_tech_battery) + (self.n_obs_battery_cap_total * self.n_tech_battery) + self.n_observation_time + 9] = np.clip((self.CfD_increment_target / np.max(self.CfD_target_increments_steps))*2 - 1, -1, 1)

			# flexibility Balance 

			obs_temp_g[i, self.n_obs_short_term_prices + self.n_obs_flexibility_prices + self.n_obs_resources + (self.n_obs_cap_ind * self.n_tech) + (self.n_obs_cap_total * self.n_tech) + (self.n_obs_battery_cap_ind * self.n_tech_battery) + (self.n_obs_battery_cap_total * self.n_tech_battery) + self.n_observation_time + 10] = np.clip(self.flexibility_balance/self.percentile_demand,-1,1)

			# flexibility Balance Real

			obs_temp_g[i, self.n_obs_short_term_prices + self.n_obs_flexibility_prices + self.n_obs_resources + (self.n_obs_cap_ind * self.n_tech) + (self.n_obs_cap_total * self.n_tech) + (self.n_obs_battery_cap_ind * self.n_tech_battery) + (self.n_obs_battery_cap_total * self.n_tech_battery) + self.n_observation_time + 11] = np.clip(self.flexibility_balance_real/self.percentile_demand,-1,1)

			# Flexibility Price 

			self.flexibility_price_aggregated = np.clip((np.sum(self.flexibility_price_pond)/(np.sum(self.capacity_flexibility_battery) + delta)), 0, self.flexibility_price_cap)

			obs_temp_g[i, self.n_obs_short_term_prices + self.n_obs_flexibility_prices + self.n_obs_resources + (self.n_obs_cap_ind * self.n_tech) + (self.n_obs_cap_total * self.n_tech) + (self.n_obs_battery_cap_ind * self.n_tech_battery) + (self.n_obs_battery_cap_total * self.n_tech_battery) + self.n_observation_time + 12] = np.clip(self.flexibility_price_aggregated/self.flexibility_price_cap * 2 - 1, -1, 1)

			# Flexibility scarcity

			obs_temp_g[i, self.n_obs_short_term_prices + self.n_obs_flexibility_prices + self.n_obs_resources + (self.n_obs_cap_ind * self.n_tech) + (self.n_obs_cap_total * self.n_tech) + (self.n_obs_battery_cap_ind * self.n_tech_battery) + (self.n_obs_battery_cap_total * self.n_tech_battery) + self.n_observation_time + 13] = np.clip((self.flexibility_scarcity * 2) - 1, -1, 1)

			# Flexibility target

			obs_temp_g[i, self.n_obs_short_term_prices + self.n_obs_flexibility_prices + self.n_obs_resources + (self.n_obs_cap_ind * self.n_tech) + (self.n_obs_cap_total * self.n_tech) + (self.n_obs_battery_cap_ind * self.n_tech_battery) + (self.n_obs_battery_cap_total * self.n_tech_battery) + self.n_observation_time + 14] = np.clip((self.flexibility_increment_target / np.max(self.flexibility_target_increments_steps))*2 - 1, -1, 1)

			## Market concentration

			# All markets

			agent_capacities = (np.sum(self.inv_g, axis=1) + np.sum(self.capacity_merchant_battery + self.capacity_cm_battery + self.capacity_flexibility_battery + self.inv_init_battery, axis = 1) + self.capacity_existing_storage_lt)
			
			# HHI calculation
			total_capacity_all = np.sum(agent_capacities)
			market_shares = agent_capacities / (total_capacity_all + delta)
			herfindahl_index = np.sum(market_shares ** 2)
			
			obs_temp_g[i, self.n_obs_short_term_prices + self.n_obs_flexibility_prices + self.n_obs_resources + (self.n_obs_cap_ind * self.n_tech) + (self.n_obs_cap_total * self.n_tech) + (self.n_obs_battery_cap_ind * self.n_tech_battery) + (self.n_obs_battery_cap_total * self.n_tech_battery) + self.n_observation_time + 15] = np.clip(herfindahl_index * 2 - 1, -1, 1)

			# Aggregate HHI for all existing (generation + long-term storage + initial battery)
			agent_capacity_existing_all = (
				np.sum(self.inv_g_aging, axis=1) 
				+ self.capacity_existing_storage_lt 
				+ np.sum(self.inv_init_battery, axis = 1)
			)
			total_capacity_existing_all = np.sum(agent_capacity_existing_all)
			market_shares_existing_all = agent_capacity_existing_all / (total_capacity_existing_all + delta)
			hhi_existing_all = np.sum(market_shares_existing_all ** 2)
			
			obs_temp_g[i, self.n_obs_short_term_prices + self.n_obs_flexibility_prices + self.n_obs_resources + (self.n_obs_cap_ind * self.n_tech) + (self.n_obs_cap_total * self.n_tech) + (self.n_obs_battery_cap_ind * self.n_tech_battery) + (self.n_obs_battery_cap_total * self.n_tech_battery) + self.n_observation_time + 16] = np.clip(hhi_existing_all * 2 - 1, -1, 1)

			# Aggregate HHI for all merchant (generation + battery)
			agent_capacity_merchant_all = (
				np.sum(self.capacity_merchant, axis=1) 
				+ np.sum(self.capacity_merchant_battery, axis = 1)
			)
			total_capacity_merchant_all = np.sum(agent_capacity_merchant_all)
			market_shares_merchant_all = agent_capacity_merchant_all / (total_capacity_merchant_all + delta)
			hhi_merchant_all = np.sum(market_shares_merchant_all ** 2)
			
			obs_temp_g[i, self.n_obs_short_term_prices + self.n_obs_flexibility_prices + self.n_obs_resources + (self.n_obs_cap_ind * self.n_tech) + (self.n_obs_cap_total * self.n_tech) + (self.n_obs_battery_cap_ind * self.n_tech_battery) + (self.n_obs_battery_cap_total * self.n_tech_battery) + self.n_observation_time + 17] = np.clip(hhi_merchant_all * 2 - 1, -1, 1)

			# Aggregate HHI for all CM (generation + battery)
			agent_capacity_cm_all = np.sum(self.capacity_cm, axis=1) + np.sum(self.capacity_cm_battery, axis = 1)
			total_capacity_cm_all = np.sum(agent_capacity_cm_all)
			market_shares_cm_all = agent_capacity_cm_all / (total_capacity_cm_all + delta)
			hhi_cm_all = np.sum(market_shares_cm_all ** 2)
			
			obs_temp_g[i, self.n_obs_short_term_prices + self.n_obs_flexibility_prices + self.n_obs_resources + (self.n_obs_cap_ind * self.n_tech) + (self.n_obs_cap_total * self.n_tech) + (self.n_obs_battery_cap_ind * self.n_tech_battery) + (self.n_obs_battery_cap_total * self.n_tech_battery) + self.n_observation_time + 18] = np.clip(hhi_cm_all * 2 - 1, -1, 1)

			# Aggregate HHI for all CfD (generation only)
			agent_capacity_cfd_all = np.sum(self.capacity_CfD_uc, axis=1)
			total_capacity_cfd_all = np.sum(agent_capacity_cfd_all)
			market_shares_cfd_all = agent_capacity_cfd_all / (total_capacity_cfd_all + delta)
			hhi_cfd_all = np.sum(market_shares_cfd_all ** 2)
			
			obs_temp_g[i, self.n_obs_short_term_prices + self.n_obs_flexibility_prices + self.n_obs_resources + (self.n_obs_cap_ind * self.n_tech) + (self.n_obs_cap_total * self.n_tech) + (self.n_obs_battery_cap_ind * self.n_tech_battery) + (self.n_obs_battery_cap_total * self.n_tech_battery) + self.n_observation_time + 19] = np.clip(hhi_cfd_all * 2 - 1, -1, 1)

			# Aggregate HHI for flexibility market (batteries only)
			agent_capacity_flexibility_all = np.sum(self.capacity_flexibility_battery, axis = 1)
			total_capacity_flexibility_all = np.sum(agent_capacity_flexibility_all)
			market_shares_flexibility_all = agent_capacity_flexibility_all / (total_capacity_flexibility_all + delta)
			hhi_flexibility_all = np.sum(market_shares_flexibility_all ** 2)
			
			obs_temp_g[i, self.n_obs_short_term_prices + self.n_obs_flexibility_prices + self.n_obs_resources + (self.n_obs_cap_ind * self.n_tech) + (self.n_obs_cap_total * self.n_tech) + (self.n_obs_battery_cap_ind * self.n_tech_battery) + (self.n_obs_battery_cap_total * self.n_tech_battery) + self.n_observation_time + 20] = np.clip(hhi_flexibility_all * 2 - 1, -1, 1)

			## Adding observations to dictionary

			observations_temp = {"observations": np.float32(obs_temp_g[i,:]),
							"action_mask":self.get_action_masks_g(self.investments_enabled[i,:], self.entry_enabled[i], self.capacity_existing_storage_lt[i], self.inv_g_aging[i,:])} 
			
			observations.update({a:observations_temp}) 

			i += 1
		
		## Observations planner 

		obs_temp_p = np.zeros([self.agents_p, self.n_obs_p])

		# Short-term prices

		index_start_a = 0

		index_last_a = self.n_obs_short_term_prices

		index_start_p = 0

		index_last_p = self.n_obs_short_term_prices

		obs_temp_p[0,index_start_p:index_last_p] = obs_temp_g[0, index_start_a:index_last_a]

		# Resources

		index_start_a = self.n_obs_short_term_prices + self.n_obs_flexibility_prices

		index_last_a = self.n_obs_short_term_prices + self.n_obs_flexibility_prices + self.n_obs_resources

		index_start_p = self.n_obs_short_term_prices_p + self.n_obs_flexibility_prices_p

		index_last_p = self.n_obs_short_term_prices_p + self.n_obs_flexibility_prices_p + self.n_obs_resources_p

		obs_temp_p[0,index_start_p:index_last_p] = obs_temp_g[0, index_start_a:index_last_a]

		# Installed capacities techonologies

		for j in range(self.n_tech):

			index_start_a = self.n_obs_short_term_prices + self.n_obs_flexibility_prices+ self.n_obs_resources + (self.n_obs_cap_ind * self.n_tech) + (self.n_obs_cap_total * j) + 0

			index_last_a = self.n_obs_short_term_prices + self.n_obs_flexibility_prices+ self.n_obs_resources + (self.n_obs_cap_ind * self.n_tech) + (self.n_obs_cap_total * j) + self.n_obs_cap_total

			index_start_p = self.n_obs_short_term_prices_p + self.n_obs_flexibility_prices_p + self.n_obs_resources_p + (self.n_obs_cap_ind_p * self.n_tech) + (self.n_obs_cap_total_p * j) + 0

			index_last_p = self.n_obs_short_term_prices_p + self.n_obs_flexibility_prices_p + self.n_obs_resources_p + (self.n_obs_cap_ind_p * self.n_tech) + (self.n_obs_cap_total_p * j) + self.n_obs_cap_total_p

			obs_temp_p[0,index_start_p:index_last_p] = obs_temp_g[0, index_start_a:index_last_a]

		# Installed capacities batteries

		for j in range(self.n_tech_battery):

			index_start_a = self.n_obs_short_term_prices + self.n_obs_flexibility_prices + self.n_obs_resources + (self.n_obs_cap_ind * self.n_tech) + (self.n_obs_cap_total * self.n_tech) + (self.n_obs_battery_cap_ind * self.n_tech_battery) + (self.n_obs_battery_cap_total * j) + 0

			index_last_a = self.n_obs_short_term_prices + self.n_obs_flexibility_prices + self.n_obs_resources + (self.n_obs_cap_ind * self.n_tech) + (self.n_obs_cap_total * self.n_tech) + (self.n_obs_battery_cap_ind * self.n_tech_battery) + (self.n_obs_battery_cap_total * j) + self.n_obs_battery_cap_total

			index_start_p = self.n_obs_short_term_prices_p + self.n_obs_flexibility_prices_p + self.n_obs_resources_p + (self.n_obs_cap_ind_p * self.n_tech) + (self.n_obs_cap_total_p * self.n_tech) + (self.n_obs_battery_cap_ind_p * self.n_tech_battery) + (self.n_obs_battery_cap_total_p * j) + 0

			index_last_p = self.n_obs_short_term_prices_p + self.n_obs_flexibility_prices_p+ self.n_obs_resources_p + (self.n_obs_cap_ind_p * self.n_tech) + (self.n_obs_cap_total_p * self.n_tech) + (self.n_obs_battery_cap_ind_p * self.n_tech_battery) + (self.n_obs_battery_cap_total_p * j) + self.n_obs_battery_cap_total_p

			obs_temp_p[0,index_start_p:index_last_p] = obs_temp_g[0, index_start_a:index_last_a]

		# Time

		index_start_a = self.n_obs_short_term_prices + self.n_obs_flexibility_prices + self.n_obs_resources + (self.n_obs_cap_ind * self.n_tech) + (self.n_obs_cap_total * self.n_tech) + (self.n_obs_battery_cap_ind * self.n_tech_battery) + (self.n_obs_battery_cap_total * self.n_tech_battery) + 0

		index_last_a = self.n_obs_short_term_prices + self.n_obs_flexibility_prices + self.n_obs_resources + (self.n_obs_cap_ind * self.n_tech) + (self.n_obs_cap_total * self.n_tech) + (self.n_obs_battery_cap_ind * self.n_tech_battery) + (self.n_obs_battery_cap_total * self.n_tech_battery) + self.n_observation_time

		index_start_p = self.n_obs_short_term_prices_p + self.n_obs_flexibility_prices_p + self.n_obs_resources_p + (self.n_obs_cap_ind_p * self.n_tech) + (self.n_obs_cap_total_p * self.n_tech) + (self.n_obs_battery_cap_ind_p * self.n_tech_battery) + (self.n_obs_battery_cap_total_p * self.n_tech_battery) + 0

		index_last_p = self.n_obs_short_term_prices_p + self.n_obs_flexibility_prices_p + self.n_obs_resources_p + (self.n_obs_cap_ind_p * self.n_tech) + (self.n_obs_cap_total_p * self.n_tech) + (self.n_obs_battery_cap_ind_p * self.n_tech_battery) + (self.n_obs_battery_cap_total_p * self.n_tech_battery) + self.n_observation_time_p

		obs_temp_p[0,index_start_p:index_last_p] = obs_temp_g[0, index_start_a:index_last_a]

		# CfD + CM markets 

		index_start_a = self.n_obs_short_term_prices + self.n_obs_flexibility_prices + self.n_obs_resources + (self.n_obs_cap_ind * self.n_tech) + (self.n_obs_cap_total * self.n_tech) + (self.n_obs_battery_cap_ind * self.n_tech_battery) + (self.n_obs_battery_cap_total * self.n_tech_battery) + self.n_observation_time + 0

		index_last_a = self.n_obs_short_term_prices + self.n_obs_flexibility_prices + self.n_obs_resources + (self.n_obs_cap_ind * self.n_tech) + (self.n_obs_cap_total * self.n_tech) + (self.n_obs_battery_cap_ind * self.n_tech_battery) + (self.n_obs_battery_cap_total * self.n_tech_battery) + self.n_observation_time + self.n_observation_cm_CfD

		index_start_p = self.n_obs_short_term_prices_p + self.n_obs_flexibility_prices_p + self.n_obs_resources_p + (self.n_obs_cap_ind_p * self.n_tech) + (self.n_obs_cap_total_p * self.n_tech) + (self.n_obs_battery_cap_ind_p * self.n_tech_battery) + (self.n_obs_battery_cap_total_p * self.n_tech_battery) + self.n_observation_time_p + 0

		index_last_p = self.n_obs_short_term_prices_p + self.n_obs_flexibility_prices_p + self.n_obs_resources_p + (self.n_obs_cap_ind_p * self.n_tech) + (self.n_obs_cap_total_p * self.n_tech) + (self.n_obs_battery_cap_ind_p * self.n_tech_battery) + (self.n_obs_battery_cap_total_p * self.n_tech_battery) + self.n_observation_time_p + self.n_observation_cm_CfD_p

		obs_temp_p[0,index_start_p:index_last_p] = obs_temp_g[0, index_start_a:index_last_a]
		
		# Reward

		obs_temp_p[0,index_last_p + 0] = 0

		obs_temp_p[0,index_last_p + 1] = 0

		obs_temp_p[0,index_last_p + 2] = 0

		obs_temp_p[0,index_last_p + 3] = 0

		obs_temp_p[0,index_last_p + 4] = 0

		obs_temp_p[0,index_last_p + 5] = 0

		obs_temp_p[0,index_last_p + 6] = 0

		obs_temp_p[0,index_last_p + 7] = self.discount_factor_planner

		self.hourly_prices = np.zeros(self.short_t)
		
		# TO DO: Reset loop for planner observations
		
		i = 0

		for a in self.possible_agents_p:

			observations_temp = {"observations": np.float32(obs_temp_p[0,:]),
							"action_mask":self.get_action_masks_p()} 
			
			observations.update({a:observations_temp}) 

			i += 1

		return observations, infos  # reward, done, info can't be included
		
	def step(self, action_dict):

		## Shock multiplier

		self.shock_multiplier_techs  = np.ones([self.n_tech])

		if self.shock_flag == True and self.year_int >= self.year_shock:

			self.shock.step(active = True)

			self.shock_multiplier_techs[[self.n_tech - 2, self.n_tech -1]] = self.shock.multiplier

		# Carbon tax

		self.CO2_tax = self.CO2_tax_scenario[self.year_int]
		
		# Mechanism costs
		
		self.cost_spot_existing_step = 0
		self.cost_total_step = 0
		self.cost_merchant_step = 0
		self.cost_existing_step = 0
		self.cost_carbon_tax_return_step = 0
		self.cost_scarcity_step = 0
		self.cost_cm_step = 0
		self.cost_CfD_step = 0
		self.cost_Flexibility_step = 0
		
		# time 

		self.year_int = int(np.floor(self.year))

		# Calculate aggregated failures

		self.aggregated_failures = self.calculate_aggregated_failures()# Random scenario selection

		n_base = 12              # k-means representative days
		n_peak = 1               # peak demand extreme day
		n_scenarios = n_base + n_peak   # 13 total

		peak_prob = 0.017
		base_prob = (1 - peak_prob) / n_base

		probs = [base_prob] * n_base + [peak_prob] * n_peak
		self.scenario = np.random.choice(n_scenarios, p=probs)

		# Availability and demand per scenario

		bimester_block = self.short_t * n_scenarios   # hours in one bimester across all RDs
		year_block = bimester_block * self.yearly_resolution

		index_init = (self.year_int * year_block
					+ self.month * bimester_block
					+ self.scenario * self.short_t)

		index_final = index_init + self.short_t

		self.availability_tech_step = self.availability_tech[int(index_init):int(index_final), :]
		self.demand_step = self.demand[int(index_init):int(index_final)]

		## Reward step resetting

		# Generation technologies

		self.reward_tech_merchant_step = np.zeros([self.agents_g, self.n_tech])
		self.reward_tech_cm_step = np.zeros([self.agents_g, self.n_tech])
		self.reward_tech_CfD_step = np.zeros([self.agents_g, self.n_tech])
		self.reward_tech_existing_step = np.zeros([self.agents_g, self.n_tech])

		# Storage

		self.reward_tech_merchant_battery_step = np.zeros([self.agents_g, self.n_tech_battery])
		self.reward_tech_cm_battery_step = np.zeros([self.agents_g, self.n_tech_battery])
		self.reward_tech_flexibility_battery_step = np.zeros([self.agents_g, self.n_tech_battery])
		self.reward_tech_existing_storage_lt_step = np.zeros([self.agents_g])

		self.reward_planner_step = 0

		self.reward_planner_adequacy_step = 0
		self.reward_planner_system_cost_step = 0
		self.reward_planner_emissions_step = 0
		self.reward_planner_depreciation_step = 0
		self.reward_planner_flexibility_step = 0
		self.reward_planner_taxes_step = 0
		self.taxes_step = 0
		self.taxes_agents = np.zeros([self.agents_g])

		self.energy_not_served_step = 0
		
		
		self.taxes_tech_existing_step = np.zeros([self.agents_g, self.n_tech])
		self.taxes_tech_merchant_step = np.zeros([self.agents_g, self.n_tech])
		self.taxes_tech_cm_step = np.zeros([self.agents_g, self.n_tech])
		self.taxes_tech_CfD_step = np.zeros([self.agents_g, self.n_tech])

		self.taxes_tech_merchant_battery_step = np.zeros([self.agents_g, self.n_tech_battery])
		self.taxes_tech_cm_battery_step = np.zeros([self.agents_g, self.n_tech_battery])
		self.taxes_tech_flexibility_battery_step = np.zeros([self.agents_g, self.n_tech_battery])
		self.taxes_tech_existing_storage_lt_step = np.zeros([self.agents_g])

		self.tax_rent_step = 0

		# Normalization of rewards

		self.total_production_tech = np.zeros([self.n_tech])

		# Emissions

		self.CO2_emissions_step = 0

		# Small constant 

		delta = 0.001 

		# Variables for correct stepping 
		info = {a: {} for a in self.agents}
		terminated = {a: False for a in self.agents}
		truncated = {a: False for a in self.agents}

		# Hourly prices

		self.hourly_prices_net = np.zeros(self.short_t)

		# Collecting actions - GENCOs
		actions_g = []
		for i in self.possible_agents_g:
			actions_g.append(action_dict[i])

		# Collecting actions - Planner

		actions_p = []
		for i in self.possible_agents_p:
			actions_p.append(action_dict[i])

		action_temp = np.zeros([self.agents_n,  self.n_act_g])

		action_temp_p = np.zeros([self.agents_p,  self.n_act_p])

		# Discount Factor

		self.discount_factor = 1/((1 + self.opportunity_cost) ** self.year)
		self.discount_factor_merchant =  1/((1 + (self.opportunity_cost_merchant)) ** self.year)
		self.discount_factor_cm = self.discount_factor
		self.discount_factor_CfD = self.discount_factor
		self.discount_factor_flexibility = self.discount_factor

		# Discount factor planner

		self.discount_factor_planner = 1/((1 + self.opportunity_cost_planner) ** self.year)
			
		reward_temp = np.zeros(self.agents_g)

		## Populating observations and rewards for the first time 
	 
	 	# Observations - GENCOs
		 
		obs_temp_g = np.zeros([self.agents_g, self.n_obs_g])

		observations_temp_g = OrderedDict()
		observations_temp_g = {"observations": np.float32(obs_temp_g[0,:]),
						"action_mask":self.get_action_masks_g(self.investments_enabled[0,:], self.entry_enabled[0], self.capacity_existing_storage_lt[0], self.inv_g_aging[0,:])} 

		observations_g = {a: observations_temp_g for a in self.possible_agents_g}
		rewards_g = {a: 0 for a in self.possible_agents_g}

		# Observations - Planner 

		obs_temp_p = np.zeros([self.agents_p, self.n_obs_p])

		observations_temp_p = OrderedDict()

		observations_temp_p = {"observations": np.float32(obs_temp_p[0,:]),
						"action_mask":self.get_action_masks_p()} 

		observations_p = {a: observations_temp_p for a in self.possible_agents_p}
		rewards_p = {a: 0 for a in self.possible_agents_p}

		## Applying actions from central planner (only if the default action during masking was not selected)

		action_temp_p[0,:] = actions_p[0] 

		## CfD market

		# CfD Target

		action_CfD_target = (action_temp_p[0, 0] - 1)/(self.step_planner - 2)

		if action_temp_p[0, 0]  > 0:
			
			self.CfD_increment_target = self.CfD_target_increments_steps[int(action_temp_p[0, 0] - 1)]

		else:
			# Negative to ensure no auction
			#self.CfD_increment_target = -5
			pass

		## CM Market
		
		self.cm_demand_target = 0.00008

		## Flexibility market
		if action_temp_p[0, self.act_p_CfD_market] > 0:
			self.flexibility_increment_target = self.flexibility_target_increments_steps[int(action_temp_p[0, self.act_p_CfD_market] - 1)]
		else:
			pass
			#self.flexibility_increment_target = -5  # no-op: no 
			
		## Capacity market auction

		p_g_cm = np.zeros([self.agents_g, self.act_bids_cm])
		q_g_cm = np.zeros([self.agents_g, self.act_bids_cm])

		## Capacity market auction (in case auction indicator was activated in the last period)

		dummy_cm = np.zeros(self.act_bids_cm)

		if self.cm_auction_indicator and self.month == 3 and self.cm_activation == True:

			# Looping through agents and technologies to get quantity and price bids to the markets

			i = 0

			for a in self.possible_agents_g:

				action_temp[i,:] = actions_g[i]  

				for j in range(self.act_bids_cm):

					## Price bids for capacity market

					action_temp_price = action_temp[i, self.act_inv + j]
					
					if action_temp_price > 0:
					
						p_g_cm[i,j] = (action_temp_price - 1)/(self.step_g_bids - 2) * self.cm_price_cap
					
					# Selecting maximum investment

					action_tmp_investment = int(action_temp[i, self.act_inv + self.act_bids_cm + j])

					if action_tmp_investment > 0:

						if j < self.n_tech: 

							q_g_cm[i, j] = (self.inv_max[action_tmp_investment - 1, j] * self.cm_tech_cc[0,j])
							
						else: 

							q_g_cm[i, j] = (self.inv_max_battery[action_tmp_investment - 1, j - self.n_tech] * self.cm_tech_cc[0,j])

					else:
					
						q_g_cm[i, j] = 0
										
				i += 1

			p_g_cm = np.where(p_g_cm < 0, 0, p_g_cm)

			q_g_cm = np.where(q_g_cm < 0, 0, q_g_cm)

			p_cm_demand = np.ones(2) * self.cm_price_cap

			q_cm_demand = np.ones(2) * max(self.cm_balance, 0)/2

			q_g_cm_flatt = q_g_cm.flatten()

			p_g_cm_flatt = p_g_cm.flatten()
	
			# Running double-sided auction (Capacity Market)
						
			cm_price_flatt, q_accepted_u_cm, q_accepted_g_cm_flatt, self.cm_scarcity = self.double_side_auction_pay_as_bid(p_cm_demand,
																				  q_cm_demand, p_g_cm_flatt, q_g_cm_flatt, self.random_g_cm, self.cm_price_cap)
			
			q_accepted_g_cm = q_accepted_g_cm_flatt.reshape((self.agents_g, self.act_bids_cm))

			self.cm_price = cm_price_flatt.reshape((self.agents_g, self.act_bids_cm))

			# Price for existing is adjusted

			total_accepted = np.sum(q_accepted_g_cm)

			## If price was defined it is calculated, otherwise constant from previous iteration

			if total_accepted > delta:
				self.cm_price_existing = np.sum(self.cm_price * q_accepted_g_cm) / total_accepted / 2

			i = 0

			for a in self.possible_agents_g:

				inv_under_construction = self.under_construction.get(a)

				for j in range(self.act_bids_cm):

					# Vector for new investment (if action was taken to invest). All winning projects are built

					cc_inv_temp = q_accepted_g_cm[i, j]
					cc_inv_total = q_g_cm[i, j]/(self.cm_tech_cc[0,j] + delta)

					if j < self.n_tech: 

						construction_time = self.construction_time[j].copy()
						inv_cost = self.inv_cost[self.year_int, j]/(construction_time * self.yearly_resolution)

					else: 

						construction_time = self.construction_time_battery[j - self.n_tech].copy()
						inv_cost = self.inv_cost_battery[self.year_int, j - self.n_tech]/(construction_time * self.yearly_resolution)

					if cc_inv_temp > 0 and self.max_year - self.year >  construction_time and self.cm_tech_cc[0,j] > 0.02:

						dummy_cm[j] += cc_inv_total

						inv_vector = np.zeros([1,6])
						inv_vector[0,0] = j
						inv_vector[0,1] = cc_inv_total  
						inv_vector[0,2] = construction_time
						inv_vector[0,3] = 1
						inv_vector[0,4] = self.cm_price[i,j]
						inv_vector[0,5] = cc_inv_total * inv_cost / self.normalization_factor

						inv_under_construction = np.vstack([inv_under_construction, inv_vector])

						if j < self.n_tech: 
						
							# Firm capacity for assets under construction is updated
								
							self.cm_fc_under_construction[j] += cc_inv_total

							# Accumulated investment from capacity market

							self.capacity_cm_uc[i,j] += cc_inv_total

							# Accumulated costs of investment in a particular technology

							self.inv_cost_accum_cm[i,j] += inv_vector[0,5]
						
						else:

							# Storage capacity for capacity market estimation

							self.cm_fc_under_construction_battery_cm[j - self.n_tech] += cc_inv_total

							# Accumulated cm investment

							self.capacity_cm_battery_uc[i, j - self.n_tech] += cc_inv_total

							# Accumulated costs of investment in a particular technology

							self.inv_cost_accum_cm_battery[i, j - self.n_tech] += inv_vector[0,5]

				# Updating 
					
				self.under_construction.update({a:inv_under_construction})

				i += 1


		elif not self.cm_auction_indicator and self.month == 3:

			self.cm_price = 0

		## Calculation of capacity market indicators for existing assets

		if self.cm_activation and self.month == 3:

			i = 0

			for a in self.possible_agents_g:

				for j in range(self.n_tech):

					# Quantities for the option

					self.cm_inv_options_existing[i,j] = self.inv_g_aging[i,j] * self.cm_tech_cc[0,j]

					# Income for the option 

					self.cm_income_existing[i,j] = self.inv_g_aging[i,j] * self.cm_tech_cc[0,j] * self.cm_price_existing * self.short_t * self.hour_month / self.normalization_factor

				i += 1

		## Contrafs for Difference auctions

		p_g_CfD = np.zeros([self.agents_g, self.act_bids_CfD])
		q_g_CfD = np.zeros([self.agents_g, self.act_bids_CfD])

		## Capacity market auction (in case auction indicator was activated in the last period)

		dummy_CfD = np.zeros([self.act_bids_CfD])

		if self.CfD_auction_indicator and self.month == 5 and self.CfD_activation == True:

			# Looping through agents and technologies to get quantity and price bids to the markets

			i = 0

			for a in self.possible_agents_g:

				action_temp[i,:] = actions_g[i]  

				for j in range(self.n_tech_RES):
					
					## Price bids for CfD market

					# Normal bids

					action_temp_price = action_temp[i, self.act_inv + self.act_inv_cm + self.act_bids_cm + j]

					if action_temp_price > 0: 

						p_g_CfD[i, j] = (action_temp_price - 1)/(self.step_g_bids - 2) * self.CfD_price_cap

					# Quantity bids for CfD market

					action_tmp_investment = int(action_temp[i, self.act_inv + self.act_inv_cm + self.act_bids_cm + self.act_bids_CfD + j])
					
					if action_tmp_investment > 0:
						
						q_g_CfD[i, j] = (self.inv_max[action_tmp_investment - 1, j] * self.availability_tech_average_yearly[self.year_int, j] * self.average_failures[0,j])
					
					else:

						q_g_CfD[i, j] = 0
				
				i += 1

			p_g_CfD = np.where(p_g_CfD < 0, 0, p_g_CfD)

			q_g_CfD = np.where(q_g_CfD < 0, 0, q_g_CfD)

			p_CfD_demand = np.ones(2) * self.CfD_price_cap

			q_CfD_demand = np.ones(2) * max(self.CfD_balance, 0)/2

			p_g_CfD_flatt = p_g_CfD.flatten()

			q_g_CfD_flatt = q_g_CfD.flatten()

			self.supply = np.sum(q_g_CfD)

			self.demand_CfD = np.sum(q_CfD_demand)

			self.delta_auction = np.sum(q_CfD_demand) - np.sum(q_g_CfD)
						
			CfD_price_flatt, q_accepted_u_CfD, q_accepted_g_CfD_flatt, self.CfD_scarcity = self.double_side_auction_pay_as_bid(p_CfD_demand,
																				  q_CfD_demand, p_g_CfD_flatt, q_g_CfD_flatt, self.random_g_CfD, self.CfD_price_cap)
									
			self.CfD_price = CfD_price_flatt.reshape((self.agents_g, self.act_bids_CfD))

			q_accepted_g_CfD = q_accepted_g_CfD_flatt.reshape((self.agents_g, self.act_bids_CfD))

			i = 0
			
			for a in self.possible_agents_g:

				inv_under_construction = self.under_construction.get(a)

				for j in range(self.n_tech_RES):

					# Vector for new investment (if action was taken to invest). All winning projects are built

					CfD_inv_temp = q_accepted_g_CfD[i,j]
					CfD_inv_total = q_g_CfD[i, j] / (self.availability_tech_average_yearly[self.year_int, j] * self.average_failures[0,j] + delta)
					
					# Construction time and investment costs

					construction_time = self.construction_time[j].copy()
					inv_cost = self.inv_cost[self.year_int, j]/(construction_time * self.yearly_resolution)

					if CfD_inv_temp > 0 and self.max_year - self.year >  construction_time:

						dummy_CfD[j] += CfD_inv_total

						inv_vector = np.zeros([1,6])
						inv_vector[0,0] = j
						inv_vector[0,1] = CfD_inv_total  
						inv_vector[0,2] = construction_time
						inv_vector[0,3] = 2
						inv_vector[0,4] = self.CfD_price[i,j]
						inv_vector[0,5] = CfD_inv_total * inv_cost / self.normalization_factor

						inv_under_construction = np.vstack([inv_under_construction, inv_vector])

						# Firm capacity for assets under construction is updated
							
						self.cm_fc_under_construction[j] += CfD_inv_total

						# Accumulated investment from capacity market

						self.capacity_CfD_uc[i,j] += CfD_inv_total

						# Accumulated costs of investment in a particular technology

						self.inv_cost_accum[i,j] += inv_vector[0,5]
						self.inv_cost_accum_CfD[i,j] += inv_vector[0,5]

				# Updating 
					
				self.under_construction.update({a:inv_under_construction})

				i += 1
			
		elif not self.CfD_auction_indicator and self.month == 5:

			self.CfD_price = 0

		## Flexibility auctions

		p_g_flexibility = np.zeros([self.agents_g, self.act_bids_flexibility])
		q_g_flexibility = np.zeros([self.agents_g, self.act_bids_flexibility])

		if self.flexibility_auction_indicator and self.month == 5 and self.flexibility_activation == True:

			# Looping through agents and technologies to get quantity and price bids to the markets

			i = 0

			for a in self.possible_agents_g:

				action_temp[i,:] = actions_g[i]  

				for j in range(self.n_tech_battery):

					action_temp_price = action_temp[i, self.act_inv + self.act_inv_cm + self.act_bids_cm + self.act_inv_CfD + self.act_bids_CfD + j]
					
					if action_temp_price > 0: 

						p_g_flexibility[i, j] = (action_temp_price - 1)/(self.step_g_bids - 2) * self.flexibility_price_cap

					# Quantity bids for flexibility

					action_tmp_investment = int(action_temp[i, self.act_inv + self.act_inv_cm + self.act_bids_cm + self.act_inv_CfD + self.act_bids_CfD + self.act_bids_flexibility + j])
						
					if action_tmp_investment > 0:
							
						q_g_flexibility[i, j] = self.inv_max_battery[action_tmp_investment - 1, j]
						
					else:

						q_g_flexibility[i, j] = 0
				
				i += 1

			p_g_flexibility = np.where(p_g_flexibility < 0, 0, p_g_flexibility)

			q_g_flexibility = np.where(q_g_flexibility < 0, 0, q_g_flexibility)

			p_flexibility_demand = np.ones(2) * self.flexibility_price_cap

			q_flexibility_demand = np.ones(2) * max(self.flexibility_balance, 0)/2

			p_g_flexibility_flatt = p_g_flexibility.flatten()

			q_g_flexibility_flatt = q_g_flexibility.flatten()
						
			flexibility_price_flatt, q_accepted_u_flexibility, q_accepted_g_flexibility_flatt, self.flexibility_scarcity = self.double_side_auction_pay_as_bid(p_flexibility_demand,
																				  q_flexibility_demand, p_g_flexibility_flatt, q_g_flexibility_flatt, self.random_g_flexibility, self.flexibility_price_cap)
									
			self.flexibility_price = flexibility_price_flatt.reshape((self.agents_g, self.act_bids_flexibility))

			q_accepted_g_flexibility = q_accepted_g_flexibility_flatt.reshape((self.agents_g, self.act_bids_flexibility))

			i = 0
			
			for a in self.possible_agents_g:

				inv_under_construction = self.under_construction.get(a)

				for j in range(self.n_tech_battery):

					flexibility_inv_temp = q_accepted_g_flexibility[i, j]
					flexibility_inv_total = q_g_flexibility[i, j] 

					construction_time = self.construction_time_battery[j].copy()
					inv_cost = self.inv_cost_battery[self.year_int, j]/(construction_time * self.yearly_resolution)

					if flexibility_inv_temp > 0 and self.max_year - self.year >  construction_time:

						inv_vector = np.zeros([1,6])
						inv_vector[0,0] = j + self.n_tech
						inv_vector[0,1] = flexibility_inv_total  
						inv_vector[0,2] = construction_time
						inv_vector[0,3] = 3
						inv_vector[0,4] = self.flexibility_price[i,j]
						inv_vector[0,5] = flexibility_inv_total * inv_cost / self.normalization_factor

						inv_under_construction = np.vstack([inv_under_construction, inv_vector])

						# Firm capacity for assets under construction is updated

						self.cm_fc_under_construction_battery_flexibility[j] += flexibility_inv_total

						# Accumulated investment from capacity market

						self.capacity_flexibility_battery_uc[i, j] += flexibility_inv_total

						# Accumulated costs of investment in a particular technology

						self.inv_cost_accum_flexibility_battery[i, j] += inv_vector[0,5]

				# Updating 
						
				self.under_construction.update({a:inv_under_construction})

				i += 1
			
		elif not self.flexibility_auction_indicator and self.month == 5:

			self.flexibility_price = 0

		## Merchant investments

		# In case the period corresponds to investments, under constructions assets are checked. Otherwise section is ignored

		i = 0

		m_inv_temp = np.zeros([self.agents_g, self.act_inv])

		if self.month == 1 and self.merchant_activation == True:

			for a in self.possible_agents_g:

				inv_under_construction = self.under_construction.get(a)
					
				action_temp[i,:] = actions_g[i]  

				# Generation assets bids

				for j in range(self.act_inv):

					action_tmp_investment = int(action_temp[i,j])
					
					if action_temp[i,j] > 0:

						# Checking if the investment is a generation asset

						if j  < self.n_tech:

							m_inv_temp[i, j] = self.inv_max[action_tmp_investment - 1, j] 

							construction_time = self.construction_time[j].copy()
							inv_cost = self.inv_cost[self.year_int, j]/(construction_time * self.yearly_resolution)

						else:

							m_inv_temp[i, j] = self.inv_max_battery[action_tmp_investment - 1, j - self.n_tech]

							construction_time = self.construction_time_battery[j - self.n_tech].copy()
							inv_cost = self.inv_cost_battery[self.year_int, j - self.n_tech]/(construction_time * self.yearly_resolution)

						# Vector for new investment (if action was taken to invest)
							
						if m_inv_temp[i, j] > 0 and self.max_year - self.year >  construction_time:

							inv_vector = np.zeros([1,6])
							inv_vector[0,0] = j
							inv_vector[0,1] = m_inv_temp[i, j]  
							inv_vector[0,2] = construction_time
							inv_vector[0,3] = 0
							inv_vector[0,4] = 0
							inv_vector[0,5] = m_inv_temp[i, j] * inv_cost / self.normalization_factor

							inv_under_construction = np.vstack([inv_under_construction, inv_vector])

							if j  < self.n_tech:

								# Firm capacity for assets under construction is updated

								self.cm_fc_under_construction[j] += m_inv_temp[i, j]

								# Accumulated merchant investment

								self.capacity_merchant_uc[i,j] += m_inv_temp[i, j]

								# Accumulated costs of investment in a particular technology

								self.inv_cost_accum_merchant[i,j] += inv_vector[0,5]

							else:

								self.cm_fc_under_construction_battery_merchant[j - self.n_tech] += m_inv_temp[i, j]

								# Accumulated merchant investment

								self.capacity_merchant_battery_uc[i, j - self.n_tech] += m_inv_temp[i, j]

								# Accumulated costs of investment in a particular technology

								self.inv_cost_accum_merchant_battery[i, j - self.n_tech] += inv_vector[0,5]

				# Updating 
						
				self.under_construction.update({a:inv_under_construction})

				i += 1

		## Decom of existing aasests

		# In case the period corresponds to investments, under constructions assets are checked. Otherwise section is ignored

		i = 0

		m_decom_temp = np.zeros([self.agents_g, self.act_decom])

		if self.month == 1 and self.decom_activation == True:

			for a in self.possible_agents_g:

				inv_under_construction = self.under_construction.get(a)
					
				action_temp[i,:] = actions_g[i]  

				# Generation assets bids

				for j in range(self.act_decom):
					
					action_tmp_investment = int(action_temp[i, self.act_inv + self.act_inv_cm + self.act_bids_cm + self.act_inv_CfD + self.act_bids_CfD + self.act_bids_flexibility + self.act_inv_flexibility + j])
					
					if action_tmp_investment > 0:

						## Decomissioning allowed only in half the size of investments. Otherwise agents decomission too early in the cycle
						
						m_decom_temp[i, j] = self.inv_max[action_tmp_investment - 1, self.n_tech_decom_offset + j]/2
						
						decom_time = self.decom_time[j]
						decom_cost = self.decom_cost[self.year_int, j]/(decom_time * self.yearly_resolution)

						# Vector for new investment (if action was taken to invest)
							
						if m_decom_temp[i, j] > 0 and self.max_year - self.year >  decom_time:

							inv_vector = np.zeros([1,6])
							inv_vector[0,0] = self.n_tech_decom_offset + j 
							inv_vector[0,1] = m_decom_temp[i, j]  
							inv_vector[0,2] = decom_time
							inv_vector[0,3] = 4
							inv_vector[0,4] = 0
							inv_vector[0,5] = m_decom_temp[i, j] * decom_cost / self.normalization_factor

							inv_under_construction = np.vstack([inv_under_construction, inv_vector])

							# Firm capacity for assets under construction is updated

							self.cm_fc_under_construction[self.n_tech_decom_offset + j] -= m_decom_temp[i, j]

							# Accumulated merchant investment

							self.capacity_decom_uc[i, self.n_tech_decom_offset + j] += m_decom_temp[i, j]

							# Accumulated costs of investment in a particular technology

							self.inv_cost_accum_decom[i, self.n_tech_decom_offset + j] += inv_vector[0,5]

				# Updating 
						
				self.under_construction.update({a:inv_under_construction})

				i += 1
				
		## Construction of assets
		
		i = 0

		for a in self.possible_agents_g:

			rows_to_delete = []	

			# Get the investments under construction and append new investment

			inv_under_construction = self.under_construction.get(a)

			inv_under_construction[:, 2] -= 1/self.yearly_resolution

			k = 0
							
			for vector_under_construction in inv_under_construction:

				if inv_under_construction[k, 2] <= 0 and int(vector_under_construction[0]) < self.n_tech:

					capacity_under_construction = vector_under_construction[1].copy()	

					# In case construction has finished, the corresponding investments are added to effective investments (per technology)

					## Capacity Market

					if vector_under_construction[3] == 1:

						self.inv_g[i, int(vector_under_construction[0])] += vector_under_construction[1]

						self.cm_fc_under_construction[int(vector_under_construction[0])] -= capacity_under_construction

						self.inv_age_cm[i,int(vector_under_construction[0])] = self.inv_age_cm[i,int(vector_under_construction[0])] * self.capacity_cm[i,int(vector_under_construction[0])]/(vector_under_construction[1] + self.capacity_cm[i,int(vector_under_construction[0])] + delta)

						self.cm_income[i,int(vector_under_construction[0])] += vector_under_construction[1] * vector_under_construction[4] * self.cm_tech_cc[0, int(vector_under_construction[0])] * self.short_t * self.hour_month / self.normalization_factor

						self.cm_inv_options[i, int(vector_under_construction[0])] += vector_under_construction[1] * self.cm_tech_cc[0, int(vector_under_construction[0])] 

						self.capacity_cm[i,int(vector_under_construction[0])] += vector_under_construction[1]

						self.inv_cost_accum_cm[i,int(vector_under_construction[0])] -= vector_under_construction[5]

					## CfD
						
					elif vector_under_construction[3] == 2:

						self.inv_g[i, int(vector_under_construction[0])] += vector_under_construction[1]

						self.cm_fc_under_construction[int(vector_under_construction[0])] -= capacity_under_construction

						self.inv_age_CfD[i,int(vector_under_construction[0])] = self.inv_age_CfD[i,int(vector_under_construction[0])] * self.capacity_CfD[i,int(vector_under_construction[0])]/(vector_under_construction[1] + self.capacity_CfD[i,int(vector_under_construction[0])] + delta)

						self.capacity_CfD[i,int(vector_under_construction[0])] += vector_under_construction[1]

						self.CfD_price_pond[i,int(vector_under_construction[0])] += vector_under_construction[1] * vector_under_construction[4]

						self.CfD_price_agents[i,int(vector_under_construction[0])] = self.CfD_price_pond[i,int(vector_under_construction[0])]/(self.capacity_CfD[i,int(vector_under_construction[0])] + delta)

						self.inv_cost_accum_CfD[i,int(vector_under_construction[0])] -= vector_under_construction[5]

					## Merchant
						
					elif vector_under_construction[3] == 0:

						self.inv_g[i, int(vector_under_construction[0])] += vector_under_construction[1]

						self.cm_fc_under_construction[int(vector_under_construction[0])] -= capacity_under_construction

						self.inv_age_merchant[i,int(vector_under_construction[0])] = self.inv_age_merchant[i,int(vector_under_construction[0])] * self.capacity_merchant[i,int(vector_under_construction[0])]/(vector_under_construction[1] + self.capacity_merchant[i,int(vector_under_construction[0])] + delta)

						self.capacity_merchant[i,int(vector_under_construction[0])] += vector_under_construction[1]

						self.inv_cost_accum_merchant[i,int(vector_under_construction[0])] -= vector_under_construction[5]

					## Decom
						
					elif vector_under_construction[3] == 4:

						capacity_tmp = self.inv_g_aging[i, int(vector_under_construction[0])] - vector_under_construction[1]

						# Existing capacity remaining to be decomissioned

						if capacity_tmp > 0:

							self.inv_g_aging[i, int(vector_under_construction[0])] -= vector_under_construction[1]
							
							self.inv_g[i, int(vector_under_construction[0])] -= vector_under_construction[1]

							self.cm_fc_under_construction[int(vector_under_construction[0])] += vector_under_construction[1]

							self.capacity_decom[i,int(vector_under_construction[0])] += vector_under_construction[1]

						# Maximum existing capacity decomissioned
						
						else:
							
							remaining_aging = self.inv_g_aging[i, int(vector_under_construction[0])].copy()
							
							self.inv_g[i, int(vector_under_construction[0])] -= remaining_aging
							
							self.inv_g_aging[i, int(vector_under_construction[0])] = 0
							
							self.cm_fc_under_construction[int(vector_under_construction[0])] += remaining_aging
							
							self.capacity_decom[i,int(vector_under_construction[0])] += remaining_aging
							
						self.inv_cost_accum_decom[i,int(vector_under_construction[0])] -= vector_under_construction[5]

					else:
						raise ValueError(f"Error in resource construction, value in vector {vector_under_construction[3]}")

					rows_to_delete.append(k)

				elif inv_under_construction[k, 2] <= 0 and int(vector_under_construction[0]) >= self.n_tech:

					# Check for projects dying

					capacity_under_construction = vector_under_construction[1].copy()	
					
					# In case construction has finished, the corresponding investments are added to effective investments (per technology)

						# Adding income for new plants coming from the Capacity Mechanism (in monthly resolution)
						
					if vector_under_construction[3] == 0:

						self.cm_fc_under_construction_battery_merchant[int(vector_under_construction[0]) - self.n_tech] -= capacity_under_construction

						self.inv_age_merchant_battery[i,int(vector_under_construction[0]) - self.n_tech] = self.inv_age_merchant_battery[i,int(vector_under_construction[0]) - self.n_tech] * self.capacity_merchant_battery[i, int(vector_under_construction[0]) - self.n_tech]/(vector_under_construction[1] + self.capacity_merchant_battery[i, int(vector_under_construction[0]) - self.n_tech] + delta)

						self.capacity_merchant_battery[i,int(vector_under_construction[0]) - self.n_tech] += vector_under_construction[1]

						self.inv_cost_accum_merchant_battery[i,int(vector_under_construction[0]) - self.n_tech] -= vector_under_construction[5]
						
					elif vector_under_construction[3] == 1:

						self.cm_income_battery[i, int(vector_under_construction[0]) - self.n_tech] += vector_under_construction[1] * vector_under_construction[4] * self.cm_tech_cc[0, int(vector_under_construction[0])] * self.short_t * self.hour_month / self.normalization_factor

						self.cm_inv_options_battery[i, int(vector_under_construction[0]) - self.n_tech] += vector_under_construction[1] * self.cm_tech_cc[0, int(vector_under_construction[0])] 

						self.inv_age_cm_battery[i, int(vector_under_construction[0]) - self.n_tech] = self.inv_age_cm_battery[i, int(vector_under_construction[0]) - self.n_tech] * self.capacity_cm_battery[i, int(vector_under_construction[0]) - self.n_tech]/(vector_under_construction[1] + self.capacity_cm_battery[i, int(vector_under_construction[0]) - self.n_tech] + delta)

						self.capacity_cm_battery[i, int(vector_under_construction[0]) - self.n_tech] += vector_under_construction[1]

						self.inv_cost_accum_cm_battery[i, int(vector_under_construction[0]) - self.n_tech] -= vector_under_construction[5]

						self.cm_fc_under_construction_battery_cm[int(vector_under_construction[0]) - self.n_tech] -= capacity_under_construction

					elif vector_under_construction[3] == 3:

						self.inv_age_flexibility_battery[i, int(vector_under_construction[0]) - self.n_tech] = self.inv_age_flexibility_battery[i, int(vector_under_construction[0]) - self.n_tech] * self.capacity_flexibility_battery[i, int(vector_under_construction[0]) - self.n_tech]/(vector_under_construction[1] + self.capacity_flexibility_battery[i, int(vector_under_construction[0]) - self.n_tech] + delta)

						self.capacity_flexibility_battery[i, int(vector_under_construction[0]) - self.n_tech] += vector_under_construction[1]

						self.cm_fc_under_construction_battery_flexibility[int(vector_under_construction[0]) - self.n_tech] -= capacity_under_construction

						self.flexibility_price_pond[i, int(vector_under_construction[0]) - self.n_tech] += vector_under_construction[1] * vector_under_construction[4]

						self.flexibility_price_agents[i, int(vector_under_construction[0]) - self.n_tech] = self.flexibility_price_pond[i, int(vector_under_construction[0]) - self.n_tech]/(self.capacity_flexibility_battery[i, int(vector_under_construction[0]) - self.n_tech] + delta) 

						self.inv_cost_accum_flexibility_battery[i, int(vector_under_construction[0]) - self.n_tech] -= vector_under_construction[5]

					rows_to_delete.append(k)

				k += 1
						
				# Deleting rows for constructed projects
					
			inv_under_construction = np.delete(inv_under_construction, rows_to_delete, axis=0)

			# Under construction dict
					
			self.under_construction.update({a:inv_under_construction})
					
			i += 1 
		
		## Capacity Market - Deficit calculation

		if self.month == 5:

			self.cm_balance, self.cm_auction_indicator, self.cm_tech_cc, self.hourly_balance, self.cm_balance_real = self.elcc_calculator.cm_balance_estimation()
			## TODO
			self.cm_tech_cc = np.nan_to_num(self.cm_tech_cc, nan=0.0, posinf=0.0, neginf=0.0)

		## CfD - Deficit calculation

		if self.month == 0:

			self.CfD_balance, self.CfD_auction_indicator, self.CfD_balance_real, self.CfD_target_penetration = self.CfD_balance_estimation()

		## Flexibility - Deficit calculation

		if self.month == 0:

			self.flexibility_balance, self.flexibility_auction_indicator, self.flexibility_balance_real, self.flexibility_target_penetration = self.flexibility_balance_estimation()

		## SoC target for storage

		i = 0

		SoC_target = np.zeros([self.agents_g])

		SoC_target_daily = np.zeros([self.agents_g])

		adjusted_inflow = np.zeros([self.agents_g])

		self.adjusted_inflow_charge = np.zeros([self.agents_g])

		self.adjusted_inflow_discharge = np.zeros([self.agents_g])

		self.storage_actions = np.zeros([self.agents_g])

		for a in self.possible_agents_g:

			action_temp[i,:] = actions_g[i]  
			
			SoC_target[i] = (0.2 + (action_temp[i, self.act_inv + self.act_inv_cm + self.act_bids_cm + self.act_inv_CfD + self.act_bids_CfD + self.act_inv_flexibility + self.act_bids_flexibility + self.act_decom])/(self.step_SoC_control - 1) * 0.6) * self.SoC_max_merchant_storage_lt[i]

			SoC_target_daily[i] = (SoC_target[i] - self.SoC_merchant_storage_lt[i])/self.hour_month + self.SoC_merchant_storage_lt[i]

			self.storage_actions[i] = action_temp[i, self.act_inv + self.act_inv_cm + self.act_bids_cm + self.act_inv_CfD + self.act_bids_CfD + self.act_inv_flexibility + self.act_bids_flexibility + self.act_decom]

			i += 1

		# Short-term storage bids

		p_g = np.zeros([self.short_t, self.n_tech + 2])
		q_g = np.zeros([self.short_t, self.n_tech + 2])

		q_g_agents_share = np.zeros([self.short_t, self.agents_g, self.n_tech])

		demand_day = np.zeros(self.short_t)

		spot_payment_trimester = 0
		demand_trimester = 0

		carbon_tax_returns = 0
		
		inflows_storage_lt = np.zeros([self.short_t, self.agents_g])

		flexibility_cost = 0

		spot_payment_hour = np.zeros([self.short_t])

		energy_not_served_costs_step_tmp = 0
		energy_not_served_costs_step_hour_tmp = np.zeros([self.short_t])

		for t in range (self.short_t):

			## Capacity market costs (hourly)

			spot_payment_trimester += np.sum(self.cm_income) * self.normalization_factor/(self.hour_month * self.short_t)
			spot_payment_hour[t] += np.sum(self.cm_income) * self.normalization_factor/(self.hour_month * self.short_t)
			self.cost_cm_step += np.sum(self.cm_income) * self.normalization_factor/(self.hour_month * self.short_t)

			spot_payment_trimester += np.sum(self.cm_income_battery) * self.normalization_factor/(self.hour_month * self.short_t)
			spot_payment_hour[t] += np.sum(self.cm_income_battery) * self.normalization_factor/(self.hour_month * self.short_t)
			self.cost_cm_step += np.sum(self.cm_income_battery) * self.normalization_factor/(self.hour_month * self.short_t)

			spot_payment_trimester += np.sum(self.cm_income_existing) * self.normalization_factor/(self.hour_month * self.short_t)
			spot_payment_hour[t] += np.sum(self.cm_income_existing) * self.normalization_factor/(self.hour_month * self.short_t)
			self.cost_cm_step += np.sum(self.cm_income_existing) * self.normalization_factor/(self.hour_month * self.short_t)

			## Flexibility cost (hourly)
			i = 0

			for a in self.possible_agents_g:

				for j in range(self.n_tech_battery):

					flexibility_cost += self.flexibility_price_agents[i,j] * self.capacity_flexibility_battery[i,j]/(self.hour_month  * self.short_t)
					spot_payment_trimester += self.flexibility_price_agents[i,j] * self.capacity_flexibility_battery[i,j]/(self.hour_month  * self.short_t)
					spot_payment_hour[t] += self.flexibility_price_agents[i,j] * self.capacity_flexibility_battery[i,j]/(self.hour_month  * self.short_t)
					self.cost_Flexibility_step += self.flexibility_price_agents[i,j] * self.capacity_flexibility_battery[i,j]/(self.hour_month  * self.short_t)
					
				# Generation assets bids

				for j in range(self.n_tech):
					
					# q bids:
					
					q_g[t,j] += self.inv_g[i, j] * (self.availability_tech_step[t, j]) * self.aggregated_failures[i,j]
					q_g_agents_share[t,i,j] = self.inv_g[i, j] * (self.availability_tech_step[t, j]) * self.aggregated_failures[i,j]	

					# Aggregated variable cost (including carbon tax)

					if (self.scenario_tax_decree == 1 or self.scenario_tax_decree == 3) and j == self.n_tech - 1 and (self.year_int <= self.scenario_tax_decree_deactivation):
						
						aggregated_vc = ((self.v_c_g[self.year_int, j]) * self.shock_multiplier_techs[j])

					elif (self.scenario_tax_decree == 2) and (self.year_int <= self.scenario_tax_decree_deactivation):

						aggregated_vc = ((self.v_c_g[self.year_int, j]) * self.shock_multiplier_techs[j])

					else:
						
						aggregated_vc = ((self.v_c_g[self.year_int, j]) * self.shock_multiplier_techs[j]) + self.CO2_tech[j] * self.CO2_tax

					p_g[t,j] = (np.double(aggregated_vc))

				inflows_storage_lt[t,i] = (self.availability_tech_step[t, self.n_tech]) * self.capacity_existing_storage_lt[i]

				i += 1

			# Demand bid utilites (minimum 2 for the auction algorithm to work)
			
			demand_day[t] = self.demand_step[t] 
			
			# Bids run of the river
			
			p_g[t,self.n_tech] = 0
			
			q_g[t,self.n_tech] = self.availability_tech_step[t, self.n_tech + 1]

			# Bids others

			p_g[t, self.n_tech + 1] = 1.5
			
			q_g[t,self.n_tech + 1] = self.availability_tech_step[t, self.n_tech + 2]
		
		## Run optimization for daily dispatch

		short_term_storage_P = np.array([
    		np.sum(self.capacity_merchant_battery[:,0] + self.capacity_cm_battery[:,0] + self.capacity_flexibility_battery[:,0] + self.inv_init_battery[:,0]) + delta,
    		np.sum(self.capacity_merchant_battery[:,1] + self.capacity_cm_battery[:,1] + self.capacity_flexibility_battery[:,1] + self.inv_init_battery[:,1]) + delta
		])
		
		short_term_storage_E = short_term_storage_P * np.array([3, 8])
		short_term_storage_init = short_term_storage_E * 0.0

		#

		flexibility_credits_dispatched = np.array([0.0, 0.1, 0.12, 0.8, 0.9, 0.8, 0.1, 0.8])

		flexibility_credit_slack = np.array([0.9])
		flexibility_credits_short_term_storage = np.array([0.9, 0.9])
		flexibility_credits_long_term_storage = np.array([0.8])
		flexibility_demand = np.array([0.2])

		P_min_coal_SAR = 250.0

		# Call the optimization
		
		results = self.model.solve(p_g, q_g, demand_day, 
					short_term_storage_P, short_term_storage_E, short_term_storage_init, 0.95, 0.95,
					self.capacity_existing_storage_lt[self.index_agents_long_term_storage], self.SoC_max_merchant_storage_lt[self.index_agents_long_term_storage], SoC_target_daily[self.index_agents_long_term_storage], self.SoC_merchant_storage_lt[self.index_agents_long_term_storage], inflows_storage_lt[:,self.index_agents_long_term_storage], 0.95, 0.95, 
					self.VoLL_max, 
					flexibility_credits_dispatched, flexibility_credit_slack,
					flexibility_credits_short_term_storage, flexibility_credits_long_term_storage, flexibility_demand, P_min_coal_SAR)

		# Handling results 

		Q_dispatched_solution, charge_short_term_battery_solution, discharge_short_term_battery_solution, charge_long_term_battery_solution_tmp, discharge_long_term_battery_solution_tmp, price, Q_slack_solution, ll , SoC_short_term_battery_solution, SoC_long_term_battery_solution, dumping_long_term_battery_solution, price_flexibility, price_coal_must_run = results

		charge_long_term_battery_solution = np.zeros([self.short_t, self.agents_g])

		charge_long_term_battery_solution[:,self.index_agents_long_term_storage] = charge_long_term_battery_solution_tmp

		discharge_long_term_battery_solution = np.zeros([self.short_t, self.agents_g])

		discharge_long_term_battery_solution[:,self.index_agents_long_term_storage] = discharge_long_term_battery_solution_tmp

		# Aggregate RES energy and curtailed generation

		self.RES_curtailed_step = np.sum(q_g[:,0:3]) + np.sum(q_g[:,self.n_tech]) - np.sum(Q_dispatched_solution[:,0:3]) - np.sum(Q_dispatched_solution[:,self.n_tech])

		self.SoC_merchant_storage_lt = SoC_target

		self.hourly_prices = price + price_flexibility * flexibility_demand

		self.price_flexibility = price_flexibility

		self.SoC_short_term_storage = SoC_short_term_battery_solution

		self.discharge_short_term_battery = discharge_short_term_battery_solution

		self.charge_short_term_battery = charge_short_term_battery_solution

		self.energy_not_served_step = np.sum(Q_slack_solution)

		# Populating observations and rewards (GENCOs) after running dispatch

		for t in range (self.short_t):

			demand_trimester += demand_day[t]
			demand_hour = demand_day[t]
			
			# Energy cost
			spot_payment_trimester += price[t] * (demand_day[t] - Q_slack_solution[t])
			spot_payment_hour[t] += price[t] * (demand_day[t] - Q_slack_solution[t])

			# Temporal Energy not served costs

			energy_not_served_costs_step_tmp += self.VoLL_planner * Q_slack_solution[t]
			energy_not_served_costs_step_hour_tmp[t] += self.VoLL_planner * Q_slack_solution[t]
			self.cost_scarcity_step += self.VoLL_planner * Q_slack_solution[t]
			
			i = 0

			for a in self.possible_agents_g:

				q_accepted_u_short_term_merchant = np.zeros([self.n_tech_battery])
				q_accepted_u_short_term_cm = np.zeros([self.n_tech_battery])
				q_accepted_u_short_term_flexibility = np.zeros([self.n_tech_battery])
				q_accepted_u_short_term_merchant_only = np.zeros([self.n_tech_battery])
				q_accepted_u_short_term_existing_only = np.zeros([self.n_tech_battery])

				# Short-term battery accepted bids

				for j in range(self.n_tech_battery):

					q_accepted_u_short_term_merchant[j] =  (discharge_short_term_battery_solution[t,j] - charge_short_term_battery_solution[t,j]) * (self.capacity_merchant_battery[i,j] + self.inv_init_battery[i,j]) / (np.sum(self.capacity_merchant_battery[:,j] + self.capacity_cm_battery[:,j] + self.capacity_flexibility_battery[:,j] + self.inv_init_battery[:,j]) + delta)
					q_accepted_u_short_term_cm[j] =  (discharge_short_term_battery_solution[t,j] - charge_short_term_battery_solution[t,j]) * self.capacity_cm_battery[i,j] / (np.sum(self.capacity_merchant_battery[:,j] + self.capacity_cm_battery[:,j] + self.capacity_flexibility_battery[:,j] + self.inv_init_battery[:,j]) + delta)
					q_accepted_u_short_term_flexibility[j] =  (discharge_short_term_battery_solution[t,j] - charge_short_term_battery_solution[t,j]) * self.capacity_flexibility_battery[i,j] / (np.sum(self.capacity_merchant_battery[:,j] + self.capacity_cm_battery[:,j] + self.capacity_flexibility_battery[:,j] + self.inv_init_battery[:,j]) + delta)
					q_accepted_u_short_term_merchant_only[j] =  (discharge_short_term_battery_solution[t,j] - charge_short_term_battery_solution[t,j]) * (self.capacity_merchant_battery[i,j]) / (np.sum(self.capacity_merchant_battery[:,j] + self.capacity_cm_battery[:,j] + self.capacity_flexibility_battery[:,j] + self.inv_init_battery[:,j]) + delta)
					q_accepted_u_short_term_existing_only[j] =  (discharge_short_term_battery_solution[t,j] - charge_short_term_battery_solution[t,j]) * (self.inv_init_battery[i,j]) / (np.sum(self.capacity_merchant_battery[:,j] + self.capacity_cm_battery[:,j] + self.capacity_flexibility_battery[:,j] + self.inv_init_battery[:,j]) + delta)

				q_accepted_u_long_term =  discharge_long_term_battery_solution[t,i] - charge_long_term_battery_solution[t,i]	

				# Reward from technology 

				for j in range(self.n_tech):

					price_aggregated = price[t] + self.price_flexibility[t] * flexibility_credits_dispatched[j]

					q_accepted_g = q_g_agents_share[t,i,j]/(q_g[t,j] + delta) * Q_dispatched_solution[t,j]

					flexibility_cost += self.price_flexibility[t] * flexibility_credits_dispatched[j] * q_accepted_g
					spot_payment_trimester += self.price_flexibility[t] * flexibility_credits_dispatched[j] * q_accepted_g
					spot_payment_hour[t] += self.price_flexibility[t] * flexibility_credits_dispatched[j] * q_accepted_g
					
					self.cost_merchant_step += price_aggregated * q_accepted_g * (self.capacity_merchant[i,j])/(self.capacity_merchant[i,j] + self.capacity_cm[i,j] + self.capacity_CfD[i,j] + self.inv_g_aging[i,j] + delta) 
					self.cost_existing_step += price_aggregated * q_accepted_g * (self.inv_g_aging[i,j])/(self.capacity_merchant[i,j] + self.capacity_cm[i,j] + self.capacity_CfD[i,j] + self.inv_g_aging[i,j] + delta) 

					profit_temp_spot = 0

					if (self.scenario_tax_decree == 1 or self.scenario_tax_decree == 3) and j == self.n_tech - 1 and (self.year_int <= self.scenario_tax_decree_deactivation):
						
						profit_temp_spot = (((price_aggregated - 
					  (self.v_c_g[self.year_int, j] * self.shock_multiplier_techs[j])) * 
						q_accepted_g) - self.fixed_cost[self.year_int, j] * self.inv_g[i,j])/(self.normalization_factor) * self.hour_month

						spot_payment_trimester += q_accepted_g * self.CO2_tech[j] * self.CO2_tax

						carbon_tax_returns += q_accepted_g * self.CO2_tech[j] * self.CO2_tax

						self.cost_carbon_tax_return_step += q_accepted_g * self.CO2_tech[j] * self.CO2_tax

					elif (self.scenario_tax_decree == 2) and (self.year_int <= self.scenario_tax_decree_deactivation):
						
						profit_temp_spot = (((price_aggregated - 
					  (self.v_c_g[self.year_int, j] * self.shock_multiplier_techs[j]) - 
					  self.CO2_tech[j] * self.CO2_tax) * 
						q_accepted_g) - self.fixed_cost[self.year_int, j] * self.inv_g[i,j])/(self.normalization_factor) * self.hour_month

						spot_payment_trimester += q_accepted_g * self.CO2_tech[j] * self.CO2_social_cost[self.year_int]

						carbon_tax_returns += q_accepted_g * self.CO2_tech[j] * self.CO2_social_cost[self.year_int]

						self.cost_carbon_tax_return_step += q_accepted_g * self.CO2_tech[j] * self.CO2_social_cost[self.year_int]

					else:
						
						profit_temp_spot = (((price_aggregated - 
					  (self.v_c_g[self.year_int, j] * self.shock_multiplier_techs[j]) - 
					  self.CO2_tech[j] * self.CO2_tax) * 
						q_accepted_g) - self.fixed_cost[self.year_int, j] * self.inv_g[i,j])/(self.normalization_factor) * self.hour_month

					self.reward_tech_merchant[i,j] += profit_temp_spot * (self.capacity_merchant[i,j])/(self.capacity_merchant[i,j] + self.capacity_cm[i,j] + self.capacity_CfD[i,j] + self.inv_g_aging[i,j] + delta) * self.discount_factor_merchant 
					self.reward_tech_merchant_step[i,j] += profit_temp_spot * (self.capacity_merchant[i,j])/(self.capacity_merchant[i,j] + self.capacity_cm[i,j] + self.capacity_CfD[i,j] + self.inv_g_aging[i,j] + delta) * self.discount_factor_merchant 
					
					self.reward_tech_existing[i,j] += profit_temp_spot * (self.inv_g_aging[i,j])/(self.capacity_merchant[i,j] + self.capacity_cm[i,j] + self.capacity_CfD[i,j] + self.inv_g_aging[i,j] + delta) * self.discount_factor 
					self.reward_tech_existing_step[i,j] += profit_temp_spot * (self.inv_g_aging[i,j])/(self.capacity_merchant[i,j] + self.capacity_cm[i,j] + self.capacity_CfD[i,j] + self.inv_g_aging[i,j] + delta) * self.discount_factor 

					self.reward_tech_cm[i,j] += profit_temp_spot * self.capacity_cm[i,j]/(self.capacity_merchant[i,j] + self.capacity_cm[i,j] + self.capacity_CfD[i,j] + self.inv_g_aging[i,j] + delta) * self.discount_factor_cm 
					self.reward_tech_cm_step[i,j] += profit_temp_spot * self.capacity_cm[i,j]/(self.capacity_merchant[i,j] + self.capacity_cm[i,j] + self.capacity_CfD[i,j] + self.inv_g_aging[i,j] + delta) * self.discount_factor_cm 

					self.reward_tech_CfD[i,j] += profit_temp_spot * self.capacity_CfD[i,j]/(self.capacity_merchant[i,j] + self.capacity_cm[i,j] + self.capacity_CfD[i,j] + self.inv_g_aging[i,j] + delta) * self.discount_factor_CfD 
					self.reward_tech_CfD_step[i,j] += profit_temp_spot * self.capacity_CfD[i,j]/(self.capacity_merchant[i,j] + self.capacity_cm[i,j] + self.capacity_CfD[i,j] + self.inv_g_aging[i,j] + delta) * self.discount_factor_CfD 

					# Capacity Market Reliability option

					cm_option_value = np.maximum(price[t] - self.cm_strike, 0) * self.cm_inv_options[i,j] /(self.normalization_factor) * self.hour_month

					spot_payment_trimester -= np.maximum(price[t] - self.cm_strike, 0) * self.cm_inv_options[i,j]
					spot_payment_hour[t] -= np.maximum(price[t] - self.cm_strike, 0) * self.cm_inv_options[i,j]
					self.cost_cm_step -= np.maximum(price[t] - self.cm_strike, 0) * self.cm_inv_options[i,j]

					# Reliability Option for existing generators

					cm_option_value_existing = np.maximum(price[t] - self.cm_strike, 0) * self.cm_inv_options_existing[i,j] /(self.normalization_factor) * self.hour_month

					spot_payment_trimester -= np.maximum(price[t] - self.cm_strike, 0) * self.cm_inv_options_existing[i,j]
					spot_payment_hour[t] -= np.maximum(price[t] - self.cm_strike, 0) * self.cm_inv_options_existing[i,j]
					self.cost_cm_step -= np.maximum(price[t] - self.cm_strike, 0) * self.cm_inv_options_existing[i,j]

					# Contracts for Difference settlement

					CfD_option_value = (price[t] - self.CfD_price_agents[i,j]) * q_accepted_g * self.capacity_CfD[i,j]/(self.capacity_merchant[i,j] + self.capacity_cm[i,j] + self.capacity_CfD[i,j] + self.inv_g_aging[i,j] + delta)/self.normalization_factor * self.hour_month

					spot_payment_trimester -= (price[t] - self.CfD_price_agents[i,j]) * q_accepted_g * self.capacity_CfD[i,j]/(self.capacity_merchant[i,j] + self.capacity_cm[i,j] + self.capacity_CfD[i,j] + self.inv_g_aging[i,j] + delta)
					spot_payment_hour[t] -= (price[t] - self.CfD_price_agents[i,j]) * q_accepted_g * self.capacity_CfD[i,j]/(self.capacity_merchant[i,j] + self.capacity_cm[i,j] + self.capacity_CfD[i,j] + self.inv_g_aging[i,j] + delta)
					self.cost_CfD_step -= (price[t] - self.CfD_price_agents[i,j]) * q_accepted_g * self.capacity_CfD[i,j]/(self.capacity_merchant[i,j] + self.capacity_cm[i,j] + self.capacity_CfD[i,j] + self.inv_g_aging[i,j] + delta)

					# Emissions and production

					self.CO2_emissions_step += q_accepted_g * self.CO2_tech[j] * self.hour_month

					self.total_production_tech[j] += q_accepted_g * self.hour_month
					
					# Option reward

					self.reward_tech_cm[i,j] -= cm_option_value * self.discount_factor_cm 
					self.reward_tech_cm_step[i,j] -= cm_option_value * self.discount_factor_cm 

					self.reward_tech_CfD[i,j] -= CfD_option_value * self.discount_factor_CfD 
					self.reward_tech_CfD_step[i,j] -= CfD_option_value * self.discount_factor_CfD 

					self.reward_tech_existing[i,j] -= cm_option_value_existing * self.discount_factor_merchant
					self.reward_tech_existing_step[i,j] -= cm_option_value_existing * self.discount_factor_merchant

					if self.year > self.max_year - self.years_slack_profits:

						self.accumulated_profit_merchant[j] += profit_temp_spot * (self.capacity_merchant[i,j])/(self.capacity_merchant[i,j] + self.capacity_cm[i,j] + self.capacity_CfD[i,j] + self.inv_g_aging[i,j] + delta)
						
						self.accumulated_profit_cm[j] += profit_temp_spot * (self.capacity_cm[i,j])/(self.capacity_merchant[i,j] + self.capacity_cm[i,j] + self.capacity_CfD[i,j] + self.inv_g_aging[i,j] + delta)

						self.accumulated_profit_cm[j] -= cm_option_value 

						self.accumulated_profit_CfD[j] += profit_temp_spot * (self.capacity_CfD[i,j])/(self.capacity_merchant[i,j] + self.capacity_cm[i,j] + self.capacity_CfD[i,j] + self.inv_g_aging[i,j] + delta)

						self.accumulated_profit_CfD[j] -= CfD_option_value

				## Long-term storage
				
				income_storage_lt = (price[t] * (q_accepted_u_long_term) - self.fixed_cost_storage_lt[self.year_int] * self.capacity_existing_storage_lt[i])/(self.normalization_factor) * self.hour_month 	

				income_storage_lt += (self.price_flexibility[t] * flexibility_credits_long_term_storage * (discharge_long_term_battery_solution[t,i] + charge_long_term_battery_solution[t,i]))/ (self.normalization_factor) * self.hour_month

				flexibility_cost += (self.price_flexibility[t] * flexibility_credits_long_term_storage * (discharge_long_term_battery_solution[t,i] + charge_long_term_battery_solution[t,i]))	
				
				spot_payment_trimester += (self.price_flexibility[t] * flexibility_credits_long_term_storage * (discharge_long_term_battery_solution[t,i] + charge_long_term_battery_solution[t,i]))	
				spot_payment_hour[t] += (self.price_flexibility[t] * flexibility_credits_long_term_storage * (discharge_long_term_battery_solution[t,i] + charge_long_term_battery_solution[t,i]))	
								
				self.cost_existing_step += (price[t] * (q_accepted_u_long_term)) 	
				self.cost_existing_step += (self.price_flexibility[t] * flexibility_credits_long_term_storage * (discharge_long_term_battery_solution[t,i] + charge_long_term_battery_solution[t,i]))

				## run of the river

				income_storage_run_of_river = price[t] * Q_dispatched_solution[t,self.n_tech] * (self.capacity_existing_storage_lt[i]/np.sum(self.capacity_existing_storage_lt))/(self.normalization_factor) * self.hour_month 	

				flexibility_cost += self.price_flexibility[t] * flexibility_credits_dispatched[self.n_tech] * Q_dispatched_solution[t,self.n_tech] * (self.capacity_existing_storage_lt[i]/np.sum(self.capacity_existing_storage_lt))
				spot_payment_trimester += self.price_flexibility[t] * flexibility_credits_dispatched[self.n_tech] * Q_dispatched_solution[t,self.n_tech] * (self.capacity_existing_storage_lt[i]/np.sum(self.capacity_existing_storage_lt))
				spot_payment_hour[t] += self.price_flexibility[t] * flexibility_credits_dispatched[self.n_tech] * Q_dispatched_solution[t,self.n_tech] * (self.capacity_existing_storage_lt[i]/np.sum(self.capacity_existing_storage_lt))
				income_storage_run_of_river += self.price_flexibility[t] * flexibility_credits_dispatched[self.n_tech] * Q_dispatched_solution[t,self.n_tech] * (self.capacity_existing_storage_lt[i]/np.sum(self.capacity_existing_storage_lt))/(self.normalization_factor) * self.hour_month

				self.cost_existing_step += price[t] * Q_dispatched_solution[t,self.n_tech] * (self.capacity_existing_storage_lt[i]/np.sum(self.capacity_existing_storage_lt))
				self.cost_existing_step += self.price_flexibility[t] * flexibility_credits_dispatched[self.n_tech] * Q_dispatched_solution[t,self.n_tech] * (self.capacity_existing_storage_lt[i]/np.sum(self.capacity_existing_storage_lt))

				## Others

				income_others = price[t] * Q_dispatched_solution[t,self.n_tech + 1] * (self.capacity_existing_storage_lt[i]/np.sum(self.capacity_existing_storage_lt))/(self.normalization_factor) * self.hour_month 	

				flexibility_cost += self.price_flexibility[t] * flexibility_credits_dispatched[self.n_tech + 1] * Q_dispatched_solution[t,self.n_tech + 1] * (self.capacity_existing_storage_lt[i]/np.sum(self.capacity_existing_storage_lt))
				spot_payment_trimester += self.price_flexibility[t] * flexibility_credits_dispatched[self.n_tech + 1] * Q_dispatched_solution[t,self.n_tech + 1]  * (self.capacity_existing_storage_lt[i]/np.sum(self.capacity_existing_storage_lt))
				spot_payment_hour[t] += self.price_flexibility[t] * flexibility_credits_dispatched[self.n_tech + 1] * Q_dispatched_solution[t,self.n_tech + 1] * (self.capacity_existing_storage_lt[i]/np.sum(self.capacity_existing_storage_lt))
				income_others += self.price_flexibility[t] * flexibility_credits_dispatched[self.n_tech + 1] * Q_dispatched_solution[t,self.n_tech + 1] * (self.capacity_existing_storage_lt[i]/np.sum(self.capacity_existing_storage_lt))/(self.normalization_factor) * self.hour_month

				self.cost_existing_step += price[t] * Q_dispatched_solution[t,self.n_tech + 1] * (self.capacity_existing_storage_lt[i]/np.sum(self.capacity_existing_storage_lt))
				self.cost_existing_step += self.price_flexibility[t] * flexibility_credits_dispatched[self.n_tech + 1] * Q_dispatched_solution[t,self.n_tech + 1] * (self.capacity_existing_storage_lt[i]/np.sum(self.capacity_existing_storage_lt))

				# Emission from others

				self.CO2_emissions_step += Q_dispatched_solution[t,self.n_tech + 1] * (self.capacity_existing_storage_lt[i]/np.sum(self.capacity_existing_storage_lt)) * self.CO2_tech[self.n_tech + 1] * self.hour_month

				## Aggregating profits
				
				self.reward_tech_existing_storage_lt[i] += (income_storage_lt + income_storage_run_of_river + income_others)* self.discount_factor_merchant 
				self.reward_tech_existing_storage_lt_step[i] += (income_storage_lt + income_storage_run_of_river + income_others) * self.discount_factor_merchant

				## Short-term storage

				for j in range(self.n_tech_battery):

					# Merchant

					flexibility_contribution_short_term_storage_merchant = ((discharge_short_term_battery_solution[t, j] + charge_short_term_battery_solution[t, j])) * (self.capacity_merchant_battery[i, j] + self.inv_init_battery[i, j]) / (np.sum(self.capacity_merchant_battery[:, j] + self.capacity_cm_battery[:, j] + self.capacity_flexibility_battery[:, j] + self.inv_init_battery[:, j]) + delta)

					income_storage_merchant = (price[t] * (q_accepted_u_short_term_merchant[j]) - self.fixed_cost_battery[self.year_int, j] * (self.capacity_merchant_battery[i, j] + self.inv_init_battery[i, j]))/(self.normalization_factor) * self.hour_month 	
					income_storage_merchant += self.price_flexibility[t] * flexibility_contribution_short_term_storage_merchant * flexibility_credits_short_term_storage[j]/(self.normalization_factor) * self.hour_month 	

					flexibility_cost += self.price_flexibility[t] * flexibility_contribution_short_term_storage_merchant * flexibility_credits_short_term_storage[j]
					spot_payment_trimester += self.price_flexibility[t] * flexibility_contribution_short_term_storage_merchant * flexibility_credits_short_term_storage[j]
					spot_payment_hour[t] += self.price_flexibility[t] * flexibility_contribution_short_term_storage_merchant * flexibility_credits_short_term_storage[j]

					flexibility_contribution_short_term_storage_merchant_only = ((discharge_short_term_battery_solution[t, j] + charge_short_term_battery_solution[t, j])) * (self.capacity_merchant_battery[i, j]) / (np.sum(self.capacity_merchant_battery[:, j] + self.capacity_cm_battery[:, j] + self.capacity_flexibility_battery[:, j] + self.inv_init_battery[:, j]) + delta)
					self.cost_merchant_step += (price[t] * (q_accepted_u_short_term_merchant_only[j])) 
					self.cost_merchant_step += self.price_flexibility[t] * flexibility_contribution_short_term_storage_merchant_only * flexibility_credits_short_term_storage[j]

					flexibility_contribution_short_term_storage_existing_only = ((discharge_short_term_battery_solution[t, j] + charge_short_term_battery_solution[t, j])) * (self.inv_init_battery[i, j]) / (np.sum(self.capacity_merchant_battery[:, j] + self.capacity_cm_battery[:, j] + self.capacity_flexibility_battery[:, j] + self.inv_init_battery[:, j]) + delta)
					self.cost_existing_step += (price[t] * (q_accepted_u_short_term_existing_only[j])) 
					self.cost_existing_step += self.price_flexibility[t] * flexibility_contribution_short_term_storage_existing_only * flexibility_credits_short_term_storage[j]

					# cm

					flexibility_contribution_short_term_storage_cm = ((discharge_short_term_battery_solution[t, j] + charge_short_term_battery_solution[t, j])) * self.capacity_cm_battery[i, j] / (np.sum(self.capacity_merchant_battery[:, j] + self.capacity_cm_battery[:, j] + self.capacity_flexibility_battery[:, j] + self.inv_init_battery[:, j]) + delta)

					income_storage_cm = (price[t] * (q_accepted_u_short_term_cm[j]) - self.fixed_cost_battery[self.year_int, j] * self.capacity_cm_battery[i, j])/(self.normalization_factor) * self.hour_month 	

					income_storage_cm += self.price_flexibility[t] * flexibility_contribution_short_term_storage_cm * flexibility_credits_short_term_storage[j]/(self.normalization_factor) * self.hour_month 	

					flexibility_cost +=  self.price_flexibility[t] * flexibility_contribution_short_term_storage_cm * flexibility_credits_short_term_storage[j]
					spot_payment_trimester += self.price_flexibility[t] * flexibility_contribution_short_term_storage_cm * flexibility_credits_short_term_storage[j]
					spot_payment_hour[t] += self.price_flexibility[t] * flexibility_contribution_short_term_storage_cm * flexibility_credits_short_term_storage[j]

					# Flexibility storage

					income_storage_flexibility = (price[t] * (q_accepted_u_short_term_flexibility[j]) - self.fixed_cost_battery[self.year_int, j] * self.capacity_flexibility_battery[i, j])/(self.normalization_factor) * self.hour_month 	
	
					#

					self.reward_tech_merchant_battery[i, j] += income_storage_merchant * self.discount_factor_merchant 
					self.reward_tech_merchant_battery_step[i, j] += income_storage_merchant * self.discount_factor_merchant 

					self.reward_tech_cm_battery[i, j] += income_storage_cm * self.discount_factor_cm 
					self.reward_tech_cm_battery_step[i, j] += income_storage_cm * self.discount_factor_cm

					self.reward_tech_flexibility_battery[i, j] += income_storage_flexibility * self.discount_factor_flexibility 
					self.reward_tech_flexibility_battery_step[i, j] += income_storage_flexibility * self.discount_factor_flexibility  

					cm_option_value_battery = np.maximum(price[t] - self.cm_strike, 0) * self.cm_inv_options_battery[i, j] /(self.normalization_factor) * self.hour_month

					spot_payment_trimester -= np.maximum(price[t] - self.cm_strike, 0) * self.cm_inv_options_battery[i, j]
					spot_payment_hour[t] -= np.maximum(price[t] - self.cm_strike, 0) * self.cm_inv_options_battery[i, j]
					self.cost_cm_step -= np.maximum(price[t] - self.cm_strike, 0) * self.cm_inv_options_battery[i, j]

					self.reward_tech_cm_battery[i, j] -= cm_option_value_battery * self.discount_factor_cm 
					self.reward_tech_cm_battery_step[i, j] -= cm_option_value_battery * self.discount_factor_cm 

					if self.year > self.max_year - self.years_slack_profits:

						self.accumulated_profit_merchant_battery[j] += income_storage_merchant 
						self.accumulated_profit_cm_battery[j] += income_storage_cm 
						self.accumulated_profit_cm_battery[j] -= cm_option_value_battery
						self.accumulated_profit_flexibility_battery[j] += income_storage_flexibility

				# Short-term prices observations to agents

				i += 1
			
			self.hourly_prices_net[t] = (spot_payment_hour[t] + energy_not_served_costs_step_hour_tmp[t]) / (demand_hour + delta)

			# Updating time	
			self.time += 1
			self.hour_year += 1
			self.time_random += 7

		# Updating counter for random matrix

		self.taxes_step += self.CO2_emissions_step * self.CO2_tax

		if self.time_random >= 503:

			self.time_random = 0
						
		# Demand projection

		average_demand_projection_short = self.demand_average_bimester[int(self.year_int * self.yearly_resolution + self.month)]

		average_demand_projection_long = self.demand_average_year[self.year_int + self.planning_horizon]
		
		# Resource availability (short)

		average_solar_availability_short = self.availability_tech_average_bimester[int(self.year_int * self.yearly_resolution + self.month), 0]

		average_wind_availability_short = self.availability_tech_average_bimester[int(self.year_int * self.yearly_resolution + self.month), 1]

		average_hydro_availability_short = self.availability_tech_average_bimester[int(self.year_int * self.yearly_resolution + self.month), self.n_tech]

		average_hydro_ror_availability_short = self.availability_tech_average_bimester[int(self.year_int * self.yearly_resolution + self.month), self.n_tech + 1]

		average_solar_availability_long = self.availability_tech_average_yearly[self.year_int, 0]

		average_wind_availability_long = self.availability_tech_average_yearly[self.year_int, 1]

		average_hydro_availability_long = self.availability_tech_average_yearly[self.year_int, self.n_tech]

		average_hydro_ror_availability_long = self.availability_tech_average_yearly[self.year_int, self.n_tech + 1]
		
		i = 0

		for a in self.possible_agents_g:
			
			## Normal price observations

			obs_temp_g[i, 0] = np.clip((np.mean(price) * 2 / self.VoLL_norm - 1), -1, 1)
			obs_temp_g[i, 1] = np.clip((np.std(price) / self.VoLL_norm), 0, 1)
			obs_temp_g[i, 2] = np.clip((np.min(price) * 2 / self.VoLL_norm - 1), -1, 1)
			obs_temp_g[i, 3] = np.clip((np.max(price) * 2 / self.VoLL_norm - 1), -1, 1)

			## Flexibility price observations

			obs_temp_g[i, self.n_obs_short_term_prices + 0] = np.clip((np.mean(self.price_flexibility) * 2 / self.VoLL_norm - 1), -1, 1)
			obs_temp_g[i, self.n_obs_short_term_prices + 1] = np.clip((np.std(self.price_flexibility) / self.VoLL_norm), 0, 1)
			obs_temp_g[i, self.n_obs_short_term_prices + 2] = np.clip((np.min(self.price_flexibility) * 2 / self.VoLL_norm - 1), -1, 1)
			obs_temp_g[i, self.n_obs_short_term_prices + 3] = np.clip((np.max(self.price_flexibility) * 2 / self.VoLL_norm - 1), -1, 1)
			
			## Installed capacity (individual and total)

			for j in range(self.n_tech):

				ind_installed_capacity = self.inv_g[i,j]

				ind_total_installed_capacity = np.sum(self.inv_g[i,:])

				total_installed_capacity = np.sum(self.inv_g[:,j])

				ind_installed_capacity_existing = self.inv_g_aging[i,j]

				total_installed_capacity_existing = np.sum(self.inv_g_aging[:,j])

				ind_installed_capacity_merchant = self.capacity_merchant_uc[i,j]

				total_installed_capacity_merchant = np.sum(self.capacity_merchant_uc[:,j])

				ind_installed_capacity_cm = self.capacity_cm_uc[i,j]

				total_installed_capacity_cm = np.sum(self.capacity_cm_uc[:,j])

				ind_installed_capacity_CfD = self.capacity_CfD_uc[i,j]

				total_installed_capacity_CfD = np.sum(self.capacity_CfD_uc[:,j])

				## Demand projections with respect to install technologies

				# Individual	
						
				obs_temp_g[i, self.n_obs_short_term_prices + self.n_obs_flexibility_prices  + self.n_obs_resources + (self.n_obs_cap_ind * j) + 0] = np.clip((average_demand_projection_long/self.agents_g - ind_installed_capacity)/(average_demand_projection_long/self.agents_g + ind_installed_capacity), -1, 1)

				# Individual	
						
				obs_temp_g[i, self.n_obs_short_term_prices + self.n_obs_flexibility_prices  + self.n_obs_resources + (self.n_obs_cap_ind * j) + 1] = np.clip((average_demand_projection_long - total_installed_capacity)/(average_demand_projection_long + total_installed_capacity), -1, 1)
				
				# Individual	
						
				obs_temp_g[i, self.n_obs_short_term_prices + self.n_obs_flexibility_prices  + self.n_obs_resources + (self.n_obs_cap_ind * j) + 2] = np.clip((ind_installed_capacity)/(ind_total_installed_capacity + delta) * 2 - 1, -1, 1)

				# ind - Existing

				obs_temp_g[i, self.n_obs_short_term_prices + self.n_obs_flexibility_prices  + self.n_obs_resources + (self.n_obs_cap_ind * j) + 3] = np.clip(ind_installed_capacity_existing / (ind_installed_capacity + delta) * 2 - 1, -1, 1)
				
				# ind - Merchant

				obs_temp_g[i, self.n_obs_short_term_prices + self.n_obs_flexibility_prices  + self.n_obs_resources + (self.n_obs_cap_ind * j) + 4] = np.clip(ind_installed_capacity_merchant / (ind_installed_capacity + delta) * 2 - 1, -1, 1)
				
				# ind - CM

				obs_temp_g[i, self.n_obs_short_term_prices + self.n_obs_flexibility_prices  + self.n_obs_resources + (self.n_obs_cap_ind * j) + 5] = np.clip(ind_installed_capacity_cm / (ind_installed_capacity + delta) * 2 - 1, -1, 1)

				# ind - CfD

				obs_temp_g[i, self.n_obs_short_term_prices + self.n_obs_flexibility_prices  + self.n_obs_resources + (self.n_obs_cap_ind * j) + 6] = np.clip(ind_installed_capacity_CfD / (ind_installed_capacity + delta) * 2 - 1, -1, 1)

				# Individual

				obs_temp_g[i, self.n_obs_short_term_prices + self.n_obs_flexibility_prices  + self.n_obs_resources + (self.n_obs_cap_ind * j) + 7] = np.clip((ind_installed_capacity)/(total_installed_capacity + delta) * 2 - 1, -1, 1)

				# total - existing

				obs_temp_g[i, self.n_obs_short_term_prices + self.n_obs_flexibility_prices  + self.n_obs_resources + (self.n_obs_cap_ind * self.n_tech) + (self.n_obs_cap_total * j) + 0] = np.clip((total_installed_capacity_existing)/(total_installed_capacity + delta) * 2 - 1, -1, 1)

				# total - Merchant

				obs_temp_g[i, self.n_obs_short_term_prices + self.n_obs_flexibility_prices  + self.n_obs_resources + (self.n_obs_cap_ind * self.n_tech) + (self.n_obs_cap_total * j) + 1] = np.clip((total_installed_capacity_merchant)/(total_installed_capacity + delta) * 2 - 1, -1, 1)

				# total - CM

				obs_temp_g[i, self.n_obs_short_term_prices + self.n_obs_flexibility_prices  + self.n_obs_resources + (self.n_obs_cap_ind * self.n_tech) + (self.n_obs_cap_total * j) + 2] = np.clip((total_installed_capacity_cm)/(total_installed_capacity + delta) * 2 - 1, -1, 1)

				# total - CfD

				obs_temp_g[i, self.n_obs_short_term_prices + self.n_obs_flexibility_prices  + self.n_obs_resources + (self.n_obs_cap_ind * self.n_tech) + (self.n_obs_cap_total * j) + 3] = np.clip((total_installed_capacity_CfD)/(total_installed_capacity + delta) * 2 - 1, -1, 1)

				# Capacity credits - CM

				obs_temp_g[i, self.n_obs_short_term_prices + self.n_obs_flexibility_prices  + self.n_obs_resources + (self.n_obs_cap_ind * self.n_tech) + (self.n_obs_cap_total * j) + 4] = np.clip((self.cm_tech_cc[0,j]) * 2 - 1, -1, 1)

				## Adjustment of rewards due to nvestment costs (Generation)

				self.reward_tech_merchant[i,j] -= self.inv_cost_accum_merchant[i,j] * self.discount_factor_merchant 
				self.reward_tech_merchant_step[i,j] -= self.inv_cost_accum_merchant[i,j] * self.discount_factor_merchant 

				self.reward_tech_cm[i,j] -= self.inv_cost_accum_cm[i,j] * self.discount_factor_cm 
				self.reward_tech_cm_step[i,j] -= self.inv_cost_accum_cm[i,j] * self.discount_factor_cm 

				self.reward_tech_CfD[i,j] -= self.inv_cost_accum_CfD[i,j] * self.discount_factor_CfD 
				self.reward_tech_CfD_step[i,j] -= self.inv_cost_accum_CfD[i,j] * self.discount_factor_CfD 

				self.reward_tech_existing[i,j] -= self.inv_cost_accum_decom[i,j] * self.discount_factor_merchant
				self.reward_tech_existing_step[i,j] -= self.inv_cost_accum_decom[i,j] * self.discount_factor_merchant

				## Including income from Capacity Market
					
				self.reward_tech_cm[i,j] += self.cm_income[i,j] * self.discount_factor_cm 
				self.reward_tech_cm_step[i,j] += self.cm_income[i,j] * self.discount_factor_cm 

				self.reward_tech_existing[i,j] += self.cm_income_existing[i,j] * self.discount_factor_merchant
				self.reward_tech_existing_step[i,j] += self.cm_income_existing[i,j] * self.discount_factor_merchant 	

				## Accumulated profit 

				if self.year > self.max_year - self.years_slack_profits:

					self.accumulated_profit_cm[j] += self.cm_income[i,j]

			## Adjustment of rewards due to investment costs (Batteries)
			
			for j in range(self.n_tech_battery):

				self.reward_tech_cm_battery[i,j] += self.cm_income_battery[i,j] * self.discount_factor_cm 
				self.reward_tech_cm_battery_step[i,j] += self.cm_income_battery[i,j] * self.discount_factor_cm 

				self.reward_tech_cm_battery[i,j] -= self.inv_cost_accum_cm_battery[i,j] * self.discount_factor_cm
				self.reward_tech_cm_battery_step[i,j] -= self.inv_cost_accum_cm_battery[i,j] * self.discount_factor_cm  

				self.reward_tech_merchant_battery[i,j] -= self.inv_cost_accum_merchant_battery[i,j] * self.discount_factor_merchant 
				self.reward_tech_merchant_battery_step[i,j] -= self.inv_cost_accum_merchant_battery[i,j] * self.discount_factor_merchant 

				self.reward_tech_flexibility_battery[i,j] += self.flexibility_price_agents[i,j] * self.capacity_flexibility_battery[i,j] /(self.normalization_factor) * self.discount_factor_flexibility
				self.reward_tech_flexibility_battery_step[i,j] +=  self.flexibility_price_agents[i,j] * self.capacity_flexibility_battery[i,j] /(self.normalization_factor) * self.discount_factor_flexibility

				self.reward_planner_flexibility += self.flexibility_price_agents[i,j] * self.capacity_flexibility_battery[i,j] /(self.normalization_factor) * self.discount_factor_flexibility

				self.reward_tech_flexibility_battery[i,j] -= self.inv_cost_accum_flexibility_battery[i,j] * self.discount_factor_flexibility
				self.reward_tech_flexibility_battery_step[i,j] -= self.inv_cost_accum_flexibility_battery[i,j] * self.discount_factor_flexibility 

				if self.year > self.max_year - self.years_slack_profits:
					self.accumulated_profit_cm_battery[j] += self.cm_income_battery[i,j]
					self.accumulated_profit_flexibility_battery[j] += self.flexibility_price_agents[i,j] * self.capacity_flexibility_battery[i,j] /(self.normalization_factor)

				ind_installed_capacity_merchant_battery = self.capacity_merchant_battery[i,j] +  self.inv_init_battery[i,j]
				total_installed_capacity_merchant_battery = np.sum(self.capacity_merchant_battery[:,j] + self.inv_init_battery[:,j])

				ind_installed_capacity_cm_battery = self.capacity_cm_battery[i,j]
				total_installed_capacity_cm_battery = np.sum(self.capacity_cm_battery[:,j])

				ind_installed_capacity_flexibility_battery = self.capacity_flexibility_battery[i,j]
				total_installed_capacity_flexibility_battery = np.sum(self.capacity_flexibility_battery[:,j])

				ind_installed_capacity_existing_storage_lt = self.capacity_existing_storage_lt[i]
				total_installed_capacity_existing_storage_lt = np.sum(self.capacity_existing_storage_lt)

				ind_SoC_existing_storage_lt = self.SoC_merchant_storage_lt[i]/(self.SoC_max_merchant_storage_lt[i] + delta)
				total_SoC_existing_storage_lt = np.sum(self.SoC_merchant_storage_lt)/(np.sum(self.SoC_max_merchant_storage_lt) + delta)

				# Individual - merchant
							
				obs_temp_g[i, self.n_obs_short_term_prices + self.n_obs_flexibility_prices + self.n_obs_resources + (self.n_obs_cap_ind * self.n_tech) + (self.n_obs_cap_total * self.n_tech) + (self.n_obs_battery_cap_ind * j) + 0] = np.clip((average_demand_projection_long/self.agents_g - ind_installed_capacity_merchant_battery)/(average_demand_projection_long/self.agents_g + ind_installed_capacity_merchant_battery), -1, 1)

				# Individual - cm
							
				obs_temp_g[i, self.n_obs_short_term_prices + self.n_obs_flexibility_prices + self.n_obs_resources + (self.n_obs_cap_ind * self.n_tech) + (self.n_obs_cap_total * self.n_tech) + (self.n_obs_battery_cap_ind * j) + 1] = np.clip((average_demand_projection_long/self.agents_g - ind_installed_capacity_cm_battery)/(average_demand_projection_long/self.agents_g + ind_installed_capacity_cm_battery), -1, 1)

				# Individual - flexibility
							
				obs_temp_g[i, self.n_obs_short_term_prices + self.n_obs_flexibility_prices + self.n_obs_resources + (self.n_obs_cap_ind * self.n_tech) + (self.n_obs_cap_total * self.n_tech) + (self.n_obs_battery_cap_ind * j) + 2] = np.clip((average_demand_projection_long/self.agents_g - ind_installed_capacity_flexibility_battery)/(average_demand_projection_long/self.agents_g + ind_installed_capacity_flexibility_battery), -1, 1)

				# Individual - Existing Merchant Long-term
							
				obs_temp_g[i, self.n_obs_short_term_prices + self.n_obs_flexibility_prices + self.n_obs_resources + (self.n_obs_cap_ind * self.n_tech) + (self.n_obs_cap_total * self.n_tech) + (self.n_obs_battery_cap_ind * j) + 3] = np.clip((average_demand_projection_long/self.agents_g - ind_installed_capacity_existing_storage_lt)/(average_demand_projection_long/self.agents_g + ind_installed_capacity_existing_storage_lt), -1, 1)

				# Individual - SoC Existing Merchant Long-term
							
				obs_temp_g[i, self.n_obs_short_term_prices + self.n_obs_flexibility_prices + self.n_obs_resources + (self.n_obs_cap_ind * self.n_tech) + (self.n_obs_cap_total * self.n_tech) + (self.n_obs_battery_cap_ind * j) + 4] = np.clip(ind_SoC_existing_storage_lt * 2 - 1, -1, 1)

				# Total	- merchant
							
				obs_temp_g[i, self.n_obs_short_term_prices + self.n_obs_flexibility_prices + self.n_obs_resources + (self.n_obs_cap_ind * self.n_tech) + (self.n_obs_cap_total * self.n_tech) + (self.n_obs_battery_cap_ind * self.n_tech_battery) + (self.n_obs_battery_cap_total * j) + 0]= np.clip((average_demand_projection_long - total_installed_capacity_merchant_battery)/(average_demand_projection_long + total_installed_capacity_merchant_battery), -1, 1)

				# Total	- cm
							
				obs_temp_g[i, self.n_obs_short_term_prices + self.n_obs_flexibility_prices + self.n_obs_resources + (self.n_obs_cap_ind * self.n_tech) + (self.n_obs_cap_total * self.n_tech) + (self.n_obs_battery_cap_ind * self.n_tech_battery) + (self.n_obs_battery_cap_total * j) + 1]= np.clip((average_demand_projection_long - total_installed_capacity_cm_battery)/(average_demand_projection_long + total_installed_capacity_cm_battery), -1, 1)

				# Total	- flexibility
							
				obs_temp_g[i, self.n_obs_short_term_prices + self.n_obs_flexibility_prices + self.n_obs_resources + (self.n_obs_cap_ind * self.n_tech) + (self.n_obs_cap_total * self.n_tech) + (self.n_obs_battery_cap_ind * self.n_tech_battery) + (self.n_obs_battery_cap_total * j) + 2]= np.clip((average_demand_projection_long - total_installed_capacity_flexibility_battery)/(average_demand_projection_long + total_installed_capacity_flexibility_battery), -1, 1)

				# Capacity factor battery 

				obs_temp_g[i, self.n_obs_short_term_prices + self.n_obs_flexibility_prices + self.n_obs_resources + (self.n_obs_cap_ind * self.n_tech) + (self.n_obs_cap_total * self.n_tech) + (self.n_obs_battery_cap_ind * self.n_tech_battery) + (self.n_obs_battery_cap_total * j) + 3]= np.clip((self.cm_tech_cc[0, self.n_tech + j]) * 2 - 1, -1, 1)

			# Total	- Existing Merchant Long-term
						
			obs_temp_g[i, self.n_obs_short_term_prices + self.n_obs_flexibility_prices + self.n_obs_resources + (self.n_obs_cap_ind * self.n_tech) + (self.n_obs_cap_total * self.n_tech) + (self.n_obs_battery_cap_ind * self.n_tech_battery) + (self.n_obs_battery_cap_total * self.n_tech_battery) + 0]= np.clip((average_demand_projection_long - total_installed_capacity_existing_storage_lt)/(average_demand_projection_long + total_installed_capacity_existing_storage_lt), -1, 1)

			# Total	- SoC Existing Merchant Long-term
						
			obs_temp_g[i, self.n_obs_short_term_prices + self.n_obs_flexibility_prices + self.n_obs_resources + (self.n_obs_cap_ind * self.n_tech) + (self.n_obs_cap_total * self.n_tech) + (self.n_obs_battery_cap_ind * self.n_tech_battery) + (self.n_obs_battery_cap_total * self.n_tech_battery) + 1]= np.clip(total_SoC_existing_storage_lt * 2 - 1, -1, 1)

			## Time observations

			# Month

			obs_temp_g[i, self.n_obs_short_term_prices + self.n_obs_flexibility_prices + self.n_obs_resources + (self.n_obs_cap_ind * self.n_tech) + (self.n_obs_cap_total * self.n_tech) + (self.n_obs_battery_cap_ind * self.n_tech_battery) + (self.n_obs_battery_cap_total * self.n_tech_battery) + 2] = np.clip((self.month/self.yearly_resolution) * 2 - 1, -1, 1)
   
			# Year
   
			obs_temp_g[i, self.n_obs_short_term_prices + self.n_obs_flexibility_prices + self.n_obs_resources + (self.n_obs_cap_ind * self.n_tech) + (self.n_obs_cap_total * self.n_tech) + (self.n_obs_battery_cap_ind * self.n_tech_battery) + (self.n_obs_battery_cap_total * self.n_tech_battery) + 3] = np.clip((self.year/self.max_year) * 2 - 1, -1, 1)
   
			# Time
   
			obs_temp_g[i, self.n_obs_short_term_prices + self.n_obs_flexibility_prices + self.n_obs_resources + (self.n_obs_cap_ind * self.n_tech) + (self.n_obs_cap_total * self.n_tech) + (self.n_obs_battery_cap_ind * self.n_tech_battery) + (self.n_obs_battery_cap_total * self.n_tech_battery) + 4] = np.clip(self.hour_year/(self.max_year * self.yearly_resolution) * 2 - 1, -1, 1)

			# CO2 tax
   
			obs_temp_g[i, self.n_obs_short_term_prices + self.n_obs_flexibility_prices + self.n_obs_resources + (self.n_obs_cap_ind * self.n_tech) + (self.n_obs_cap_total * self.n_tech) + (self.n_obs_battery_cap_ind * self.n_tech_battery) + (self.n_obs_battery_cap_total * self.n_tech_battery) + 5] = np.clip((self.CO2_tax_scenario[self.year_int + 4]/300) * 2 - 1, -1, 1)

			# Corporate tax rate
   
			obs_temp_g[i, self.n_obs_short_term_prices + self.n_obs_flexibility_prices + self.n_obs_resources + (self.n_obs_cap_ind * self.n_tech) + (self.n_obs_cap_total * self.n_tech) + (self.n_obs_battery_cap_ind * self.n_tech_battery) + (self.n_obs_battery_cap_total * self.n_tech_battery) + 6] = np.clip((self.corporate_tax_rate) * 2 - 1, -1, 1)
			
			## Capacity Market and CfD observations

			# CM Balance 

			obs_temp_g[i, self.n_obs_short_term_prices + self.n_obs_flexibility_prices + self.n_obs_resources + (self.n_obs_cap_ind * self.n_tech) + (self.n_obs_cap_total * self.n_tech) + (self.n_obs_battery_cap_ind * self.n_tech_battery) + (self.n_obs_battery_cap_total * self.n_tech_battery) + self.n_observation_time + 0] = np.clip(self.cm_balance/self.percentile_demand,-1,1)

			# CM Balance Real

			obs_temp_g[i, self.n_obs_short_term_prices + self.n_obs_flexibility_prices + self.n_obs_resources + (self.n_obs_cap_ind * self.n_tech) + (self.n_obs_cap_total * self.n_tech) + (self.n_obs_battery_cap_ind * self.n_tech_battery) + (self.n_obs_battery_cap_total * self.n_tech_battery) + self.n_observation_time + 1] = np.clip(self.cm_balance_real/self.percentile_demand,-1,1)

			# CM price

			self.cm_price_aggregated = np.clip(((((np.sum(self.cm_income) + np.sum(self.cm_income_battery)) * self.normalization_factor /(self.short_t * self.hour_month))/(np.sum(self.cm_inv_options) + np.sum(self.cm_inv_options_battery) + delta))), 0, self.cm_price_cap_max)

			obs_temp_g[i, self.n_obs_short_term_prices + self.n_obs_flexibility_prices + self.n_obs_resources + (self.n_obs_cap_ind * self.n_tech) + (self.n_obs_cap_total * self.n_tech) + (self.n_obs_battery_cap_ind * self.n_tech_battery) + (self.n_obs_battery_cap_total * self.n_tech_battery) + self.n_observation_time + 2] = np.clip(self.cm_price_aggregated/self.cm_price_cap_max * 2 - 1, -1, 1)

			# CM scarcity

			obs_temp_g[i, self.n_obs_short_term_prices + self.n_obs_flexibility_prices + self.n_obs_resources + (self.n_obs_cap_ind * self.n_tech) + (self.n_obs_cap_total * self.n_tech) + (self.n_obs_battery_cap_ind * self.n_tech_battery) + (self.n_obs_battery_cap_total * self.n_tech_battery) + self.n_observation_time + 3] = np.clip((self.cm_scarcity * 2) - 1, -1, 1)

			# CM target

			obs_temp_g[i, self.n_obs_short_term_prices + self.n_obs_flexibility_prices + self.n_obs_resources + (self.n_obs_cap_ind * self.n_tech) + (self.n_obs_cap_total * self.n_tech) + (self.n_obs_battery_cap_ind * self.n_tech_battery) + (self.n_obs_battery_cap_total * self.n_tech_battery) + self.n_observation_time + 4] = np.clip((self.cm_demand_target / np.max(self.cm_reliability_target_values))*2 - 1, -1, 1)

			# CfD Balance 

			obs_temp_g[i, self.n_obs_short_term_prices + self.n_obs_flexibility_prices + self.n_obs_resources + (self.n_obs_cap_ind * self.n_tech) + (self.n_obs_cap_total * self.n_tech) + (self.n_obs_battery_cap_ind * self.n_tech_battery) + (self.n_obs_battery_cap_total * self.n_tech_battery) + self.n_observation_time + 5] = np.clip(self.CfD_balance/self.percentile_demand,-1,1)

			# CfD Balance Real

			obs_temp_g[i, self.n_obs_short_term_prices + self.n_obs_flexibility_prices + self.n_obs_resources + (self.n_obs_cap_ind * self.n_tech) + (self.n_obs_cap_total * self.n_tech) + (self.n_obs_battery_cap_ind * self.n_tech_battery) + (self.n_obs_battery_cap_total * self.n_tech_battery) + self.n_observation_time + 6] = np.clip(self.CfD_balance_real/self.percentile_demand,-1,1)

			# CfD Price 

			self.CfD_price_aggregated = np.clip((np.sum(self.CfD_price_pond)/(np.sum(self.capacity_CfD) + delta)), 0, self.CfD_price_cap_max)

			obs_temp_g[i, self.n_obs_short_term_prices + self.n_obs_flexibility_prices + self.n_obs_resources + (self.n_obs_cap_ind * self.n_tech) + (self.n_obs_cap_total * self.n_tech) + (self.n_obs_battery_cap_ind * self.n_tech_battery) + (self.n_obs_battery_cap_total * self.n_tech_battery) + self.n_observation_time + 7] = np.clip(self.CfD_price_aggregated/self.CfD_price_cap_max * 2 - 1, -1, 1)

			# CfD scarcity

			obs_temp_g[i, self.n_obs_short_term_prices + self.n_obs_flexibility_prices + self.n_obs_resources + (self.n_obs_cap_ind * self.n_tech) + (self.n_obs_cap_total * self.n_tech) + (self.n_obs_battery_cap_ind * self.n_tech_battery) + (self.n_obs_battery_cap_total * self.n_tech_battery) + self.n_observation_time + 8] = np.clip((self.CfD_scarcity * 2) - 1, -1, 1)

			# CfD target

			obs_temp_g[i, self.n_obs_short_term_prices + self.n_obs_flexibility_prices + self.n_obs_resources + (self.n_obs_cap_ind * self.n_tech) + (self.n_obs_cap_total * self.n_tech) + (self.n_obs_battery_cap_ind * self.n_tech_battery) + (self.n_obs_battery_cap_total * self.n_tech_battery) + self.n_observation_time + 9] = np.clip((self.CfD_increment_target / np.max(self.CfD_target_increments_steps))*2 - 1, -1, 1)

			# flexibility Balance 

			obs_temp_g[i, self.n_obs_short_term_prices + self.n_obs_flexibility_prices + self.n_obs_resources + (self.n_obs_cap_ind * self.n_tech) + (self.n_obs_cap_total * self.n_tech) + (self.n_obs_battery_cap_ind * self.n_tech_battery) + (self.n_obs_battery_cap_total * self.n_tech_battery) + self.n_observation_time + 10] = np.clip(self.flexibility_balance/self.percentile_demand,-1,1)

			# flexibility Balance Real

			obs_temp_g[i, self.n_obs_short_term_prices + self.n_obs_flexibility_prices + self.n_obs_resources + (self.n_obs_cap_ind * self.n_tech) + (self.n_obs_cap_total * self.n_tech) + (self.n_obs_battery_cap_ind * self.n_tech_battery) + (self.n_obs_battery_cap_total * self.n_tech_battery) + self.n_observation_time + 11] = np.clip(self.flexibility_balance_real/self.percentile_demand,-1,1)

			# Flexibility Price 

			self.flexibility_price_aggregated = np.clip((np.sum(self.flexibility_price_pond)/(np.sum(self.capacity_flexibility_battery) + delta)), 0, self.flexibility_price_cap)

			obs_temp_g[i, self.n_obs_short_term_prices + self.n_obs_flexibility_prices + self.n_obs_resources + (self.n_obs_cap_ind * self.n_tech) + (self.n_obs_cap_total * self.n_tech) + (self.n_obs_battery_cap_ind * self.n_tech_battery) + (self.n_obs_battery_cap_total * self.n_tech_battery) + self.n_observation_time + 12] = np.clip(self.flexibility_price_aggregated/self.flexibility_price_cap * 2 - 1, -1, 1)

			# Flexibility scarcity

			obs_temp_g[i, self.n_obs_short_term_prices + self.n_obs_flexibility_prices + self.n_obs_resources + (self.n_obs_cap_ind * self.n_tech) + (self.n_obs_cap_total * self.n_tech) + (self.n_obs_battery_cap_ind * self.n_tech_battery) + (self.n_obs_battery_cap_total * self.n_tech_battery) + self.n_observation_time + 13] = np.clip((self.flexibility_scarcity * 2) - 1, -1, 1)

			# Flexibility target

			obs_temp_g[i, self.n_obs_short_term_prices + self.n_obs_flexibility_prices + self.n_obs_resources + (self.n_obs_cap_ind * self.n_tech) + (self.n_obs_cap_total * self.n_tech) + (self.n_obs_battery_cap_ind * self.n_tech_battery) + (self.n_obs_battery_cap_total * self.n_tech_battery) + self.n_observation_time + 14] = np.clip((self.flexibility_increment_target / np.max(self.flexibility_target_increments_steps))*2 - 1, -1, 1)

			## Market concentration

			# All markets

			agent_capacities = (np.sum(self.inv_g, axis=1) + np.sum(self.capacity_merchant_battery + self.capacity_cm_battery + self.capacity_flexibility_battery + self.inv_init_battery, axis = 1) + self.capacity_existing_storage_lt)
			
			# HHI calculation
			total_capacity_all = np.sum(agent_capacities)
			market_shares = agent_capacities / (total_capacity_all + delta)
			herfindahl_index = np.sum(market_shares ** 2)
			
			obs_temp_g[i, self.n_obs_short_term_prices + self.n_obs_flexibility_prices + self.n_obs_resources + (self.n_obs_cap_ind * self.n_tech) + (self.n_obs_cap_total * self.n_tech) + (self.n_obs_battery_cap_ind * self.n_tech_battery) + (self.n_obs_battery_cap_total * self.n_tech_battery) + self.n_observation_time + 15] = np.clip(herfindahl_index * 2 - 1, -1, 1)

			# Aggregate HHI for all existing (generation + long-term storage + initial battery)
			agent_capacity_existing_all = (
				np.sum(self.inv_g_aging, axis=1) 
				+ self.capacity_existing_storage_lt 
				+ np.sum(self.inv_init_battery, axis = 1)
			)
			total_capacity_existing_all = np.sum(agent_capacity_existing_all)
			market_shares_existing_all = agent_capacity_existing_all / (total_capacity_existing_all + delta)
			hhi_existing_all = np.sum(market_shares_existing_all ** 2)
			
			obs_temp_g[i, self.n_obs_short_term_prices + self.n_obs_flexibility_prices + self.n_obs_resources + (self.n_obs_cap_ind * self.n_tech) + (self.n_obs_cap_total * self.n_tech) + (self.n_obs_battery_cap_ind * self.n_tech_battery) + (self.n_obs_battery_cap_total * self.n_tech_battery) + self.n_observation_time + 16] = np.clip(hhi_existing_all * 2 - 1, -1, 1)

			# Aggregate HHI for all merchant (generation + battery)
			agent_capacity_merchant_all = (
				np.sum(self.capacity_merchant, axis=1) 
				+ np.sum(self.capacity_merchant_battery, axis = 1)
			)
			total_capacity_merchant_all = np.sum(agent_capacity_merchant_all)
			market_shares_merchant_all = agent_capacity_merchant_all / (total_capacity_merchant_all + delta)
			hhi_merchant_all = np.sum(market_shares_merchant_all ** 2)
			
			obs_temp_g[i, self.n_obs_short_term_prices + self.n_obs_flexibility_prices + self.n_obs_resources + (self.n_obs_cap_ind * self.n_tech) + (self.n_obs_cap_total * self.n_tech) + (self.n_obs_battery_cap_ind * self.n_tech_battery) + (self.n_obs_battery_cap_total * self.n_tech_battery) + self.n_observation_time + 17] = np.clip(hhi_merchant_all * 2 - 1, -1, 1)

			# Aggregate HHI for all CM (generation + battery)
			agent_capacity_cm_all = np.sum(self.capacity_cm, axis=1) + np.sum(self.capacity_cm_battery, axis = 1)
			total_capacity_cm_all = np.sum(agent_capacity_cm_all)
			market_shares_cm_all = agent_capacity_cm_all / (total_capacity_cm_all + delta)
			hhi_cm_all = np.sum(market_shares_cm_all ** 2)
			
			obs_temp_g[i, self.n_obs_short_term_prices + self.n_obs_flexibility_prices + self.n_obs_resources + (self.n_obs_cap_ind * self.n_tech) + (self.n_obs_cap_total * self.n_tech) + (self.n_obs_battery_cap_ind * self.n_tech_battery) + (self.n_obs_battery_cap_total * self.n_tech_battery) + self.n_observation_time + 18] = np.clip(hhi_cm_all * 2 - 1, -1, 1)

			# Aggregate HHI for all CfD (generation only)
			agent_capacity_cfd_all = np.sum(self.capacity_CfD_uc, axis=1)
			total_capacity_cfd_all = np.sum(agent_capacity_cfd_all)
			market_shares_cfd_all = agent_capacity_cfd_all / (total_capacity_cfd_all + delta)
			hhi_cfd_all = np.sum(market_shares_cfd_all ** 2)
			
			obs_temp_g[i, self.n_obs_short_term_prices + self.n_obs_flexibility_prices + self.n_obs_resources + (self.n_obs_cap_ind * self.n_tech) + (self.n_obs_cap_total * self.n_tech) + (self.n_obs_battery_cap_ind * self.n_tech_battery) + (self.n_obs_battery_cap_total * self.n_tech_battery) + self.n_observation_time + 19] = np.clip(hhi_cfd_all * 2 - 1, -1, 1)

			# Aggregate HHI for flexibility market (batteries only)
			agent_capacity_flexibility_all = np.sum(self.capacity_flexibility_battery, axis = 1)
			total_capacity_flexibility_all = np.sum(agent_capacity_flexibility_all)
			market_shares_flexibility_all = agent_capacity_flexibility_all / (total_capacity_flexibility_all + delta)
			hhi_flexibility_all = np.sum(market_shares_flexibility_all ** 2)
			
			obs_temp_g[i, self.n_obs_short_term_prices + self.n_obs_flexibility_prices + self.n_obs_resources + (self.n_obs_cap_ind * self.n_tech) + (self.n_obs_cap_total * self.n_tech) + (self.n_obs_battery_cap_ind * self.n_tech_battery) + (self.n_obs_battery_cap_total * self.n_tech_battery) + self.n_observation_time + 20] = np.clip(hhi_flexibility_all * 2 - 1, -1, 1)

			for j in range(self.n_tech):

				## Depreciation values: (including tax adjustment)

				if self.time >= self.max_t: 
					
					npv_annuaity_MW_merchant = ((self.accumulated_profit_merchant[j]) / (np.sum(self.capacity_merchant[:,j]) + delta)) * ((1 - (1 + self.opportunity_cost_merchant)**(-(np.maximum(self.life_time[j] - int(self.inv_age_merchant[i,j]),0))))/(self.opportunity_cost_merchant))

					npv_annuaity_MW_cm = ((self.accumulated_profit_cm[j]) / (np.sum(self.capacity_cm[:,j]) + delta)) * ((1 - (1 + self.opportunity_cost_cm)**(-(np.maximum(self.life_time[j] - int(self.inv_age_cm[i,j]),0))))/(self.opportunity_cost_cm))

					npv_annuaity_MW_CfD = ((self.accumulated_profit_CfD[j]) / (np.sum(self.capacity_CfD[:,j]) + delta)) * ((1 - (1 + self.opportunity_cost_CfD)**(-(np.maximum(self.life_time[j] - int(self.inv_age_CfD[i,j]),0))))/(self.opportunity_cost_CfD))
					

					depreciation_value_merchant = npv_annuaity_MW_merchant * (self.capacity_merchant[i,j])/(self.count_last[j])

					depreciation_value_cm = npv_annuaity_MW_cm * (self.capacity_cm[i,j])/(self.count_last[j]) 

					depreciation_value_CfD = npv_annuaity_MW_CfD * (self.capacity_CfD[i,j])/(self.count_last[j]) 

					# Accumulating reward

					self.reward_tech_merchant[i,j] += depreciation_value_merchant * self.discount_factor_merchant 
					self.reward_tech_merchant_step[i,j] += depreciation_value_merchant * self.discount_factor_merchant 

					self.reward_planner_depreciation += depreciation_value_merchant * self.discount_factor_planner * self.normalization_factor
					self.reward_planner_depreciation_step += depreciation_value_merchant * self.discount_factor_planner * self.normalization_factor
					self.reward_planner_depreciation_not_discounted += depreciation_value_merchant * self.normalization_factor

					self.reward_tech_cm[i,j] += depreciation_value_cm * self.discount_factor_cm
					self.reward_tech_cm_step[i,j] += depreciation_value_cm * self.discount_factor_cm

					self.reward_planner_depreciation += depreciation_value_cm * self.discount_factor_planner * self.normalization_factor
					self.reward_planner_depreciation_step += depreciation_value_cm * self.discount_factor_planner * self.normalization_factor
					self.reward_planner_depreciation_not_discounted += depreciation_value_cm * self.normalization_factor

					self.reward_tech_CfD[i,j] += depreciation_value_CfD * self.discount_factor_CfD
					self.reward_tech_CfD_step[i,j] += depreciation_value_CfD * self.discount_factor_CfD

					self.reward_planner_depreciation += depreciation_value_CfD * self.discount_factor_planner * self.normalization_factor
					self.reward_planner_depreciation_step += depreciation_value_CfD * self.discount_factor_planner * self.normalization_factor
					self.reward_planner_depreciation_not_discounted += depreciation_value_CfD * self.normalization_factor

			## Depreciation value battery

			if self.time >= self.max_t: 
				
				for j in range(self.n_tech_battery):

					# Merchant
					
					npv_annuaity_MW_merchant_battery = (self.accumulated_profit_merchant_battery[j] / (np.sum(self.capacity_merchant_battery[:, j]) + delta)) * ((1 - (1 + self.opportunity_cost_merchant)**(-(np.maximum(self.life_time_battery[j] - int(self.inv_age_merchant_battery[i, j]), 0))))/(self.opportunity_cost_merchant))
					depreciation_value_merchant_battery = npv_annuaity_MW_merchant_battery * (self.capacity_merchant_battery[i, j])/(self.count_last_battery[j])

					self.reward_tech_merchant_battery[i, j] += depreciation_value_merchant_battery * self.discount_factor_merchant 
					self.reward_tech_merchant_battery_step[i, j] += depreciation_value_merchant_battery * self.discount_factor_merchant 

					self.reward_planner_depreciation += depreciation_value_merchant_battery * self.discount_factor_planner * self.normalization_factor
					self.reward_planner_depreciation_step += depreciation_value_merchant_battery * self.discount_factor_planner * self.normalization_factor
					self.reward_planner_depreciation_not_discounted += depreciation_value_merchant_battery * self.normalization_factor

					# Capacity Market

					npv_annuaity_MW_cm_battery = (self.accumulated_profit_cm_battery[j] / (np.sum(self.capacity_cm_battery[:, j]) + delta)) * ((1 - (1 + self.opportunity_cost_cm)**(-(np.maximum(self.life_time_battery[j] - int(self.inv_age_cm_battery[i, j]), 0))))/(self.opportunity_cost_cm))
					depreciation_value_cm_battery = npv_annuaity_MW_cm_battery * (self.capacity_cm_battery[i, j])/(self.count_last_battery[j])

					self.reward_tech_cm_battery[i, j] += depreciation_value_cm_battery * self.discount_factor_cm 
					self.reward_tech_cm_battery_step[i, j] += depreciation_value_cm_battery * self.discount_factor_cm

					self.reward_planner_depreciation += depreciation_value_cm_battery * self.discount_factor_planner * self.normalization_factor
					self.reward_planner_depreciation_step += depreciation_value_cm_battery * self.discount_factor_planner * self.normalization_factor
					self.reward_planner_depreciation_not_discounted += depreciation_value_cm_battery * self.normalization_factor
					
					# Flexibility

					npv_annuaity_MW_flexibility_battery = (self.accumulated_profit_flexibility_battery[j] / (np.sum(self.capacity_flexibility_battery[:, j]) + delta)) * ((1 - (1 + self.opportunity_cost_flexibility)**(-(np.maximum(self.life_time_battery[j] - int(self.inv_age_flexibility_battery[i, j]), 0))))/(self.opportunity_cost_flexibility))
					depreciation_value_flexibility_battery = npv_annuaity_MW_flexibility_battery * (self.capacity_flexibility_battery[i, j])/(self.count_last_battery[j])

					self.reward_tech_flexibility_battery[i, j] += depreciation_value_flexibility_battery * self.discount_factor_flexibility 
					self.reward_tech_flexibility_battery_step[i, j] += depreciation_value_flexibility_battery * self.discount_factor_flexibility

					self.reward_planner_depreciation += depreciation_value_flexibility_battery * self.discount_factor_planner * self.normalization_factor
					self.reward_planner_depreciation_step += depreciation_value_flexibility_battery * self.discount_factor_planner * self.normalization_factor
					self.reward_planner_depreciation_not_discounted += depreciation_value_flexibility_battery * self.normalization_factor

			for j in range(self.n_tech):

				## Reward observations 

				obs_temp_g[i, self.n_obs_short_term_prices + self.n_obs_flexibility_prices + self.n_obs_resources + (self.n_obs_cap_ind * self.n_tech) + (self.n_obs_cap_total * self.n_tech) + (self.n_obs_battery_cap_ind * self.n_tech_battery) + (self.n_obs_battery_cap_total * self.n_tech_battery) + self.n_observation_time + self.n_observation_cm_CfD + (j * self.n_observation_reward_tech) + 0] = np.clip(self.reward_tech_merchant[i,j], -1, 1)
				
				obs_temp_g[i, self.n_obs_short_term_prices + self.n_obs_flexibility_prices + self.n_obs_resources + (self.n_obs_cap_ind * self.n_tech) + (self.n_obs_cap_total * self.n_tech) + (self.n_obs_battery_cap_ind * self.n_tech_battery) + (self.n_obs_battery_cap_total * self.n_tech_battery) + self.n_observation_time + self.n_observation_cm_CfD + (j * self.n_observation_reward_tech) + 1] = np.clip(self.reward_tech_cm[i,j], -1, 1)

				obs_temp_g[i, self.n_obs_short_term_prices + self.n_obs_flexibility_prices + self.n_obs_resources + (self.n_obs_cap_ind * self.n_tech) + (self.n_obs_cap_total * self.n_tech) + (self.n_obs_battery_cap_ind * self.n_tech_battery) + (self.n_obs_battery_cap_total * self.n_tech_battery) + self.n_observation_time + self.n_observation_cm_CfD + (j * self.n_observation_reward_tech) + 2] = np.clip(self.reward_tech_existing[i,j], -1, 1)

				# Specfic price per technology

				obs_temp_g[i, self.n_obs_short_term_prices + self.n_obs_flexibility_prices + self.n_obs_resources + (self.n_obs_cap_ind * self.n_tech) + (self.n_obs_cap_total * self.n_tech) + (self.n_obs_battery_cap_ind * self.n_tech_battery) + (self.n_obs_battery_cap_total * self.n_tech_battery) + self.n_observation_time + self.n_observation_cm_CfD + (j * self.n_observation_reward_tech) + 3] = np.clip(((self.cm_income[i,j] * self.normalization_factor /((self.short_t * self.hour_month)) / (self.capacity_cm[i,j] + delta)) / self.cm_price_cap_max) * 2 - 1, -1, 1)

			for j in range(self.n_tech_RES):

				# Reward observations
				
				obs_temp_g[i, self.n_obs_short_term_prices + self.n_obs_flexibility_prices + self.n_obs_resources + (self.n_obs_cap_ind * self.n_tech) + (self.n_obs_cap_total * self.n_tech) + (self.n_obs_battery_cap_ind * self.n_tech_battery) + (self.n_obs_battery_cap_total * self.n_tech_battery) + self.n_observation_time + self.n_observation_cm_CfD + self.n_tech * self.n_observation_reward_tech + (j * self.n_observation_reward_tech_CfD) + 0] = np.clip(self.reward_tech_CfD[i,j], -1, 1)

				# Specific price per technology

				obs_temp_g[i, self.n_obs_short_term_prices + self.n_obs_flexibility_prices + self.n_obs_resources + (self.n_obs_cap_ind * self.n_tech) + (self.n_obs_cap_total * self.n_tech) + (self.n_obs_battery_cap_ind * self.n_tech_battery) + (self.n_obs_battery_cap_total * self.n_tech_battery) + self.n_observation_time + self.n_observation_cm_CfD + self.n_tech * self.n_observation_reward_tech + (j * self.n_observation_reward_tech_CfD) + 1] = np.clip((self.CfD_price_agents[i,j] / self.CfD_price_cap_max) * 2 - 1, -1, 1)

			# Reward observation batteries (TODO)

			for j in range(self.n_tech_battery):

				obs_temp_g[i, self.n_obs_short_term_prices + self.n_obs_flexibility_prices + self.n_obs_resources + (self.n_obs_cap_ind * self.n_tech) + (self.n_obs_cap_total * self.n_tech) + (self.n_obs_battery_cap_ind * self.n_tech_battery) + (self.n_obs_battery_cap_total * self.n_tech_battery) + self.n_observation_time + self.n_observation_cm_CfD + self.n_tech * self.n_observation_reward_tech + self.n_tech_RES * self.n_observation_reward_tech_CfD + (self.n_observation_reward_tech_battery * j) + 0] = np.clip(self.reward_tech_merchant_battery[i,j], -1, 1)

				obs_temp_g[i, self.n_obs_short_term_prices + self.n_obs_flexibility_prices + self.n_obs_resources + (self.n_obs_cap_ind * self.n_tech) + (self.n_obs_cap_total * self.n_tech) + (self.n_obs_battery_cap_ind * self.n_tech_battery) + (self.n_obs_battery_cap_total * self.n_tech_battery) + self.n_observation_time + self.n_observation_cm_CfD + self.n_tech * self.n_observation_reward_tech + self.n_tech_RES * self.n_observation_reward_tech_CfD + (self.n_observation_reward_tech_battery * j) + 1] = np.clip(self.reward_tech_cm_battery[i,j], -1, 1)

				obs_temp_g[i, self.n_obs_short_term_prices + self.n_obs_flexibility_prices + self.n_obs_resources + (self.n_obs_cap_ind * self.n_tech) + (self.n_obs_cap_total * self.n_tech) + (self.n_obs_battery_cap_ind * self.n_tech_battery) + (self.n_obs_battery_cap_total * self.n_tech_battery) + self.n_observation_time + self.n_observation_cm_CfD + self.n_tech * self.n_observation_reward_tech + self.n_tech_RES * self.n_observation_reward_tech_CfD + (self.n_observation_reward_tech_battery * j) + 2] = np.clip(((self.cm_income_battery[i,j] * self.normalization_factor /((self.short_t * self.hour_month)) / (self.capacity_cm_battery[i,j] + delta)) / self.cm_price_cap_max) * 2 - 1, -1, 1)

				obs_temp_g[i, self.n_obs_short_term_prices + self.n_obs_flexibility_prices + self.n_obs_resources + (self.n_obs_cap_ind * self.n_tech) + (self.n_obs_cap_total * self.n_tech) + (self.n_obs_battery_cap_ind * self.n_tech_battery) + (self.n_obs_battery_cap_total * self.n_tech_battery) + self.n_observation_time + self.n_observation_cm_CfD + self.n_tech * self.n_observation_reward_tech + self.n_tech_RES * self.n_observation_reward_tech_CfD + (self.n_observation_reward_tech_battery * j) + 3] = np.clip(self.reward_tech_existing_storage_lt_step[i], -1, 1)

				obs_temp_g[i, self.n_obs_short_term_prices + self.n_obs_flexibility_prices + self.n_obs_resources + (self.n_obs_cap_ind * self.n_tech) + (self.n_obs_cap_total * self.n_tech) + (self.n_obs_battery_cap_ind * self.n_tech_battery) + (self.n_obs_battery_cap_total * self.n_tech_battery) + self.n_observation_time + self.n_observation_cm_CfD + self.n_tech * self.n_observation_reward_tech + self.n_tech_RES * self.n_observation_reward_tech_CfD + (self.n_observation_reward_tech_battery * j) + 4] = np.clip(self.reward_tech_flexibility_battery[i,j], -1, 1)

				obs_temp_g[i, self.n_obs_short_term_prices + self.n_obs_flexibility_prices + self.n_obs_resources + (self.n_obs_cap_ind * self.n_tech) + (self.n_obs_cap_total * self.n_tech) + (self.n_obs_battery_cap_ind * self.n_tech_battery) + (self.n_obs_battery_cap_total * self.n_tech_battery) + self.n_observation_time + self.n_observation_cm_CfD + self.n_tech * self.n_observation_reward_tech + self.n_tech_RES * self.n_observation_reward_tech_CfD + (self.n_observation_reward_tech_battery * j) + 5] = np.clip((self.flexibility_price_agents[i,j] / self.flexibility_price_cap) * 2 - 1, -1, 1)

			# Regret penalty - merchant only (applied at end of episode)

			# Compute regret averages once (before per-agent application, uses pre-tax reward arrays)

			if self.time >= self.max_t and i == 0:

				self.average_rewards_merchant = np.zeros([self.n_tech])

				self.average_rewards_merchant_battery = np.zeros([self.n_tech_battery])

			# Accumulating reward

			## Taxes 

			reward_step_total = np.sum(self.reward_tech_existing[i,:] + self.reward_tech_merchant[i,:] + self.reward_tech_cm[i,:] + self.reward_tech_CfD[i,:]) + np.sum(self.reward_tech_merchant_battery[i,:] + self.reward_tech_cm_battery[i,:] + self.reward_tech_flexibility_battery[i,:]) + self.reward_tech_existing_storage_lt[i]

			reward_step_total_step = np.sum(self.reward_tech_existing_step[i,:] + self.reward_tech_merchant_step[i,:] + self.reward_tech_cm_step[i,:] + self.reward_tech_CfD_step[i,:]) + np.sum(self.reward_tech_merchant_battery_step[i,:] + self.reward_tech_cm_battery_step[i,:] + self.reward_tech_flexibility_battery_step[i,:]) + self.reward_tech_existing_storage_lt_step[i]

			taxes_tmp = 0

			if reward_step_total_step > 0:

				taxes_tmp = reward_step_total_step * self.corporate_tax_rate

				self.taxes_agents[i] += reward_step_total_step * self.corporate_tax_rate

				self.taxes_step += taxes_tmp * self.normalization_factor/self.discount_factor

				self.tax_rent_step += taxes_tmp * self.normalization_factor/self.discount_factor

			## Calculate taxes per market and technology (Just for estimating IRR)
			
			self.taxes_tech_existing_step[i,:] = np.maximum(0, self.reward_tech_existing_step[i,:]) * self.corporate_tax_rate
			self.taxes_tech_merchant_step[i,:] = np.maximum(0, self.reward_tech_merchant_step[i,:]) * self.corporate_tax_rate
			self.taxes_tech_cm_step[i,:] = np.maximum(0, self.reward_tech_cm_step[i,:]) * self.corporate_tax_rate
			self.taxes_tech_CfD_step[i,:] = np.maximum(0, self.reward_tech_CfD_step[i,:]) * self.corporate_tax_rate
			
			# Battery/storage taxes
			self.taxes_tech_merchant_battery_step[i,:] = np.maximum(0, self.reward_tech_merchant_battery_step[i,:]) * self.corporate_tax_rate
			self.taxes_tech_cm_battery_step[i,:] = np.maximum(0, self.reward_tech_cm_battery_step[i,:]) * self.corporate_tax_rate
			self.taxes_tech_flexibility_battery_step[i,:] = np.maximum(0, self.reward_tech_flexibility_battery_step[i,:]) * self.corporate_tax_rate
			self.taxes_tech_existing_storage_lt_step[i] = np.maximum(0, self.reward_tech_existing_storage_lt_step[i]) * self.corporate_tax_rate
    
			## Acumulating rewards

			if self.terminal_reward_flag == True:

				if self.time >= self.max_t: 

					reward_temp[i] = reward_step_total - self.taxes_agents[i]

				else:

					reward_temp[i] = 0

			else:

				reward_temp[i] = reward_step_total_step - taxes_tmp

			## Regret penalty - merchant only (applied after taxes, end of episode only)

			if self.time >= self.max_t:

				# Proportional entry scaling: agents entering late face a reduced penalty
				# reflecting the fraction of the simulation they were eligible to participate.
				# Non-entrants have entry_enabled[i] == 0, so they receive the full penalty.

				entry_scale = (self.max_year - self.entry_enabled[i]) / self.max_year

				# Apply regret penalty to merchant generation technologies

				for j in range(self.n_tech):

					if self.average_rewards_merchant[j] > 0 and self.investments_enabled[i, j] == 1 and self.average_rewards_merchant[j] > self.reward_tech_merchant[i, j]:

						self.penalty_merchant[i, j] = ((self.average_rewards_merchant[j] - self.reward_tech_merchant[i, j])*2) * entry_scale

						reward_temp[i] -= self.penalty_merchant[i, j]

				# Apply regret penalty to merchant battery technologies

				for j in range(self.n_tech_battery):

					if self.average_rewards_merchant_battery[j] > 0 and self.investments_enabled[i, self.n_tech] == 1 and self.average_rewards_merchant_battery[j] > self.reward_tech_merchant_battery[i, j]:

						self.penalty_merchant_battery[i, j] = ((self.average_rewards_merchant_battery[j] - self.reward_tech_merchant_battery[i, j])*2) * entry_scale

						reward_temp[i] -= self.penalty_merchant_battery[i, j]

				# Accumulate penalty term for tracking

				self.penalty_term[i, :self.n_tech] += self.penalty_merchant[i, :]

				self.penalty_term[i, self.n_tech] += np.sum(self.penalty_merchant_battery[i, :])

			self.reward_agents_g[i] = reward_temp[i]

			rewards_g.update({a:np.float32(reward_temp[i])})

			## Observation total reward

			obs_temp_g[i, self.n_obs_short_term_prices + self.n_obs_flexibility_prices + self.n_obs_resources + (self.n_obs_cap_ind * self.n_tech) + (self.n_obs_cap_total * self.n_tech) + (self.n_obs_battery_cap_ind * self.n_tech_battery) + (self.n_obs_battery_cap_total * self.n_tech_battery) + self.n_observation_time + self.n_observation_cm_CfD + self.n_tech * self.n_observation_reward_tech + self.n_tech_RES * self.n_observation_reward_tech_CfD + (self.n_observation_reward_tech_battery * self.n_tech_battery) + 0] = np.clip((reward_step_total - self.taxes_agents[i]), -1, 1)

			observations_temp = {"observations": np.float32(obs_temp_g[i,:]),
									"action_mask":self.get_action_masks_g(self.investments_enabled[i,:], self.entry_enabled[i], self.capacity_existing_storage_lt[i], self.inv_g_aging[i,:])} 
			

			if np.any(~np.isfinite(obs_temp_g[i,:])):
				bad_indices = np.where(~np.isfinite(obs_temp_g[i,:]))[0]
				print(f"Agent {i}, year {self.year}, month {self.month}: NaN/Inf at indices {bad_indices}")
				print(f"  price contains nan: {np.any(~np.isfinite(price))}")
				print(f"  avg_demand: {average_demand_projection_long}")
				print(f"  normalization_factor: {self.normalization_factor}")
				raise ValueError(f"NaN in observation for agent {i}")

			observations_g.update({a:observations_temp})
			
			i += 1

		self.cost_total_step = (spot_payment_trimester + energy_not_served_costs_step_tmp)

		self.price_net = (spot_payment_trimester + energy_not_served_costs_step_tmp)/(demand_trimester + delta)

		self.carbon_tax_returns_total = carbon_tax_returns/(demand_trimester + delta)

		self.energy_not_served_percentage = self.energy_not_served_step/(demand_trimester + delta)

		self.demand_trimester_average = demand_trimester * self.hour_month

		## Observations planner 

		# Short-term prices

		index_start_a = 0

		index_last_a = self.n_obs_short_term_prices + self.n_obs_flexibility_prices

		index_start_p = 0

		index_last_p = self.n_obs_short_term_prices_p + self.n_obs_flexibility_prices_p

		obs_temp_p[0,index_start_p:index_last_p] = obs_temp_g[0, index_start_a:index_last_a]

		# Resources

		index_start_a = self.n_obs_short_term_prices + self.n_obs_flexibility_prices

		index_last_a = self.n_obs_short_term_prices + self.n_obs_flexibility_prices + self.n_obs_resources

		index_start_p = self.n_obs_short_term_prices_p + self.n_obs_flexibility_prices_p

		index_last_p = self.n_obs_short_term_prices_p + self.n_obs_flexibility_prices_p + self.n_obs_resources_p

		obs_temp_p[0,index_start_p:index_last_p] = obs_temp_g[0, index_start_a:index_last_a]

		# Installed capacities techonologies

		for j in range(self.n_tech):

			index_start_a = self.n_obs_short_term_prices + self.n_obs_flexibility_prices+ self.n_obs_resources + (self.n_obs_cap_ind * self.n_tech) + (self.n_obs_cap_total * j) + 0

			index_last_a = self.n_obs_short_term_prices + self.n_obs_flexibility_prices+ self.n_obs_resources + (self.n_obs_cap_ind * self.n_tech) + (self.n_obs_cap_total * j) + self.n_obs_cap_total

			index_start_p = self.n_obs_short_term_prices_p + self.n_obs_flexibility_prices_p + self.n_obs_resources_p + (self.n_obs_cap_ind_p * self.n_tech) + (self.n_obs_cap_total_p * j) + 0

			index_last_p = self.n_obs_short_term_prices_p + self.n_obs_flexibility_prices_p + self.n_obs_resources_p + (self.n_obs_cap_ind_p * self.n_tech) + (self.n_obs_cap_total_p * j) + self.n_obs_cap_total_p

			obs_temp_p[0,index_start_p:index_last_p] = obs_temp_g[0, index_start_a:index_last_a]

		# Installed capacities batteries

		for j in range(self.n_tech_battery):

			index_start_a = self.n_obs_short_term_prices + self.n_obs_flexibility_prices + self.n_obs_resources + (self.n_obs_cap_ind * self.n_tech) + (self.n_obs_cap_total * self.n_tech) + (self.n_obs_battery_cap_ind * self.n_tech_battery) + (self.n_obs_battery_cap_total * j) + 0

			index_last_a = self.n_obs_short_term_prices + self.n_obs_flexibility_prices + self.n_obs_resources + (self.n_obs_cap_ind * self.n_tech) + (self.n_obs_cap_total * self.n_tech) + (self.n_obs_battery_cap_ind * self.n_tech_battery) + (self.n_obs_battery_cap_total * j) + self.n_obs_battery_cap_total

			index_start_p = self.n_obs_short_term_prices_p + self.n_obs_flexibility_prices_p + self.n_obs_resources_p + (self.n_obs_cap_ind_p * self.n_tech) + (self.n_obs_cap_total_p * self.n_tech) + (self.n_obs_battery_cap_ind_p * self.n_tech_battery) + (self.n_obs_battery_cap_total_p * j) + 0

			index_last_p = self.n_obs_short_term_prices_p + self.n_obs_flexibility_prices_p+ self.n_obs_resources_p + (self.n_obs_cap_ind_p * self.n_tech) + (self.n_obs_cap_total_p * self.n_tech) + (self.n_obs_battery_cap_ind_p * self.n_tech_battery) + (self.n_obs_battery_cap_total_p * j) + self.n_obs_battery_cap_total_p

			obs_temp_p[0,index_start_p:index_last_p] = obs_temp_g[0, index_start_a:index_last_a]

		# Time

		index_start_a = self.n_obs_short_term_prices + self.n_obs_flexibility_prices + self.n_obs_resources + (self.n_obs_cap_ind * self.n_tech) + (self.n_obs_cap_total * self.n_tech) + (self.n_obs_battery_cap_ind * self.n_tech_battery) + (self.n_obs_battery_cap_total * self.n_tech_battery) + 0

		index_last_a = self.n_obs_short_term_prices + self.n_obs_flexibility_prices + self.n_obs_resources + (self.n_obs_cap_ind * self.n_tech) + (self.n_obs_cap_total * self.n_tech) + (self.n_obs_battery_cap_ind * self.n_tech_battery) + (self.n_obs_battery_cap_total * self.n_tech_battery) + self.n_observation_time

		index_start_p = self.n_obs_short_term_prices_p + self.n_obs_flexibility_prices_p + self.n_obs_resources_p + (self.n_obs_cap_ind_p * self.n_tech) + (self.n_obs_cap_total_p * self.n_tech) + (self.n_obs_battery_cap_ind_p * self.n_tech_battery) + (self.n_obs_battery_cap_total_p * self.n_tech_battery) + 0

		index_last_p = self.n_obs_short_term_prices_p + self.n_obs_flexibility_prices_p + self.n_obs_resources_p + (self.n_obs_cap_ind_p * self.n_tech) + (self.n_obs_cap_total_p * self.n_tech) + (self.n_obs_battery_cap_ind_p * self.n_tech_battery) + (self.n_obs_battery_cap_total_p * self.n_tech_battery) + self.n_observation_time_p

		obs_temp_p[0,index_start_p:index_last_p] = obs_temp_g[0, index_start_a:index_last_a]

		# CfD + CM markets 

		index_start_a = self.n_obs_short_term_prices + self.n_obs_flexibility_prices + self.n_obs_resources + (self.n_obs_cap_ind * self.n_tech) + (self.n_obs_cap_total * self.n_tech) + (self.n_obs_battery_cap_ind * self.n_tech_battery) + (self.n_obs_battery_cap_total * self.n_tech_battery) + self.n_observation_time + 0

		index_last_a = self.n_obs_short_term_prices + self.n_obs_flexibility_prices + self.n_obs_resources + (self.n_obs_cap_ind * self.n_tech) + (self.n_obs_cap_total * self.n_tech) + (self.n_obs_battery_cap_ind * self.n_tech_battery) + (self.n_obs_battery_cap_total * self.n_tech_battery) + self.n_observation_time + self.n_observation_cm_CfD

		index_start_p = self.n_obs_short_term_prices_p + self.n_obs_flexibility_prices_p + self.n_obs_resources_p + (self.n_obs_cap_ind_p * self.n_tech) + (self.n_obs_cap_total_p * self.n_tech) + (self.n_obs_battery_cap_ind_p * self.n_tech_battery) + (self.n_obs_battery_cap_total_p * self.n_tech_battery) + self.n_observation_time_p + 0

		index_last_p = self.n_obs_short_term_prices_p + self.n_obs_flexibility_prices_p + self.n_obs_resources_p + (self.n_obs_cap_ind_p * self.n_tech) + (self.n_obs_cap_total_p * self.n_tech) + (self.n_obs_battery_cap_ind_p * self.n_tech_battery) + (self.n_obs_battery_cap_total_p * self.n_tech_battery) + self.n_observation_time_p + self.n_observation_cm_CfD_p

		obs_temp_p[0,index_start_p:index_last_p] = obs_temp_g[0, index_start_a:index_last_a]

		## Accumulating rewards planner

		self.cost_total_step = self.cost_total_step * self.hour_month
		self.cost_cm_step = self.cost_cm_step * self.hour_month
		self.cost_CfD_step = self.cost_CfD_step * self.hour_month
		self.cost_Flexibility_step  = self.cost_Flexibility_step * self.hour_month
		self.cost_scarcity_step = self.cost_scarcity_step * self.hour_month
		self.cost_carbon_tax_return_step = self.cost_carbon_tax_return_step * self.hour_month
		self.cost_merchant_step = self.cost_merchant_step * self.hour_month
		self.cost_existing_step = self.cost_existing_step * self.hour_month

		self.cost_total += self.cost_total_step * self.discount_factor_planner
		self.cost_cm += self.cost_cm_step * self.discount_factor_planner
		self.cost_CfD += self.cost_CfD_step * self.discount_factor_planner
		self.cost_Flexibility += self.cost_Flexibility_step * self.discount_factor_planner

		self.reward_planner_system_cost += spot_payment_trimester * self.discount_factor_planner  * self.hour_month
		self.reward_planner_system_cost_step = spot_payment_trimester * self.discount_factor_planner * self.hour_month
		self.reward_planner_system_cost_not_discounted += spot_payment_trimester * self.hour_month

		self.reward_planner_emissions += self.CO2_emissions_step * self.CO2_social_cost[self.year_int] * self.discount_factor_planner 
		self.reward_planner_emissions_step += self.CO2_emissions_step * self.CO2_social_cost[self.year_int]  * self.discount_factor_planner	
		self.reward_planner_emissions_not_discounted += self.CO2_emissions_step

		self.reward_planner_flexibility += flexibility_cost * self.discount_factor_planner * self.hour_month
		self.reward_planner_flexibility_step += flexibility_cost * self.discount_factor_planner * self.hour_month
		self.reward_planner_flexibility_not_discounted += flexibility_cost * self.hour_month

		self.reward_planner_taxes += self.taxes_step * self.discount_factor_planner 
		self.reward_planner_taxes_step = self.taxes_step * self.discount_factor_planner	
		self.reward_planner_taxes_not_discounted += self.taxes_step

		self.reward_planner_rent += self.tax_rent_step * self.discount_factor_planner 

		self.reward_planner_adequacy += self.energy_not_served_step * self.VoLL_planner * self.discount_factor_planner * self.hour_month
		self.reward_planner_adequacy_step += self.energy_not_served_step * self.VoLL_planner * self.discount_factor_planner * self.hour_month
		
		## Reward planner --  Total and step

		# Total
		
		reward_planner_temp = float(0)

		reward_planner_temp -= (self.reward_planner_system_cost + self.reward_planner_emissions + self.reward_planner_depreciation + self.reward_planner_adequacy) / self.normalization_factor_planner
			
		reward_planner_temp += (self.reward_planner_taxes) / self.normalization_factor_planner

		# Step

		reward_planner_temp_step = float(0)

		reward_planner_temp_step -= (self.reward_planner_system_cost_step + self.reward_planner_emissions_step + self.reward_planner_adequacy_step) / self.normalization_factor_planner
		
		# Taxes and discounting the equivalent taxes of the depreciation for agents

		reward_planner_temp_step += (self.reward_planner_taxes_step - self.reward_planner_depreciation_step * self.corporate_tax_rate) / self.normalization_factor_planner

		# Total reward planner

		obs_temp_p[0,index_last_p + 0] = np.clip((-reward_planner_temp), 0 , 1)

		# Emissions

		obs_temp_p[0,index_last_p + 1] = np.clip((self.reward_planner_emissions) / self.normalization_factor_planner, 0 , 1)

		# Taxes

		obs_temp_p[0,index_last_p + 2] = np.clip(self.reward_planner_taxes / self.normalization_factor_planner, 0, 1)

		# Merchant

		obs_temp_p[0,index_last_p + 3] = np.clip((self.cost_total) / self.normalization_factor_planner, 0 , 1)

		# CM

		obs_temp_p[0,index_last_p + 4] = np.clip(self.cost_cm / self.normalization_factor_planner, 0, 1)
		
		# CfD

		obs_temp_p[0,index_last_p + 5] = np.clip(self.cost_CfD / self.normalization_factor_planner, 0, 1)

		# Flexibility

		obs_temp_p[0,index_last_p + 6] = np.clip(self.cost_Flexibility / self.normalization_factor_planner, 0, 1)

		# Discount rate

		obs_temp_p[0,index_last_p + 7] = np.clip((self.discount_factor_planner) * 2 - 1, -1, 1)

		self.obs_p = obs_temp_p
		
		i = 0

		for a in self.possible_agents_p:

			observations_temp = {"observations": np.float32(obs_temp_p[0,:]),
									"action_mask":self.get_action_masks_p()} 
			
			observations_p.update({a:observations_temp})
			
			# rewards_p.update({a:reward_planner_temp})

			i += 1
		
		i = 0

		for a in self.possible_agents_p:

			# Terminal reward active

			if self.terminal_reward_flag == True:

				if self.time >= self.max_t: 
					
					if self.public_generators == True:

						original_reward_planner = float(reward_planner_temp)
						
						# Sum rewards from all public agents (0 if empty)
						
						total_public_agent_rewards = sum(
							float(reward_temp[agent_idx]) for agent_idx in self.agent_g_public
						)
						
						# Updating reward of planner

						updated_reward_planner = original_reward_planner + total_public_agent_rewards * self.agents_g
						rewards_p.update({a: float(updated_reward_planner)})
						
						# Update each public generator

						for agent_idx in self.agent_g_public:
								a_agent_g_public = self.possible_agents_g[agent_idx]
								original_reward_agent = float(reward_temp[agent_idx])
								updated_reward_agent = original_reward_planner/self.agents_g + original_reward_agent
								rewards_g.update({a_agent_g_public: float(updated_reward_agent)})

					elif self.lobby == True:

						original_reward_planner = float(reward_planner_temp)
						
						# Sum rewards from all public agents (0 if empty)
						
						total_lobby_agent_rewards = sum(
							float(reward_temp[agent_idx]) for agent_idx in self.agent_g_lobby
						)
						
						# Updating reward of planner (No need to mixed rewards of agents)

						updated_reward_planner = original_reward_planner + total_lobby_agent_rewards * self.agents_g
						rewards_p.update({a: float(updated_reward_planner)})

					else:

						updated_reward_planner = float(reward_planner_temp)
						rewards_p.update({a: float(updated_reward_planner)})
					
				else:
				
					rewards_p.update({a: float(0)})

			# Terminal reward not active
			
			else:
					
				if self.public_generators == True:

					original_reward_planner = float(reward_planner_temp_step)
						
					# Sum rewards from all public agents (0 if empty)
						
					total_public_agent_rewards = sum(
							float(reward_temp[agent_idx]) for agent_idx in self.agent_g_public
						)
						
					# Updating reward of planner

					updated_reward_planner = original_reward_planner + total_public_agent_rewards * self.agents_g
					rewards_p.update({a: float(updated_reward_planner)})
						
					# Update each public generator

					for agent_idx in self.agent_g_public:
								a_agent_g_public = self.possible_agents_g[agent_idx]
								original_reward_agent = float(reward_temp[agent_idx])
								updated_reward_agent = original_reward_planner/self.agents_g + original_reward_agent
								rewards_g.update({a_agent_g_public: float(updated_reward_agent)})

				elif self.lobby == True:

					original_reward_planner = float(reward_planner_temp_step)
						
					# Sum rewards from all public agents (0 if empty)
						
					total_lobby_agent_rewards = sum(
							float(reward_temp[agent_idx]) for agent_idx in self.agent_g_lobby
						)
						
					# Updating reward of planner (No need to mixed rewards of agents)

					updated_reward_planner = original_reward_planner + total_lobby_agent_rewards * self.agents_g
					rewards_p.update({a: float(updated_reward_planner)})

				else:

					updated_reward_planner = float(reward_planner_temp_step)
					rewards_p.update({a: float(updated_reward_planner)})


			self.reward_agents_p = updated_reward_planner 
					
		##  Termination conditions for the Environment

		if self.time >= self.max_t: 
			
			i = 0
			
			for a in self.possible_agents:
				
				self.terminateds.add(a)
				self.truncateds.add(a) 

				i += 1

			terminated = {a: True for a in self.possible_agents}
			truncated = {a: True for a in self.possible_agents}

		observations = dict(observations_g, **observations_p)

		rewards = dict(rewards_g, ** rewards_p)

		# Counting years and months
		for j in range(self.n_tech):
			if self.year > self.max_year - self.years_slack_profits:
				self.count_last[j] += 1/self.yearly_resolution

		for j in range(self.n_tech_battery):
			if self.year > self.max_year - self.years_slack_profits:
				self.count_last_battery[j] += 1/self.yearly_resolution

		self.year += 1/self.yearly_resolution

		self.month += 1

		# New year condition

		if self.month >= self.yearly_resolution:

			self.month = 0
			self.hour_year = 0

			# Aging update for existing assets

			self.inv_age[self.inv_new > 0] += 1

			self.inv_age_cm[self.capacity_cm > 0] += 1

			self.inv_age_CfD[self.capacity_CfD > 0] += 1

			self.inv_age_merchant[self.capacity_merchant > 0] += 1

			self.inv_age_merchant_battery[self.capacity_merchant_battery > 0] += 1

			self.inv_age_cm_battery[self.capacity_cm_battery > 0] += 1

			self.inv_age_flexibility_battery[self.capacity_flexibility_battery > 0] += 1

			# Aging update for existing assets

			for j in range(self.n_tech):

				self.inv_g[:,j] -= self.inv_init_g[:,j] * self.aging[self.year_int , j]
				self.inv_g_aging[:,j] -= self.inv_init_g[:,j] * self.aging[self.year_int , j]
			
			# Avoiding zeros
			self.inv_g =np.maximum(self.inv_g, 0)
			self.inv_g [self.inv_g  < 1] = 0
			self.inv_g_aging = np.maximum(self.inv_g_aging, 0)
			self.inv_g_aging [self.inv_g_aging  < 1] = 0

		# Eliminating agents after termination/truncation (for RLLIB compatibility)
		terminated["__all__"] = len(self.terminateds) == len(self.agents)
		truncated["__all__"] = len(self.truncateds) == len(self.agents)

		return observations, rewards, terminated, truncated, info
	
	def double_side_auction(self, price_u, q_u_o, price_g, q_g_o, random, price_cap):
		
		total_demand = np.sum(q_u_o)
		total_supply = np.sum(q_g_o)

		q_g = copy.deepcopy(q_g_o)
		q_u = copy.deepcopy(q_u_o)
		
		if total_supply >= total_demand:

			energy_not_served = 0
			
			LL = False  # No load loss

            # Step 2: Sort buyers by descending price (high to low)
			sorted_buy_indices = np.argsort(-price_u)
			sorted_price_u = price_u[sorted_buy_indices]
			sorted_q_u = q_u[sorted_buy_indices]

			sorted_sell_indices = np.lexsort((np.random.random(len(price_g)), price_g))
			sorted_price_g = price_g[sorted_sell_indices]
			sorted_q_g = q_g[sorted_sell_indices]

            # Initialize variables for matching
			buyer_idx = 0
			seller_idx = 0
			final_price = 0

            # Step 4: Matching buyers and sellers
			while buyer_idx < len(sorted_price_u) and seller_idx < len(sorted_price_g):
                # Get the buyer and seller's prices
				buyer_price = sorted_price_u[buyer_idx]
				seller_price = sorted_price_g[seller_idx]

                # Check if buyer is willing to pay more or equal to seller's price
				if buyer_price >= seller_price:
                    # Determine the quantity to be matched
					matched_quantity = min(sorted_q_u[buyer_idx], sorted_q_g[seller_idx])

                    # Reduce quantities of both buyer and seller
					sorted_q_u[buyer_idx] -= matched_quantity
					sorted_q_g[seller_idx] -= matched_quantity

                    # Update the final transaction price as seller's price
					final_price = seller_price

                    # Move to the next buyer if their quantity is fully matched
					if sorted_q_u[buyer_idx] == 0:
						buyer_idx += 1

                    # Move to the next seller if their quantity is fully matched
					if sorted_q_g[seller_idx] == 0:
						seller_idx += 1
				else:
                    # If no match is possible, stop the auction
					break

            # Final accepted quantities (already adjusted in sorted_q_u and sorted_q_g)
			q_accepted_u = np.zeros_like(q_u)
			q_accepted_u[sorted_buy_indices] = q_u[sorted_buy_indices] - sorted_q_u

			q_accepted_g = np.zeros_like(q_g)
			q_accepted_g[sorted_sell_indices] = q_g[sorted_sell_indices] - sorted_q_g

		else:

			energy_not_served = total_demand - total_supply

            # Case where total supply is insufficient to meet demand
			
			LL = True  # Load loss
			
			final_price = price_cap  # Set price to cap

            # Accept all seller quantities
			q_accepted_g = q_g_o

            # Adjust buyer quantities proportionally to available supply
			q_accepted_u = q_u * total_supply / total_demand

		return final_price, q_accepted_u, q_accepted_g, LL, energy_not_served

	def double_side_auction_pay_as_bid(self, price_u, q_u_o, price_g, q_g_o, random, price_cap):
		
		total_demand = np.sum(q_u_o)
		total_supply = np.sum(q_g_o)

		q_g = copy.deepcopy(q_g_o)
		q_u = copy.deepcopy(q_u_o)
		
		if total_supply >= total_demand:	

			LL = False  # No load loss

            # Step 2: Sort buyers by descending price (high to low)
			sorted_buy_indices = np.argsort(-price_u)
			sorted_price_u = price_u[sorted_buy_indices]
			sorted_q_u = q_u[sorted_buy_indices]

			sorted_sell_indices = np.lexsort((np.random.random(len(price_g)), price_g))
			sorted_price_g = price_g[sorted_sell_indices]
			sorted_q_g = q_g[sorted_sell_indices]

            # Initialize variables for matching
			buyer_idx = 0
			seller_idx = 0
			final_price = 0

			# Initialize variables for matching
			buyer_idx = 0
			seller_idx = 0
			final_prices_sellers = np.zeros_like(price_g)  # To store final prices paid to sellers

			# Step 4: Matching buyers and sellers
			while buyer_idx < len(sorted_price_u) and seller_idx < len(sorted_price_g):
				# Get the buyer and seller's prices
				buyer_price = sorted_price_u[buyer_idx]
				seller_price = sorted_price_g[seller_idx]

				# Check if buyer is willing to pay more or equal to seller's price
				if buyer_price >= seller_price:
					# Determine the quantity to be matched
					matched_quantity = min(sorted_q_u[buyer_idx], sorted_q_g[seller_idx])

					# Reduce quantities of both buyer and seller
					sorted_q_u[buyer_idx] -= matched_quantity
					sorted_q_g[seller_idx] -= matched_quantity

					# Record the seller's price for the matched quantity
					final_prices_sellers[sorted_sell_indices[seller_idx]] = seller_price

					# Move to the next buyer if their quantity is fully matched
					if sorted_q_u[buyer_idx] == 0:
						buyer_idx += 1

					# Move to the next seller if their quantity is fully matched
					if sorted_q_g[seller_idx] == 0:
						seller_idx += 1
				else:
					# If no match is possible, stop the auction
					break

			# Final accepted quantities (already adjusted in sorted_q_u and sorted_q_g)
			q_accepted_u = np.zeros_like(q_u)
			q_accepted_u[sorted_buy_indices] = q_u[sorted_buy_indices] - sorted_q_u

			q_accepted_g = np.zeros_like(q_g)
			q_accepted_g[sorted_sell_indices] = q_g[sorted_sell_indices] - sorted_q_g

		else:
			# Case where total supply is insufficient to meet demand
			LL = True  # Load loss
			
			final_prices_sellers = np.full_like(price_g, price_cap)  # Sellers get the capped price
			final_price = price_cap  # Set price to cap

			# Accept all seller quantities
			q_accepted_g = q_g_o

			# Adjust buyer quantities proportionally to available supply
			q_accepted_u = q_u * total_supply / (total_demand + 0.0001)
			
		return final_prices_sellers, q_accepted_u, q_accepted_g, LL
		
	def get_action_masks_g(self, investments_enabled, entry_enabled, capacity_existing_storage_lt, inv_g_aging):
		

		possible_actions = np.ones(int(self.step_g_inv * self.act_inv + self.step_g_bids * self.act_bids_cm + self.step_g_inv * self.act_inv_cm
											  + self.step_g_bids * self.act_bids_CfD + self.step_g_inv * self.act_inv_CfD 
											  + self.step_g_bids * self.act_bids_flexibility + self.step_g_inv * self.act_inv_flexibility
											  + self.step_g_inv * self.act_decom 
											  + self.step_SoC_control * self.act_SoC))


		invalid_actions = []

		# Investment not enabled out of period

		for j in range(self.act_inv):

			if j < self.n_tech:

				construction_time = self.construction_time[j]
			
			else:

				construction_time = self.construction_time_battery[j - self.n_tech]

			if self.month != 0 or self.max_year - self.year - self.years_slack_termination <=  construction_time or self.merchant_activation == False or investments_enabled[j] == 0 or self.year_int < entry_enabled: 

				# Merchant investment 
			
				invalid_actions_temp = list(range(1 + self.step_g_inv * (j), 
									  self.step_g_inv * (j + 1)))
				
				invalid_actions = invalid_actions + invalid_actions_temp
		
			else:
				
				# Merchant investment 
			
				invalid_actions_temp = list(range(self.step_g_inv * (j) + 0, 
									  self.step_g_inv * (j) + 1))
				
				invalid_actions = invalid_actions + invalid_actions_temp

		##  Capacity market masking 

		for j in range(self.act_bids_cm):

			if j < self.n_tech:

				construction_time = self.construction_time[j]
			
			else:

				construction_time = self.construction_time_battery[j - self.n_tech]
		
			if self.year <= self.cm_activation_year or self.month != 2 or self.cm_auction_indicator == False or self.cm_tech_cc[0,j] <= 0.02 or self.max_year - self.year - self.years_slack_termination <= construction_time or investments_enabled[j] == 0 or self.cm_activation == False or self.year_int < entry_enabled: 

				# Capacity Market prices 
				
				invalid_actions_temp = list(range(1 + self.step_g_inv * self.act_inv + self.step_g_bids * (j), 
									  self.step_g_inv * self.act_inv + self.step_g_bids * (j + 1)))
					
				invalid_actions = invalid_actions + invalid_actions_temp

				# Capacity Market Quantities 
				
				invalid_actions_temp = list(range(1 + self.step_g_inv * self.act_inv + self.step_g_bids * self.act_bids_cm + self.step_g_inv * (j),
									  self.step_g_inv * self.act_inv + self.step_g_bids * self.act_bids_cm + self.step_g_inv * (j + 1)))
					
				invalid_actions = invalid_actions + invalid_actions_temp
			
			else:

				# Capacity Market prices 
				
				invalid_actions_temp = list(range(self.step_g_inv * self.act_inv + self.step_g_bids * (j) + 0, 
												+ self.step_g_inv * self.act_inv + self.step_g_bids * (j) + 1))
					
				invalid_actions = invalid_actions + invalid_actions_temp

				# Capacity Market quantities 
				
				invalid_actions_temp = list(range(self.step_g_inv * self.act_inv + self.step_g_bids * self.act_bids_cm + self.step_g_inv * (j) + 0,
									  self.step_g_inv * self.act_inv + self.step_g_bids * self.act_bids_cm + self.step_g_inv * (j) + 1))
					
				invalid_actions = invalid_actions + invalid_actions_temp

		##  CfD masking

		for j in range(self.act_bids_CfD):
		
			if self.year <= self.CfD_activation_year or self.month != 4 or self.CfD_auction_indicator == False or self.max_year - self.year - self.years_slack_termination <= self.construction_time[j] or investments_enabled[j] == 0 or self.CfD_activation == False or self.year_int < entry_enabled: 

				# CfD Market prices 
				
				invalid_actions_temp = list(range(1 + self.step_g_inv * self.act_inv + self.step_g_bids * self.act_bids_cm + self.step_g_inv * self.act_inv_cm
												+ self.step_g_bids * (j),
										self.step_g_inv * self.act_inv + self.step_g_bids * self.act_bids_cm + self.step_g_inv * self.act_inv_cm
												+ self.step_g_bids * (j + 1),))
					
				invalid_actions = invalid_actions + invalid_actions_temp

				# CfD Market quantities 
				
				invalid_actions_temp = list(range(1 + self.step_g_inv * self.act_inv + self.step_g_bids * self.act_bids_cm + self.step_g_inv * self.act_inv_cm
												+ self.step_g_bids * self.act_bids_CfD + self.step_g_inv * (j), 
										self.step_g_inv * self.act_inv + self.step_g_bids * self.act_bids_cm + self.step_g_inv * self.act_inv_cm
												+ self.step_g_bids * self.act_bids_CfD + self.step_g_inv * (j + 1)))
					
				invalid_actions = invalid_actions + invalid_actions_temp
			
			else:

				# CfD Market prices
				
				invalid_actions_temp = list(range(self.step_g_inv * self.act_inv + self.step_g_bids * self.act_bids_cm + self.step_g_inv * self.act_inv_cm 
												+ self.step_g_bids * (j) + 0,
										self.step_g_inv * self.act_inv + self.step_g_bids * self.act_bids_cm + self.step_g_inv * self.act_inv_cm
												+ self.step_g_bids * (j) + 1,))
					
				invalid_actions = invalid_actions + invalid_actions_temp

				# CfD Market quantities 
				
				invalid_actions_temp = list(range(self.step_g_inv * self.act_inv + self.step_g_bids * self.act_bids_cm + self.step_g_inv * self.act_inv_cm
												+ self.step_g_bids * self.act_bids_CfD + self.step_g_inv * (j) + 0, 
										self.step_g_inv * self.act_inv + self.step_g_bids * self.act_bids_cm + self.step_g_inv * self.act_inv_cm
												+ self.step_g_bids * self.act_bids_CfD + self.step_g_inv * (j) + 1))
				
				invalid_actions = invalid_actions + invalid_actions_temp

		##  flexibility masking

		for j in range(self.act_bids_flexibility):

			construction_time = self.construction_time_battery[j]
		
			if self.year <= self.flexibility_activation_year or self.month != 4 or self.flexibility_auction_indicator == False or self.max_year - self.year - self.years_slack_termination <= construction_time or investments_enabled[self.n_tech + j] == 0 or self.flexibility_activation == False or self.year_int < entry_enabled: 

				# flexibility Market prices 
				
				invalid_actions_temp = list(range(1 + self.step_g_inv * self.act_inv + self.step_g_bids * self.act_bids_cm + self.step_g_inv * self.act_inv_cm + self.step_g_bids * self.act_bids_CfD + self.step_g_inv * self.act_inv_CfD
												+ self.step_g_bids * (j),
										self.step_g_inv * self.act_inv + self.step_g_bids * self.act_bids_cm + self.step_g_inv * self.act_inv_cm + self.step_g_bids * self.act_bids_CfD + self.step_g_inv * self.act_inv_CfD
												+ self.step_g_bids * (j + 1),))
					
				invalid_actions = invalid_actions + invalid_actions_temp

				# flexibility Market quantities 
				
				invalid_actions_temp = list(range(1 + self.step_g_inv * self.act_inv + self.step_g_bids * self.act_bids_cm + self.step_g_inv * self.act_inv_cm + self.step_g_bids * self.act_bids_CfD + self.step_g_inv * self.act_inv_CfD
												+ self.step_g_bids * self.act_bids_flexibility + self.step_g_inv * (j), 
										self.step_g_inv * self.act_inv + self.step_g_bids * self.act_bids_cm + self.step_g_inv * self.act_inv_cm + self.step_g_bids * self.act_bids_CfD + self.step_g_inv * self.act_inv_CfD
												+ self.step_g_bids * self.act_bids_flexibility + self.step_g_inv * (j + 1)))
					
				invalid_actions = invalid_actions + invalid_actions_temp
			
			else:

				# flexibility Market prices
				
				invalid_actions_temp = list(range(self.step_g_inv * self.act_inv + self.step_g_bids * self.act_bids_cm + self.step_g_inv * self.act_inv_cm + self.step_g_bids * self.act_bids_CfD + self.step_g_inv * self.act_inv_CfD
												+ self.step_g_bids * (j) + 0,
										self.step_g_inv * self.act_inv + self.step_g_bids * self.act_bids_cm + self.step_g_inv * self.act_inv_cm + self.step_g_bids * self.act_bids_CfD + self.step_g_inv * self.act_inv_CfD
												+ self.step_g_bids * (j) + 1,))
					
				invalid_actions = invalid_actions + invalid_actions_temp

				# flexibility Market quantities 
				
				invalid_actions_temp = list(range(self.step_g_inv * self.act_inv + self.step_g_bids * self.act_bids_cm + self.step_g_inv * self.act_inv_cm + self.step_g_bids * self.act_bids_CfD + self.step_g_inv * self.act_inv_CfD
												+ self.step_g_bids * self.act_bids_flexibility + self.step_g_inv * (j) + 0, 
										self.step_g_inv * self.act_inv + self.step_g_bids * self.act_bids_cm + self.step_g_inv * self.act_inv_cm + self.step_g_bids * self.act_bids_CfD + self.step_g_inv * self.act_inv_CfD
												+ self.step_g_bids * self.act_bids_flexibility + self.step_g_inv * (j) + 1))
				
				invalid_actions = invalid_actions + invalid_actions_temp

		##  Decom

		for j in range(self.act_decom):

			decom_time = self.decom_time[j]
		
			if self.month != 0 or self.max_year - self.year - self.years_slack_termination <= decom_time or inv_g_aging[self.n_tech_decom_offset + j] < 10 or self.decom_activation == False: 

				# Decom quantities 
				
				invalid_actions_temp = list(range(1 + self.step_g_inv * self.act_inv + self.step_g_bids * self.act_bids_cm + self.step_g_inv * self.act_inv_cm + self.step_g_bids * self.act_bids_CfD + self.step_g_inv * self.act_inv_CfD
												+ self.step_g_bids * self.act_bids_flexibility + self.step_g_inv * self.act_inv_flexibility + self.step_g_inv * (j), 
										self.step_g_inv * self.act_inv + self.step_g_bids * self.act_bids_cm + self.step_g_inv * self.act_inv_cm + self.step_g_bids * self.act_bids_CfD + self.step_g_inv * self.act_inv_CfD
												+ self.step_g_bids * self.act_bids_flexibility + self.step_g_inv * self.act_inv_flexibility + self.step_g_inv * (j + 1)))
					
				invalid_actions = invalid_actions + invalid_actions_temp
				
			else:
				
				invalid_actions_temp = list(range(
					self.step_g_inv * self.act_inv + self.step_g_bids * self.act_bids_cm + self.step_g_inv * self.act_inv_cm + self.step_g_bids * self.act_bids_CfD + self.step_g_inv * self.act_inv_CfD
						+ self.step_g_bids * self.act_bids_flexibility + self.step_g_inv * self.act_inv_flexibility + self.step_g_inv * (j) + 0,
					self.step_g_inv * self.act_inv + self.step_g_bids * self.act_bids_cm + self.step_g_inv * self.act_inv_cm + self.step_g_bids * self.act_bids_CfD + self.step_g_inv * self.act_inv_CfD
						+ self.step_g_bids * self.act_bids_flexibility + self.step_g_inv * self.act_inv_flexibility + self.step_g_inv * (j) + 1))
				
				invalid_actions = invalid_actions + invalid_actions_temp

		if capacity_existing_storage_lt <= 0.2:
			
			invalid_actions_temp = list(range(1 + self.step_g_inv * self.act_inv + self.step_g_bids * self.act_bids_cm + self.step_g_inv * self.act_inv_cm
												+ self.step_g_bids * self.act_bids_CfD + self.step_g_inv * self.act_inv_CfD + self.step_g_bids * self.act_bids_flexibility + self.step_g_inv * self.act_inv_flexibility + self.step_g_inv * self.act_decom,
										self.step_g_inv * self.act_inv + self.step_g_bids * self.act_bids_cm + self.step_g_inv * self.act_inv_cm
												+ self.step_g_bids * self.act_bids_CfD + self.step_g_inv * self.act_inv_CfD + self.step_g_bids * self.act_bids_flexibility + self.step_g_inv * self.act_inv_flexibility + self.step_g_inv * self.act_decom
												+ self.act_SoC * self.step_SoC_control))
			
			invalid_actions = invalid_actions + invalid_actions_temp

		## Aggregating invalid actions

		for i in invalid_actions:   

			possible_actions[i] = 0
			
		return np.float32(possible_actions)
		
	def get_action_masks_p(self):
    
		possible_actions = np.ones(int(self.step_planner * self.act_p_CfD_market + self.step_planner * self.act_p_flexibility
										))

		invalid_actions = []

		# Masking for CfD Market

		for j in range(self.act_p_CfD_market):

			base_idx =  self.step_planner * j

			if self.month != 1 or self.year <= self.CfD_activation_year or self.max_year - self.year <= self.years_slack_termination or self.CfD_activation == False:
			
				invalid_actions_temp = list(range(1  + self.step_planner * (j), 
										 self.step_planner * (j + 1)))
				
				invalid_actions = invalid_actions + invalid_actions_temp
			
			else:
			
				invalid_actions_temp = list(range(self.step_planner * (j) + 0, 
									 self.step_planner * (j) + 1))
				
				invalid_actions = invalid_actions + invalid_actions_temp

		# Masking for Flexibility market

		for j in range(self.act_p_flexibility):

			base_idx = self.step_planner * self.act_p_CfD_market + self.step_planner * j

			if self.month != 4 or self.year <= self.flexibility_activation_year or self.max_year - self.year <= self.years_slack_termination or self.flexibility_activation == False:
			
				invalid_actions_temp = list(range(1  + self.step_planner * (self.act_p_CfD_market) + self.step_planner * (j), 
										self.step_planner * (self.act_p_CfD_market) + self.step_planner * (j + 1)))
				
				invalid_actions = invalid_actions + invalid_actions_temp
			
			else:
			
				invalid_actions_temp = list(range(self.step_planner * (self.act_p_CfD_market) + self.step_planner * (j) + 0, 
									self.step_planner * (self.act_p_CfD_market) + self.step_planner * (j) + 1))
				
				invalid_actions = invalid_actions + invalid_actions_temp
		
		## Aggregating invalid actions

		for i in invalid_actions:   

			possible_actions[i] = 0
			
		return np.float32(possible_actions)

	def render(self):
		pass
		
	def calculate_aggregated_failures(self):

		# Finding multiple of 500 (for failure aggregation)

		agent_multiple = np.ceil(self.inv_g / 250).astype(int)

		aggregated_failures = np.zeros([self.agents_g, self.n_tech])

		for i in range(self.agents_g):

			for j in range(self.n_tech):

				failure_temp = 0

				for t in range(agent_multiple[i,j]):

					s_failures = self.random_number_failures[self.time_random_failures]

					failure_temp += (1 + self.scenario_failures[j, s_failures]/100)/agent_multiple[i,j] 

					self.time_random_failures += 1

					if self.time_random_failures >= 52100:
						self.time_random_failures = 0

				aggregated_failures[i,j] = failure_temp

		return aggregated_failures
	
	def top_n_rows(self, matrix, n, high):

		# Sort elements by value (ascending for lowest, descending for highest)
		sorted_indices = np.argsort(matrix, axis=None)
		if high:
			sorted_indices = sorted_indices[::-1]  # Reverse for descending order
		
		# Convert flattened indices to row indices without repetition
		unique_rows = set()
		row_indexes = []
		for idx in sorted_indices:
			row, col = divmod(idx, matrix.shape[1])
			if row not in unique_rows:
				unique_rows.add(row)
				row_indexes.append(row)
			if len(row_indexes) == n:  # Stop after collecting `n` unique rows
				break
		
		return row_indexes

	## CfD Market - Incremental Penetration Approach

	def CfD_balance_estimation(self):
		"""
		Calculate CfD balance based on incremental target above current penetration.
		Auction is triggered only if regulator chose to intervene AND there's a gap.
		"""
		# Current and projected RES energy

		projected_RES_penetration, future_demand, projected_RES_energy = self.get_projected_RES_energy_penetration()

		# Get target penetration (From planner actions)
		
		target_penetration = projected_RES_penetration + self.CfD_increment_target
		target_penetration = min(target_penetration, self.CfD_maximum_target)

		CfD_target_penetration = target_penetration

		# Calculate balance
		target_energy = future_demand * target_penetration / 100
		CfD_balance = target_energy - projected_RES_energy
		
		# Auction only if: (1) regulator chose to intervene, AND (2) gap exists
		CfD_auction_indicator = CfD_balance > 0
		
		# Real balance for metrics (full demand vs RES)
		CfD_balance_real = future_demand - projected_RES_energy
		
		return CfD_balance, CfD_auction_indicator, CfD_balance_real, CfD_target_penetration

	def get_projected_RES_energy_penetration(self):
		"""
		Project RES energy at planning horizon including under construction.
		"""
		projected_RES_energy = 0.0
		CfD_fc_aging = 0.0
		
		for j in range(self.n_tech_RES):
			availability = self.availability_tech_average_yearly[
				self.year_int + self.planning_horizon, j
			]
			failure_rate = self.average_failures[0, j]
			
			# Existing capacity
			projected_RES_energy += np.sum(self.inv_g[:, j]) * availability * failure_rate
			
			# Under construction
			projected_RES_energy += self.cm_fc_under_construction[j] * availability * failure_rate
			
			# Aging deductions
			inv_sum = np.sum(self.inv_init_g[:, j])
			for year_temp in range(int(self.planning_horizon)):
				CfD_fc_aging += (
					inv_sum 
					* self.aging[self.year_int + year_temp, j] 
					* availability 
					* failure_rate
				)
		
		projected_RES_energy -= CfD_fc_aging
		
		# Hydro
		projected_RES_energy += (
			self.availability_tech_average_yearly[
				self.year_int + self.planning_horizon, self.n_tech
			]
			* np.sum(self.capacity_existing_storage_lt)
			* self.average_failures[0, self.n_tech]
		)
		projected_RES_energy += (
			self.availability_tech_average_yearly[
				self.year_int + self.planning_horizon, self.n_tech + 1
			]
			* self.average_failures[0, self.n_tech]
		)
		
		# Calculate penetration percentage
		future_demand = self.demand_average_year[self.year_int + self.planning_horizon]
		projected_RES_penetration = (projected_RES_energy / future_demand) * 100 if future_demand > 0 else 0.0
		
		return projected_RES_penetration, future_demand, projected_RES_energy

	## Flexibility Market - Incremental Capacity Approach (Short-Term Storage Only)

	def flexibility_balance_estimation(self):
		"""
		Calculate flexibility balance based on incremental target above projected flexibility.
		Auction is triggered only if regulator chose to intervene AND there's a gap.
		"""
		# Current and projected flexibility capacity
		projected_flexibility_penetration, future_flexibility_demand, projected_flexibility_capacity = \
			self.get_projected_flexibility_capacity_penetration()

		# Get target penetration (From planner actions)

		target_penetration = projected_flexibility_penetration + self.flexibility_increment_target
		target_penetration = min(target_penetration, self.flexibility_maximum_target)

		flexibility_target_penetration = target_penetration

		# Calculate balance
		target_capacity = future_flexibility_demand * target_penetration / 100
		flexibility_balance = target_capacity - projected_flexibility_capacity

		# Auction only if: (1) regulator chose to intervene, AND (2) gap exists
		flexibility_auction_indicator = flexibility_balance > 0

		# Real balance for metrics (full demand vs flexibility)
		flexibility_balance_real = future_flexibility_demand - projected_flexibility_capacity

		return flexibility_balance, flexibility_auction_indicator, flexibility_balance_real, flexibility_target_penetration


	def get_projected_flexibility_capacity_penetration(self):
		"""
		Project flexibility capacity at planning horizon.
		Considers all battery sources: merchant, capacity market, flexibility market, and initial.
		"""
		# Total projected flexibility from all battery sources
		projected_flexibility_capacity_3_hours = np.sum(
			self.capacity_merchant_battery_uc[:,0] 
			+ self.capacity_cm_battery_uc[:,0] 
			+ self.capacity_flexibility_battery_uc[:,0] 
			+ self.inv_init_battery[:,0]
		) * self.average_failures[0, self.n_tech]

		projected_flexibility_capacity_8_hours = np.sum(
			self.capacity_merchant_battery_uc[:,1] 
			+ self.capacity_cm_battery_uc[:,1] 
			+ self.capacity_flexibility_battery_uc[:,1] 
			+ self.inv_init_battery[:,1]
		) * self.average_failures[0, self.n_tech] * 8/3

		projected_flexibility_capacity = projected_flexibility_capacity_3_hours + projected_flexibility_capacity_8_hours

		# Future demand at planning horizon
		future_flexibility_demand = self.demand_average_year[self.year_int + self.planning_horizon]

		# Calculate penetration percentage
		projected_flexibility_penetration = (
			(projected_flexibility_capacity / future_flexibility_demand) * 100 
			if future_flexibility_demand > 0 else 0.0
		)

		return projected_flexibility_penetration, future_flexibility_demand, projected_flexibility_capacity
