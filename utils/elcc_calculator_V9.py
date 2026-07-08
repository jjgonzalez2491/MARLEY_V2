# elcc_calculator.py

"""
ELCC-based Capacity Market Calculator.
Uses pre-built HiGHS model for fast repeated solves.
Includes short-term batteries (3h and 8h) and long-term hydro storage (8h with inflows).
"""

import numpy as np
import highspy
from scipy.sparse import coo_matrix
from highspy import HighsModelStatus

class ELCCCalculator:
    """
    ELCC-based capacity credit and reliability calculator.
    
    Pre-builds LP structure once, then updates parameters for fast solves.
    
    Parameters
    ----------
    model : object
        Reference to main RL environment model
    debug : bool
        Enable debug printing
    """
    
    def __init__(self, model, debug=False):
        self.model = model
        self.debug = debug
        self.n_tech = model.n_tech
        self.n_tech_battery = model.n_tech_battery  # Number of battery technologies (2: 3h and 8h)
        self.short_t = model.short_t  # 24 hours
        self.yearly_resolution = model.yearly_resolution  # 6 bi-months
        self.max_index = self.short_t * self.yearly_resolution  # 144 hours
        self.n_scenarios = 13

        # Scenario probabilities: 12 k-means clusters + 1 peak demand extreme day
        # Must match the sampling distribution used in the main environment.
        n_base = 12
        n_peak = 1
        self.peak_prob = 0.017
        self.base_prob = (1.0 - self.peak_prob) / n_base
        self.scenario_probs = np.array(
            [self.base_prob] * n_base + [self.peak_prob] * n_peak,
            dtype=np.float64,
        )
        # Sanity: probabilities sum to 1
        assert abs(self.scenario_probs.sum() - 1.0) < 1e-9

        # Threshold below which a scenario's base EUE is considered "no stress"
        # and tech/storage perturbations are skipped (they contribute 0 to ELCC).
        self.eue_skip_threshold = 1.0  # MWh, post-weight
        
        # Reliability parameters
        self.target_eue_fraction = 0.00001
        self.delta_capacity = 10.0
        
        # Short-term storage parameters (per technology)
        # Column 0: 3-hour battery, Column 1: 8-hour battery
        self.st_storage_duration = np.array([3.0, 8.0])  # [3h, 8h]
        self.st_storage_efficiency = np.array([0.95, 0.95])  # Same efficiency for both
        
        # Long-term storage parameters (8-hour hydro pump-storage)
        self.lt_storage_duration = 8.0
        self.lt_storage_efficiency = 0.85
        
        # Period weights
        self.period_weights = np.zeros(self.max_index)
        for p in range(self.yearly_resolution):
            self.period_weights[p * self.short_t:(p + 1) * self.short_t] = 1460.0 / 24.0
        
        # Variable and constraint indexing
        self.var_idx = {}
        self.idx = 0
        
        # Build HiGHS model once
        self.lp = highspy.Highs()
        self.lp.setOptionValue("log_to_console", False)
        
        if self.debug:
            print("=" * 60)
            print("ELCC CALCULATOR INITIALIZATION")
            print("=" * 60)
            print(f"  n_tech:              {self.n_tech}")
            print(f"  n_tech_battery:      {self.n_tech_battery}")
            print(f"  short_t:             {self.short_t}")
            print(f"  yearly_resolution:   {self.yearly_resolution}")
            print(f"  max_index:           {self.max_index}")
            print(f"  delta_capacity:      {self.delta_capacity} MW")
            print(f"  ST storage durations: {self.st_storage_duration} hours")
            print(f"  LT storage duration: {self.lt_storage_duration} hours")
            print("-" * 60)
            print("Building LP model...")
        
        self._build_model()
        
        if self.debug:
            print(f"  Variables:    {self.n_vars}")
            print(f"  Constraints:  {self.n_constraints}")
            print("LP model built successfully.")
            print("=" * 60)
    
    def _add_var(self, name, t):
        """Add variable to index map."""
        self.var_idx[(name, t)] = self.idx
        self.idx += 1
    
    def _build_model(self):
        """Build LP structure once - only topology, not values."""
        
        hours = list(range(self.max_index))
        techs = list(range(self.n_tech))
        storage_techs = list(range(self.n_tech_battery))
        
        row, col, data = [], [], []
        rhs_lower, rhs_upper = [], []
        constraint_idx = 0
        
        # ===== VARIABLES =====
        
        # Generation per technology per hour
        for t in hours:
            for j in techs:
                self._add_var(f"gen_{j}", t)
        
        # Unserved energy per hour
        for t in hours:
            self._add_var("unserved", t)
        
        # Perfect capacity (single variable)
        self._add_var("perfect", 0)
        
        # Short-term storage (for each battery technology)
        for s in storage_techs:
            for t in hours:
                self._add_var(f"st_charge_{s}", t)
                self._add_var(f"st_discharge_{s}", t)
                self._add_var(f"st_soc_{s}", t)
        
        # Long-term storage (8h hydro pump-storage)
        for t in hours:
            self._add_var("lt_charge", t)
            self._add_var("lt_discharge", t)
            self._add_var("lt_soc", t)
        
        self.n_vars = self.idx
        
        # Objective coefficients (minimize weighted unserved energy)
        cost = np.zeros(self.n_vars, dtype=np.float64)
        for t in hours:
            cost[self.var_idx[("unserved", t)]] = self.period_weights[t]
        
        lb = np.zeros(self.n_vars, dtype=np.float64)
        ub = np.full(self.n_vars, highspy.kHighsInf, dtype=np.float64)
        
        # ===== CONSTRAINTS =====
        
        # 1. Generation limits: gen[t,j] <= availability[t,j] * capacity[j] * failure_rate[j]
        self.gen_limit_rows = []
        for t in hours:
            for j in techs:
                self.gen_limit_rows.append(constraint_idx)
                row.append(constraint_idx)
                col.append(self.var_idx[(f"gen_{j}", t)])
                data.append(1.0)
                rhs_lower.append(0.0)
                rhs_upper.append(0.0)  # Will update
                constraint_idx += 1
        
        # 2. Energy balance:
        # sum(gen) + unserved + perfect + sum_s(st_discharge_s - st_charge_s) + lt_discharge - lt_charge >= demand
        self.balance_rows = []
        for t in hours:
            self.balance_rows.append(constraint_idx)
            
            # Generation from all technologies
            for j in techs:
                row.append(constraint_idx)
                col.append(self.var_idx[(f"gen_{j}", t)])
                data.append(1.0)
            
            # Unserved energy
            row.append(constraint_idx)
            col.append(self.var_idx[("unserved", t)])
            data.append(1.0)
            
            # Perfect capacity
            row.append(constraint_idx)
            col.append(self.var_idx[("perfect", 0)])
            data.append(1.0)
            
            # Short-term storage (all technologies)
            for s in storage_techs:
                row.append(constraint_idx)
                col.append(self.var_idx[(f"st_discharge_{s}", t)])
                data.append(1.0)
                row.append(constraint_idx)
                col.append(self.var_idx[(f"st_charge_{s}", t)])
                data.append(-1.0)
            
            # Long-term storage
            row.append(constraint_idx)
            col.append(self.var_idx[("lt_discharge", t)])
            data.append(1.0)
            row.append(constraint_idx)
            col.append(self.var_idx[("lt_charge", t)])
            data.append(-1.0)
            
            rhs_lower.append(0.0)  # Will update with demand
            rhs_upper.append(highspy.kHighsInf)
            constraint_idx += 1
        
        # ===== SHORT-TERM STORAGE CONSTRAINTS (per technology) =====
        
        self.st_charge_limit_rows = {}
        self.st_discharge_limit_rows = {}
        self.st_soc_limit_rows = {}
        self.st_soc_init_rows = {}
        self.st_soc_dynamics_rows = {}
        
        for s in storage_techs:
            # 3. ST charge limit
            self.st_charge_limit_rows[s] = []
            for t in hours:
                self.st_charge_limit_rows[s].append(constraint_idx)
                row.append(constraint_idx)
                col.append(self.var_idx[(f"st_charge_{s}", t)])
                data.append(1.0)
                rhs_lower.append(0.0)
                rhs_upper.append(0.0)  # Will update
                constraint_idx += 1
            
            # 4. ST discharge limit
            self.st_discharge_limit_rows[s] = []
            for t in hours:
                self.st_discharge_limit_rows[s].append(constraint_idx)
                row.append(constraint_idx)
                col.append(self.var_idx[(f"st_discharge_{s}", t)])
                data.append(1.0)
                rhs_lower.append(0.0)
                rhs_upper.append(0.0)  # Will update
                constraint_idx += 1
            
            # 5. ST SOC upper limit
            self.st_soc_limit_rows[s] = []
            for t in hours:
                self.st_soc_limit_rows[s].append(constraint_idx)
                row.append(constraint_idx)
                col.append(self.var_idx[(f"st_soc_{s}", t)])
                data.append(1.0)
                rhs_lower.append(0.0)
                rhs_upper.append(0.0)  # Will update
                constraint_idx += 1
            
            # 6. ST SOC initial (first hour of each period)
            self.st_soc_init_rows[s] = []
            for p in range(self.yearly_resolution):
                t = p * self.short_t
                self.st_soc_init_rows[s].append(constraint_idx)
                row.append(constraint_idx)
                col.append(self.var_idx[(f"st_soc_{s}", t)])
                data.append(1.0)
                row.append(constraint_idx)
                col.append(self.var_idx[(f"st_charge_{s}", t)])
                data.append(-self.st_storage_efficiency[s])
                row.append(constraint_idx)
                col.append(self.var_idx[(f"st_discharge_{s}", t)])
                data.append(1.0)
                rhs_lower.append(0.0)  # Will update with init_soc
                rhs_upper.append(0.0)
                constraint_idx += 1
            
            # 7. ST SOC dynamics (within period)
            self.st_soc_dynamics_rows[s] = []
            for p in range(self.yearly_resolution):
                for h in range(1, self.short_t):
                    t = p * self.short_t + h
                    t_prev = t - 1
                    self.st_soc_dynamics_rows[s].append(constraint_idx)
                    row.append(constraint_idx)
                    col.append(self.var_idx[(f"st_soc_{s}", t)])
                    data.append(1.0)
                    row.append(constraint_idx)
                    col.append(self.var_idx[(f"st_soc_{s}", t_prev)])
                    data.append(-1.0)
                    row.append(constraint_idx)
                    col.append(self.var_idx[(f"st_charge_{s}", t)])
                    data.append(-self.st_storage_efficiency[s])
                    row.append(constraint_idx)
                    col.append(self.var_idx[(f"st_discharge_{s}", t)])
                    data.append(1.0)
                    rhs_lower.append(0.0)
                    rhs_upper.append(0.0)
                    constraint_idx += 1
        
        # ===== LONG-TERM STORAGE CONSTRAINTS (Hydro with inflows) =====
        
        # 8. LT charge limit
        self.lt_charge_limit_rows = []
        for t in hours:
            self.lt_charge_limit_rows.append(constraint_idx)
            row.append(constraint_idx)
            col.append(self.var_idx[("lt_charge", t)])
            data.append(1.0)
            rhs_lower.append(0.0)
            rhs_upper.append(0.0)  # Will update
            constraint_idx += 1
        
        # 9. LT discharge limit
        self.lt_discharge_limit_rows = []
        for t in hours:
            self.lt_discharge_limit_rows.append(constraint_idx)
            row.append(constraint_idx)
            col.append(self.var_idx[("lt_discharge", t)])
            data.append(1.0)
            rhs_lower.append(0.0)
            rhs_upper.append(0.0)  # Will update
            constraint_idx += 1
        
        # 10. LT SOC upper limit
        self.lt_soc_limit_rows = []
        for t in hours:
            self.lt_soc_limit_rows.append(constraint_idx)
            row.append(constraint_idx)
            col.append(self.var_idx[("lt_soc", t)])
            data.append(1.0)
            rhs_lower.append(0.0)
            rhs_upper.append(0.0)  # Will update
            constraint_idx += 1
        
        # 11. LT SOC initial (first hour of each period) - includes inflows
        # soc[t] = init_soc + inflow[t] + eff*charge[t] - discharge[t]
        self.lt_soc_init_rows = []
        for p in range(self.yearly_resolution):
            t = p * self.short_t
            self.lt_soc_init_rows.append(constraint_idx)
            row.append(constraint_idx)
            col.append(self.var_idx[("lt_soc", t)])
            data.append(1.0)
            row.append(constraint_idx)
            col.append(self.var_idx[("lt_charge", t)])
            data.append(-self.lt_storage_efficiency)
            row.append(constraint_idx)
            col.append(self.var_idx[("lt_discharge", t)])
            data.append(1.0)
            rhs_lower.append(0.0)  # Will update with init_soc + inflow[t]
            rhs_upper.append(0.0)
            constraint_idx += 1
        
        # 12. LT SOC dynamics (within period) - includes inflows
        # soc[t] = soc[t-1] + inflow[t] + eff*charge[t] - discharge[t]
        self.lt_soc_dynamics_rows = []
        for p in range(self.yearly_resolution):
            for h in range(1, self.short_t):
                t = p * self.short_t + h
                t_prev = t - 1
                self.lt_soc_dynamics_rows.append(constraint_idx)
                row.append(constraint_idx)
                col.append(self.var_idx[("lt_soc", t)])
                data.append(1.0)
                row.append(constraint_idx)
                col.append(self.var_idx[("lt_soc", t_prev)])
                data.append(-1.0)
                row.append(constraint_idx)
                col.append(self.var_idx[("lt_charge", t)])
                data.append(-self.lt_storage_efficiency)
                row.append(constraint_idx)
                col.append(self.var_idx[("lt_discharge", t)])
                data.append(1.0)
                rhs_lower.append(0.0)  # Will update with inflow[t]
                rhs_upper.append(0.0)
                constraint_idx += 1
        
        # 13. Perfect capacity bound
        self.perfect_cap_row = constraint_idx
        row.append(constraint_idx)
        col.append(self.var_idx[("perfect", 0)])
        data.append(1.0)
        rhs_lower.append(0.0)
        rhs_upper.append(0.0)
        constraint_idx += 1
        
        self.n_constraints = constraint_idx
        
        # Build sparse matrix
        A_coo = coo_matrix((data, (row, col)), shape=(self.n_constraints, self.n_vars))
        A = A_coo.tocsr()
        
        rhs_lower = np.array(rhs_lower, dtype=np.float64)
        rhs_upper = np.array(rhs_upper, dtype=np.float64)
        
        # Add to HiGHS model
        self.lp.addCols(
            self.n_vars,
            cost.astype(np.float64),
            lb.astype(np.float64),
            ub.astype(np.float64),
            0,
            np.array([], dtype=np.int32),
            np.array([], dtype=np.int32),
            np.array([], dtype=np.float64)
        )
        
        self.lp.addRows(
            A.shape[0],
            rhs_lower,
            rhs_upper,
            A.nnz,
            A.indptr.astype(np.int32),
            A.indices.astype(np.int32),
            A.data.astype(np.float64)
        )
    
    def _update_and_solve(self, capacity_vector, demand, availability,
                          st_storage_power=None, lt_storage_power=0, lt_inflows=None,
                          perfect_capacity=0, label=""):
        """Update model parameters and solve."""
        
        hours = list(range(self.max_index))
        techs = list(range(self.n_tech))
        storage_techs = list(range(self.n_tech_battery))
        
        # Default st_storage_power to zeros if not provided
        if st_storage_power is None:
            st_storage_power = np.zeros(self.n_tech_battery)
        
        # Short-term storage (per technology)
        st_energy_capacity = st_storage_power * self.st_storage_duration
        st_init_soc = st_energy_capacity  # Start at 100%
        
        # Long-term storage
        lt_energy_capacity = lt_storage_power * self.lt_storage_duration
        lt_init_soc = 0.5 * lt_energy_capacity  # Start at 50% (will be filled by inflows)
        
        # Default inflows to zero if not provided
        if lt_inflows is None:
            lt_inflows = np.zeros(self.max_index)
        
        if self.debug:
            print(f"\n  [SOLVE] {label}")
            print(f"    Capacity vector:  {np.round(capacity_vector, 1)}")
            for s in storage_techs:
                print(f"    ST storage {s} ({self.st_storage_duration[s]}h): {st_storage_power[s]:.1f} MW ({st_energy_capacity[s]:.1f} MWh)")
            print(f"    LT storage power: {lt_storage_power:.1f} MW ({lt_energy_capacity:.1f} MWh)")
            print(f"    LT inflows:       min={lt_inflows.min():.1f}, max={lt_inflows.max():.1f}, sum={lt_inflows.sum():.1f} MWh")
            print(f"    Perfect cap:      {perfect_capacity:.1f} MW")
            print(f"    Demand range:     [{demand.min():.1f}, {demand.max():.1f}] MW")
        
        # 1. Update generation limits
        count = 0
        for t in hours:
            for j in techs:
                max_gen = (
                    availability[t, j] *
                    capacity_vector[j]
                )
                self.lp.changeRowBounds(self.gen_limit_rows[count], 0.0, max_gen)
                count += 1
        
        # 2. Update balance constraints (demand on RHS)
        for i, t in enumerate(hours):
            self.lp.changeRowBounds(self.balance_rows[i], demand[t], highspy.kHighsInf)
        
        # 3. Update short-term storage limits (per technology)
        for s in storage_techs:
            for i, t in enumerate(hours):
                self.lp.changeRowBounds(self.st_charge_limit_rows[s][i], 0.0, st_storage_power[s])
                self.lp.changeRowBounds(self.st_discharge_limit_rows[s][i], 0.0, st_storage_power[s])
                self.lp.changeRowBounds(self.st_soc_limit_rows[s][i], 0.0, st_energy_capacity[s])
            
            # 4. Update ST SOC initial conditions (100% at start of each period)
            for i in range(self.yearly_resolution):
                self.lp.changeRowBounds(self.st_soc_init_rows[s][i], st_init_soc[s], st_init_soc[s])
        
        # 5. Update long-term storage limits
        for i, t in enumerate(hours):
            self.lp.changeRowBounds(self.lt_charge_limit_rows[i], 0.0, lt_storage_power)
            self.lp.changeRowBounds(self.lt_discharge_limit_rows[i], 0.0, lt_storage_power)
            self.lp.changeRowBounds(self.lt_soc_limit_rows[i], 0.0, lt_energy_capacity)
        
        # 6. Update LT SOC initial conditions (includes inflows)
        for i, p in enumerate(range(self.yearly_resolution)):
            t = p * self.short_t
            # Initial SOC + inflow at first hour
            rhs_value = lt_init_soc + lt_inflows[t]
            self.lp.changeRowBounds(self.lt_soc_init_rows[i], rhs_value, rhs_value)
        
        # 7. Update LT SOC dynamics (includes inflows on RHS)
        count = 0
        for p in range(self.yearly_resolution):
            for h in range(1, self.short_t):
                t = p * self.short_t + h
                # RHS = inflow[t] (gets added to SOC)
                rhs_value = lt_inflows[t]
                self.lp.changeRowBounds(self.lt_soc_dynamics_rows[count], rhs_value, rhs_value)
                count += 1
        
        # 8. Update perfect capacity bound
        self.lp.changeRowBounds(self.perfect_cap_row, perfect_capacity, perfect_capacity)
        
        # Solve
        self.lp.run()
        status = self.lp.getModelStatus()
        
        if status == HighsModelStatus.kOptimal:
            solution = self.lp.getSolution()
            sol = np.array(solution.col_value, dtype=np.float64)
            
            unserved = np.array([
                sol[self.var_idx[("unserved", t)]] for t in hours
            ])
            
            eue = self.lp.getInfo().objective_function_value
            
            if self.debug:
                print(f"    Status:           OPTIMAL")
                print(f"    EUE:              {eue:.2f} MWh")
                print(f"    Max unserved:     {unserved.max():.2f} MW")
                print(f"    Hours w/ deficit: {np.sum(unserved > 0.01)}")
            
            return eue, unserved, 'optimal'
        else:
            # Return a large finite sentinel instead of np.inf to prevent
            # downstream NaN/Inf propagation in ELCC ratio calculations.
            # Sentinel = 10x the theoretical maximum EUE (full demand unserved
            # across all weighted hours), large enough to signal failure but
            # safe for arithmetic operations.
            max_possible_eue = np.sum(demand * self.period_weights)
            fallback_eue = 10.0 * max_possible_eue if max_possible_eue > 0 else 1e9
            if self.debug:
                print(f"    Status:           {status} (NOT OPTIMAL)")
                print(f"    WARNING: Using fallback EUE sentinel = {fallback_eue:.2f} MWh")
            return fallback_eue, np.zeros(self.max_index), str(status)

    def set_reliability_target(self):

        self.target_eue_fraction = self.model.cm_demand_target
    
    def prepare_demand_availability(self, scenario_cm=None):
        """Extract demand, availability, and hydro inflows from model for one scenario.

        Parameters
        ----------
        scenario_cm : int, optional
            Index of the representative day to use. With 12 k-means clusters + 1
            peak day (n_scenarios=13), valid indices are 0..12. Index 12 is the
            peak demand extreme day. If None, defaults to the peak day.
        """
        if scenario_cm is None:
            scenario_cm = self.n_scenarios - 1  # peak day
        
        if self.debug:
            print(f"\n[PREPARE DATA] Scenario {scenario_cm}")
        
        availability = np.zeros([self.max_index, self.n_tech + 1])
        demand = np.zeros([self.max_index])
        lt_inflows = np.zeros([self.max_index])
        
        # Block sizes (assumes self.n_scenarios is set in __init__, e.g., 13)
        bimester_block = self.short_t * self.n_scenarios
        year_block = bimester_block * self.yearly_resolution
        
        for month in range(self.yearly_resolution):
            idx_init = (
                (self.model.year_int + self.model.planning_horizon) * year_block +
                month * bimester_block +
                scenario_cm * self.short_t
            )
            idx_final = idx_init + self.short_t
            
            t_start = self.short_t * month
            t_end = self.short_t * (month + 1)
            
            availability[t_start:t_end, :] = \
                self.model.availability_tech[int(idx_init):int(idx_final), :self.n_tech + 1]
            
            # Demand (already subtracts some availability in original)
            demand[t_start:t_end] = (
                self.model.demand[int(idx_init):int(idx_final)] -
                self.model.availability_tech[int(idx_init):int(idx_final), self.n_tech + 1]
            )
            
            # Hydro inflows (availability_tech[:, n_tech] is the hydro availability factor)
            # Convert to MWh inflow: availability * capacity * hours
            # For now, treat as hourly energy inflow
            lt_inflows[t_start:t_end] = (
                self.model.availability_tech[int(idx_init):int(idx_final), self.n_tech] *
                np.sum(self.model.capacity_existing_storage_lt)
            )
        
        if self.debug:
            print(f"  Demand:       min={demand.min():.1f}, max={demand.max():.1f}, mean={demand.mean():.1f} MW")
            print(f"  LT Inflows:   min={lt_inflows.min():.1f}, max={lt_inflows.max():.1f}, sum={lt_inflows.sum():.1f} MWh")
            print(f"  Availability: shape={availability.shape}")
            for j in range(min(self.n_tech, 5)):
                print(f"    Tech {j}: min={availability[:,j].min():.3f}, max={availability[:,j].max():.3f}, mean={availability[:,j].mean():.3f}")
        
        return demand, availability, lt_inflows
    
    def get_base_capacity(self):
        """Calculate current capacity vector."""
        
        capacity = np.zeros(self.n_tech)
        
        for j in range(self.n_tech):
            capacity[j] = (
                np.sum(self.model.inv_g[:, j]) +
                self.model.cm_fc_under_construction[j]
            )
            
            for year in range(self.model.planning_horizon):
                capacity[j] -= (
                    np.sum(self.model.inv_init_g[:, j]) *
                    self.model.aging[int(self.model.year_int + year), j]
                )
            
            capacity[j] *= self.model.average_failures[0, j]
        
        capacity = np.maximum(capacity, 0)
        
        if self.debug:
            print(f"\n[BASE CAPACITY]")
            for j in range(self.n_tech):
                print(f"  Tech {j}: {capacity[j]:.1f} MW")
        
        return capacity
    
    def get_storage_capacities(self):
        """Get short-term and long-term storage power capacities."""
        
        # Short-term: batteries (per technology)
        st_storage_power = np.zeros(self.n_tech_battery)
        for s in range(self.n_tech_battery):
            st_storage_power[s] = (
                np.sum(self.model.capacity_merchant_battery[:, s] + self.model.capacity_cm_battery[:, s] + self.model.capacity_flexibility_battery[:, s] + self.model.inv_init_battery[:, s]) +
                self.model.cm_fc_under_construction_battery_cm[s]
            ) * self.model.average_failures[0, self.n_tech]
        
        # Long-term: Existing hydro pump-storage (8-hour)
        lt_storage_power = (
            np.sum(self.model.capacity_existing_storage_lt) *
            self.model.average_failures[0, self.n_tech]
        )
        
        if self.debug:
            print(f"\n[STORAGE CAPACITIES]")
            for s in range(self.n_tech_battery):
                print(f"  Short-term {s} ({self.st_storage_duration[s]}h battery):")
                print(f"    Power:  {st_storage_power[s]:.1f} MW")
                print(f"    Energy: {st_storage_power[s] * self.st_storage_duration[s]:.1f} MWh")
            print(f"  Long-term (8h hydro):")
            print(f"    Power:  {lt_storage_power:.1f} MW")
            print(f"    Energy: {lt_storage_power * self.lt_storage_duration:.1f} MWh")
        
        return st_storage_power, lt_storage_power
        
    def _prepare_all_scenarios(self):
        """Extract demand, availability, inflows for every scenario.

        Returns
        -------
        scenarios : list of dict, length n_scenarios
            Each dict has 'idx', 'prob', 'demand', 'availability',
            'lt_inflows', and 'demand_total' (used for warm-start ordering).
        """
        scenarios = []
        for s in range(self.n_scenarios):
            demand_s, avail_s, inflows_s = self.prepare_demand_availability(
                scenario_cm=s
            )
            scenarios.append({
                'idx': s,
                'prob': self.scenario_probs[s],
                'demand': demand_s,
                'availability': avail_s,
                'lt_inflows': inflows_s,
                'demand_total': float(np.sum(demand_s * self.period_weights)),
            })
        return scenarios

    def _compute_scenario_cache(self, scenarios, base_capacity,
                                st_storage_power, lt_storage_power):
        """Pre-compute base_eue and eue_perfect for every scenario.

        Solves are ordered by ascending total demand to maximize warm-start
        basis reuse between consecutive solves.

        Returns
        -------
        cache : dict keyed by scenario index, with fields:
            'base_eue', 'base_unserved', 'base_status',
            'perfect_eue', 'perfect_status'
        """
        cache = {}

        # Solve order: ascending total demand (warm-start friendly)
        order = sorted(range(self.n_scenarios),
                       key=lambda s: scenarios[s]['demand_total'])

        for s in order:
            sc = scenarios[s]
            # Base case (no perfect capacity)
            base_eue, base_unserved, base_status = self._update_and_solve(
                base_capacity, sc['demand'], sc['availability'],
                st_storage_power=st_storage_power,
                lt_storage_power=lt_storage_power,
                lt_inflows=sc['lt_inflows'],
                perfect_capacity=0,
                label=f"[scen {s}] base"
            )

            # Perfect-capacity reference (delta_capacity MW)
            perfect_eue, _, perfect_status = self._update_and_solve(
                base_capacity, sc['demand'], sc['availability'],
                st_storage_power=st_storage_power,
                lt_storage_power=lt_storage_power,
                lt_inflows=sc['lt_inflows'],
                perfect_capacity=self.delta_capacity,
                label=f"[scen {s}] perfect"
            )

            cache[s] = {
                'base_eue': base_eue,
                'base_unserved': base_unserved,
                'base_status': base_status,
                'perfect_eue': perfect_eue,
                'perfect_status': perfect_status,
            }

        return cache

    def calculate_marginal_elcc(self, scenarios, base_capacity,
                                 st_storage_power, lt_storage_power,
                                 cache):
        """Marginal ELCC using the expected-EUE-ratio formulation.

        Instead of averaging per-scenario ELCC ratios, we aggregate the
        probability-weighted EUE reductions in the numerator and denominator
        separately, then take a single ratio:

            ELCC[j] = E_s[ DeltaEUE_s(j) ] / E_s[ DeltaEUE_s(perfect) ]
                    = sum_s p_s * (base_eue_s - perturbed_eue_s)
                      / sum_s p_s * (base_eue_s - perfect_eue_s)

        This matches the standard Garver-style ELCC under a stochastic load /
        availability model: scenarios with high EUE (typically the peak day)
        dominate both numerator and denominator, so a single low-probability
        stress scenario is not diluted by mild scenarios with no EUE.

        Scenarios with base_eue below self.eue_skip_threshold are skipped for
        perturbations (they would contribute 0 to both numerator and
        denominator).
        """
        if self.debug:
            print("\n" + "=" * 60)
            print("CALCULATING MARGINAL ELCC (multi-scenario, E-ratio)")
            print("=" * 60)

        # Probability-weighted EUE reductions, summed across scenarios
        weighted_red_tech = np.zeros(self.n_tech)
        weighted_red_perfect = 0.0

        # Diagnostic per-scenario reductions (not used in aggregation)
        per_scen_red_tech = np.zeros((self.n_scenarios, self.n_tech))
        per_scen_red_perfect = np.zeros(self.n_scenarios)

        for s in range(self.n_scenarios):
            sc = scenarios[s]
            cs = cache[s]

            if cs['base_status'] != 'optimal' or cs['perfect_status'] != 'optimal':
                if self.debug:
                    print(f"  [scen {s}] non-optimal cache, skipping")
                continue

            eue_red_perfect = cs['base_eue'] - cs['perfect_eue']
            per_scen_red_perfect[s] = eue_red_perfect

            # Mild scenario: no stress -> no contribution to either side
            if cs['base_eue'] < self.eue_skip_threshold or eue_red_perfect <= 0:
                if self.debug:
                    print(f"  [scen {s}] base_eue={cs['base_eue']:.2f} "
                          f"below threshold or no perfect reduction; skip perturbations")
                continue

            # This scenario contributes to the denominator
            weighted_red_perfect += self.scenario_probs[s] * eue_red_perfect

            # Tech perturbations contribute to the numerator
            for j in range(self.n_tech):
                perturbed = base_capacity.copy()
                perturbed[j] += self.delta_capacity * self.model.average_failures[0, j]

                eue_perturbed, _, status = self._update_and_solve(
                    perturbed, sc['demand'], sc['availability'],
                    st_storage_power=st_storage_power,
                    lt_storage_power=lt_storage_power,
                    lt_inflows=sc['lt_inflows'],
                    perfect_capacity=0,
                    label=f"[scen {s}] tech {j} +delta"
                )

                if status != 'optimal':
                    continue

                eue_red_tech = cs['base_eue'] - eue_perturbed
                if eue_red_tech > 0:
                    per_scen_red_tech[s, j] = eue_red_tech
                    weighted_red_tech[j] += self.scenario_probs[s] * eue_red_tech

            if self.debug:
                print(f"  [scen {s}] base_eue={cs['base_eue']:.2f} "
                      f"perf_red={eue_red_perfect:.2f} "
                      f"tech_red={np.round(per_scen_red_tech[s], 2)}")

        # Aggregate ratio
        if weighted_red_perfect <= 0:
            if self.debug:
                print("  No scenario contributes positive perfect-capacity "
                      "reduction; system is fully reliable -> ELCC = 0")
            return np.zeros(self.n_tech), per_scen_red_tech

        marginal_elcc = np.minimum(weighted_red_tech / weighted_red_perfect, 1.0)
        # Floor at 0 (negative reductions, if any from solver noise)
        marginal_elcc = np.maximum(marginal_elcc, 0.0)

        if self.debug:
            print(f"\n  Weighted perfect EUE reduction: {weighted_red_perfect:.4f} MWh")
            print("  Aggregated marginal ELCC (E[DeltaEUE_j] / E[DeltaEUE_perfect]):")
            for j in range(self.n_tech):
                print(f"    Tech {j}: weighted_red={weighted_red_tech[j]:.4f}, "
                      f"ELCC={marginal_elcc[j]:.4f} ({marginal_elcc[j]*100:.1f}%)")

        return marginal_elcc, per_scen_red_tech

    def calculate_storage_elcc(self, scenarios, base_capacity,
                                st_storage_power, lt_storage_power,
                                cache):
        """Storage ELCC using the expected-EUE-ratio formulation.

        Same logic as calculate_marginal_elcc: probability-weighted EUE
        reductions are summed across scenarios in numerator and denominator
        independently, then divided. Final ratio capped at 0.95.
        """
        if self.debug:
            print("\n" + "=" * 60)
            print("CALCULATING STORAGE ELCC (multi-scenario, E-ratio)")
            print("=" * 60)

        weighted_red_storage = np.zeros(self.n_tech_battery)
        weighted_red_perfect = 0.0

        per_scen_red_storage = np.zeros((self.n_scenarios, self.n_tech_battery))

        storage_delta = self.delta_capacity * self.model.average_failures[0, self.n_tech]

        for s in range(self.n_scenarios):
            sc = scenarios[s]
            cs = cache[s]

            if cs['base_status'] != 'optimal' or cs['perfect_status'] != 'optimal':
                continue

            eue_red_perfect = cs['base_eue'] - cs['perfect_eue']

            if cs['base_eue'] < self.eue_skip_threshold or eue_red_perfect <= 0:
                continue

            weighted_red_perfect += self.scenario_probs[s] * eue_red_perfect

            for st in range(self.n_tech_battery):
                st_perturbed = st_storage_power.copy()
                st_perturbed[st] += storage_delta

                eue_more_storage, _, status = self._update_and_solve(
                    base_capacity, sc['demand'], sc['availability'],
                    st_storage_power=st_perturbed,
                    lt_storage_power=lt_storage_power,
                    lt_inflows=sc['lt_inflows'],
                    perfect_capacity=0,
                    label=f"[scen {s}] storage {st} +delta"
                )

                if status != 'optimal':
                    continue

                eue_red_storage = cs['base_eue'] - eue_more_storage
                if eue_red_storage > 0:
                    per_scen_red_storage[s, st] = eue_red_storage
                    weighted_red_storage[st] += self.scenario_probs[s] * eue_red_storage

            if self.debug:
                print(f"  [scen {s}] base_eue={cs['base_eue']:.2f} "
                      f"perf_red={eue_red_perfect:.2f} "
                      f"storage_red={np.round(per_scen_red_storage[s], 2)}")

        if weighted_red_perfect <= 0:
            if self.debug:
                print("  No scenario contributes positive perfect-capacity "
                      "reduction; storage ELCC = 0")
            return np.zeros(self.n_tech_battery), per_scen_red_storage

        storage_elcc = np.minimum(weighted_red_storage / weighted_red_perfect, 0.95)
        storage_elcc = np.maximum(storage_elcc, 0.0)

        if self.debug:
            print(f"\n  Weighted perfect EUE reduction: {weighted_red_perfect:.4f} MWh")
            print("  Aggregated storage ELCC (E[DeltaEUE_st] / E[DeltaEUE_perfect]):")
            for st in range(self.n_tech_battery):
                print(f"    Storage {st} ({self.st_storage_duration[st]}h): "
                      f"weighted_red={weighted_red_storage[st]:.4f}, "
                      f"ELCC={storage_elcc[st]:.4f} ({storage_elcc[st]*100:.1f}%)")

        return storage_elcc, per_scen_red_storage

    def calculate_reliability_target(self, scenarios, base_capacity,
                                     st_storage_power, lt_storage_power,
                                     marginal_elcc, cache):
        """Firm-capacity gap using expected EUE across scenarios.

        Consistent with the expected-EUE-ratio ELCC: rather than averaging
        per-scenario additional-MW estimates, we aggregate EUE first and
        then compute one additional-MW figure from the expectation:

            E[current_eue]   = sum_s p_s * base_eue_s
            E[target_eue]    = sum_s p_s * (demand_total_s * target_fraction)
            E[reduction/MW]  = sum_s p_s * (base_eue_s - perfect_eue_s) / delta_capacity
            additional_needed = max(0, (E[current_eue] - E[target_eue]) /
                                       E[reduction/MW])

        Hourly_unserved is still returned as a probability-weighted average
        across scenarios for downstream consumers.
        """
        if self.debug:
            print("\n" + "=" * 60)
            print("CALCULATING RELIABILITY TARGET (multi-scenario, E-based)")
            print("=" * 60)

        weighted_current_eue = 0.0
        weighted_target_eue = 0.0
        weighted_red_perfect = 0.0
        weighted_unserved = np.zeros(self.max_index)
        valid_prob_mass = 0.0

        for s in range(self.n_scenarios):
            sc = scenarios[s]
            cs = cache[s]

            if cs['base_status'] != 'optimal':
                if self.debug:
                    print(f"  [scen {s}] base non-optimal, skipping")
                continue

            p_s = self.scenario_probs[s]
            valid_prob_mass += p_s

            weighted_current_eue += p_s * cs['base_eue']
            weighted_target_eue += p_s * (sc['demand_total'] * self.target_eue_fraction)
            weighted_unserved += p_s * cs['base_unserved']

            if cs['perfect_status'] == 'optimal':
                eue_red_perfect_s = cs['base_eue'] - cs['perfect_eue']
                if eue_red_perfect_s > 0:
                    weighted_red_perfect += p_s * eue_red_perfect_s

            if self.debug:
                print(f"  [scen {s}] p={p_s:.4f} "
                      f"base_eue={cs['base_eue']:.2f} "
                      f"target_eue_s={sc['demand_total'] * self.target_eue_fraction:.2f}")

        if valid_prob_mass <= 0:
            current_firm = float(np.sum(base_capacity * marginal_elcc))
            return current_firm, 0.0, 0.0, np.zeros(self.max_index)

        # Renormalize aggregates if some scenarios were dropped due to
        # non-optimal solves (preserves the expectation interpretation)
        if valid_prob_mass < 1.0:
            weighted_current_eue /= valid_prob_mass
            weighted_target_eue /= valid_prob_mass
            weighted_red_perfect /= valid_prob_mass
            weighted_unserved /= valid_prob_mass

        # Expected reduction per MW from a delta_capacity-MW perfect injection
        weighted_red_per_mw = weighted_red_perfect / self.delta_capacity

        eue_gap = weighted_current_eue - weighted_target_eue
        if eue_gap > 0 and weighted_red_per_mw > 0:
            additional_needed = eue_gap / weighted_red_per_mw
        else:
            additional_needed = 0.0

        current_firm = float(np.sum(base_capacity * marginal_elcc))
        target_firm = current_firm + additional_needed

        if self.debug:
            print(f"\n  E[current EUE]:           {weighted_current_eue:.4f} MWh")
            print(f"  E[target EUE]:            {weighted_target_eue:.4f} MWh")
            print(f"  E[gap]:                   {eue_gap:.4f} MWh")
            print(f"  E[reduction per MW]:      {weighted_red_per_mw:.6f} MWh/MW")
            print(f"  Additional firm needed:   {additional_needed:.1f} MW")
            print(f"  Current firm equivalent:  {current_firm:.1f} MW")
            print(f"  Target firm capacity:     {target_firm:.1f} MW")

        return target_firm, additional_needed, weighted_current_eue, weighted_unserved

    def cm_balance_estimation(self):
        """
        Multi-scenario CM balance estimation.

        Returns
        -------
        cm_balance : float
        cm_auction_indicator : bool
        cm_tech_cc : np.ndarray [1, n_tech + n_tech_battery]
        hourly_balance : np.ndarray
        cm_balance_real : float
        """
        if self.debug:
            print("\n" + "#" * 60)
            print("# CM BALANCE ESTIMATION - START (multi-scenario)")
            print("#" * 60)

        # 1. Build per-scenario data once
        scenarios = self._prepare_all_scenarios()

        # 2. Capacities (independent of scenario)
        base_capacity = self.get_base_capacity()
        st_storage_power, lt_storage_power = self.get_storage_capacities()

        # 3. Reliability target fraction (uses model.cm_demand_target as before)
        self.set_reliability_target()

        if self.debug:
            print(f"\n[DEMAND TARGET]")
            print(f"  cm_demand_target: {self.model.cm_demand_target}")
            for s in range(self.n_scenarios):
                d = scenarios[s]['demand']
                print(f"  scen {s} (p={self.scenario_probs[s]:.4f}): "
                      f"demand min={d.min():.1f} max={d.max():.1f}")

        # 4. Cache base_eue and perfect_eue per scenario (warm-start friendly order)
        cache = self._compute_scenario_cache(
            scenarios, base_capacity, st_storage_power, lt_storage_power
        )

        # 5. Marginal ELCC across scenarios (uses cache; selective perturbation)
        marginal_elcc, _ = self.calculate_marginal_elcc(
            scenarios, base_capacity, st_storage_power, lt_storage_power, cache
        )

        # 6. Reliability target across scenarios (uses cache)
        target_firm, additional_needed, current_eue, hourly_unserved = \
            self.calculate_reliability_target(
                scenarios, base_capacity, st_storage_power, lt_storage_power,
                marginal_elcc, cache
            )

        # 7. Storage ELCC across scenarios (uses cache; selective perturbation)
        storage_elcc, _ = self.calculate_storage_elcc(
            scenarios, base_capacity, st_storage_power, lt_storage_power, cache
        )

        # 8. Format results (same shape as V8)
        cm_balance = additional_needed
        cm_auction_indicator = cm_balance > 0

        cm_tech_cc = np.zeros([1, self.n_tech + self.n_tech_battery])
        cm_tech_cc[0, :self.n_tech] = marginal_elcc
        cm_tech_cc[0, self.n_tech:self.n_tech + self.n_tech_battery] = storage_elcc

        hourly_balance = -hourly_unserved

        # 9. "Real" balance under fixed reliability fraction (1e-5).
        #    Re-uses the same cache because base/perfect EUEs do not depend
        #    on target_eue_fraction.
        if self.debug:
            print("\n" + "=" * 60)
            print("CALCULATING REAL BALANCE (fixed target_eue_fraction)")
            print("=" * 60)

        prev_target = self.target_eue_fraction
        self.target_eue_fraction = 0.00001
        try:
            _, additional_real, _, _ = self.calculate_reliability_target(
                scenarios, base_capacity, st_storage_power, lt_storage_power,
                marginal_elcc, cache
            )
        finally:
            self.target_eue_fraction = prev_target
        cm_balance_real = additional_real

        if self.debug:
            print("\n" + "#" * 60)
            print("# CM BALANCE ESTIMATION - RESULTS (multi-scenario)")
            print("#" * 60)
            print(f"  target_eue_frac:      {self.target_eue_fraction}")
            print(f"  CM Balance (market):  {cm_balance:.1f} MW")
            print(f"  CM Balance (real):    {cm_balance_real:.1f} MW")
            print(f"  Auction needed:       {cm_auction_indicator}")
            print(f"  Target firm capacity: {target_firm:.1f} MW")
            print(f"\n  Capacity Credits:")
            for j in range(self.n_tech):
                print(f"    Tech {j}: {cm_tech_cc[0, j]:.4f} "
                      f"({cm_tech_cc[0, j]*100:.1f}%)")
            for st in range(self.n_tech_battery):
                print(f"    ST Storage {st} ({self.st_storage_duration[st]}h): "
                      f"{cm_tech_cc[0, self.n_tech + st]:.4f} "
                      f"({cm_tech_cc[0, self.n_tech + st]*100:.1f}%)")
            print("#" * 60 + "\n")

        return (cm_balance, cm_auction_indicator, cm_tech_cc,
                hourly_balance, cm_balance_real)
