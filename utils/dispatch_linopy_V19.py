import numpy as np
import highspy
from scipy.sparse import coo_matrix
from highspy import HighsModelStatus

class DispatchHighsRL:
    def __init__(self, n_agents, short_t, n_tech, short_term_eff_charge, short_term_eff_discharge, long_term_eff_charge, long_term_eff_discharge, coal_tech_idx):
        self.n_batteries = 2
        self.n_agents = n_agents
        self.short_t = short_t
        self.n_tech = n_tech
        # Tech indices considered as Sardinian must-run coal (aggregated).
        # Stored as a numpy int array so it can be used directly in the model construction.
        self.coal_tech_idx = np.atleast_1d(np.asarray(coal_tech_idx, dtype=np.int64))
        self.var_idx = {}
        self.idx = 0
        self.model = highspy.Highs()

        self._build_model(short_term_eff_charge, short_term_eff_discharge, long_term_eff_charge, long_term_eff_discharge)

    def _add_var(self, name, t, i=0):
        self.var_idx[(name, t, i)] = self.idx
        self.idx += 1

    def _build_model(self, short_term_eff_charge, short_term_eff_discharge, long_term_eff_charge, long_term_eff_discharge):
        time_steps = np.arange(self.short_t)
        tech = np.arange(self.n_tech)
        agents = np.arange(self.n_agents)
        n_batteries = self.n_batteries
        
        row, col, data = [], [], []
        rhs_lower, rhs_upper = [], []
        constraint_idx = 0

        # --- Map variables ---
        for t in time_steps:
            for i in tech:
                self._add_var("Q_dispatched", t, i)
        for t in time_steps:
            self._add_var("Q_slack", t)
        for t in time_steps:
            for b in range(n_batteries):
                self._add_var("charge_short", t, b)
                self._add_var("discharge_short", t, b)
                self._add_var("SoC_short", t, b)
        for t in time_steps:
            for a in agents:
                self._add_var("charge_long", t, a)
                self._add_var("discharge_long", t, a)
                self._add_var("SoC_long", t, a)
                self._add_var("dumping", t, a)

        self.n_vars = self.idx
        
        cost = np.zeros(self.n_vars, dtype=np.float64)
        lb = np.zeros(self.n_vars, dtype=np.float64)
        ub = np.full(self.n_vars, float(highspy.kHighsInf), dtype=np.float64)
        
        self.Q_dispatched_limit_rows = []
        # --- Generator limits ---
        for t in time_steps:
            for i in tech:
                self.Q_dispatched_limit_rows.append(constraint_idx)
                row.append(constraint_idx)
                col.append(self.var_idx[("Q_dispatched", t, i)])
                data.append(1)
                rhs_lower.append(0)
                rhs_upper.append(1)
                constraint_idx += 1

        # --- Slack limits ---

        self.Q_slack_limit_rows = []
        for t in time_steps:
            self.Q_slack_limit_rows.append(constraint_idx)
            row.append(constraint_idx)
            col.append(self.var_idx[("Q_slack", t, 0)])
            data.append(1)
            rhs_lower.append(-highspy.kHighsInf)
            rhs_upper.append(0)
            constraint_idx += 1

        # --- Short-term battery power limits
        self.charge_short_limits_rows = []
        for t in time_steps:
            for b in range(n_batteries):
                self.charge_short_limits_rows.append(constraint_idx)
                # Charge limit
                row.append(constraint_idx)
                col.append(self.var_idx[("charge_short", t, b)])
                data.append(1)
                rhs_lower.append(-highspy.kHighsInf)
                rhs_upper.append(0)  # Will update to short_term_battery_P
                constraint_idx += 1


        self.discharge_short_limits_rows = []
        for t in time_steps:
            for b in range(n_batteries):
                self.discharge_short_limits_rows.append(constraint_idx)
                # Discharge limit
                row.append(constraint_idx)
                col.append(self.var_idx[("discharge_short", t, b)])
                data.append(1)
                rhs_lower.append(-highspy.kHighsInf)
                rhs_upper.append(0)  # Will update
                constraint_idx += 1
 
        # --- Short-term SoC bounds (energy limits)
        self.SoC_short_limits_rows = []
        for t in time_steps:
            for b in range(n_batteries):
                self.SoC_short_limits_rows.append(constraint_idx)
                row.append(constraint_idx)
                col.append(self.var_idx[("SoC_short", t, b)])
                data.append(1)
                rhs_lower.append(-highspy.kHighsInf)
                rhs_upper.append(0)  # Will update to short_term_battery_E
                constraint_idx += 1


        # --- Long-term battery limits (charge/discharge)
        self.charge_long_limits_rows = []
        for t in time_steps:
            for a in agents:
                self.charge_long_limits_rows.append(constraint_idx)
                row.append(constraint_idx)
                col.append(self.var_idx[("charge_long", t, a)])
                data.append(1)
                rhs_lower.append(-highspy.kHighsInf)
                rhs_upper.append(0)
                constraint_idx += 1

        # --- Long-term battery limits (charge/discharge)
        self.discharge_long_limits_rows = []
        for t in time_steps:
            for a in agents:
                self.discharge_long_limits_rows.append(constraint_idx)
                row.append(constraint_idx)
                col.append(self.var_idx[("discharge_long", t, a)])
                data.append(1)
                rhs_lower.append(-highspy.kHighsInf)
                rhs_upper.append(0)
                constraint_idx += 1

        # --- Long-term SoC bounds
        self.SoC_long_limits_rows = []
        for t in time_steps:
            for a in agents:
                self.SoC_long_limits_rows.append(constraint_idx)
                row.append(constraint_idx)
                col.append(self.var_idx[("SoC_long", t, a)])
                data.append(1)
                rhs_lower.append(-highspy.kHighsInf)
                rhs_upper.append(0)
                constraint_idx += 1

        # --- Long-term Dumping limits
        self.long_dumping_limits_rows = []
        for t in time_steps:
            for a in agents:
                self.long_dumping_limits_rows.append(constraint_idx)
                row.append(constraint_idx)
                col.append(self.var_idx[("dumping", t, a)])
                data.append(1)
                rhs_lower.append(-highspy.kHighsInf)
                rhs_upper.append(0)
                constraint_idx += 1

        # --- Short-term SoC cyclic dynamic at t=0
        # Mirrors the dynamics rows at t=1..T-1 but wraps around: SoC[t-1] = SoC[T-1].
        # Single row enforces both cyclicity (SoC[0]=SoC[T-1] net of dispatch) and
        # correctly attaches charge[0]/discharge[0] to the SoC transition at t=0,
        # replacing the former free initial row and the separate cyclic equality row.
        self.SoC_short_initial_rows = []
        for b in range(n_batteries):
            self.SoC_short_initial_rows.append(constraint_idx)
            row += [constraint_idx]*4
            col += [
                self.var_idx[("SoC_short", 0,              b)],
                self.var_idx[("SoC_short", self.short_t-1, b)],
                self.var_idx[("charge_short",    0, b)],
                self.var_idx[("discharge_short", 0, b)]
            ]
            data += [1, -1, -short_term_eff_charge, 1/short_term_eff_discharge]
            rhs_lower.append(0)
            rhs_upper.append(0)
            constraint_idx += 1

        # --- Short-term SoC dynamics
        self.SoC_short_dynamics_rows = []
        for t in range(1, self.short_t):
            for b in range(n_batteries):
                self.SoC_short_dynamics_rows.append(constraint_idx)
                row += [constraint_idx]*4
                col += [
                    self.var_idx[("SoC_short", t, b)],
                    self.var_idx[("SoC_short", t-1, b)],
                    self.var_idx[("charge_short", t, b)],
                    self.var_idx[("discharge_short", t, b)]
                ]
                data += [1, -1, -short_term_eff_charge, 1/short_term_eff_discharge]
                rhs_lower.append(0)
                rhs_upper.append(0)
                constraint_idx += 1

        # --- Long-term SoC initial
        # inflows[0] is consumed here as it arrives before any dispatch at t=0.
        # Flow variables at t=0 are included so the full first-period balance is captured.
        self.SoC_long_initial_rows = []
        for a in agents:
            self.SoC_long_initial_rows.append(constraint_idx)
            t = 0
            row += [constraint_idx]*4
            col += [
                self.var_idx[("SoC_long",       t, a)],
                self.var_idx[("charge_long",     t, a)],
                self.var_idx[("discharge_long",  t, a)],
                self.var_idx[("dumping",         t, a)]
            ]
            data += [1, -long_term_eff_charge, 1/long_term_eff_discharge, 1]
            rhs_lower.append(0)   # updated in solve() to SoC_init[a] + inflows[0, a]
            rhs_upper.append(0)
            constraint_idx += 1

        # --- Long-term SoC dynamics: t=1..T-1, RHS = inflows[t]
        self.SoC_long_dynamics_rows = []
        for t in range(1, self.short_t):
            for a in agents:
                self.SoC_long_dynamics_rows.append(constraint_idx)
                row += [constraint_idx]*5
                col += [
                    self.var_idx[("SoC_long",       t,   a)],
                    self.var_idx[("SoC_long",        t-1, a)],
                    self.var_idx[("charge_long",     t,   a)],
                    self.var_idx[("discharge_long",  t,   a)],
                    self.var_idx[("dumping",         t,   a)]
                ]
                data += [1, -1, -long_term_eff_charge, 1/long_term_eff_discharge, 1]
                rhs_lower.append(0)   # updated in solve() to inflows[t, a]
                rhs_upper.append(0)
                constraint_idx += 1

        # --- Power balance
        self.power_balance_rows = []
        for t in time_steps:
            self.power_balance_rows.append(constraint_idx)
            for i in tech:
                row.append(constraint_idx)
                col.append(self.var_idx[("Q_dispatched", t, i)])
                data.append(1)
            for b in range(n_batteries):
                row.append(constraint_idx)
                col.append(self.var_idx[("discharge_short", t, b)])
                data.append(1)
                row.append(constraint_idx)
                col.append(self.var_idx[("charge_short", t, b)])
                data.append(-1)
            for a in agents:
                row.append(constraint_idx)
                col.append(self.var_idx[("discharge_long", t, a)])
                data.append(1)
                row.append(constraint_idx)
                col.append(self.var_idx[("charge_long", t, a)])
                data.append(-1)
            row.append(constraint_idx)
            col.append(self.var_idx[("Q_slack", t, 0)])
            data.append(1)
            rhs_lower.append(0)
            rhs_upper.append(0)
            constraint_idx += 1

        # --- Coal must-run constraint (Sardinian essential plants) ---
        # Hard lower bound: sum of Q_dispatched over coal tech indices >= P_min_coal_SAR
        # RHS (P_min_coal_SAR) is set in solve(); upper bound is +inf.
        # No slack: the user has confirmed availability will always exceed the floor.
        self.coal_must_run_rows = []
        for t in time_steps:
            self.coal_must_run_rows.append(constraint_idx)
            for i in self.coal_tech_idx:
                row.append(constraint_idx)
                col.append(self.var_idx[("Q_dispatched", t, int(i))])
                data.append(1)
            rhs_lower.append(0)  # updated in solve() to P_min_coal_SAR
            rhs_upper.append(highspy.kHighsInf)
            constraint_idx += 1

        # --- Long-term SoC final condition
        self.SoC_long_final_rows = []
        for a in agents:
            self.SoC_long_final_rows.append(constraint_idx)
            t = self.short_t - 1
            row.append(constraint_idx)
            col.append(self.var_idx[("SoC_long", t, a)])
            data.append(1)
            rhs_lower.append(0)  # Will be long_term_SoC_target
            rhs_upper.append(0)
            constraint_idx += 1

        # Long term flexibility through charge and discharge (discharge)

        self.long_flexibility_capabilities_rows = []
        for t in time_steps:
            for a in agents:
                self.long_flexibility_capabilities_rows.append(constraint_idx)
                row.append(constraint_idx)
                col.append(self.var_idx[("discharge_long", t, a)])
                data.append(1)  # Will update with flexibility_credits_dispatched[i] in solve()
                row.append(constraint_idx)
                col.append(self.var_idx[("charge_long", t, a)])
                data.append(1)
                rhs_lower.append(0)
                rhs_upper.append(1)
                constraint_idx += 1

        # Short term flexibility through charge and discharge (discharge)

        self.short_flexibility_capabilities_rows = []
        for t in time_steps:
            for b in range(n_batteries):
                self.short_flexibility_capabilities_rows.append(constraint_idx)
                row.append(constraint_idx)
                col.append(self.var_idx[("discharge_short", t, b)])
                data.append(1)  # Will update with flexibility_credits_dispatched[i] in solve()
                row.append(constraint_idx)
                col.append(self.var_idx[("charge_short", t, b)])
                data.append(1)
                rhs_lower.append(0)
                rhs_upper.append(1)
                constraint_idx += 1

        ## Flexibility constraints 
        self.flex_constraint_rows = []
        for t in time_steps:
            self.flex_constraint_rows.append(constraint_idx)
            # Q_dispatched
            for i in tech:
                row.append(constraint_idx)
                col.append(self.var_idx[("Q_dispatched", t, i)])
                data.append(1.0)  # Will update with flexibility_credits_dispatched[i] in solve()
            # Q_slack
            row.append(constraint_idx)
            col.append(self.var_idx[("Q_slack", t, 0)])
            data.append(1.0)  # Will update

            # discharge_short_term_battery
            for b in range(n_batteries):
                row.append(constraint_idx)
                col.append(self.var_idx[("discharge_short", t, b)])
                data.append(1.0)

                row.append(constraint_idx)
                col.append(self.var_idx[("charge_short", t, b)])
                data.append(1.0)

            # discharge_long_term_battery
            for a in agents:
                row.append(constraint_idx)
                col.append(self.var_idx[("discharge_long", t, a)])
                data.append(1.0)

                row.append(constraint_idx)
                col.append(self.var_idx[("charge_long", t, a)])
                data.append(1.0)

            rhs_lower.append(0)   # Will update in solve()
            rhs_upper.append(highspy.kHighsInf)
            constraint_idx += 1

        self.constraint_idx = constraint_idx
        self.n_constraints = constraint_idx

        A_coo = coo_matrix((data, (row, col)), shape=(self.n_constraints, self.n_vars))
        A = A_coo.tocsr()

        rhs_lower = np.array(rhs_lower, dtype=np.float64)
        rhs_upper = np.array(rhs_upper, dtype=np.float64)
        
        self.model.addCols(
            self.n_vars,
            cost.astype(np.float64),
            lb.astype(np.float64),
            ub.astype(np.float64),
            0,
            np.array([], dtype=np.int32),
            np.array([], dtype=np.int32),
            np.array([], dtype=np.float64)
        )
        self.model.addRows(
                A.shape[0],                     # number of constraints
                rhs_lower,                      # lower bounds
                rhs_upper,                      # upper bounds
                A.nnz,                          # number of nonzeros
                A.indptr.astype(np.int32),     # starts (row pointers)
                A.indices.astype(np.int32),    # column indices
                A.data.astype(np.float64)      # coefficients
            )

    def solve(self, 
              P_bid, Q_bid, Demand, 
              short_term_battery_P, short_term_battery_E, short_term_battery_init, short_term_eff_charge, short_term_eff_discharge,
              long_term_battery_P, long_term_battery_E, long_term_SoC_target, long_term_SoC_init, long_term_inflows, long_term_eff_charge, long_term_eff_discharge,
              VoLL,
              flexibility_credits_dispatched,flexibility_credit_slack,
              flexibility_credits_short_term_storage, flexibility_credits_long_term_storage, 
              flexibility_demand,
              P_min_coal_SAR):
        
        time_steps = np.arange(self.short_t)
        tech = np.arange(self.n_tech)
        agents = np.arange(self.n_agents)
        n_batteries = self.n_batteries

        # --- Update costs ---
        new_costs = np.zeros(self.n_vars)
        for t in time_steps:
            for i in tech:
                new_costs[self.var_idx[("Q_dispatched", t, i)]] = P_bid[t, i]
            new_costs[self.var_idx[("Q_slack", t, 0)]] = VoLL
        
        self.model.changeColsCost(
            self.n_vars,
            np.arange(self.n_vars, dtype=np.int32),
            new_costs.astype(np.float64)
        )

        # --- Update RHS bounds dynamically ---
        idx = 0

        # --- Generator upper bounds (Q_dispatched <= Q_bid) ---
        count_tmp = 0
        for t in time_steps:
            for i in tech:
                row_id = self.Q_dispatched_limit_rows[count_tmp]
                # Update rows 
                self.model.changeRowBounds(row_id, 0, Q_bid[t,i])
                idx += 1
                count_tmp += 1

        # --- Slack limits (Q_slack <= Q_slack_max) ---
        Q_slack_max = 1e8
        count_tmp = 0
        for t in time_steps:
            row_id = self.Q_slack_limit_rows[count_tmp]
            self.model.changeRowBounds(row_id, 0, Q_slack_max)
            idx += 1
            count_tmp += 1

        # --- Short-term battery: charge/discharge power limits ---
        count_tmp_charge = 0
        count_tmp_discharge = 0
        for t in time_steps:
            for b in range(n_batteries):
                # Charge limit
                row_id = self.charge_short_limits_rows[count_tmp_charge]
                self.model.changeRowBounds(row_id, 0, short_term_battery_P[b])
                idx += 1
                count_tmp_charge += 1

                # Discharge limit
                row_id = self.discharge_short_limits_rows[count_tmp_discharge]
                self.model.changeRowBounds(row_id, 0, short_term_battery_P[b])
                idx += 1
                count_tmp_discharge += 1

        # --- Short-term SoC energy capacity limits ---
        count_tmp = 0
        for t in time_steps:
            for b in range(n_batteries):
                row_id = self.SoC_short_limits_rows[count_tmp]
                self.model.changeRowBounds(row_id, 0, short_term_battery_E[b])
                idx += 1
                count_tmp += 1

        # --- Long-term battery: charge/discharge power limits ---
        count_tmp_charge = 0
        count_tmp_discharge = 0
        for t in time_steps:
            for a in agents:
                # Charge limit
                row_id = self.charge_long_limits_rows[count_tmp_charge]
                self.model.changeRowBounds(row_id, 0, long_term_battery_P[a])
                idx += 1
                count_tmp_charge += 1

                # Discharge limit
                row_id = self.discharge_long_limits_rows[count_tmp_discharge]
                self.model.changeRowBounds(row_id, 0, long_term_battery_P[a])
                idx += 1
                count_tmp_discharge += 1

        # --- Long-term SoC energy capacity limits ---
        count_tmp = 0
        for t in time_steps:
            for a in agents:
                row_id = self.SoC_long_limits_rows[count_tmp]
                self.model.changeRowBounds(row_id, 0, long_term_battery_E[a])
                idx += 1
                count_tmp += 1

        # --- Long-term Dumping limits ---
        count_tmp = 0
        for t in time_steps:
            for a in agents:
                row_id = self.long_dumping_limits_rows[count_tmp]
                self.model.changeRowBounds(row_id, 0, long_term_battery_E[a])
                idx += 1
                count_tmp += 1

        # --- Short-term battery: cyclic dynamic at t=0 --- tight equality, no RHS update needed.
        # Increment idx to keep constraint count consistent with n_constraints.
        for b in range(n_batteries):
            idx += 1

        # --- Short-term battery: SoC dynamics ---
        count_tmp = 0
        for t in range(1, self.short_t):
            for b in range(n_batteries):
                row_id = self.SoC_short_dynamics_rows[count_tmp]
                self.model.changeRowBounds(row_id, 0, 0)
                idx += 1
                count_tmp += 1

        # --- Long-term battery: initial SoC --- RHS = SoC_init + inflows[0]
        count_tmp = 0
        for a in agents:
            row_id = self.SoC_long_initial_rows[count_tmp]
            self.model.changeRowBounds(row_id, long_term_SoC_init[a] + long_term_inflows[0, a], long_term_SoC_init[a] + long_term_inflows[0, a])
            idx += 1
            count_tmp += 1

        # --- Long-term battery: SoC dynamics --- t=1..T-1, RHS = inflows[t]
        count_tmp = 0
        for t in range(1, self.short_t):
            for a in agents:
                row_id = self.SoC_long_dynamics_rows[count_tmp]
                self.model.changeRowBounds(row_id, long_term_inflows[t, a], long_term_inflows[t, a])
                idx += 1
                count_tmp += 1

        # --- Power balance constraints ---
        count_tmp = 0
        for t in time_steps:
            row_id = self.power_balance_rows[count_tmp]
            self.model.changeRowBounds(row_id, Demand[t], Demand[t])
            idx += 1
            count_tmp += 1

        # --- Coal must-run constraint (Sardinian essential plants) ---
        # RHS = P_min_coal_SAR (constant across t for the conservative horizon-wide setting).
        count_tmp = 0
        for t in time_steps:
            row_id = self.coal_must_run_rows[count_tmp]
            self.model.changeRowBounds(row_id, float(P_min_coal_SAR), float(highspy.kHighsInf))
            idx += 1
            count_tmp += 1

        # --- Short-term SoC cyclic condition: merged into initial row (V20), no separate counter needed

        # --- Long-term SoC final condition
        # Relaxed from equality to inequality (>= target) to ensure feasibility.
        # Target is clamped to the physically reachable range: [0, min(battery_E, init + sum(inflows))].
        count_tmp = 0
        for a in agents:
            row_id = self.SoC_long_final_rows[count_tmp]
            reachable_max = long_term_SoC_init[a] + np.sum(long_term_inflows[:, a])
            effective_target = np.clip(long_term_SoC_target[a], 0, min(long_term_battery_E[a], reachable_max))
            self.model.changeRowBounds(row_id, effective_target, float(highspy.kHighsInf))
            idx += 1
            count_tmp += 1

        # Maximum flexibility limtis -- Long
        count_tmp = 0
        for t in range(self.short_t):
            for a in agents:
                row_id = self.long_flexibility_capabilities_rows[count_tmp]
                self.model.changeRowBounds(row_id, 0, long_term_battery_P[a])
                idx += 1
                count_tmp += 1

        # Maximum flexibility limits -- Short
        count_tmp = 0
        for t in range(self.short_t):
            for a in range(n_batteries):
                row_id = self.short_flexibility_capabilities_rows[count_tmp]
                self.model.changeRowBounds(row_id, 0, short_term_battery_P[a])
                idx += 1
                count_tmp += 1

        ### Flexibility constraints 

        for t in time_steps:
            row_id = self.flex_constraint_rows[t]

            # compute RHS
            rhs_val = Demand[t] * flexibility_demand
            self.model.changeRowBounds(row_id, rhs_val, float(highspy.kHighsInf))

            # Q_dispatched
            for i in tech:
                col_id = self.var_idx[("Q_dispatched", t, i)]
                coeff = flexibility_credits_dispatched[i]
                self.model.changeCoeff(row_id, col_id, float(coeff))

            # Q_slack
            col_id = self.var_idx[("Q_slack", t, 0)]
            self.model.changeCoeff(row_id, col_id, float(flexibility_credit_slack))

            # discharge_short_term_battery
            for b in range(n_batteries):
                col_id = self.var_idx[("charge_short", t, b)]
                coeff = flexibility_credits_short_term_storage[b]
                self.model.changeCoeff(row_id, col_id, float(coeff))

                col_id = self.var_idx[("discharge_short", t, b)]
                coeff = flexibility_credits_short_term_storage[b]
                self.model.changeCoeff(row_id, col_id, float(coeff))

            # discharge_long_term_battery
            for a in agents:
                col_id = self.var_idx[("charge_long", t, a)]
                coeff = flexibility_credits_long_term_storage
                self.model.changeCoeff(row_id, col_id, float(coeff))

                col_id = self.var_idx[("discharge_long", t, a)]
                coeff = flexibility_credits_long_term_storage
                self.model.changeCoeff(row_id, col_id, float(coeff))
            
            idx += 1

        assert idx == self.n_constraints, f"Mismatch in constraint count: {idx} vs {self.n_constraints}"

        self.model.run()
        model_status = self.model.getModelStatus()

        if model_status != HighsModelStatus.kOptimal:

            SoC_targets_norm = long_term_SoC_target/long_term_battery_E

            raise Exception(f"Model is infeasible__{model_status} SoC_Targets_{long_term_SoC_target} SoC init {long_term_SoC_init} SoC_short_init {short_term_battery_init} Q_bids {np.sum(Q_bid, axis = 1)} P_bid{P_bid} Demand {Demand} Short_term P {short_term_battery_P} Short_term E {short_term_battery_E} Inflows {long_term_inflows} Long_term P {long_term_battery_P} Long_Term E {long_term_battery_E} P_min_coal_SAR {P_min_coal_SAR}")
        
        else:
            solution = self.model.getSolution()
            sol = np.array(solution.col_value, dtype=np.float64)
            duals = np.array(solution.row_dual, dtype=np.float64)

            # --- Extract solution ---
            def get_var(name, t, idx=0):
                return sol[self.var_idx[(name, t, idx)]]

            Q_dispatched_solution = np.array([[get_var("Q_dispatched", t, i) for i in tech] for t in time_steps])
            charge_short_term_battery_solution = np.array([[get_var("charge_short", t, b) for b in range(n_batteries)] for t in time_steps])
            discharge_short_term_battery_solution = np.array([[get_var("discharge_short", t, b) for b in range(n_batteries)] for t in time_steps])
            charge_long_term_battery_solution = np.array([[get_var("charge_long", t, a) for a in agents] for t in time_steps])
            discharge_long_term_battery_solution = np.array([[get_var("discharge_long", t, a) for a in agents] for t in time_steps])
            SoC_short_term_battery_solution = np.array([[get_var("SoC_short", t, b) for b in range(n_batteries)] for t in time_steps])
            SoC_long_term_battery_solution = np.array([[get_var("SoC_long", t, a) for a in agents] for t in time_steps])
            dumping_long_term_battery_solution = np.array([[get_var("dumping", t, a) for a in agents] for t in time_steps])
            Q_slack_solution = np.array([get_var("Q_slack", t) for t in time_steps])

            ll = np.sum(Q_slack_solution) > 1e-5

            price = np.array([duals[row_id] for row_id in self.power_balance_rows])
            price_flexibility = np.array([duals[row_id] for row_id in self.flex_constraint_rows])
            # Shadow price of the coal must-run constraint per timestep.
            # Typically <= 0: when the floor binds in low-net-load hours, raising the floor
            # increases system cost (positive sign convention can vary by solver — check sign in post-processing).
            price_coal_must_run = np.array([duals[row_id] for row_id in self.coal_must_run_rows])

        return (Q_dispatched_solution, 
                charge_short_term_battery_solution, 
                discharge_short_term_battery_solution, 
                charge_long_term_battery_solution, 
                discharge_long_term_battery_solution, 
                price, 
                Q_slack_solution, 
                ll, 
                SoC_short_term_battery_solution, 
                SoC_long_term_battery_solution, 
                dumping_long_term_battery_solution, price_flexibility,
                price_coal_must_run)