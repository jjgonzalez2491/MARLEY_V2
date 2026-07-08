"""
gas_price_shock.py
------------------
Markov regime-switching shock model for natural gas prices in MARLEY.

States:  0=Normal  1=Shock  2=Recovery

Calibration (bimester time step):
    - Shock arrival rate lam = 0.017 -> ~1 shock per ~12 years on average
    - p_stay = 11/12  -> expected Shock duration = 12 bimesters (~2 years)
                         (plus 1 bimester of Recovery -> total elevated ~13 bimesters)
    - Peak multiplier ~ LogNormal(mu_log=ln(2), sigma_log=0.25), clipped to [1.3, 3.5]
      -> median peak ~ 2x long-term price, with realistic dispersion

Activation:
    The module can be deactivated during periods where gas prices are driven
    by an exogenous source (e.g. real forward curve at the start of the
    simulation). While inactive, step() returns 1.0 and no state transitions
    occur, so the chain remains in Normal and cannot "secretly" enter a shock
    that would surface when activation flips on.

Usage:
    shock = GasPriceShockModel(seed=42)
    shock.reset()
    for t in range(T):
        active = (t >= warmup_bimesters)         # e.g. forward curve covers warmup
        mult = shock.step(active=active)
        v_c_g[t] = v_c_g_baseline[t] * mult
"""

from typing import Optional, Tuple

import numpy as np


class GasPriceShockModel:
    # State constants
    S_NORMAL, S_SHOCK, S_RECOVERY = 0, 1, 2

    def __init__(
        self,
        lam: float = 0.017,
        p_stay: float = 11.0 / 12.0,
        mu_log: float = np.log(2.0),
        sigma_log: float = 0.25,
        peak_clip: Tuple[float, float] = (1.3, 3.5),
        active: bool = True,
        seed: Optional[int] = None,
    ):
        """
        Parameters
        ----------
        lam : float
            Per-bimester probability of a shock arriving while in Normal state.
            Default 0.017 -> roughly one shock every ~12 years (accounting for
            time spent in Shock/Recovery states).
        p_stay : float
            Per-bimester probability of remaining in the Shock state.
            Expected Shock duration = 1 / (1 - p_stay) bimesters.
            Default 11/12 -> ~12 bimesters (~2 years).
        mu_log, sigma_log : float
            Parameters of the LogNormal distribution from which the peak
            multiplier is drawn. Default mu_log = ln(2) -> median peak = 2x.
        peak_clip : (float, float)
            Hard bounds on the peak multiplier to avoid extreme tails.
        active : bool
            Initial activation state. If False, step() returns 1.0 and
            performs no state transitions until set_active(True) is called
            (or step(active=True) is invoked).
        seed : int or None
            RNG seed for reproducibility.
        """
        self.lam = lam
        self.p_stay = p_stay
        self.mu_log = mu_log
        self.sigma_log = sigma_log
        self.peak_clip = peak_clip
        self.active = bool(active)
        self.rng = np.random.default_rng(seed)
        self.reset()

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------
    def reset(self):
        self.state = self.S_NORMAL
        self.multiplier = 1.0
        self._peak = 1.0

    def set_active(self, active: bool):
        """Enable or disable the shock module."""
        self.active = bool(active)

    def step(self, active: Optional[bool] = None) -> float:
        """
        Advance one bimester and return the multiplier.

        Parameters
        ----------
        active : bool or None
            If provided, overrides the stored activation flag for this step
            (and updates the stored flag). If None, uses the current flag.

        Returns
        -------
        float
            The current multiplier on the long-term gas price.
            Always 1.0 when the module is inactive.
        """
        if active is not None:
            self.active = bool(active)

        if not self.active:
            # Force baseline; do not advance the Markov chain.
            self.state = self.S_NORMAL
            self.multiplier = 1.0
            return self.multiplier

        u = self.rng.random()

        if self.state == self.S_NORMAL:
            if u < self.lam:
                self._peak = float(np.clip(
                    self.rng.lognormal(self.mu_log, self.sigma_log),
                    self.peak_clip[0], self.peak_clip[1],
                ))
                self.state = self.S_SHOCK
                self.multiplier = self._peak

        elif self.state == self.S_SHOCK:
            if u >= self.p_stay:
                self.state = self.S_RECOVERY
                # midpoint recovery between peak and baseline
                self.multiplier = 1.0 + (self._peak - 1.0) * 0.5

        else:  # S_RECOVERY
            self.state = self.S_NORMAL
            self.multiplier = 1.0

        return self.multiplier