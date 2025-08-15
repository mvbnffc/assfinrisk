import dataclasses
from typing import Optional, List

import numpy as np
import pandas as pd
from scipy.stats import poisson
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

from prisk.kernel.message import FloodEvent

@dataclasses.dataclass
class FloodExposure:
    """
    The FloodExposure class represents the exposure of an entity to a flood event expressed
    in terms of return period and depth.
    """
    return_period: float
    depth: float

    @property
    def probability(self) -> float:
        """ The probability of the flood event based on the return period"""
        return 1 / self.return_period

    @property
    def poisson_probability(self) -> float:
        """ The probability of the flood event based on the return period under the poisson distribution.
        Both of the probabilities converge to the same value as the return period increases"""
        return 1 - np.exp(-1 / self.return_period)

    def __str__(self) -> str:
        return f"FloodExposure({self.return_period}, {self.depth})"
    
class FloodExceedanceCurve:
    """
    Constructs and manages an exceedance curve from multiple FloodExposure points
    """

    def __init__(self, flood_exposures: List[FloodExposure]):
        self.exposures = flood_exposures
        self._build_curve()

    def _build_curve(self):
        """
        Build the exceedance curve from the flood exposures
        """
        if not self.exposures:
            self.curve_function = None
            self.min_depth = 0
            self.max_depth = 0
            return
        
        # Want to make exceedance probability curve go to RP=2 (which will equal 0 depth)
        all_exposures = self.exposures.copy() # Add zero depth exposure for RP=2
        zero_depth_exposure = FloodExposure(return_period=2, depth=0)
        all_exposures.append(zero_depth_exposure)
        
        # Sort by depth (ascending)
        sorted_exposures = sorted(all_exposures, key=lambda x: x.depth)

        # Extract depths and probabilities
        depths = np.array([exposure.depth for exposure in sorted_exposures])
        probabilities = np.array([exposure.probability for exposure in sorted_exposures])

        # Store for reference
        self.depths = depths
        self.probabilities = probabilities
        self.min_depth = depths.min()
        self.max_depth = depths.max()

        # Create interpolation function
        # We'll interpolate in logarithmic space for better handling of small probabilities
        log_probs = np.log(probabilities)

        # Create interpolation function
        self.curve_function = interp1d(
            depths, 
            log_probs, 
            kind='linear',
            bounds_error=False, 
            fill_value=(log_probs[0], log_probs[-1])
        )

    def exceedance_probability(self, depth: float) -> float:
        """
        Get the exceedance probability for a given depth
        """
        if self.curve_function is None:
            return 0.0
        
        # For depths below zero, return the RP=2 probability (0.5)
        if depth < 0:
            return 0.5
        
        # Calculate the log probability and then exponentiate to get the actual probability
        log_prob = self.curve_function(depth)
        return np.exp(log_prob)
    
    def return_period(self, depth: float) -> float:
        """
        Get the return period for a given depth
        """
        if self.curve_function is None:
            return float('inf')
        
        prob = self.exceedance_probability(depth)
        if prob == 0:
            return float('inf')
        
        return 1 / prob
    
    def sample_depth_from_probability(self, probability: float) -> float:
        """
        Sample a depth from a given exceedance probability
        """
        if self.curve_function is None or probability <= 0 or probability >= 1:
            return 0.0
        
        # If probability is 0.5 or higher, return 0 depth (RP <= 2 years)
        if probability >= 0.5:
            return 0.0
        
        # If probability is outside the range of the curve, return min or max depth
        if probability <= self.probabilities.min():
            return self.max_depth  # Higher depths for lower probabilities
        if probability >= self.probabilities.max():
            return self.min_depth  # Lower depths for higher probabilities
        
        # Binary search for the log probability
        log_prob = np.log(probability)

        # Search for the closest depth corresponding to the log probability
        depth_range = np.linspace(self.min_depth, self.max_depth, 1000)
        log_probs = self.curve_function(depth_range)

        # Find closest depth
        idx = np.argmin(np.abs(log_probs - log_prob))
        return depth_range[idx]
    
    def plot(self, figsize=(12, 5)):
        """
        Plot the exceedance curve
        """
        
        if self.curve_function is None:
            print("No curve to plot.")
            return
        
        # Create depth range for plotting
        depth_range = np.linspace(self.min_depth, self.max_depth, 1000)
        
        # Calculate probabilities for each depth using the curve function
        probabilities = []
        for d in depth_range:
            probabilities.append(self.exceedance_probability(d))
        
        probabilities = np.array(probabilities)

        # Create the plot
        plt.figure(figsize=figsize)
        plt.plot(depth_range, probabilities, label='Exceedance Probability', color='blue')
        plt.scatter(self.depths, self.probabilities, color='red', s=50, label='Original Data Points')
        plt.yscale('log')
        plt.xlabel('Flood Depth (m)')
        plt.ylabel('Annual Exceedance Probability')
        plt.title('Flood Exceedance Curve')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.legend()
        plt.show()


class FloodBasinSim:
    def __init__(self, entity, events):
        """
        The FloodBasinSim class simulates the flood events based on the return periods
        and the depth associated to these return periods. The simulations are done
        at the basin-level.
        """
        self.entity = entity
        self.events = events

    def simulate(self, kernel):
        """
        Simulates flood events and adds them to the kernal queue

        Parameters
        ----------
        kernel : Kernel
            The kernel object that manages the simulation
        """
        for exposure in self.entity.flood_exposure:
            rp = exposure.return_period
            rp_events = self.events[self.events.return_period == rp].to_dict("records")
            for event in rp_events:
                for i in range(int(event["events"])):
                    FloodEvent(event["year"]-1, exposure.depth, self.entity).send(kernel=kernel)

    @classmethod
    def generate_events_set(
            self, 
            random_numbers: pd.DataFrame,
            years:int = 25):
        """
        Generate a set of flood events based on the return periods and the number of events
        for each return period

        Parameters
        ----------
        years : int
            The number of years to generate the events for
        random_numbers : pd.DataFrame
            The random numbers to use for the simulation.
        """
        return_periods = [5, 10, 25, 50, 100, 200, 500, 1000]
        events = pd.DataFrame()
        for return_period in return_periods:
            simulated_data = random_numbers.sample(years)
            simulated_data = simulated_data.apply(lambda x: poisson.ppf(x, 1/return_period)).reset_index().clip(0, 1)
            simulated_data = simulated_data.replace(0, pd.NA).melt(id_vars="index").dropna()
            if simulated_data.empty:
                continue
            simulated_data.loc[:, "return_period"] = return_period
            events = pd.concat([events, simulated_data])
        events.columns = ["year", "basin", "events", "return_period"]
        events.basin = events.basin.astype(str)
        return events

    @classmethod
    def events_df(self, random_numbers, years=25):
        return_periods = [5, 10, 25, 50, 100, 200, 500, 1000]
        events = pd.DataFrame()
        for return_period in return_periods:
            simulated_data = random_numbers.sample(years)
            simulated_data = simulated_data.apply(lambda x: poisson.ppf(x, 1/return_period)).reset_index().clip(0, 1)
            simulated_data = simulated_data.replace(0, pd.NA).melt(id_vars="index").dropna()
            if simulated_data.empty:
                continue
            simulated_data.loc[:, "return_period"] = return_period
            events = pd.concat([events, simulated_data])
        events.columns = ["year", "basin", "events", "return_period"]
        events.basin = events.basin.astype(str)
        return events


class FloodEntitySim:
    """ The FloodEntitySim allows the simulation of floods based on
    the exposures of a certain entity """
    def __init__(self, entity, model: str = "poisson", random_seed: Optional[int] = None):
        self.entity = entity
        self.exposures = entity.flood_exposure
        self.model = model
        self.random_seed = random_seed


    def _simulate_poisson(self, time_horizon: float, kernel):
        """ Simulate the floodings using the Poisson model 
        
        Parameters
        ----------
        time_horizon : float
            The time horizon of the simulation in years
        """
        for exposure in self.exposures:
            time = np.random.exponential(exposure.return_period)
            while time < time_horizon:
                FloodEvent(time, exposure.depth, self.entity).send(kernel=kernel)
                time += np.random.exponential(exposure.return_period)

    def simulate(self, time_horizon: float, kernel):
        """ Simulate the floodings 
        
        Parameters
        ----------
        time_horizon : float
            The time horizon of the simulation in years
        """
        if self.random_seed:
            np.random.seed(self.random_seed)
        if self.model == "poisson":
            self._simulate_poisson(time_horizon, kernel)

