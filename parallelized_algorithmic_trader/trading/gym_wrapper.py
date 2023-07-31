from parallelized_algorithmic_trader.trading.simulated_broker import SimulatedStockBrokerCandleData

import gym
from gym import spaces
import numpy as np

import logging


class StockBrokerGymEnv(gym.Env, SimulatedStockBrokerCandleData):
    """OpenAI Gym environment for the SimulatedStockBrokerCandleData."""

    def __init__(self, *args, **kwargs):
        SimulatedStockBrokerCandleData.__init__(self, *args, **kwargs)

        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions and observations:
        self.action_space = spaces.Discrete(3)  # Buy, sell, hold
        self.observation_space = spaces.Box(low=0, high=float('inf'), shape=(n,), dtype=np.float32)  # n-dimensional price vector

    def step(self, action):
        """
        Run one timestep of the environment's dynamics.
        Accepts an action and returns a tuple (observation, reward, done, info).
        """
        # Execute one time step within the environment
        # Here you should apply the action to the environment and return the results
        # This is just a placeholder and you should replace it with your own logic
        
        # process pending orders from the last loop and update the value history
        brokerage.set_prices(row)
        brokerage.process_orders_for_account(account)  # processes any pending orders
        account.value_history[ts] = brokerage.get_account_value(account)

        # generate new orders and submit them to the brokerage
        logger.debug(f'Getting state for {algorithm.indicator_mapping}')
        state = get_state(algorithm.indicator_mapping, row)
        new_orders = algorithm.act(account, state)
        if new_orders is not None:
            account.submit_new_orders(new_orders)

        if logger.getEffectiveLevel() < 20 and display_progress_bar and i%10 == 0: 
            printProgressBar(i, n_rows)
        brokerage.clean_up()
        
        observation = None
        reward = None
        done = False
        info = {}

        return observation, reward, done, info

    def render(self, mode='human'):
        """
        Render the environment to the screen.
        """
        # This is just a placeholder and you should replace it with your own logic
        pass

