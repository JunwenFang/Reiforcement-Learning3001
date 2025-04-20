"""Player based on a trained neural network"""
# pylint: disable=wrong-import-order,invalid-name,import-error,missing-function-docstring
# poetry run python main.py selfplay dqn_train --funds_plot --log log.txt --name=my_model

import logging
import time

import numpy as np

from gym_env.enums import Action

import tensorflow as tf
import json

from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.callbacks import TensorBoard, CSVLogger
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from rl.agents import DQNAgent
from rl.core import Processor

autoplay = True  # play automatically if played against keras-rl

window_length = 1
nb_max_start_steps = 1  # random action
train_interval = 50  # train every 100 steps
nb_steps_warmup = 150  # before training starts, should be higher than start steps
nb_steps = 1000 # 10000   
memory_limit = int(nb_steps / 2)
batch_size = 128  # items sampled from memory to train
enable_double_dqn = False

log = logging.getLogger(__name__)

# Monkey patch: add get_updates method to Adam
def adam_get_updates(self, loss, params):
    grads = self.get_gradients(loss, params)
    self.updates = []
    self.weights = []
    for p, g in zip(params, grads):
        self.updates.append(p.assign_sub(self.learning_rate * g))
        self.weights.append(p)
    return self.updates

# only patch if missing
if not hasattr(Adam, "get_updates"):
    Adam.get_updates = adam_get_updates



class Player:
    """Mandatory class with the player methods"""

    def __init__(self, name='DQN', load_model=None, env=None):
        """Initiaization of an agent"""
        self.equity_alive = 0
        self.actions = []
        self.last_action_in_stage = ''
        self.temp_stack = []
        self.name = name
        self.autoplay = True

        self.dqn = None
        self.model = None
        self.env = env

        if load_model:
            self.load(load_model)

    def initiate_agent(self, env):
        """initiate a deep Q agent"""
        # tf.compat.v1.disable_eager_execution()

        self.env = env
        # if observation is None:
        #     observation = env.reset()[5]
        nb_actions = self.env.action_space.n

        self.model = Sequential()
        self.model.add(Dense(512, activation='relu', input_shape=(env.observation_space[0],)))
        
        
        log.info(f">>> OBS SHAPE: {env.observation_space[0]}")


        self.model.add(Dropout(0.2))
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(nb_actions, activation='linear'))

        # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
        # even the metrics!
        memory = SequentialMemory(limit=memory_limit, window_length=window_length)
        policy = TrumpPolicy()

        nb_actions = env.action_space.n

        self.dqn = DQNAgent(model=self.model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=nb_steps_warmup,
                            target_model_update=1e-2, policy=policy,
                            processor=CustomProcessor(),
                            batch_size=batch_size, train_interval=train_interval, enable_double_dqn=enable_double_dqn)
        opt = Adam(learning_rate=1e-4)
        opt._name = "Adam"
        self.dqn.compile(opt, metrics=['mae'])


    def start_step_policy(self, observation):
        """Custom policy for random decisions for warm up."""
        if np.any(np.isnan(observation)):
            log.warning("🚨 Observation contains NaN! Obs = %s", observation)
        try:
            q_vals = self.model.predict(np.array([observation]), verbose=0)[0]
            log.info("🧪 Predicted Q-values during warmup: %s", q_vals)
            if np.any(np.isnan(q_vals)):
                log.warning("❌ Model predict Q-values contain NaN during warmup! Check model weights or input.")
        except Exception as e:
            log.error("💥 Model prediction failed in start_step_policy: %s", str(e))

        log.info("Random action")
        _ = observation
        action = self.env.action_space.sample()
        log.info(">>> StartStep Obs:", observation)
        return action

    def train(self, env_name):
        """Train a model"""
        print(">>> start !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")  
        # initiate training loop
        timestr = time.strftime("%Y%m%d-%H%M%S") + "_" + str(env_name)
        # tensorboard = TensorBoard(log_dir='./Graph/{}'.format(timestr), histogram_freq=0, write_graph=True,
                                #   write_images=False)
        csv_logger = CSVLogger(f'output/training_log_{env_name}.csv', append=True)
        log.info(f"Training log path: output/output_training_log_{env_name}.csv, logger: {csv_logger}")
        self.dqn.fit(self.env, 
                     nb_max_start_steps=nb_max_start_steps, 
                     nb_steps=nb_steps, visualize=False, verbose=2,
                     start_step_policy=self.start_step_policy, 
                     callbacks=[csv_logger])

        # Save the architecture
        dqn_json = self.model.to_json()
        with open("dqn_{}_json.json".format(env_name), "w") as json_file:
            json.dump(dqn_json, json_file)

        # After training is done, we save the final weights.
        self.dqn.save_weights('dqn_{}_weights.h5'.format(env_name), overwrite=True)

        # Finally, evaluate our algorithm for 5 episodes.
        self.dqn.test(self.env, nb_episodes=5, visualize=False)

    def load(self, env_name):
        """Load a model"""

        # Load the architecture
        with open('dqn_{}_json.json'.format(env_name), 'r') as architecture_json:
            dqn_json = json.load(architecture_json)

        self.model = model_from_json(dqn_json)
        self.model.load_weights('dqn_{}_weights.h5'.format(env_name))

    def play(self, nb_episodes=5, render=False):
        """Let the agent play"""
        memory = SequentialMemory(limit=memory_limit, window_length=window_length)
        policy = TrumpPolicy()

        class CustomProcessor(Processor):  # pylint: disable=redefined-outer-name
            """The agent and the environment"""

            def process_state_batch(self, batch):
                """
                Given a state batch, I want to remove the second dimension, because it's
                useless and prevents me from feeding the tensor into my CNN
                """
                return np.squeeze(batch, axis=1)

            def process_info(self, info):
                processed_info = info['player_data']
                if 'stack' in processed_info:
                    processed_info = {'x': 1}
                return processed_info

        nb_actions = self.env.action_space.n

        self.dqn = DQNAgent(model=self.model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=nb_steps_warmup,
                            target_model_update=1e-2, policy=policy,
                            processor=CustomProcessor(),
                            batch_size=batch_size, train_interval=train_interval, enable_double_dqn=enable_double_dqn)
        opt = Adam(learning_rate=1e-3)
        opt._name = "Adam"
        self.dqn.compile(opt, metrics=['mae'])  # pylint: disable=no-member

        self.dqn.test(self.env, nb_episodes=nb_episodes, visualize=render)

    def action(self, action_space, observation, info):  # pylint: disable=no-self-use
        """Mandatory method that calculates the move based on the observation array and the action space."""
        _ = observation  # not using the observation for random decision
        _ = info

        this_player_action_space = {Action.FOLD, Action.CHECK, Action.CALL, Action.RAISE_POT, Action.RAISE_HALF_POT,
                                    Action.RAISE_2POT}
        _ = this_player_action_space.intersection(set(action_space))

        action = None
        return action


class TrumpPolicy(BoltzmannQPolicy):
    """Custom policy when making decision based on neural network."""

    def select_action(self, q_values):
        """Return the selected action

        # Arguments
            q_values (np.ndarray): List of the estimations of Q for each action

        # Returns
            Selection action
        """
        assert q_values.ndim == 1
        q_values = q_values.astype('float64')
        nb_actions = q_values.shape[0]

        exp_values = np.exp(np.clip(q_values / self.tau, self.clip[0], self.clip[1]))
        probs = exp_values / np.sum(exp_values)
        if np.any(np.isnan(q_values)):
            log.warning("❌ Q-values contain NaN: %s", q_values)
        else:
            log.info("✅ Q-values are valid: %s", q_values)

        try:
            scaled_q = np.clip(q_values / self.tau, -50, 50)  
            exp_values = np.exp(scaled_q)
            probs = exp_values / np.sum(exp_values)

            if np.any(np.isnan(probs)) or np.sum(probs) == 0:
                raise ValueError("Invalid probs")
        except Exception as e:
            log.warning(f"[TrumpPolicy] fallback to uniform due to: {e}, q_values={q_values}")
            probs = np.ones(nb_actions) / nb_actions


        action = np.random.choice(range(nb_actions), p=probs)
        log.info(f"Chosen action by keras-rl {action} - probabilities: {probs}")
        return action


class CustomProcessor(Processor):
    """The agent and the environment"""

    def __init__(self):
        """initizlie properties"""
        self.legal_moves_limit = None

    def process_state_batch(self, batch):
        """Remove second dimension to make it possible to pass it into cnn"""
        return np.squeeze(batch, axis=1)

    def process_info(self, info):
        if 'legal_moves' in info.keys():
            self.legal_moves_limit = info['legal_moves']
        else:
            self.legal_moves_limit = None
        return {'x': 1}  # on arrays allowed it seems

    def process_action(self, action):
        """Find nearest legal action"""
        if 'legal_moves_limit' in self.__dict__ and self.legal_moves_limit is not None:
            self.legal_moves_limit = [move.value for move in self.legal_moves_limit]
            if action not in self.legal_moves_limit:
                for i in range(5):
                    action += i
                    if action in self.legal_moves_limit:
                        break
                    action -= i * 2
                    if action in self.legal_moves_limit:
                        break
                    action += i

        return action
