import logging
import numpy as np
import tensorflow as tf
import os


from tf_agents.environments import tf_py_environment
from tf_agents.networks.q_network import QNetwork
from tf_agents.agents.dqn.dqn_agent import DqnAgent
from tf_agents.utils import common
from tf_agents.trajectories.trajectory import from_transition
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer
from tf_agents.policies import policy_saver
from tf_agents.trajectories import time_step as ts
from tf_agents.environments import py_environment
from tf_agents.specs import BoundedTensorSpec
from tf_agents.trajectories.policy_step import PolicyStep

from gym_env.env import Action

log = logging.getLogger(__name__)

# Global training hyperparameters
WINDOW_LENGTH = 1
NB_MAX_START_STEPS = 1
TRAIN_INTERVAL = 5
NB_STEPS_WARMUP = 10
NB_STEPS = 150
MEMORY_LIMIT = int(NB_STEPS / 2)
BATCH_SIZE = 64
ENABLE_DOUBLE_DQN = False

class RawWrapper(py_environment.PyEnvironment):
    def __init__(self, raw_env):
        self._env = raw_env
        init_obs = raw_env.reset()
        self._observation_spec = tf.TensorSpec(shape=init_obs.shape, dtype=tf.float32)
        self._action_spec = BoundedTensorSpec(
            shape=(),
            dtype=tf.int64,
            minimum=0,
            maximum=len(Action) - 1
        )

    def observation_spec(self):
        return self._observation_spec

    def action_spec(self):
        return self._action_spec

    def _reset(self):
        obs = self._env.reset()
        obs = np.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6)

        return ts.restart(obs.astype(np.float32))

    def _step(self, action):
        obs, reward, done, info = self._env.step(action)
        obs = obs.astype(np.float32)
        obs = np.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6)

        if done:
            return ts.termination(obs, reward)
        else:
            return ts.transition(obs, reward)
        
    def reward_spec(self):
        return tf.TensorSpec(shape=(), dtype=tf.float32)

class SoftmaxPolicy:
    def __init__(self, q_network, temperature=1.0, fallback_limit=50):
        self.q_network = q_network
        self.temperature = temperature
        self.nan_counter = 0
        self.fallback_limit = fallback_limit

    def action(self, time_step, legal_actions=None):
        obs = time_step.observation
        raw_output = self.q_network(obs)
        q_values = raw_output[0] if isinstance(raw_output, tuple) else raw_output
        logits = tf.squeeze(q_values)

        if legal_actions is None:
            legal_actions = list(range(len(Action)))

        mask = tf.ones_like(logits) * -1e9
        indices = tf.constant(legal_actions, dtype=tf.int32)
        mask = tf.tensor_scatter_nd_update(mask, tf.reshape(indices, [-1, 1]), tf.zeros_like(indices, dtype=tf.float32))
        masked_logits = logits + mask

        if tf.math.reduce_any(tf.math.is_nan(masked_logits)):
            self.nan_counter += 1
            log.warning(f"Q-values contain NaN (#{self.nan_counter}) -- fallback to uniform")
            probs = tf.ones_like(masked_logits) / tf.cast(tf.shape(masked_logits)[0], tf.float32)
            sample = tf.random.categorical(tf.math.log([probs]), 1)[0, 0]
            return PolicyStep(action=tf.expand_dims(sample, 0), state=(), info={})

        probs = tf.nn.softmax(masked_logits / self.temperature)
        sample = tf.random.categorical(tf.math.log([probs]), 1)[0, 0]
        return PolicyStep(action=tf.expand_dims(sample, 0), state=(), info={})


class Player:
    def __init__(self, name='DQN', load_model=None, env=None):
        self.name = name
        self.equity_alive = 0
        self.actions = []
        self.last_action_in_stage = ''
        self.temp_stack = []
        self.autoplay = True
        self.action_wrapper = None 

        self.tf_env = None
        self.eval_env = None
        self.agent = None
        self.policy = None
        self.replay_buffer = None
        self.train_step = None
        self.env_wrapper = None
        self.observation_space = None
        self.policy_dir = "output/saved_policy"
        self.episode_reward = 0.0
        self.episode_step_count = 0
        
        self.load_model = load_model
        self.external_env = env

    def initiate_agent(self, env):
        log.info("Initializing TF-Agents DQN agent")

        self.env_wrapper = RawWrapper(env)
        self.action_wrapper = ActionProcessorWrapper(env_action_space_size=len(Action))

        self.tf_env = tf_py_environment.TFPyEnvironment(self.env_wrapper)
        self.eval_env = tf_py_environment.TFPyEnvironment(RawWrapper(env))
        self.observation_space = env.observation_space

        q_net = QNetwork(
            self.tf_env.observation_spec(),
            self.tf_env.action_spec(),
            fc_layer_params=(512, 512, 512)
        )

        optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
        self.train_step = tf.Variable(0)

        self.agent = DqnAgent(
            self.tf_env.time_step_spec(),
            self.tf_env.action_spec(),
            q_network=q_net,
            optimizer=optimizer,
            td_errors_loss_fn=common.element_wise_squared_loss,
            train_step_counter=self.train_step
        )
        self.agent.initialize()
        self.policy = SoftmaxPolicy(self.agent._q_network)

        self.replay_buffer = TFUniformReplayBuffer(
            data_spec=self.agent.collect_data_spec,
            batch_size=self.tf_env.batch_size,
            max_length=MEMORY_LIMIT
        )

        if self.load_model:
            self.load()

    def play(self, nb_episodes=5, render=False):
        log.info(f"[{self.name}] Start playing for {nb_episodes} episodes.")
        for i in range(nb_episodes):
            time_step = self.eval_env.reset()
            episode_reward = 0
            step = 0

            while not time_step.is_last():
                legal_actions = list(range(len(Action)))  # or you extract it from env step if available
                action_step = self.policy.action(time_step, legal_actions=legal_actions)

                time_step = self.eval_env.step(action_step.action)
                episode_reward += time_step.reward.numpy()[0]
                step += 1

            log.info(f"[{self.name}] Episode {i + 1}: Reward = {episode_reward}, Steps = {step}")

    def action(self, action_space, observation, info):
        if observation is None or np.any(np.isnan(observation)):
            log.warning("[NaN WARNING] Observation contains NaN or is None")
            return np.random.choice(action_space)

        observation = np.expand_dims(observation, axis=0).astype(np.float32)
        time_step = ts.restart(observation)

        action_step = self.policy.action(time_step)
        raw_action = int(action_step.action.numpy()[0])

        self.action_wrapper.update_legal_actions(info)
        final_action = self.action_wrapper.process_action(raw_action)

        next_time_step = self.tf_env.step(tf.constant([final_action], dtype=tf.int64))

        reward = next_time_step.reward.numpy()[0]
        self.episode_reward += reward
        self.episode_step_count += 1

        traj = from_transition(
            time_step,
            PolicyStep(action_step.action, action_step.state, ()), 
            next_time_step
        )
        self.replay_buffer.add_batch(traj)

        if self.train_step.numpy() > NB_STEPS_WARMUP and self.train_step.numpy() % TRAIN_INTERVAL == 0:
            dataset = self.replay_buffer.as_dataset(
                num_parallel_calls=3,
                sample_batch_size=BATCH_SIZE,
                num_steps=2).prefetch(3)
            iterator = iter(dataset)
            experience, _ = next(iterator)
            loss = self.agent.train(experience).loss

            try:
                q_values = self.agent._q_network(next_time_step.observation)[0].numpy()[0]
                q_str = np.array2string(q_values, precision=2, separator=', ')
                weights = self.agent._q_network.trainable_weights
                total_norm = float(tf.linalg.global_norm(weights))
            except Exception as e:
                q_str = f"Q-error: {e}"
                total_norm = -1

            log.info(f"[Train] step={self.train_step.numpy()} loss={loss:.4f} q={q_str} w_norm={total_norm:.2f} ep_reward={self.episode_reward:.2f} ep_len={self.episode_step_count}")

        log.info(f"Agent action: {Action(final_action).name if final_action < len(Action) else final_action}, stack: {info.get('stack', 'N/A')}")
        return final_action


    def evaluate_league_table(self):
        env = self.env_wrapper.pyenv.envs[0]
        if hasattr(env, 'stats_wins') and hasattr(env, 'stats_total_chips'):
            log.info("\n=== League Table ===")
            table = []
            total_rounds = sum(env.stats_wins.values())
            for player_id, win_count in env.stats_wins.items():
                win_rate = win_count / total_rounds if total_rounds > 0 else 0
                chips = env.stats_total_chips[player_id]
                mean_stack = np.mean(chips) if chips else 0
                median_stack = np.median(chips) if chips else 0
                table.append((player_id, win_rate, mean_stack, median_stack))
                log.info(f"Player {player_id}: WinRate={win_rate:.3f}, MeanStack={mean_stack:.1f}, Median={median_stack:.1f}")

            best_player = max(table, key=lambda x: x[1])[0]
            log.info(f"Best Player: {best_player}")
        else:
            log.warning("No league statistics available in environment.")

    def train(self, num_iterations=NB_STEPS):
        log.info("Starting full training loop")
        for i in range(num_iterations):
            self.episode_reward = 0.0
            self.episode_step_count = 0

            time_step = self.tf_env.reset()
            while not time_step.is_last():
                action_step = self.policy.action(time_step)
                next_time_step = self.tf_env.step(action_step.action)

                traj = from_transition(
                    time_step,
                    PolicyStep(action_step.action, action_step.state, ()),  
                    next_time_step
                )
                self.replay_buffer.add_batch(traj)

                self.episode_reward += next_time_step.reward.numpy()[0]
                self.episode_step_count += 1

                if self.replay_buffer.num_frames() >= BATCH_SIZE:
                    dataset = self.replay_buffer.as_dataset(
                        num_parallel_calls=3,
                        sample_batch_size=BATCH_SIZE,
                        num_steps=2).prefetch(3)
                    iterator = iter(dataset)
                    experience, _ = next(iterator)
                    train_loss = self.agent.train(experience).loss

                    q_values = self.agent._q_network(next_time_step.observation)[0].numpy()[0]
                    q_str = np.array2string(q_values, precision=2, separator=', ')
                    weights = self.agent._q_network.trainable_weights
                    total_norm = float(tf.linalg.global_norm(weights))
                    log.info(f"Train Episode {i} step={self.train_step.numpy()} loss={train_loss:.4f} q={q_str} w_norm={total_norm:.2f} ep_reward={self.episode_reward:.2f} ep_len={self.episode_step_count}")

                time_step = next_time_step

        policy_saver.PolicySaver(self.agent.policy).save(self.policy_dir)
        log.info(f"Saved policy to {self.policy_dir}")
        self.play()
        self.evaluate_league_table()
        log.info("Evaluating trained agent with play()...")
        print("Training completed successfully.")
        log.info("Training completed successfully.")

    def load(self):
        if os.path.exists(self.policy_dir):
            self.policy = tf.saved_model.load(self.policy_dir)
            log.info(f"Loaded saved policy from {self.policy_dir}")
        else:
            log.warning(f"Policy directory {self.policy_dir} not found.")


class ActionProcessorWrapper:
    def __init__(self, env_action_space_size):
        self.last_legal_actions = list(range(env_action_space_size))

    def update_legal_actions(self, info):
        if 'legal_moves' in info:
            self.last_legal_actions = [a.value if hasattr(a, 'value') else a for a in info['legal_moves']]
        else:
            self.last_legal_actions = list(range(len(Action)))

    def process_action(self, action):
        a = int(action)
        if a not in self.last_legal_actions:
            log.warning(f"Illegal action {a}, fallback to legal.")
            fallback_action = np.random.choice(self.last_legal_actions)
            return fallback_action
        return a
