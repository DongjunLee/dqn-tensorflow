"""
Double DQN (Nature 2015)
http://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf
Notes:
    The difference is that now there are two DQNs (DQN & Target DQN)
    y_i = r_i + 𝛾 * max(Q(next_state, action; 𝜃_target))
    Loss: (y_i - Q(state, action; 𝜃))^2
    Every C step, 𝜃_target <- 𝜃
"""

from collections import deque
import logging
import random
from typing import List

import numpy as np
import tensorflow as tf
import gym

from dqn import DeepQNetwork
from config import Config


flags = tf.app.flags
flags.DEFINE_float('discount_rate', 0.99, 'Initial discount rate.')
flags.DEFINE_integer('replay_memory_length', 50000, 'Number of replay memory episode.')
flags.DEFINE_integer('target_update_count', 5, 'DQN Target Network update count.')
flags.DEFINE_integer('max_episode_count', 5000, 'Number of maximum episodes.')
flags.DEFINE_integer('batch_size', 64, 'Batch size. (Must divide evenly into the dataset sizes)')
flags.DEFINE_string('gym_result_dir', 'gym-results/', 'Directory to put the gym results.')
flags.DEFINE_string('gym_env', 'CartPole-v0', 'Name of Open Gym\'s enviroment name.')

FLAGS = flags.FLAGS

env = gym.make(FLAGS.gym_env)
env = gym.wrappers.Monitor(env, directory=FLAGS.gym_result_dir, force=True)

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Constants defining our neural network
config = Config(env, FLAGS.gym_env)

def replay_train(mainDQN: DeepQNetwork, targetDQN: DeepQNetwork, train_batch: list) -> float:
    """Trains `mainDQN` with target Q values given by `targetDQN`
    Args:
        mainDQN (DeepQNetwork``): Main DQN that will be trained
        targetDQN (DeepQNetwork): Target DQN that will predict Q_target
        train_batch (list): Minibatch of replay memory
            Each element is (s, a, r, s', done)
            [(state, action, reward, next_state, done), ...]
    Returns:
        float: After updating `mainDQN`, it returns a `loss`
    """
    states = np.vstack([x[0] for x in train_batch])
    actions = np.array([x[1] for x in train_batch])
    rewards = np.array([x[2] for x in train_batch])
    next_states = np.vstack([x[3] for x in train_batch])
    done = np.array([x[4] for x in train_batch])

    X = states

    Q_target = rewards + FLAGS.discount_rate * np.max(targetDQN.predict(next_states), axis=1) * ~done

    y = mainDQN.predict(states)
    y[np.arange(len(X)), actions] = Q_target

    # Train our network using target and predicted Q values on each episode
    return mainDQN.update(X, y)


def get_copy_var_ops(*, dest_scope_name: str, src_scope_name: str) -> List[tf.Operation]:
    """Creates TF operations that copy weights from `src_scope` to `dest_scope`
    Args:
        dest_scope_name (str): Destination weights (copy to)
        src_scope_name (str): Source weight (copy from)
    Returns:
        List[tf.Operation]: Update operations are created and returned
    """
    # Copy variables src_scope to dest_scope
    op_holder = []

    src_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope_name)
    dest_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope=dest_scope_name)

    for src_var, dest_var in zip(src_vars, dest_vars):
        op_holder.append(dest_var.assign(src_var.value()))

    return op_holder


def bot_play(mainDQN: DeepQNetwork, env: gym.Env) -> None:
    """Test runs with rendering and logger.infos the total score
    Args:
        mainDQN (DeepQNetwork): DQN agent to run a test
        env (gym.Env): Gym Environment
    """
    state = env.reset()
    reward_sum = 0

    while True:
        env.render()
        action = np.argmax(mainDQN.predict(state))
        state, reward, done, _ = env.step(action)
        reward_sum += reward

        if done:
            logger.info("Total score: {}".format(reward_sum))
            break


def main():
    logger.info("FLAGS configure.")
    logger.info(FLAGS.__flags)

    # store the previous observations in replay memory
    replay_buffer = deque(maxlen=FLAGS.replay_memory_length)

    consecutive_len = 100 # default value
    if config.solving_criteria:
        consecutive_len = config.solving_criteria[0]
    last_n_game_reward = deque(maxlen=consecutive_len)

    with tf.Session() as sess:
        mainDQN = DeepQNetwork(sess, config.input_size, config.output_size, name="main")
        targetDQN = DeepQNetwork(sess, config.input_size, config.output_size, name="target")
        sess.run(tf.global_variables_initializer())

        # initial copy q_net -> target_net
        copy_ops = get_copy_var_ops(dest_scope_name="target",
                                    src_scope_name="main")
        sess.run(copy_ops)

        for episode in range(FLAGS.max_episode_count):
            e = 1. / ((episode / 10) + 1)
            done = False
            step_count = 0

            state = env.reset()
            e_reward = 0
            model_loss = 0
            avg_reward = np.mean(last_n_game_reward)

            while not done:
                if np.random.rand() < e:
                    action = env.action_space.sample()
                else:
                    # Choose an action by greedily from the Q-network
                    action = np.argmax(mainDQN.predict(state))

                # Get new state and reward from environment
                next_state, reward, done, _ = env.step(action)

                if done and FLAGS.gym_env.startswith("CartPole"):  # Penalty
                    reward = -1

                # Save the experience to our buffer
                replay_buffer.append((state, action, reward, next_state, done))

                if len(replay_buffer) > FLAGS.batch_size:
                    minibatch = random.sample(replay_buffer, FLAGS.batch_size)
                    loss, _ = replay_train(mainDQN, targetDQN, minibatch)
                    model_loss = loss

                if step_count % FLAGS.target_update_count == 0:
                    sess.run(copy_ops)

                state = next_state
                e_reward += reward
                step_count += 1

            logger.info(f"Episode: {episode}  reward: {e_reward}  loss: {model_loss}  consecutive_{consecutive_len}_avg_reward: {avg_reward}")

            # CartPole-v0 Game Clear Checking Logic
            last_n_game_reward.append(e_reward)

            if len(last_n_game_reward) == last_n_game_reward.maxlen:
                avg_reward = np.mean(last_n_game_reward)

                if config.solving_criteria and  avg_reward > (config.solving_criteria[1]):
                    logger.info(f"Game Cleared in {episode} episodes with avg reward {avg_reward}")
                    break


if __name__ == "__main__":
    main()
