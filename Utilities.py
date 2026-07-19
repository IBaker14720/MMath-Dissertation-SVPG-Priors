import numpy as np
import tensorflow as tf
from keras import layers
import itertools
import math
import random
from matplotlib import pyplot as plt
import gymnasium as gym
from tqdm import tqdm
import os
import pickle
from matplotlib.colors import PowerNorm
from Priors import *

# ---- Policy network ----
# Single hidden layer network
class PolicyNetwork(tf.keras.Model):
    def __init__(self, hidden_units, act_space, continuous = False):
        super().__init__()
        self.continuous = continuous

        self.d1 = layers.Dense(hidden_units, activation='relu')

        if not continuous:
            self.d2 = layers.Dense(act_space, activation='softmax')
        else:
            self.mu = layers.Dense(act_space, activation=None)

            self.log_std = self.add_weight(name = "log_std", shape = (act_space, ), initializer = "zeros", trainable = True)

    def call(self, s):
        x = self.d1(s)
        if not self.continuous:
            return self.d2(x)
        else:
            mu = self.mu(x)
            log_std = tf.clip_by_value(self.log_std, -5,2)
            std = tf.exp(log_std)
            std = tf.broadcast_to(std, tf.shape(mu))
            return mu,std

# ---- Critic Network ----
class ValueNetwork(tf.keras.Model):
    def __init__(self, hidden_units):
        super().__init__()
        self.d1 = layers.Dense(hidden_units, activation="relu")
        self.d2 = layers.Dense(hidden_units, activation="relu")
        self.v  = layers.Dense(1, activation=None)

    def call(self, s):
        x = self.d1(s)
        x = self.d2(x)
        return self.v(x)


def set_seed(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)

# For continuous
def gaussian_log_prob(a, mu, std):
    log_2pi = tf.constant(np.log(2.0 * np.pi), dtype=tf.float32)
    return -0.5 * (((a - mu) / (std + 1e-8))**2 + 2.0 * tf.math.log(std + 1e-8) + log_2pi)
def arctanh(x):
    x = tf.clip_by_value(x, -0.999999, 0.999999)
    return 0.5 * (tf.math.log1p(x) - tf.math.log1p(-x))

# ---- Discounted Returns ----
def compute_returns(rewards, gamma):
    returns = np.zeros_like(rewards, dtype=np.float32)
    running = 0.0
    for t in reversed(range(len(rewards))):
        running = rewards[t] + gamma * running
        returns[t] = running
    return returns

# ---- GAE ----
def compute_gae(rewards, dones, values, next_values, gamma, lam):
    T = len(rewards)
    advantage = np.zeros(T, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(T)):
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_values[t] * mask - values[t]
        gae = delta + gamma * lam * mask * gae
        advantage[t] = gae
    v_target = advantage + values
    return advantage.astype(np.float32), v_target.astype(np.float32)

def test_critic(critic_model, states, returns):
    states_tf = tf.convert_to_tensor(states, dtype=tf.float32)
    v_pred = tf.squeeze(critic_model(states_tf), axis=1).numpy()

    mse = np.mean((v_pred - returns) ** 2)
    var_returns = np.var(returns)

    explained_var = 1.0 - mse / (var_returns + 1e-8)

    corr = np.corrcoef(v_pred, returns)[0, 1]

    print(
        f"[critic] "
        f"MSE={mse:.3f} "
        f"Var(G)={var_returns:.3f} "
        f"ExplainedVar={explained_var:.3f} "
        f"Corr(V,G)={corr:.3f} "
        f"MeanV={np.mean(v_pred):.2f} "
        f"MeanG={np.mean(returns):.2f}")


# ---- Flatten/unflatten functions ----
def flatten_vars(var_list):
    flats = [tf.reshape(v, [-1]) for v in var_list]
    flat = tf.concat(flats, axis=0)
    shapes = [v.shape for v in var_list]
    sizes = [int(np.prod(s)) for s in shapes]
    idxs = np.cumsum([0] + sizes)
    return flat, shapes, idxs

def unflatten_to_like(flat_vec, shapes, idxs):
    pieces = []
    for i in range(len(shapes)):
        sl = flat_vec[idxs[i]:idxs[i+1]]
        pieces.append(tf.reshape(sl, shapes[i]))
    return pieces

def get_param_vector(model):
    flat, _, _ = flatten_vars(model.trainable_variables)
    return tf.identity(flat)

def set_param_vector_add_(model, delta_flat):
    _, shapes, idxs = flatten_vars(model.trainable_variables)
    deltas = unflatten_to_like(delta_flat, shapes, idxs)
    for v, dv in zip(model.trainable_variables, deltas):
        v.assign_add(dv)

def set_param_vector_from_flat(model, flat_vec: tf.Tensor):
    _, shapes, idxs = flatten_vars(model.trainable_variables)
    pieces = unflatten_to_like(flat_vec, shapes, idxs)
    for v, new_val in zip(model.trainable_variables, pieces):
        v.assign(new_val)

def flatten_gradients(grads, variables):
    flat_grads = []
    for g, v in zip(grads, variables):
        if g is None:
            flat_grads.append(tf.zeros([tf.size(v)], dtype=tf.float32))
        else:
            flat_grads.append(tf.reshape(g, [-1]))
    return tf.concat(flat_grads, axis=0)


# ---- Optimiser ----
def ADAM(model, phi_flat, optimizer, step_scale=1.0):
    _, shapes, idxs = flatten_vars(model.trainable_variables)
    grads_like = unflatten_to_like(-step_scale * phi_flat, shapes, idxs)
    optimizer.apply_gradients(zip(grads_like, model.trainable_variables))


# ---- Kernel Stuff ----
def pairwise_squared_dist(mat):
    # [M, M] matrix of ||x_i - x_j||^2
    # ||x_i||^2 + ||x_j||^2 - 2 x_i x_j
    x2 = tf.reduce_sum(tf.square(mat), axis=1, keepdims=True)
    sq = x2 + tf.transpose(x2) - 2.0 * (mat @ tf.transpose(mat))
    return tf.maximum(sq, 0.0)

def rbf_kernel(theta_mat):
    # Pairwise squared distances
    sq = pairwise_squared_dist(theta_mat)  # [M, M]
    M_ = tf.shape(theta_mat)[0]

    # Chat gpt did this part
    mask = tf.linalg.band_part(tf.ones_like(sq, dtype=tf.bool), 0, -1)  # upper incl diag
    mask = tf.logical_and(mask, tf.logical_not(tf.eye(M_, dtype=tf.bool)))  # remove diag
    vals = tf.boolean_mask(sq, mask)

    def median_or_one(x):
        x = tf.sort(x)
        n = tf.size(x)
        return tf.cond(n > 0,
                       lambda: x[n // 2],
                       lambda: tf.constant(1.0, tf.float32))
    
    median_sq = median_or_one(vals)
    median_sq = tf.stop_gradient(median_sq)


    h = median_sq / tf.math.log(tf.cast(M_, tf.float32) + 1.0)
    h = tf.maximum(h, 1e-6)
    h = tf.stop_gradient(h)

    # RBF kernel: exp(-||x - x'||^2 / h)
    K = tf.exp(-sq / (h + 1e-8))
    return K, h

def kernel_grad_terms(theta_mat, K, h):
    M_, D = theta_mat.shape
    theta_j = tf.expand_dims(theta_mat, 1)     
    theta_i = tf.expand_dims(theta_mat, 0)     
    Kji = tf.transpose(K)                
    diff = theta_j - theta_i             
    grad_jk = Kji[..., None] * (-2.0/(h+1e-8)) * diff 
    grad_term_for_i = tf.reduce_mean(grad_jk, axis=0)
    return grad_term_for_i

# --- Plotting ---
# def plot_learning_curve(curves,  title="Learning curve"):
#     plt.figure(figsize=(8, 4))

#     for curve in curves:
#         x = curve["x"]
#         y = curve["y"]
#         label = curve.get("label", None)

#         plt.plot(x, y, label=label)

#         if "std" in curve and curve["std"] is not None:
#             y_np = np.asarray(y)
#             std_np = np.asarray(curve["std"])
#             x_np = np.asarray(x)

#             plt.fill_between(
#                 x_np,
#                 y_np - std_np,
#                 y_np + std_np,
#                 alpha=0.2
#             )

#     plt.xlabel("Iterations", fontsize=20)
#     plt.ylabel("Return", fontsize=20)
#     plt.title(title, fontsize=24)
#     plt.xticks(fontsize=14)
#     plt.yticks(fontsize=14)

#     plt.grid(True)
#     plt.legend(fontsize=10)

#     plt.tight_layout()
#     plt.savefig("returns.png")
#     plt.close()

def plot_learning_curve(curves, title="Learning curve", filename="returns.png", ylabel="Return"):
    fig, ax = plt.subplots(figsize=(10, 5.5))

    for i, curve in enumerate(curves):
        x = np.asarray(curve["x"])
        y = np.asarray(curve["y"])
        label = curve.get("label", f"Curve {i+1}")

        line, = ax.plot(x, y, linewidth=2.5, label=label)
        color = line.get_color()

        std = curve.get("std", None)
        if std is not None:
            std = np.asarray(std)
            ax.fill_between(x, y - std, y + std, alpha=0.08, linewidth=0, color=color)

        overlay_y = curve.get("overlay_y", None)
        if overlay_y is not None:
            overlay_y = np.asarray(overlay_y)
            overlay_label = curve.get("overlay_label", None)

            ax.plot(
                x,
                overlay_y,
                linewidth=1.5,
                alpha=0.35,
                linestyle="--",
                color=color,
                label=overlay_label
            )

    ax.set_xlabel("Iterations", fontsize=18, labelpad=10)
    ax.set_ylabel(ylabel, fontsize=18, labelpad=10)

    ax.tick_params(axis="both", labelsize=13)
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.8)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if any(curve.get("label", None) is not None or curve.get("overlay_label", None) is not None for curve in curves):
        ax.legend(fontsize=14, frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.36))

    fig.tight_layout()
    fig.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_ppc(results, cartpole_states, prior_name):
    x_edges = results["x_edges"]
    y_edges = results["y_edges"]
    entropy_map = results["entropy_map"]
    between_policy_map = results["between_policy_map"]
    occupancy_map = results["occupancy_map"]
    state_dims = results["state_dims"]

    dim_x, dim_y = state_dims
    occupancy_density_map = occupancy_map / np.nansum(occupancy_map)

    def draw(ax, z, vmin, vmax, norm=None):
        heatmap = ax.pcolormesh(x_edges, y_edges, z, shading="auto", cmap="viridis", vmin=None if norm is not None else vmin, vmax=None if norm is not None else vmax, norm=norm)
        ax.set_xlabel(cartpole_states[dim_x], fontsize=18)
        ax.set_ylabel(cartpole_states[dim_y], fontsize=18)
        ax.set_xlim(-2.4, 2.4)
        ax.set_ylim(-0.2095, 0.2095)
        ax.tick_params(axis="both", labelsize=11)
        ax.grid(alpha=0.2)
        return heatmap

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))

    im1 = draw(
        axes[0],
        entropy_map,
        0.0,
        np.log(2.0),
        norm=PowerNorm(gamma=0.5, vmin=0.0, vmax=np.log(2.0))
    )
    im2 = draw(
        axes[1],
        between_policy_map,
        0.0,
        0.25,
        norm=PowerNorm(gamma=0.5, vmin=0.0, vmax=0.25)
    )

    cbar1 = fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    cbar2 = fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    cbar1.ax.tick_params(labelsize=11)
    cbar2.ax.tick_params(labelsize=11)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    avg_mean_entropy = np.nanmean(entropy_map)
    avg_between_policy_variance = np.nanmean(between_policy_map)

    print("Average mean entropy:", avg_mean_entropy)
    print("Average variance between policies:", avg_between_policy_variance)

    avg_mean_entropy_weighted = np.nansum(entropy_map * occupancy_map) / np.nansum(occupancy_map)
    avg_between_policy_variance_weighted = np.nansum(between_policy_map * occupancy_map) / np.nansum(occupancy_map)

    print(avg_mean_entropy_weighted, " is weighted")
    print(avg_between_policy_variance_weighted, " is weighted var")

# --- Acting ---
def select_action(policy_model, state_input, env, continuous_action, act_space, greedy=False):
    if not continuous_action:
        probs = policy_model(state_input).numpy()[0]
        if greedy:
            action = int(np.argmax(probs))
        else:
            action = np.random.choice(act_space, p=probs)
        return action

    mu, policy_sd = policy_model(state_input)
    if greedy:
        act_high = env.action_space.high.astype(np.float32)
        return (act_high * np.tanh(mu.numpy()[0])).astype(np.float32)

    eps = np.random.randn(act_space).astype(np.float32)
    u = (mu.numpy()[0] + policy_sd.numpy()[0] * eps).astype(np.float32)
    act_high = env.action_space.high.astype(np.float32)
    return (act_high * np.tanh(u)).astype(np.float32)

def collect_episode_trajectory(policy_model, env, continuous_action, act_space, seed=None, greedy=False):
    if seed is None:
        state, _ = env.reset()
    else:
        state, _ = env.reset(seed=int(seed))

    states, actions, rewards = [], [], []
    dones, next_states = [], []
    done = False
    total_reward = 0.0

    while not done:
        state_input = np.asarray(state, dtype=np.float32).reshape(1, -1)
        action = select_action(
            policy_model=policy_model,
            state_input=state_input,
            env=env,
            continuous_action=continuous_action,
            act_space=act_space,
            greedy=greedy)

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        states.append(state_input[0])
        actions.append(action)
        rewards.append(float(reward))
        dones.append(float(done))
        next_states.append(np.asarray(next_state, dtype=np.float32))

        total_reward += reward
        state = next_state

    return {
        "states": np.asarray(states, dtype=np.float32),
        "actions": np.asarray(actions),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "dones": np.asarray(dones, dtype=np.float32),
        "next_states": np.asarray(next_states, dtype=np.float32),
        "total_reward": float(total_reward),
    }