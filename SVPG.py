import gymnasium as gym
import highway_env
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from keras import layers
from collections import deque
from tqdm import tqdm
import itertools
import math
import random
import matplotlib.pyplot as plt
from keras import layers
from Priors import *
from Utilities import *
from Estimators import *
from PPC import *
from gymnasium.wrappers import RecordVideo
import Experiments

SEED = 41
set_seed(SEED)

verbose = 1

# ---- Environment ----
env_name = "LunarLander-v3" # C State (4), D Action (1)
#env_name = "racetrack-v0"
#env_name = "MountainCar-v0"
#env_name = "Acrobot-v1"
#env_name = "CartPole"
#env_name = "Pendulum-v1" # C State (3), C Action (1)

env = gym.make(env_name)
env.reset(seed=SEED)

obs_space = env.observation_space.shape[0]
continuous_action = not hasattr(env.action_space, "n")
if continuous_action:
    act_space = env.action_space.shape[0]
else:
    act_space = env.action_space.n

# --- Hyperparameters ---
gamma = 0.99                     # Reward discount
M = 1                            # Number of particles
alpha = 0.1                      # Temperature

lr_actor = 0.01             
lr_critic = 1e-3


hidden_units = 64

training_batch_size = 5         # Number of episodes per update
num_episodes = 500            
normalise_advantages = False
use_gae = False

# --- Prior Settings ---
start_true = False              # Initialise prior around saved weights

if start_true:
    # Requires numpy weights
    weights_file = "LunarGAE.npy"
    theta_star = np.load(weights_file)
    prior_now = GaussianPrior(theta_star, 1)
else:
    prior_mean = 0
    prior_std = 10
    prior_now = UninformativePrior()

stuff = {
    "seed": SEED,
    "env_name": env_name,
    "gamma": gamma,
    "hidden_units": hidden_units,
    "act_space": act_space,
    "obs_space": obs_space,
    "continuous_action": continuous_action,
    "normalise_advantages": normalise_advantages,
    "training_batch_size": training_batch_size,
}


# --- SVPG UPDATE ---
def svpg_update_step(particles, critics, optims_policy, optims_value, prior: BasePrior, alpha: float, env, gamma: float, ep = 0, config = None):
    # config controls the pg estimator used
    # Set the estimator and batch size
    if config is None:
        if use_gae:
            config = {"estimator": "gae", "batch_size": training_batch_size, "lam": 0.95}
        else:
            config = {"estimator": "reinforce", "batch_size": training_batch_size, "baseline": "const_mean"}

    estimator = config.get("estimator", "reinforce")
    batch_size = config.get("batch_size", training_batch_size)

    thetas = []
    grads  = []
    ep_rewards = []

    # Compute PG estimate for each particle
    for i, p in enumerate(particles):
        theta_i = get_param_vector(p)

        # Policy gradient time
        if estimator == "gae":
            # Compute gradients and update critics
            g_i, critic_batch, total_r = GAE_pg(p, critics[i], env, stuff, lam=config.get("lam", 0.95), n_episodes=batch_size)
            if verbose >= 1 and ep % 5 == 0:
                test_critic(critics[i], critic_batch["states_tf"], critic_batch["returns_mc_np"])
            fit_critic_batch(value_model=critics[i], optimizer=optims_value[i], states_tf=critic_batch["states_tf"], v_target_tf=critic_batch["v_target_tf"])

        else:                      
            g_i, total_r = REINFORCE_pg(p, env, stuff, batch_size, baseline=config.get("baseline", "const_mean"))

        thetas.append(theta_i)
        grads.append(g_i)
        ep_rewards.append(total_r)

    theta_mat = tf.stack(thetas, axis=0)   
    grad_mat  = tf.stack(grads,  axis=0) 

    K, h = rbf_kernel(theta_mat) # Compute kernel matrix

    policy_term = (1.0 / alpha) * grad_mat
    prior_term  = prior.grad_log_prob(theta_mat)
    score = policy_term + prior_term

    attract = (tf.transpose(K) @ score) / tf.cast(len(particles), tf.float32)
    repel   = kernel_grad_terms(theta_mat, K, h)
    phi     = attract + repel

    # --- Prior Dominance Diagnostic ---
    # Norm of directional forces for each particle
    theta_norms = tf.norm(theta_mat, axis = 1) # The size of the particle
    policy_norms = tf.norm(policy_term, axis=1)
    prior_norms  = tf.norm(prior_term, axis=1)
    repel_norms  = tf.norm(repel, axis=1)
    phi_norms    = tf.norm(phi, axis=1)

    prior_fraction = prior_norms / (policy_norms + prior_norms + 1e-12)
    
    force_stats = {"theta_norm_mean": float(tf.reduce_mean(theta_norms).numpy()),
                   "policy_norm_mean": float(tf.reduce_mean(policy_norms).numpy()), "policy_norm_std": float(tf.math.reduce_std(policy_norms).numpy()),
                   "prior_norm_mean": float(tf.reduce_mean(prior_norms).numpy()), "prior_norm_std": float(tf.math.reduce_std(prior_norms).numpy()),
                   "repel_norm_mean": float(tf.reduce_mean(repel_norms).numpy()), "repel_norm_std": float(tf.math.reduce_std(repel_norms).numpy()),
                   "phi_norm_mean": float(tf.reduce_mean(phi_norms).numpy()), "phi_norm_std": float(tf.math.reduce_std(phi_norms).numpy()), 
                   "prior_fraction_mean": float(tf.reduce_mean(prior_fraction).numpy()), "prior_fraction_std": float(tf.math.reduce_std(prior_fraction).numpy())}
    
    # Update each particle
    for i, (p, opt) in enumerate(zip(particles, optims_policy)):
        ADAM(p, phi[i], opt)

    return np.array(ep_rewards), theta_mat.numpy(), force_stats

# --- EVALUATION ---
def eval_policy(policy_model, env=None, n=100, greedy = True, render = True):
    returns = []

    if env is None:
        env = gym.make(env_name, render_mode = "human" if render else None)
        env.reset(seed=SEED)

    for i in tqdm(range(n)):
        traj = collect_episode_trajectory(
            policy_model=policy_model,
            env=env,
            continuous_action=continuous_action,
            act_space=act_space,
            seed=None,
            greedy=greedy)
        returns.append(traj["total_reward"])

    return float(np.mean(returns)), float(np.std(returns))

def eval_policy_video(policy_model):
    env = gym.make(env_name, render_mode="rgb_array")
    env = RecordVideo(env, video_folder="videos", episode_trigger = lambda i: True, name_prefix="eval")

    traj = collect_episode_trajectory(policy_model=policy_model, env=env, continuous_action=continuous_action, act_space=act_space, seed=SEED, greedy=True)
    env.close()
    return traj["total_reward"]

# Stats/Helper functions
def weight_stats(model, prefix=""):
    # Return information about the weights of a given model
    out = []
    for v in model.trainable_variables:
        x = v.numpy().ravel()
        out.append({
            "name": prefix + v.name,
            "shape": v.shape,
            "rms": float(np.sqrt(np.mean(x*x))),
            "l2": float(np.linalg.norm(x)),
            "maxabs": float(np.max(np.abs(x))),
            "mean": float(np.mean(x)),
            "std": float(np.std(x))})
    return out



# TRAINING
def train_and_record(training_batch_size_override=None, run_name="run", seed_offset=0, make_plots=True):
    global training_batch_size

    old_batch = training_batch_size
    if training_batch_size_override is not None:
        training_batch_size = int(training_batch_size_override)

    run_seed = SEED + int(seed_offset)
    set_seed(run_seed)

    env_local = gym.make(env_name)
    env_local.reset(seed=run_seed)

    particles, critics, optims_policy, optims_value = initialise()

    average_over = 30
    returns_window_mean = deque(maxlen=average_over)   # mean across particles, per update
    returns_window_best = deque(maxlen=average_over)   # best particle, per update
    returns_window_std  = deque(maxlen=average_over)   # std across particles, per update

    saved_updates = []
    saved_mean = []
    saved_best = []
    saved_std = []

    plot_every = 5   # update plot every N SVPG updates (episodes)

    best_ever = -np.inf
    best_theta = None

    # --- Info about current settings ---
    if verbose >= 1:
        print(
        f"\n"
        f"=== Experiment configuration ===\n"
        f"Environment           : {env_name}\n"
        f"Particles             : {M}\n"
        f"Alpha                 : {alpha}\n"
        f"Policy Learning rate  : {lr_actor}\n"
        f"Critic Lerning rate   : {lr_critic}\n"
        f"Training batch size   : {training_batch_size} trajectories / update\n"
        f"SVPG updates          : {num_episodes}\n"
        f"Prior                 : {prior_now.name}\n"
        f"================================\n")

    # --- Info about theta_0 ---
    stats = weight_stats(particles[0], prefix = "Initial/")
    if verbose >= 2:
        for s in stats:
            print(f"{s['name']:40s}  "
            f"shape={s['shape']!s:12s}  "
            f"rms={s['rms']:.4e}  "
            f"std={s['std']:.4e}  "
            f"mean={s['mean']:.4e} "
            f"max|w|={s['maxabs']:.4e}")
        
    # --- TRAINING ---   
    rewards = np.zeros((num_episodes+1, M))
    pbar = tqdm(range(1, num_episodes + 1), desc=f"Train {run_name}, GAE PG estimator {use_gae}", leave=True)
    
    for ep in pbar:
        ep_rewards, _, _ = svpg_update_step(particles, critics, optims_policy, optims_value, prior_now, alpha, env_local, gamma, ep)

        rewards[ep] = ep_rewards

        # For this episode, average over the particles (M = 1 this is just the current reward and sd = 0)
        mean_r = float(np.mean(ep_rewards))
        best_r = float(np.max(ep_rewards))
        std_r  = float(np.std(ep_rewards))

        returns_window_mean.append(mean_r)
        returns_window_best.append(best_r)
        returns_window_std.append(std_r)

        # Save best policy
        if best_r > best_ever:
            best_ever = best_r
            best_theta = get_param_vector(particles[int(np.argmax(ep_rewards))]).numpy().copy()
            np.save("theta_star_best.npy", best_theta)

        # Logging
        if ep % plot_every == 0:
            saved_updates.append(ep)
            saved_mean.append(float(np.mean(returns_window_mean)))
            saved_best.append(float(np.mean(returns_window_best)))
            saved_std.append(float(np.mean(returns_window_std)))

            best_particle = np.argmax(ep_rewards)

            if make_plots: 
                curves = [{"x": saved_updates, "y": saved_mean, "label": "Rolling mean", "std": saved_std},
                          {"x": saved_updates, "y": saved_best, "label": "Rolling best particle"}]
                plot_learning_curve(curves, title="SVPG learning curve")
        
        if ep % 10 == 0:
            pbar.set_postfix({
                "mean": f"{np.mean(returns_window_mean):.1f}",
                "best": f"{np.mean(returns_window_best):.1f}",
                "std":  f"{np.mean(returns_window_std):.1f}",
            })

        # Eval return
        # if ep % 20 == 0:
        #     idx = int(np.argmax(ep_rewards))
        #     mean_return, std_return = eval_policy_greedy(particles[idx], env, n=20)
        #     if verbose >= 1:
        #         print(f"Episode {ep}/{num_episodes}  |  mean evaluation return: {mean_return:.1f}  |  std: {std_return:.1f}")

    env_local.close()
    

    # Info about theta^*
    if verbose >= 2:
        stats = weight_stats(particles[0], prefix = "Final/")
        for s in stats:
            print(f"{s['name']:40s}  "
            f"shape={s['shape']!s:12s}  "
            f"rms={s['rms']:.4e}  "
            f"std={s['std']:.4e}  "
            f"max|w|={s['maxabs']:.4e}")
    
    # Save final weights
    best_particle = particles[np.argmax(rewards[-1])]
    theta_star = get_param_vector(best_particle).numpy()
    np.save("theta_star.npy", theta_star) 

    training_batch_size = old_batch

    return(np.array(saved_updates), np.array(saved_mean), np.array(saved_best), np.array(saved_std))

    # Learning curve
    # for i in range(M):
    #     plt.plot(rewards[:, i], label=f"Particle {i}", alpha = 0.5)
    # plt.plot(np.max(rewards, axis=1), "r", linewidth=2, label="Best", alpha=0.3)

    # plt.xlabel("SVPG update")
    # plt.ylabel("Episode return")
    # plt.title("SVPG learning curves per particle")
    # plt.legend()
    # plt.grid(True)
    # plt.show()


def train_and_record_compare_algos(configs = None, title = None):
    set_seed(SEED)

    estimator_configs = [
            {"estimator": "reinforce", "label": "Improper Prior", "batch_size": 5, "baseline": "const_mean"},
            {"estimator": "reinforce", "label": "Gaussian Prior", "batch_size": 5, "baseline": "const_mean"},
            {"estimator": "reinforce", "label": "Laplace Prior", "batch_size": 5, "baseline": "const_mean"},
            {"estimator": "reinforce", "label": "Cauchy Prior", "batch_size": 5, "baseline": "const_mean"}]

    specs = [{"config": estimator_configs[0], "M": 1, "prior": UninformativePrior(), "alpha": alpha},
             {"config": estimator_configs[1], "M": 1, "prior": GaussianPrior(0, 1), "alpha": alpha},
             {"config": estimator_configs[2], "M": 1, "prior": LaplacePrior(0, np.round(1/np.sqrt(2), 3)), "alpha": alpha},
             {"config": estimator_configs[3], "M": 1, "prior": CauchyPrior(0, 1), "alpha": alpha}]

    max_M = max(spec.get("M", M) for spec in specs)
    particles0, critics0, _, _ = initialise(M=max_M)

    initial_thetas = [get_param_vector(p).numpy().copy() for p in particles0]
    initial_critic_thetas = [get_param_vector(c).numpy().copy() for c in critics0]

    runs = []
    average_over = 30
    plot_every = 5


    for i, test in enumerate(specs):
        set_seed(SEED)

        env_local = gym.make(env_name)
        env_local.reset(seed=SEED)

        M_i = test.get("M", M)
        particles, critics, optims_policy, optims_value = initialise(M=M_i)

        # for j in range(len(particles)):
        #     set_param_vector_from_flat(particles[j], tf.convert_to_tensor(initial_thetas[j], dtype=tf.float32))
        #     set_param_vector_from_flat(critics[j], tf.convert_to_tensor(initial_critic_thetas[j], dtype=tf.float32))

        for j in range(len(particles)):
            if test.get("init_at_theta_star", False):
                set_param_vector_from_flat(
                    particles[j],
                    tf.convert_to_tensor(theta_star, dtype=tf.float32)
                )
            else:
                set_param_vector_from_flat(
                    particles[j],
                    tf.convert_to_tensor(initial_thetas[j], dtype=tf.float32)
                )

            set_param_vector_from_flat(
                critics[j],
                tf.convert_to_tensor(initial_critic_thetas[j], dtype=tf.float32)
            )

        runs.append({
            "config": test["config"],
            "env": env_local,
            "particles": particles,
            "critics": critics,
            "prior": test.get("prior", prior_now),
            "alpha": test.get("alpha", alpha),
            "optims_policy": optims_policy,
            "optims_value": optims_value,
            "window_mean": deque(maxlen=average_over),
            "window_best": deque(maxlen=average_over),
            "window_std": deque(maxlen = average_over),
            "saved_x": [],
            "saved_mean": [],
            "saved_best": [],
            "saved_std": [],
            "initial_thetas": [get_param_vector(p).numpy().copy() for p in particles],
            "initial_critics": [get_param_vector(c).numpy().copy() for c in critics],
            "final_thetas": None,
            "final_critics": None,
            "policy_norm_mean": [],
            "policy_norm_std": [],
            "prior_norm_mean": [],
            "prior_norm_std": [],
            "repel_norm_mean": [],
            "repel_norm_std": [],
            "phi_norm_mean": [],
            "phi_norm_std": [],
            "prior_fraction_mean": [],
            "prior_fraction_std": [],
            "theta_norm_mean": []
        })

    pbar = tqdm(range(1, num_episodes + 1), desc="Train compare algos", leave=True)
    last_completed_ep = 0

    try:
        for ep in pbar:
            last_completed_ep = ep
            # Run each config at each timestep
            for run in runs:
                ep_rewards, _, forces = svpg_update_step(run["particles"], run["critics"], run["optims_policy"], run["optims_value"], run["prior"], run["alpha"], run["env"], gamma, ep, config=run["config"])

                mean_r = float(np.mean(ep_rewards))
                best_r = float(np.max(ep_rewards))
                std_r = float(np.std(ep_rewards))

                run["window_mean"].append(mean_r)
                run["window_best"].append(best_r)
                run["window_std"].append(std_r)

                run["theta_norm_mean"].append(forces["theta_norm_mean"])

                run["policy_norm_mean"].append(forces["policy_norm_mean"])
                run["policy_norm_std"].append(forces["policy_norm_std"])
                run["prior_norm_mean"].append(forces["prior_norm_mean"])
                run["prior_norm_std"].append(forces["prior_norm_std"])
                run["repel_norm_mean"].append(forces["repel_norm_mean"])
                run["repel_norm_std"].append(forces["repel_norm_std"])
                run["phi_norm_mean"].append(forces["phi_norm_mean"])
                run["phi_norm_std"].append(forces["phi_norm_std"])
                run["prior_fraction_mean"].append(forces["prior_fraction_mean"])
                run["prior_fraction_std"].append(forces["prior_fraction_std"])

                if ep % plot_every == 0:
                    run["saved_x"].append(ep)
                    run["saved_mean"].append(float(np.mean(run["window_mean"])))
                    run["saved_best"].append(float(np.mean(run["window_best"])))
                    run["saved_std"].append(float(np.mean(run["window_std"])))

            if ep % plot_every == 0:
                curves = []
                for run in runs:
                    curves.append({"x": run["saved_x"], "y": run["saved_mean"], "std": run["saved_std"], "label": run["config"]["label"]})
                plot_learning_curve(curves, title=title)

            if ep % 10 == 0:
                postfix = {}
                for i, run in enumerate(runs):
                    postfix[f"run{i+1}"] = f"{np.mean(run['window_mean']):.1f}" if len(run["window_mean"]) else "-"
                pbar.set_postfix(postfix)
    except KeyboardInterrupt:
        print(f"Interrupted at episode {last_completed_ep}")
    
    finally:
        for run in runs:
            run["final_thetas"] = [get_param_vector(p).numpy().copy() for p in run["particles"]]
            run["final_critics"] = [get_param_vector(c).numpy().copy() for c in run["critics"]]
            run["env"].close()

        results = {run["config"]["label"]: {
            "config": run["config"],
            "x": np.array(run["saved_x"]),
            "mean": np.array(run["saved_mean"]),
            "best": np.array(run["saved_best"]),
            "std": np.array(run["saved_std"]),
            "initial_thetas": run["initial_thetas"],
            "initial_critics": run["initial_critics"],
            "final_thetas": run["final_thetas"],
            "final_critics": run["final_critics"],
            "policy_norm_mean": np.array(run["policy_norm_mean"]),
            "policy_norm_std": np.array(run["policy_norm_std"]),
            "prior_norm_mean": np.array(run["prior_norm_mean"]),
            "prior_norm_std": np.array(run["prior_norm_std"]),
            "repel_norm_mean": np.array(run["repel_norm_mean"]),
            "repel_norm_std": np.array(run["repel_norm_std"]),
            "phi_norm_mean": np.array(run["phi_norm_mean"]),
            "phi_norm_std": np.array(run["phi_norm_std"]),
            "prior_fraction_mean": np.array(run["prior_fraction_mean"]),
            "prior_fraction_std": np.array(run["prior_fraction_std"]),
            "theta_norm_mean": np.array(run["theta_norm_mean"])}
        for run in runs}

        with open("compare_algos_results.pkl", "wb") as f:
            pickle.dump(results, f)
    
    return results


#def train_and_record_compare_algos_DEPRICATED(configs = None):
    set_seed(SEED)

    # Get one common initial policy
    particles0, critics0, _, _ = initialise()
    initial_thetas = [get_param_vector(p).numpy().copy() for p in particles0]
    initial_critic_thetas = [get_param_vector(c).numpy().copy() for c in critics0]

    runs = []
    average_over = 30
    plot_every = 5

    if configs == None:
        configs = [
            {"estimator": "reinforce", "label": "REINFORCE: No Baseline, B=1", "batch_size": 1, "baseline": None},
            {"estimator": "reinforce", "label": "REINFORCE: Constant Baseline, B=1", "batch_size": 1, "baseline": "const_mean"},
            {"estimator": "reinforce", "label": "REINFORCE: No Baseline, B=5", "batch_size": 5, "baseline": None},
            {"estimator": "reinforce", "label": "REINFORCE: Constant Baseline, B=5", "batch_size": 5, "baseline": "const_mean"}
        ]

    for test in configs:
        set_seed(SEED)

        env_local = gym.make(env_name)
        env_local.reset(seed=SEED)

        particles, critics, optims_policy, optims_value = initialise()

        for i in range(len(particles)):
            set_param_vector_from_flat(particles[i], tf.convert_to_tensor(initial_thetas[i], dtype=tf.float32))
            set_param_vector_from_flat(critics[i], tf.convert_to_tensor(initial_critic_thetas[i], dtype=tf.float32))

        runs.append({
            "label": test["label"],
            "config": test,
            "env": env_local,
            "particles": particles,
            "critics": critics,
            "optims_policy": optims_policy,
            "optims_value": optims_value,
            "window_mean": deque(maxlen=average_over),
            "window_best": deque(maxlen=average_over),
            "window_std": deque(maxlen = average_over),
            "saved_x": [],
            "saved_mean": [],
            "saved_best": [],
            "saved_std": []
        })

    pbar = tqdm(range(1, num_episodes + 1), desc="Train compare algos", leave=True)

    for ep in pbar:
        # Run each config at each timestep
        for run in runs:
            ep_rewards, _ = svpg_update_step(run["particles"], run["critics"], run["optims_policy"], run["optims_value"], prior_now, alpha, run["env"], gamma, ep, config=run["config"])

            mean_r = float(np.mean(ep_rewards))
            best_r = float(np.max(ep_rewards))
            std_r = float(np.std(ep_rewards))

            run["window_mean"].append(mean_r)
            run["window_best"].append(best_r)
            run["window_std"].append(std_r)

            if ep % plot_every == 0:
                run["saved_x"].append(ep)
                run["saved_mean"].append(float(np.mean(run["window_mean"])))
                run["saved_best"].append(float(np.mean(run["window_best"])))
                run["saved_std"].append(float(np.mean(run["window_std"])))

        if ep % plot_every == 0:
            curves = []
            for run in runs:
                curves.append({"x": run["saved_x"], "y": run["saved_mean"], "std": run["saved_std"], "label": run["label"]})
            plot_learning_curve(curves, title="Estimator comparison")

        if ep % 10 == 0:
            postfix = {}
            for i, run in enumerate(runs):
                postfix[f"run{i+1}"] = f"{np.mean(run['window_mean']):.1f}" if len(run["window_mean"]) else "-"
            pbar.set_postfix(postfix)

    for run in runs:
        run["env"].close()

    results = {run["label"]: {"x": np.array(run["saved_x"]), "mean": np.array(run["saved_mean"]), "best": np.array(run["saved_best"]), "std": np.array(run["saved_std"])} for run in runs}
    
    with open("compare_algos_results.pkl", "wb") as f:
        pickle.dump(results, f)
    
    return results


def initialise(M=M, obs_space=obs_space, hidden_units = hidden_units, lr_actor = lr_actor, lr_critic = lr_critic):
    particles = [PolicyNetwork(hidden_units=hidden_units, act_space=act_space, continuous=continuous_action) for _ in range(M)]
    # Build once
    _ = [p(tf.convert_to_tensor(np.zeros((1, obs_space), dtype=np.float32))) for p in particles]

    # Critic networks per particle
    critics = [ValueNetwork(hidden_units=hidden_units) for _ in range(M)]
    _ = [c(tf.convert_to_tensor(np.zeros((1, obs_space), dtype=np.float32))) for c in critics]

    if start_true and False:
        # Set around optimal
        theta_star_tf = tf.convert_to_tensor(theta_star, dtype=tf.float32)
        for p in particles:
            Dp = int(get_param_vector(p).shape[0])
            noise = tf.random.normal([Dp], stddev=0.01, dtype=tf.float32)
            set_param_vector_from_flat(p, theta_star_tf + noise)
    else:
        # Initialise weights of each particle randomly
        for p in particles:
            flat = get_param_vector(p)
            noise = tf.random.normal(tf.shape(flat), stddev=0.01)
            set_param_vector_add_(p, noise)

    # Create ADAM optimisers
    optims_policy = [tf.keras.optimizers.Adam(learning_rate=lr_actor) for _ in range(M)]
    optims_value = [tf.keras.optimizers.Adam(learning_rate = lr_critic) for _ in range(M)]

    return particles, critics, optims_policy, optims_value

if __name__ == "__main__":
    # --- LOOKING AT TRAINED AGENT ---
    learned_pol = PolicyNetwork(hidden_units=hidden_units, act_space=act_space, continuous=continuous_action)
    _ = learned_pol(tf.convert_to_tensor(np.zeros((1, obs_space), dtype=np.float32)))
    set_param_vector_from_flat(learned_pol, tf.convert_to_tensor(theta_star, dtype=tf.float32))
    print(eval_policy_video(learned_pol))

    # --- COMPARING ESTIMATOR VARIANCE ---
    #Experiments.compare_reinforce_variance(stuff, batch_sizes=(1,5), n_repeats=500, include_learned=True)

    # --- PRIOR PREDICTIVE CHECKS ---
    # prior_mean = 0
    # prior_std = 1
    # prior_now = GaussianPrior()
    # ppc = prior_predictive_check(stuff, env, prior_now, num_policies=1000)
    # ppc_plot = cross_policy_bin_eval_plot((0,2), ppc["states_visited"], ppc["policies"], cartpole_states, prior_now, res=45)

    # with open(f"PPC_{prior_now.name}.pkl", "rb") as f:
    #     results = pickle.load(f)

    # plot_ppc(results=results, cartpole_states=cartpole_states, prior_name=results["prior_name"])

    # --- TRAINING AGENTS ---
    #train_and_record_compare_algos()
    #train_and_record()

    # --- PLOT LEARNING CURVES ---
    # with open("aOneParticle_medalpha.pkl", "rb") as f:
    #     results = pickle.load(f)

    # curves = []
    # for label, data in results.items():
    #     curves.append({
    #         "x": data["x"],
    #         "y": data["mean"],
    #         "std": data["std"],
    #         "label": label
    #     })

    # plot_learning_curve(curves, title="Estimator Comparison - Lunar Lander")

    # # --- PLOT PRIOR DOMININANCE ---
    # window = 30
    # curves = []
    # for label, data in results.items():
    #     y = np.array(data["prior_fraction_mean"], dtype=float)
    #     s = np.array(data["prior_fraction_std"], dtype=float)
    #     extra = np.array(data["theta_norm_mean"], dtype=float)

    #     if len(y) >= window:
    #         y_smooth = np.array([np.mean(y[max(0, i - window + 1):i + 1]) for i in range(len(y))])
    #         s_smooth = np.array([np.mean(s[max(0, i - window + 1):i + 1]) for i in range(len(s))])
    #         extra_smooth = np.array([np.mean(extra[max(0, i-window+1):i+1]) for i in range(len(extra))])
    #     else:
    #         y_smooth = y
    #         s_smooth = s
    #         extra_smooth = extra
        
    #     extra_smooth = extra_smooth / (np.max(extra) + 1e-12)
    #     #y_smooth = y_smooth / (np.max(y_smooth) + 1e-12)
    #     x = np.arange(1, len(y_smooth) + 1)

    #     curves.append({
    #         "x": x,
    #         "y": y_smooth,
    #         "std": s_smooth,
    #         "label": label,
    #     })

    # plot_learning_curve(curves, title="Theta Norm, Alpha = 1")

    # --- PLOT AVERAGE LEARNING CURVE OVER RUNS ---
    # files = ["REINFORCE_cartpole1.pkl", "REINFORCE_cartpole2.pkl", "REINFORCE_cartpole3.pkl"]
    # all_results = []
    # for file in files:
    #     with open(file, "rb") as f:
    #         all_results.append(pickle.load(f))

    # curves = []
    # first_keys = list(all_results[0].keys())

    # new_labels = ["No baseline, B=1", "Constant baseline, B=1", "No baseline, B=5", "Constant baseline, B=5"]
    # for i in range(len(first_keys)):
    #     label = first_keys[i]
    #     x = np.array(all_results[0][label]["x"])

    #     means = []
    #     for results in all_results:
    #         this_keys = list(results.keys())
    #         this_label = this_keys[i]
    #         means.append(np.array(results[this_label]["mean"]))

    #     means = np.array(means)

    #     avg_mean = np.mean(means, axis=0)
    #     std_mean = np.std(means, axis=0)

    #     curves.append({
    #         "x": x,
    #         "y": avg_mean,
    #         "std": std_mean,
    #         "label": new_labels[i]
    #     })

    # plot_learning_curve(curves, title="REINFORCE Learning Curves: Lunar Lander")