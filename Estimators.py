import numpy as np
import tensorflow as tf
import random
import os

from Utilities import *

def GAE_pg(policy_model, value_model, env, stuff, lam=0.95, n_episodes=1):
    gamma = stuff["gamma"]
    continuous_action = stuff["continuous_action"]
    act_space = stuff["act_space"]
    normalise_advantages = stuff["normalise_advantages"]

    all_states = []
    all_actions = []
    all_rewards = []
    
    all_dones = []
    all_next_states = []
    total_rewards = []
    traj_lengths = []

    for ep in range(n_episodes):
        traj = collect_episode_trajectory(
            policy_model=policy_model,
            env=env,
            continuous_action=continuous_action,
            act_space=act_space,
            seed=None,
            greedy=False,
        )

        all_states.append(traj["states"])

        if continuous_action:
            all_actions.append(traj["actions"].astype(np.float32)) 
        else:
            all_actions.append(traj["actions"].astype(np.int32))  

        all_rewards.append(traj["rewards"])
        all_dones.append(traj["dones"])
        all_next_states.append(traj["next_states"])
        total_rewards.append(traj["total_reward"])
        #traj_lengths.append(len(traj["rewards"]))

    # Flatten into single batch
    states_tf      = tf.convert_to_tensor(np.vstack(all_states), dtype=tf.float32)           
    next_states_tf = tf.convert_to_tensor(np.vstack(all_next_states), dtype=tf.float32)

    if continuous_action:
        actions_tf = tf.convert_to_tensor(np.vstack(all_actions), dtype=tf.float32) 
    else:
        actions_tf = tf.convert_to_tensor(np.concatenate(all_actions), dtype=tf.int32)     

    rewards_np     = np.concatenate(all_rewards).astype(np.float32)                          
    dones_np       = np.concatenate(all_dones).astype(np.float32)                           

    # Compute values for GAE using current critic
    values_np      = tf.squeeze(value_model(states_tf), axis=1).numpy().astype(np.float32)       # V(s_t)
    next_values_np = tf.squeeze(value_model(next_states_tf), axis=1).numpy().astype(np.float32)  # V(s_{t+1})

    adv_np, v_target_np = compute_gae(rewards_np, dones_np, values_np, next_values_np, gamma, lam)

    if normalise_advantages:
        adv_np = (adv_np - adv_np.mean()) / (adv_np.std() + 1e-8)

    # if random.random() < 0.2:
    #     mc_returns = compute_returns(rewards_np, gamma)
    #     test_critic(value_model, states_tf.numpy(), mc_returns)
    #     print(
    #         f"[advantage] mean={np.mean(adv_np):.3f} "
    #         f"std={np.std(adv_np):.3f} "
    #         f"max={np.max(adv_np):.3f}")

    adv_tf      = tf.stop_gradient(tf.convert_to_tensor(adv_np, dtype=tf.float32))
    v_target_tf = tf.stop_gradient(tf.convert_to_tensor(v_target_np, dtype=tf.float32))

    with tf.GradientTape() as tape:

        if not continuous_action:
            # Actor log-probs
            probs = policy_model(states_tf)
            chosen = tf.gather(probs, actions_tf, axis=1, batch_dims=1)
            logp = tf.math.log(chosen + 1e-8)

        else:
            mu, policy_sd = policy_model(states_tf) 

            act_high = tf.convert_to_tensor(env.action_space.high, dtype=tf.float32)
            u = arctanh(actions_tf / act_high)

            logp_perdim_u = gaussian_log_prob(u, mu, policy_sd)  
            logp_u = tf.reduce_sum(logp_perdim_u, axis=-1)   

            log_det = tf.reduce_sum(tf.math.log(act_high + 1e-8) + tf.math.log(1.0 - tf.tanh(u) ** 2 + 1e-8), axis=-1)

            logp = logp_u - log_det

        actor_loss = -tf.reduce_mean(logp * adv_tf)

        # Critic loss (MSE to v_target)
        # v_s = tf.squeeze(value_model(states_tf), axis=1)
        # critic_loss = tf.reduce_mean(tf.square(v_target_tf - v_s))

    actor_grads  = tape.gradient(actor_loss,  policy_model.trainable_variables)

    # critic_grads = tape.gradient(critic_loss, value_model.trainable_variables)

    del tape

    g_flat = -flatten_gradients(actor_grads, policy_model.trainable_variables)

    batch_data = {"states_tf": states_tf, "v_target_tf": v_target_tf, "returns_mc_np": compute_returns(rewards_np, gamma)}
    return g_flat, batch_data, float(np.mean(total_rewards))

def fit_critic_batch(value_model, optimizer, states_tf, v_target_tf,
                     n_epochs=3, minibatch_size=256, grad_clip_norm=1.0):
    N = int(states_tf.shape[0])

    if minibatch_size is None or minibatch_size >= N:
        minibatch_size = N

    idx_all = tf.range(N)

    for _ in range(n_epochs):
        idx_all = tf.random.shuffle(idx_all)

        for start in range(0, N, minibatch_size):
            mb_idx = idx_all[start:start + minibatch_size]
            xb = tf.gather(states_tf, mb_idx)
            yb = tf.gather(v_target_tf, mb_idx)

            with tf.GradientTape() as tape:
                pred = tf.squeeze(value_model(xb), axis=1)
                loss = tf.reduce_mean(tf.square(yb - pred))

            grads = tape.gradient(loss, value_model.trainable_variables)
            grads = [
                None if g is None else tf.clip_by_norm(g, grad_clip_norm)
                for g in grads
            ]
            optimizer.apply_gradients(zip(grads, value_model.trainable_variables))

# ---- Policy gradient for a single particle via REINFORCE (now with batches) ----
def REINFORCE_pg(policy_model, env, stuff, n_episodes=1, baseline=None, seed=None):
    gamma = stuff["gamma"]
    continuous_action = stuff["continuous_action"]
    act_space = stuff["act_space"]
    normalise_advantages = stuff["normalise_advantages"]

    # Seed
    if seed is not None:
        set_seed(seed)
    
    # For each timestep, for each episode
    all_states  = []
    all_actions = []
    all_returns = []

    # For each episode
    total_rewards = []

    traj_lengths = []

    # Collect n episodes
    for ep in range(n_episodes):
        ep_seed = None if seed is None else int(seed) + ep

        traj = collect_episode_trajectory(
            policy_model=policy_model,
            env=env,
            continuous_action=continuous_action,
            act_space=act_space,
            seed=ep_seed,
            greedy=False,
        )

        returns = compute_returns(traj["rewards"], gamma)
        if normalise_advantages:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        all_states.append(traj["states"])
        all_actions.append(np.asarray(traj["actions"]))
        all_returns.append(np.asarray(returns))
        total_rewards.append(traj["total_reward"])
        #traj_lengths.append(len(traj["rewards"]))

    # Concatenate all episodes into tf tensor
    states_tf  = tf.convert_to_tensor(np.vstack(all_states), dtype=tf.float32)

    if not continuous_action:
        # all_actions is a list of arrays of shape (T_ep,) of ints
        actions_tf = tf.convert_to_tensor(np.concatenate(all_actions), dtype=tf.int32)
    else:
        # all_actions is a list of arrays of shape (T_ep, act_dim)
        actions_tf = tf.convert_to_tensor(np.vstack(all_actions), dtype=tf.float32)

    returns_tf = tf.convert_to_tensor(np.concatenate(all_returns), dtype=tf.float32)
    returns_tf = tf.stop_gradient(returns_tf) # Returns are constant w.r.t parameters when computing gradient

    # BASELINE STUFF
    if baseline is None:
        adv_tf = returns_tf # No baseline so use G_t as estimator for Q

    elif isinstance(baseline, str) and baseline == "const_mean":
        # Mean of returns in THIS batch
        b_const = tf.reduce_mean(returns_tf)
        adv_tf = returns_tf - b_const
        adv_tf = tf.stop_gradient(adv_tf)

    elif isinstance(baseline, float):
        b_const = tf.constant(baseline, dtype=tf.float32)
        adv_tf = returns_tf - b_const
        adv_tf = tf.stop_gradient(adv_tf)
    else:
        # Learned baseline
        states_norm = (states_tf - baseline.X_mean) / baseline.X_std
        b = baseline(states_norm)
        b = b * baseline.y_std + baseline.y_mean
        #b = baseline(states_tf)
        b = tf.reshape(b, [-1]) 
        adv_tf = returns_tf - b
        adv_tf = tf.stop_gradient(adv_tf)

    # Compute policy gradient estimator as loss function
    with tf.GradientTape() as tape:
        if not continuous_action:
            probs = policy_model(states_tf) # Compute pi(a|s_t) for each t (T,A)
            chosen = tf.gather(probs, actions_tf, axis=1, batch_dims=1) # Find the probability of the CHOSEN actions so pi(a_t|s_t)
            logp = tf.math.log(chosen + 1e-8) # Log it
            loss = -tf.reduce_mean(logp * adv_tf) # Minimising - E(log pi(a_t|s_t)G_t) is equivilant to maximising J(theta)
    
        else:
            mu, policy_sd = policy_model(states_tf) # Unbounded mean

            u = arctanh(actions_tf/tf.convert_to_tensor(env.action_space.high, dtype=tf.float32))

            logp_perdim_u = gaussian_log_prob(u, mu, policy_sd) # The log density of the normal N(a_t,mu_t,sigma_t)
            logp_u = tf.reduce_sum(logp_perdim_u, axis=-1) # Summin over the actions log pi(a_t,s_t) = sum_d log N(a_{t,d};mu_{t,d},sigma_d)

            # change-of-variables correction: sum log |da/du|
            # a = act_high * tanh(u) => da/du = act_high * (1 - tanh(u)^2)
            # log|da/du| = log(act_high) + log(1 - tanh(u)^2)
            # We subtract this because log p(a) = log p(u) - log|da/du|
            log_det = tf.reduce_sum(tf.math.log(tf.convert_to_tensor(env.action_space.high, dtype=tf.float32) + 1e-8) + tf.math.log(1.0 - tf.tanh(u)**2 + 1e-8), axis=-1)

            logp = logp_u - log_det  # [T]
        
            loss = -tf.reduce_mean(logp * adv_tf) #L(theta) = -J(theta) so nabla L(theta) = -nabla J(theta)

    # Flatten gradient estimates
    grads = tape.gradient(loss, policy_model.trainable_variables) # Finds estimate for - \nabla J(theta) 
    g_flat = -flatten_gradients(grads, policy_model.trainable_variables)
    return g_flat, np.mean(total_rewards)


def fit_mc_baseline(policy_model, stuff, n_episodes=1000, epochs=100, hidden_units=64, lr=1e-3, batch_size = 512):
    env_name = stuff["env_name"]
    obs_space = stuff["obs_space"]
    continuous_action = stuff["continuous_action"]
    act_space = stuff["act_space"]
    gamma = stuff["gamma"]
    seed = stuff["seed"]

    X_path = f"baseline_X_{env_name}_{seed}_{n_episodes}.npy"
    y_path = f"baseline_y_{env_name}_{seed}_{n_episodes}.npy"

    set_seed(seed)

    baseline = tf.keras.Sequential([
        layers.Input(shape=(obs_space,)),
        layers.Dense(hidden_units, activation="relu"),
        layers.Dense(hidden_units, activation="relu"),
        layers.Dense(1),
])
    opt = tf.keras.optimizers.Adam(learning_rate=lr)

    if os.path.exists(X_path) and os.path.exists(y_path):
        # So we don't need to collect the training data again
        X = np.load(X_path)
        y = np.load(y_path)
    else:
        # Collect states (X) abd returns (y) with the fixed random policy
        X, y = [], []
        env_baseline = gym.make(env_name)

        for ep in tqdm(range(n_episodes)):
            traj = collect_episode_trajectory(policy_model, env_baseline, continuous_action, act_space, seed=seed + ep, greedy=False)

            G = compute_returns(traj["rewards"], gamma).astype(np.float32)
            X.append(traj["states"].astype(np.float32))
            y.append(G)

        env_baseline.close()

        X = np.vstack(X).astype(np.float32)  
        y = np.concatenate(y).astype(np.float32)

        np.save(X_path, X)
        np.save(y_path, y)

    X_mean = X.mean(axis=0, keepdims=True)
    X_std = X.std(axis=0, keepdims=True) + 1e-8
    X = (X - X_mean) / X_std

    y_mean = y.mean()
    y_std = y.std() + 1e-8
    y = (y - y_mean) / y_std

    # For unnormalising
    baseline.X_mean = X_mean
    baseline.X_std = X_std
    baseline.y_mean = y_mean
    baseline.y_std = y_std
    baseline.fixed_const = float(y_mean)

    N = X.shape[0]
    idx = np.arange(N)
    rng = np.random.default_rng(seed + 999)
    rng.shuffle(idx)

    n_val = int(0.2 * N)
    val_idx = idx[:n_val]
    tr_idx  = idx[n_val:]

    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_va, y_va = X[val_idx], y[val_idx]

    ds_train = (tf.data.Dataset.from_tensor_slices((X_tr, y_tr)). shuffle(len(y_tr), seed=seed, reshuffle_each_iteration=True). batch(batch_size))

    # Training the value function network
    for ep in range(epochs):
        for xb, yb in ds_train:
            with tf.GradientTape() as tape:
                pred = tf.squeeze(baseline(xb), axis=-1)
                loss = tf.reduce_mean(tf.square(pred - yb))
            grads = tape.gradient(loss, baseline.trainable_variables)
            grads = [tf.clip_by_norm(g, 1.0) if g is not None else None for g in grads]
            opt.apply_gradients(zip(grads, baseline.trainable_variables))

        if (ep == 0 or (ep + 1) % 5 == 0 or ep == epochs - 1):
            tr_mse = float(tf.reduce_mean(tf.square(tf.squeeze(baseline(X_tr), -1) - y_tr)).numpy())
            va_mse = float(tf.reduce_mean(tf.square(tf.squeeze(baseline(X_va), -1) - y_va)).numpy())
            va_var = tf.math.reduce_variance(y_va)
            r2 = 1.0 - (va_mse / (va_var + 1e-8))
            print(f"Epoch {ep+1:>3d}/{epochs}  train MSE={tr_mse:.4g}  val MSE={va_mse:.4g}  R^2={r2:.4f}")

    yhat_tr = tf.squeeze(baseline(X_tr), axis=-1).numpy()
    yhat_va = tf.squeeze(baseline(X_va), axis=-1).numpy()

    def stats(y_true, y_pred):
        mse = float(np.mean((y_pred - y_true) ** 2))
        rmse = float(np.sqrt(mse))
        var = float(np.var(y_true))
        r2 = float(1.0 - mse / (var + 1e-12))
        corr = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 else float("nan")
        return mse, rmse, r2, corr

    tr_mse, tr_rmse, tr_r2, tr_corr = stats(y_tr, yhat_tr)
    va_mse, va_rmse, va_r2, va_corr = stats(y_va, yhat_va)

    print("\n=== Learned baseline fit diagnostics ===")
    print(f"Data: N={N} timesteps from {n_episodes} episodes | val_frac={0.2}")
    print(f"Target y = G_t (returns-to-go), gamma={gamma}")
    print(f"Train: MSE={tr_mse:.6g} | RMSE={tr_rmse:.6g} | R^2={tr_r2:.4f} | corr={tr_corr:.4f}")
    print(f" Val : MSE={va_mse:.6g} | RMSE={va_rmse:.6g} | R^2={va_r2:.4f} | corr={va_corr:.4f}")
    print(f"y range: [{float(np.min(y)):.3f}, {float(np.max(y)):.3f}] "
            f"| yhat(val) range: [{float(np.min(yhat_va)):.3f}, {float(np.max(yhat_va)):.3f}]")
    print("========================================\n")

    return baseline