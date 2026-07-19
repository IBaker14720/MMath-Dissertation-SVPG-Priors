from Utilities import *
import matplotlib.pyplot as plt

def prior_predictive_check(stuff, env, prior, num_policies = 4, episodes_per_policy = 5):
    # Collect n trajectories from M policies sampled from the prior in the environment
    # Returns the states visited by the trajectories and the actions taken in those states with a policy id to track which policy the trajectory is from

    hidden_units = stuff["hidden_units"]
    act_space = stuff["act_space"]
    obs_space = stuff["obs_space"]
    continuous_action = stuff["continuous_action"]

    file_name = f"prior_sample_{num_policies}_{prior.name}_{episodes_per_policy}.pkl"

    # Load the samples if already computed
    if os.path.exists(file_name):
        with open(file_name, "rb") as f:
            data = pickle.load(f)

        theta_samples_np = data["parameters"]

        # Rebuild the policies from the saved parameters
        policies = []
        for i in tqdm(range(theta_samples_np.shape[0])):
            p = PolicyNetwork(hidden_units=hidden_units, act_space=act_space, continuous=continuous_action)
            _ = p(tf.convert_to_tensor(np.zeros((1, obs_space), dtype=np.float32)))
            set_param_vector_from_flat(p, tf.convert_to_tensor(theta_samples_np[i], dtype=tf.float32))
            policies.append(p)

        data["policies"] = policies
        return data

    # Get the size of the policy and build a base policy
    base_policy = PolicyNetwork(hidden_units=hidden_units, act_space=act_space, continuous = continuous_action)
    _ = base_policy(tf.convert_to_tensor(np.zeros((1, obs_space), dtype=np.float32)))
    D = int(get_param_vector(base_policy).shape[0])

     # Collect samples of D dimensional weight vectors from the prior
    theta_samples = tf.convert_to_tensor(prior.sample(num_policies, D))

    all_states = []
    all_actions = []
    policy_ids = []

    ep_returns = [[] for _ in range(num_policies)]

    for i in tqdm(range(num_policies)): # Loop through policies
        set_param_vector_from_flat(base_policy, theta_samples[i]) # Set the policy to have parameters sampled from the prior

        returns = []
        for ep in range(episodes_per_policy): # Collect some trajectories
            trajectory = collect_episode_trajectory(base_policy, env, continuous_action, act_space)

            all_states.append(trajectory["states"])
            all_actions.append(trajectory["actions"])

            policy_ids.append(np.full(len(trajectory["states"]), i, dtype=np.int32))
            returns.append(trajectory["total_reward"])

        ep_returns[i] = returns

    all_states = np.vstack(all_states)
    if continuous_action:
        all_actions = np.vstack(all_actions)
    else:
        all_actions = np.concatenate(all_actions)
    policy_ids = np.concatenate(policy_ids)
    theta_samples_np = theta_samples.numpy()


    policies = []
    # Build them
    for i in range(num_policies):
        p = PolicyNetwork(hidden_units=hidden_units, act_space=act_space, continuous = continuous_action)
        _ = p(tf.convert_to_tensor(np.zeros((1, obs_space), dtype=np.float32)))
        set_param_vector_from_flat(p, theta_samples[i])
        policies.append(p)

    data = {"states_visited": all_states, "actions_taken": all_actions, "policy_ids": policy_ids, "episode_returns": ep_returns, "parameters": theta_samples_np}

    with open(file_name, "wb") as f:
        pickle.dump(data, f)

    data["policies"] = policies
    return data



def cross_policy_bin_eval_plot(state_dims, visited_states, policies, cartpole_states, prior, res=45, states_per_bin=10):
    rng = np.random.default_rng(100)

    save_path = f"PPC_{prior.name}.pkl"

    dim_x, dim_y = state_dims
    x = visited_states[:, dim_x]
    y = visited_states[:, dim_y]

    x_edges = np.linspace(-2.4, 2.4, res + 1)
    y_edges = np.linspace(-0.2095, 0.2095, res + 1)

    x_bins = np.digitize(x, x_edges) - 1
    y_bins = np.digitize(y, y_edges) - 1

    valid = (x_bins >= 0) & (x_bins < res) & (y_bins >= 0) & (y_bins < res)

    x_bins = x_bins[valid]
    y_bins = y_bins[valid]
    visited_states = visited_states[valid]

    mean_action_map = np.full((res, res), np.nan, dtype=np.float32)
    entropy_map = np.full((res, res), np.nan, dtype=np.float32)
    between_policy_map = np.full((res, res), np.nan, dtype=np.float32)
    occupancy_map = np.full((res, res), np.nan, dtype=np.float32)

    per_bin_policy_mean_p = [[None for _ in range(res)] for _ in range(res)]
    per_bin_policy_mean_entropy = [[None for _ in range(res)] for _ in range(res)]

    eps = 1e-8

    for yi in tqdm(range(res)):
        for xi in tqdm(range(res), leave=False):
            mask = (x_bins == xi) & (y_bins == yi)
            idx = np.where(mask)[0]

            if len(idx) == 0:
                continue

            occupancy_map[yi, xi] = len(idx)

            if len(idx) > states_per_bin:
                idx = rng.choice(idx, size=states_per_bin, replace=False)

            states_here = visited_states[idx]
            states_tf = tf.convert_to_tensor(states_here, dtype=tf.float32)

            mean_ps = []
            mean_ents = []

            for policy in policies:
                probs = policy(states_tf).numpy()
                p_right = probs[:, 1]
                ent = -(p_right * np.log(p_right + eps) + (1.0 - p_right) * np.log(1.0 - p_right + eps))
                mean_ps.append(np.mean(p_right))
                mean_ents.append(np.mean(ent))

            mean_ps = np.asarray(mean_ps, dtype=np.float32)
            mean_ents = np.asarray(mean_ents, dtype=np.float32)

            per_bin_policy_mean_p[yi][xi] = mean_ps
            per_bin_policy_mean_entropy[yi][xi] = mean_ents

            mean_action_map[yi, xi] = np.mean(mean_ps)
            entropy_map[yi, xi] = np.mean(mean_ents)
            between_policy_map[yi, xi] = np.var(mean_ps)

    results = {
        "x_edges": x_edges,
        "y_edges": y_edges,
        "state_dims": state_dims,
        "mean_action_map": mean_action_map,
        "entropy_map": entropy_map,
        "between_policy_map": between_policy_map,
        "occupancy_map": occupancy_map,
        "per_bin_policy_mean_p": per_bin_policy_mean_p,
        "per_bin_policy_mean_entropy": per_bin_policy_mean_entropy,
        "prior_name": prior.name,
    }

    with open(save_path, "wb") as f:
        pickle.dump(results, f)

    plot_ppc(results=results, cartpole_states=cartpole_states, prior_name=prior.name)

    return results