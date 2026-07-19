from Utilities import *
from Estimators import *

def compare_reinforce_variance(stuff, batch_sizes=(1, 10), n_repeats = 400, include_learned = True, ref_policy = None):
    hidden_units = stuff["hidden_units"]
    act_space = stuff["act_space"]
    obs_space = stuff["obs_space"]
    continuous_action = stuff["continuous_action"]
    seed = stuff["seed"]
    env_name = stuff["env_name"]

    baseline_train_episodes: int = 2000
    baseline_hidden: int = 64
    baseline_epochs: int = 5
    baseline_lr: float = 1e-3

    # Build reference policy theta
    if ref_policy is None:
        ref_policy = PolicyNetwork(hidden_units=hidden_units, act_space=act_space, continuous=continuous_action)
        _ = ref_policy(tf.convert_to_tensor(np.zeros((1, obs_space), dtype=np.float32)))

    theta_ref = get_param_vector(ref_policy).numpy().copy()
    D = theta_ref.shape[0]

    # Fixed random projection direction u for Var(u^T g)
    u = np.random.default_rng(seed).normal(size=D)
    u = u / (np.linalg.norm(u) + 1e-12)
    u = u.astype(np.float32)

    estimator_names = ["none", "const_fixed", "learned"]

    results = {type_estimator: {B: {"mean_return": [], "grad_norm": [], "grad_proj": []}for B in batch_sizes} for type_estimator in estimator_names}

    # Fit learned baseline once so we do not include baseline-training noise in the comparison
    learned_baseline_model = None
    if include_learned:
        pol_for_baseline = PolicyNetwork(hidden_units=hidden_units, act_space=act_space, continuous=continuous_action)
        _ = pol_for_baseline(tf.convert_to_tensor(np.zeros((1, obs_space), dtype=np.float32)))
        set_param_vector_from_flat(pol_for_baseline, tf.convert_to_tensor(theta_ref, dtype=tf.float32))

        learned_baseline_model = fit_mc_baseline(pol_for_baseline, stuff, baseline_train_episodes, baseline_epochs, baseline_hidden, lr = baseline_lr)

        fixed_const = float(learned_baseline_model.fixed_const)

    # Collect Data
    for B in batch_sizes:
        tqdm.write(f"\n=== Batch size B = {B} ===")

        for k in tqdm(range(n_repeats), desc=f"B={B}", leave=True):
            seed_k = seed + k # Different seed for each run and the same across tests 

            env_eval = gym.make(env_name)

            pol = PolicyNetwork(hidden_units=hidden_units, act_space=act_space, continuous=continuous_action)
            _ = pol(tf.convert_to_tensor(np.zeros((1, obs_space), dtype=np.float32)))
            set_param_vector_from_flat(pol, tf.convert_to_tensor(theta_ref, dtype=tf.float32))

            # --- No baseline ---
            g_flat, mean_batch_return = REINFORCE_pg(pol, env_eval, stuff, n_episodes=B, baseline=None, seed=seed_k)
            g = g_flat.numpy()
            results["none"][B]["mean_return"].append(float(mean_batch_return))
            results["none"][B]["grad_norm"].append(float(np.linalg.norm(g)))
            results["none"][B]["grad_proj"].append(float(np.dot(u, g)))

            # --- Constant baseline ---
            # g_flat, mean_batch_return = REINFORCE_pg(pol, env_eval, stuff, n_episodes=B, baseline="const_mean", seed=seed_k)
            # g = g_flat.numpy()
            # results["const_mean"][B]["mean_return"].append(float(mean_batch_return))
            # results["const_mean"][B]["grad_norm"].append(float(np.linalg.norm(g)))
            # results["const_mean"][B]["grad_proj"].append(float(np.dot(u, g)))

            # --- Constant FIXED baseline ---
            g_flat, mean_batch_return = REINFORCE_pg(pol, env_eval, stuff, n_episodes=B, baseline=fixed_const, seed=seed_k)
            g = g_flat.numpy()
            results["const_fixed"][B]["mean_return"].append(float(mean_batch_return))
            results["const_fixed"][B]["grad_norm"].append(float(np.linalg.norm(g)))
            results["const_fixed"][B]["grad_proj"].append(float(np.dot(u, g)))

            # --- Learned baseline ---
            if include_learned:
                g_flat, mean_batch_return = REINFORCE_pg(pol, env_eval, stuff, n_episodes=B, baseline=learned_baseline_model, seed=seed_k)
                g = g_flat.numpy()
                results["learned"][B]["mean_return"].append(float(mean_batch_return))
                results["learned"][B]["grad_norm"].append(float(np.linalg.norm(g)))
                results["learned"][B]["grad_proj"].append(float(np.dot(u, g)))

            env_eval.close()

    # Print stats
    print("\n=== REINFORCE estimator variance comparison (fixed theta) ===")
    for est in estimator_names:
        print(f"\n--- Estimator: {est} ---")
        for B in batch_sizes:
            mr = np.array(results[est][B]["mean_return"], dtype=np.float64)
            gp = np.array(results[est][B]["grad_proj"], dtype=np.float64)

            print(
                f"B={B:>2d} | "
                f"E[mean return]≈{mr.mean():.3f}, Var(mean return)≈{mr.var(ddof=1):.6f} | "
                f"E[u^T g]≈{gp.mean():.3e}, Var(u^T g)≈{gp.var(ddof=1):.6e}"
            )

    print("No baseline, B=1:")
    print("Var grad_norm =", np.var(results["none"][1]["grad_norm"], ddof=1))
    print("Var grad_proj =", np.var(results["none"][1]["grad_proj"], ddof=1))

    fig, ax = plt.subplots(figsize=(11, 6))

    positions = np.array([0.0, 1.0, 2.0])
    handles = []

    for j, B in enumerate(batch_sizes):
        boxplot = ax.boxplot([results["none"][B]["grad_proj"], results["const_fixed"][B]["grad_proj"], results["learned"][B]["grad_proj"]],
                             positions=positions + (j - (len(batch_sizes) - 1) / 2) * 0.25,
                             widths=0.22,
                             patch_artist=True, showfliers=False, medianprops=dict(linewidth=2, color="black"), boxprops=dict(linewidth=1.5), whiskerprops=dict(linewidth=1.5), capprops=dict(linewidth=1.5))
        color = plt.rcParams["axes.prop_cycle"].by_key()["color"][j]
        for box in boxplot["boxes"]:
            box.set_facecolor(color)
            box.set_alpha(0.6)
        handles.append(boxplot["boxes"][0])

    ax.set_xticks(positions)
    ax.set_xticklabels(["No Baseline", "Constant Baseline", "Learned Baseline"], fontsize=20)
    ax.set_ylabel(r"$u^\top g$", fontsize=20)
    ax.set_title("Variability of Projected Gradient Estimates by Estimator and Batch Size", fontsize=24, pad=16)
    ax.tick_params(axis="y", labelsize=15)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(handles, [f"$B={B}$" for B in batch_sizes], fontsize=16, title="Batch size", title_fontsize=18)

    plt.tight_layout()
    plt.show()

    data = {"seed": seed, "env_name": env_name, "batch_sizes": tuple(batch_sizes), "n_repeats": n_repeats, "include_learned": include_learned, "u": u, "theta_ref": theta_ref, "results": results,}
    with open("estimator_variance_data", "wb") as f:
        pickle.dump(data, f)

    return results