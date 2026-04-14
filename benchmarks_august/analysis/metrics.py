"""
Diagnostics and evaluation for PDMP sampler outputs.

Usage:
    from sampler_eval import sample_quality, model_performance

    # After resampling:
    t, x = resample_pdmp_path(boom, n_samples=50000)
    sample_quality(x, sklearn_coefs=lr.coef_[0], label="Boomerang")

    # Performance on held-out data:
    model_performance(x, X_train, y_train, X_test, y_test,
                      labels=["Boomerang", "Sticky PLI"],
                      samples_list=[x_boom, x_pli])
"""
import numpy as np
import matplotlib.pyplot as plt
import warnings

# ── 0. ESS ────────────────────────────────────────────────
def _ess_batch_means(chain, n_batches=50):
    """Batch-means ESS, matching Bierkens et al. convention."""
    n = len(chain)
    batch_size = n // n_batches
    if batch_size < 2:
        return float(n)
    
    # Trim to exact multiple
    trimmed = chain[:n_batches * batch_size]
    batches = trimmed.reshape(n_batches, batch_size)
    batch_means = batches.mean(axis=1)
    
    var_total = np.var(trimmed, ddof=1)
    var_bm = np.var(batch_means, ddof=1)
    
    if var_bm < 1e-14:
        return float(n)
    
    # ESS = n * var_total / (batch_size * var_bm)
    return float(n * var_total / (batch_size * var_bm))

def _ess(samples):
    """ESS per coordinate, returns array of length D."""
    return np.array([_ess_batch_means(samples[:, i]) for i in range(samples.shape[1])])

def sticky_ess(samples, burnin_frac=0.1):
    """
    ESS that respects stickiness: 
    - Inclusion ESS: ESS of the binary indicator (is coordinate active?)
    - Active ESS: ESS of coordinate values conditional on being nonzero
    - Model size ESS: ESS of the number of active coordinates
    """
    n_burn = int(len(samples) * burnin_frac)
    post = samples[n_burn:]
    D = post.shape[1]
    
    # Inclusion indicators: gamma_i = 1 if beta_i != 0
    gamma = (np.abs(post) > 1e-10).astype(float)
    
    # ESS of inclusion indicators per coordinate
    inclusion_ess = np.array([_ess_batch_means(gamma[:, i]) for i in range(D)])
    
    # ESS of model size (total number of active coordinates)
    model_size = gamma.sum(axis=1)
    model_size_ess = _ess_batch_means(model_size)
    
    # ESS of active-only values per coordinate
    active_ess = np.zeros(D)
    for i in range(D):
        active_mask = np.abs(post[:, i]) > 1e-10
        if active_mask.sum() > 10:
            active_ess[i] = _ess_batch_means(post[active_mask, i])
        else:
            active_ess[i] = np.nan
    
    return {
        'inclusion_ess': inclusion_ess,
        'active_ess': active_ess, 
        'model_size_ess': model_size_ess,
        'mean_inclusion_ess': np.mean(inclusion_ess),
        'mean_active_ess': np.nanmean(active_ess),
    }


# ── 1. Sample quality ────────────────────────────────────────────────
def sample_quality(samples, sklearn_coefs=None, label="Sampler",
                   burnin_frac=0.1):
    """
    Trace plots, posterior violins, ESS bar chart — one figure.

    Parameters
    ----------
    samples      : (n_samples, D) array from resample_*_pdmp_path
    sklearn_coefs: (D,) MAP reference (optional, shown as diamonds)
    label        : title string
    burnin_frac  : fraction of samples to discard as burn-in
    """
    n_burn = int(len(samples) * burnin_frac)
    raw = samples            # keep full chain for trace
    post = samples[n_burn:]  # after burn-in for summaries
    D = samples.shape[1]
    ess = _ess(post)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5),
                             gridspec_kw={'width_ratios': [3, 3, 2]})

    # — Trace (first 3 coords) —
    ax = axes[0]
    for i in range(min(3, D)):
        ax.plot(raw[:, i], lw=0.3, alpha=0.7, label=f"β{i}")
        if sklearn_coefs is not None:
            ax.axhline(sklearn_coefs[i], ls="--", color=f"C{i}", lw=0.8, alpha=0.5)
    ax.axvline(n_burn, color="k", ls=":", lw=1, label="burn-in")
    ax.set_xlabel("sample index")
    ax.set_ylabel("value")
    ax.set_title("Trace (first 3 coords)")
    ax.legend(fontsize=7, ncol=2)

    # — Violin of posterior —
    ax = axes[1]
    coords = np.arange(D)
    parts = ax.violinplot([post[:, i] for i in range(D)],
                          positions=coords, showmeans=False,
                          showmedians=False, showextrema=False)
    for pc in parts['bodies']:
        pc.set_facecolor("steelblue"); pc.set_alpha(0.35)
    ax.scatter(coords, post.mean(axis=0), marker="o", s=70,
               c="orange", edgecolors="k", zorder=5, label="posterior mean")
    if sklearn_coefs is not None:
        ax.scatter(coords, sklearn_coefs, marker="D", s=50,
                   c="black", zorder=5, label="sklearn MAP")
    ax.axhline(0, color="grey", ls="--", lw=0.6)
    ax.set_xticks(coords)
    ax.set_xticklabels([f"β{i}" for i in range(D)])
    ax.set_title("Posterior (after burn-in)")
    ax.legend(fontsize=7)

    # — ESS bar —
    ax = axes[2]
    colors = ["#e74c3c" if e < 100 else "steelblue" for e in ess]
    ax.barh(coords, ess, color=colors, edgecolor="k", linewidth=0.4)
    ax.set_yticks(coords)
    ax.set_yticklabels([f"β{i}" for i in range(D)])
    ax.set_xlabel("ESS")
    ax.set_title(f"ESS  (min={ess.min():.0f})")
    ax.axvline(100, color="red", ls=":", lw=0.8, alpha=0.6)

    fig.suptitle(f"{label}  |  n={len(post)}  (burn-in {n_burn})", fontsize=13)
    plt.tight_layout()
    #return fig

def sampler_efficiency(samplers_dict):
    """
    Compact efficiency table from sampler diagnostics.
    
    Parameters
    ----------
    samplers_dict : dict {name: sampler_object}
    """
    header = (f"{'Sampler':<25s} {'Wall(s)':>8s} {'Grad evals':>12s} "
              f"{'Sim time':>10s} {'N_skel':>8s} {'Bounces':>8s} {'Refresh':>8s} {'Ref ratio':>9s}")
    print(header)
    print("─" * len(header))

    for name, sampler in samplers_dict.items():
        df = sampler.diagnostics_df
        wall = df['wall_seconds'].sum()
        grads = df['rate_evals'].sum()
        sim_time = sampler.Time[sampler.iteration - 1]
        n_skel = sampler.iteration
        n_bounce = df[df['event_type'] == 'bounce'].shape[0]
        n_refresh = df[df['event_type'] == 'refresh'].shape[0]
        n_total = n_bounce + n_refresh
        ratio = n_refresh / n_total if n_total > 0 else 0.0

        print(f"{name:<25s} {wall:>8.1f} {grads:>12d} "
              f"{sim_time:>10.0f} {n_skel:>8d} {n_bounce:>8d} {n_refresh:>8d} {ratio:>9.3f}")
        

# ── 2. Model performance ─────────────────────────────────────────────
def logreg_performance(X_train, y_train, X_test, y_test,
                       samples_dict, burnin_frac=0.1):
    """
    Logistic regression metrics for each sampler on held-out data.
    """
    from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
    from sklearn.decomposition import PCA

    def _predict_proba(X, beta):
        logits = X @ beta.T
        probs = 1.0 / (1.0 + np.exp(-logits))
        return probs.mean(axis=1)

    def _ece(y_true, y_prob, n_bins=10):
        bins = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (y_prob >= lo) & (y_prob < hi)
            if mask.sum() == 0:
                continue
            ece += mask.sum() * abs(y_true[mask].mean() - y_prob[mask].mean())
        return ece / len(y_true)

    header = (f"{'Sampler':<25s} {'Acc(tr)':>8s} {'Acc(te)':>8s} "
              f"{'NLL':>8s} {'Brier':>8s} {'ECE':>8s} {'ESS_PC1':>8s} {'Sparsity':>9s}")
    print(header)
    print("─" * len(header))

    for name, samples in samples_dict.items():
        n_burn = int(len(samples) * burnin_frac)
        post = samples[n_burn:]

        p_train = _predict_proba(X_train, post)
        p_test = _predict_proba(X_test, post)

        acc_train = accuracy_score(y_train, (p_train > 0.5).astype(int))
        acc_test = accuracy_score(y_test, (p_test > 0.5).astype(int))
        nll = log_loss(y_test, np.clip(p_test, 1e-7, 1 - 1e-7))
        brier = brier_score_loss(y_test, p_test)
        ece = _ece(y_test, p_test)

        # ESS on first principal component
        if len(post) > 1:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pca = PCA(n_components=1).fit(post)
            pc1 = post @ pca.components_[0]
            ess_pc1 = _ess_batch_means(pc1)
        else:
            ess_pc1 = 1.0

        # Sparsity: fraction of coordinates at exactly zero
        sparsity = np.mean(np.abs(post) < 1e-10)
        if sparsity < 0.001:
            sp_str = f"{'—':>9s}"
        else:
            sp_str = f"{sparsity:>8.1%}"

        print(f"{name:<25s} {acc_train:>8.3f} {acc_test:>8.3f} "
              f"{nll:>8.4f} {brier:>8.4f} {ece:>8.4f} {ess_pc1:>8.0f} {sp_str}")      
        
        
# ── 3. BNN performance ───────────────────────────────────────────────
def bnn_performance(X_train, y_train, X_test, y_test,
                    samples_dict, shapes, slices, weight_mask=None,
                    burnin_frac=0.1, n_posterior=500):
    """
    BNN classification metrics on held-out data.

    Parameters
    ----------
    samples_dict : dict {name: (n_samples, D) array}
    shapes, slices : architecture info from target.meta
    weight_mask : bool array, for sparsity reporting
    n_posterior : number of posterior samples to use for predictions
    """
    from sklearn.metrics import accuracy_score, log_loss
    from sklearn.decomposition import PCA

    def _forward_np(theta, X):
        h = X
        for l in range(len(shapes) - 1):
            w_sl, b_sl = slices[l]
            W = theta[w_sl].reshape(shapes[l][0])
            b = theta[b_sl]
            h = np.tanh(h @ W + b)
        w_sl, b_sl = slices[-1]
        W = theta[w_sl].reshape(shapes[-1][0])
        b = theta[b_sl]
        return (h @ W + b).ravel()

    def _sigmoid(x):
        return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)),
                        np.exp(x) / (1.0 + np.exp(x)))

    def _predict_bnn(X, post_samples):
        idx = np.linspace(0, len(post_samples) - 1, n_posterior, dtype=int)
        all_probs = np.array([_sigmoid(_forward_np(post_samples[i], X)) for i in idx])
        return all_probs.mean(axis=0), all_probs.std(axis=0)

    def _ece(y_true, y_prob, n_bins=10):
        bins = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (y_prob >= lo) & (y_prob < hi)
            if mask.sum() == 0:
                continue
            acc_bin = y_true[mask].mean()
            conf_bin = y_prob[mask].mean()
            ece += mask.sum() * abs(acc_bin - conf_bin)
        return ece / len(y_true)

    header = (f"{'Sampler':<25s} {'Acc(tr)':>8s} {'Acc(te)':>8s} "
              f"{'NLL':>8s} {'ECE':>8s} {'ESS_PC1':>8s} {'Sparsity':>9s}")
    print(header)
    print("─" * len(header))

    for name, samples in samples_dict.items():
        n_burn = int(len(samples) * burnin_frac)
        post = samples[n_burn:]

        p_train, _ = _predict_bnn(X_train, post)
        p_test, p_std = _predict_bnn(X_test, post)

        acc_train = accuracy_score(y_train, (p_train > 0.5).astype(int))
        acc_test = accuracy_score(y_test, (p_test > 0.5).astype(int))
        nll = log_loss(y_test, np.clip(p_test, 1e-7, 1 - 1e-7))
        ece = _ece(y_test, p_test)

        # ESS on first principal component
        if len(post) > 1:
            pca = PCA(n_components=1).fit(post)
            pc1 = post @ pca.components_[0]
            ess_pc1 = _ess_batch_means(pc1)
        else:
            ess_pc1 = 1.0

        # Sparsity
        if weight_mask is not None:
            sparsity = np.mean([
                np.mean(np.abs(post[i][weight_mask]) < 1e-10)
                for i in np.linspace(0, len(post)-1, n_posterior, dtype=int)
            ])
            sp_str = f"{sparsity:>8.1%}"
        else:
            sp_str = f"{'n/a':>9s}"

        print(f"{name:<25s} {acc_train:>8.3f} {acc_test:>8.3f} "
              f"{nll:>8.4f} {ece:>8.4f} {ess_pc1:>8.0f} {sp_str}")
        
def bnn_trace(samples, label="BNN", burnin_frac=0.1):
    """Quick trace plot for high-dimensional BNN samples."""
    from sklearn.decomposition import PCA
    n_burn = int(len(samples) * burnin_frac)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pca = PCA(n_components=3).fit(samples[n_burn:])
    
    proj = samples @ pca.components_.T  # (n_samples, 3)
    
    fig, ax = plt.subplots(figsize=(10, 3))
    for i in range(3):
        ax.plot(proj[:, i], lw=0.3, alpha=0.7, label=f"PC{i+1}")
    ax.axvline(n_burn, color="k", ls=":", lw=1, label="burn-in")
    ax.legend(fontsize=8)
    ax.set_xlabel("sample index")
    ax.set_title(f"{label}  |  trace of first 3 principal components")
    plt.tight_layout()
    #return fig

def bnn_regression_performance(X_train, y_train, X_test, y_test, y_std,
                               samples_dict, shapes, slices, tau=1.0,
                               weight_mask=None, burnin_frac=0.1,
                               n_posterior=500):
    """
    BNN regression metrics on held-out data.

    Parameters
    ----------
    samples_dict : dict {name: (n_samples, D) array}
    shapes, slices : architecture info from target.meta
    tau : noise precision (1/sigma_noise^2)
    weight_mask : bool array, for sparsity reporting
    n_posterior : number of posterior samples to use for predictions
    """
    from sklearn.decomposition import PCA

    def _forward_np(theta, X):
        h = X
        for l in range(len(shapes) - 1):
            w_sl, b_sl = slices[l]
            W = theta[w_sl].reshape(shapes[l][0])
            b = theta[b_sl]
            h = np.tanh(h @ W + b)
        w_sl, b_sl = slices[-1]
        W = theta[w_sl].reshape(shapes[-1][0])
        b = theta[b_sl]
        return (h @ W + b).ravel()

    def _predict_bnn(X, post_samples):
        idx = np.linspace(0, len(post_samples) - 1, n_posterior, dtype=int)
        all_preds = np.array([_forward_np(post_samples[i], X) for i in idx])
        mean_pred = all_preds.mean(axis=0)
        var_pred = all_preds.var(axis=0) + 1.0 / tau   # epistemic + aleatoric
        return mean_pred, var_pred

    def _gaussian_nll(y, mean, var):
        return np.mean(0.5 * np.log(2 * np.pi * var)
                       + 0.5 * (y - mean) ** 2 / var)

    header = (f"{'Sampler':<25s} {'RMSE(tr)':>9s} "
              f"{'RMSE(te)':>9s} {'NLL(te)':>9s} {'ESS_PC1':>8s} {'Sparsity':>9s}")
    print(header)
    print("─" * len(header))

    for name, samples in samples_dict.items():
        n_burn = int(len(samples) * burnin_frac)
        post = samples[n_burn:]

        mean_tr, var_tr = _predict_bnn(X_train, post)
        mean_te, var_te = _predict_bnn(X_test, post)

        mse_train = np.mean((mean_tr - y_train) ** 2)
        mse_test  = np.mean((mean_te - y_test) ** 2)
        rmse_train = np.sqrt(mse_train)
        rmse_test = np.sqrt(mse_test)
        nll_test  = _gaussian_nll(y_test, mean_te, var_te)

        # ESS on first principal component
        if len(post) > 1:
            pca = PCA(n_components=1).fit(post)
            pc1 = post @ pca.components_[0]
            ess_pc1 = _ess_batch_means(pc1)
        else:
            ess_pc1 = 1.0

        # Sparsity
        if weight_mask is not None:
            sparsity = np.mean([
                np.mean(np.abs(post[i][weight_mask]) < 1e-10)
                for i in np.linspace(0, len(post)-1, n_posterior, dtype=int)
            ])
            sp_str = f"{sparsity:>8.1%}"
        else:
            sp_str = f"{'n/a':>9s}"

        print(f"{name:<25s} {rmse_train*y_std:>9.4f} "
              f"{rmse_test*y_std:>9.4f} {nll_test:>9.4f} {ess_pc1:>8.0f} {sp_str}")
        

# ── 4. refreshment rate ───────────────────────────────────────────────
def refresh_diagnostic(sampler):
    """
    Check refresh-to-event ratio against the 78.12% rule
    (Bierkens, Kamatani & Roberts 2022, eq. 2.8).
    
    Prints current ratio and suggests adjustment.
    """
    df = sampler.diagnostics_df
    n_bounce = df[df['event_type'] == 'bounce'].shape[0]
    n_refresh = df[df['event_type'] == 'refresh'].shape[0]
    n_total = n_bounce + n_refresh
    
    if n_total == 0:
        print("No events recorded.")
        return
    
    ratio = n_refresh / n_total
    wall_sec = df['wall_seconds'].sum()
    sim_time = sampler.Time[sampler.iteration - 1]
    
    print(f"Bounces: {n_bounce}, Refreshes: {n_refresh}")
    print(f"Refresh ratio: {ratio:.3f} (target: ~0.78)")
    print(f"Current refresh_rate: {sampler.refresh_rate}")
    
    if ratio < 0.7:
        suggested = sampler.refresh_rate * 0.78 / max(ratio, 0.01)
        print(f"→ Under-refreshing. Try refresh_rate ≈ {suggested:.2f}")
    elif ratio > 0.85:
        suggested = sampler.refresh_rate * 0.78 / ratio
        print(f"→ Over-refreshing. Try refresh_rate ≈ {suggested:.2f}")
    else:
        print(f"→ Looks good.")
    
        