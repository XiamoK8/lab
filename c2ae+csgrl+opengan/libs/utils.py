import os

import numpy as np
import scipy.stats as stats
import torch


def fit_gpd(data, tail_fraction=0.1):
    threshold = np.quantile(data, 1 - tail_fraction)
    exceedances = data[data > threshold] - threshold

    if len(exceedances) < 10:
        return threshold, 0, np.mean(exceedances) if len(exceedances) > 0 else 1.0

    mean_excess = np.mean(exceedances)
    var_excess = np.var(exceedances)

    if var_excess < 1e-6:
        return threshold, 0, mean_excess

    shape = 0.5 * (mean_excess**2 / var_excess - 1)
    scale = 0.5 * mean_excess * (mean_excess**2 / var_excess + 1)

    return threshold, shape, max(scale, 1e-6)


def calculate_threshold(S_m, S_nm, p_u):
    u_m, zeta_m, mu_m = fit_gpd(S_m, tail_fraction=0.1)

    S_nm_inv = -S_nm
    u_nm, zeta_nm, mu_nm = fit_gpd(S_nm_inv, tail_fraction=0.1)
    u_nm = -u_nm

    tau_min = max(S_m.min(), S_nm.min())
    tau_max = min(S_m.max(), S_nm.max())
    tau_range = np.linspace(tau_min, tau_max, 1000)

    min_error = float("inf")
    optimal_tau = (tau_min + tau_max) / 2

    for tau in tau_range:
        if tau > u_m:
            p_tail = 1 - stats.genpareto.cdf(tau - u_m, zeta_m, scale=mu_m)
            n_exceed = np.sum(S_m > u_m)
            p_m = (n_exceed / len(S_m)) * p_tail
        else:
            p_m = np.mean(S_m > tau)

        if tau < u_nm:
            p_tail = 1 - stats.genpareto.cdf(-(tau - u_nm), zeta_nm, scale=mu_nm)
            n_below = np.sum(S_nm < u_nm)
            p_nm = (n_below / len(S_nm)) * p_tail
        else:
            p_nm = np.mean(S_nm < tau)

        error_prob = (1 - p_u) * p_m + p_u * p_nm

        if error_prob < min_error:
            min_error = error_prob
            optimal_tau = tau

    return optimal_tau


def compute_reconstruction_errors(model, loader, args):
    model.eval()
    match_errors = []
    nonmatch_errors = []

    with torch.no_grad():
        for images, labels in loader:
            if args.cuda:
                images, labels = images.cuda(), labels.cuda()

            z = model.encoder(images)

            condition_match = model.create_condition_vector(labels)
            z_match = model.film(z, condition_match)
            x_recon_match = model.decoder(z_match)
            x_recon_match = torch.clamp(x_recon_match, 0, 1)
            error_match = torch.mean(torch.abs(images - x_recon_match), dim=[1, 2, 3])
            match_errors.extend(error_match.cpu().numpy())

            batch_size = images.size(0)
            errs = []
            for k in range(model.num_classes):
                labels_k = torch.full((batch_size,), k, dtype=torch.long, device=labels.device)
                condition_k = model.create_condition_vector(labels_k)
                z_k = model.film(z, condition_k)
                x_recon_k = model.decoder(z_k)
                x_recon_k = torch.clamp(x_recon_k, 0, 1)
                err_k = torch.mean(torch.abs(images - x_recon_k), dim=[1, 2, 3])
                errs.append(err_k)
            errs = torch.stack(errs, dim=1)
            mask_true = torch.zeros_like(errs, dtype=torch.bool)
            mask_true[torch.arange(batch_size), labels] = True
            big = torch.finfo(errs.dtype).max
            errs_masked = errs.masked_fill(mask_true, big)
            err_nonmatch_min, _ = torch.min(errs_masked, dim=1)
            nonmatch_errors.extend(err_nonmatch_min.cpu().numpy())

    return np.array(match_errors), np.array(nonmatch_errors)


def plot_error_histograms(S_m, S_nm, epoch, output_dir="outputs", bins=30):
    try:
        import matplotlib.pyplot as plt

        os.makedirs(output_dir, exist_ok=True)
        fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.5))

        vmin = float(min(np.min(S_m), np.min(S_nm)))
        vmax = float(max(np.max(S_m), np.max(S_nm)))
        bin_edges = np.linspace(vmin, vmax, bins + 1)

        ax.hist(S_nm, bins=bin_edges, density=True, color="#4263eb", alpha=0.85, label="Non Match")
        ax.hist(S_m, bins=bin_edges, density=True, color="#ffd43b", alpha=0.85, label="Match")

        ax.set_xlabel("Reconstruction Error")
        ax.set_ylabel("Normalized Histogram")
        ax.legend(frameon=True)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.set_title(f"Stage2 Epoch {epoch}")

        save_path = os.path.join(output_dir, f"stage2_hist_epoch_{epoch}.png")
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"[Stage2] Saved S_m/S_nm histograms to {save_path}")
        return save_path
    except Exception as e:
        print(f"[Stage2] Matplotlib not available or failed ({e}). Printing ASCII histograms.")

    def _ascii_histogram(data, bins=30, width=40):
        counts, edges = np.histogram(data, bins=bins)
        max_count = counts.max() if counts.size and counts.max() > 0 else 1
        lines = []
        for i, c in enumerate(counts):
            bar_len = int(c / max_count * width)
            left = edges[i]
            right = edges[i + 1]
            lines.append(f"[{left:8.3f}, {right:8.3f}): " + ("*" * bar_len))
        return "\n".join(lines)

    print(f"\n[Stage2] Epoch {epoch} S_m histogram (ASCII)")
    print(_ascii_histogram(S_m, bins=bins))
    print(f"\n[Stage2] Epoch {epoch} S_nm histogram (ASCII)")
    print(_ascii_histogram(S_nm, bins=bins))
    return None


def plot_known_unknown_histograms(
    known_errors,
    unknown_errors,
    output_path="outputs/stage3_known_unknown_hist.png",
    bins=30,
):
    known_errors = np.asarray(known_errors)
    unknown_errors = np.asarray(unknown_errors)

    try:
        import matplotlib.pyplot as plt

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.5))

        vmin = float(min(np.min(known_errors), np.min(unknown_errors)))
        vmax = float(max(np.max(known_errors), np.max(unknown_errors)))
        bin_edges = np.linspace(vmin, vmax, bins + 1)

        ax.hist(known_errors, bins=bin_edges, density=True, color="#69db7c", alpha=0.85, label="Known")
        ax.hist(unknown_errors, bins=bin_edges, density=True, color="#ff6b6b", alpha=0.85, label="Unknown")

        ax.set_xlabel("Reconstruction Error")
        ax.set_ylabel("Normalized Histogram")
        ax.legend(frameon=True)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.set_title("Stage3 Known vs Unknown")

        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        print(f"[Stage3] Saved Known/Unknown histograms to {output_path}")
        return output_path
    except Exception as e:
        print(f"[Stage3] Matplotlib not available or failed ({e}). Printing ASCII histograms.")

    def _ascii_histogram(data, bins=30, width=40):
        counts, edges = np.histogram(data, bins=bins)
        max_count = counts.max() if counts.size and counts.max() > 0 else 1
        lines = []
        for i, c in enumerate(counts):
            bar_len = int(c / max_count * width)
            left = edges[i]
            right = edges[i + 1]
            lines.append(f"[{left:8.3f}, {right:8.3f}): " + ("*" * bar_len))
        return "\n".join(lines)

    print("\n[Stage3] Known histogram (ASCII)")
    print(_ascii_histogram(known_errors, bins=bins))
    print("\n[Stage3] Unknown histogram (ASCII)")
    print(_ascii_histogram(unknown_errors, bins=bins))
    return None