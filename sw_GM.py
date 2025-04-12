import torch 
from ot import generate_uniform_unit_sphere_projections, Wasserstein_One_Dimension
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import matplotlib.cm as cm



def inter_batch_loss_gaussian(mu1s, Sigma1s, mu2s, Sigma2s, num_projections=10000, p=2):
    """
        Compute Sliced Wasserstein between two Mixture of Gaussians
        mu1s: R^(batch_size, dims)
        Sigma1s: R^(batch_size, dims)
        mu2s: R^(batch_size, dims)
        Sigma2s: R^(batch_size, dims)
    """
    chunk = 1000
    if num_projections < chunk:
        chunk = num_projections
        chunk_num_projections = 1
    else:
        chunk_num_projections = num_projections // chunk

    sw_chunk = 0
    d = mu1s.shape[1]
    for i in range(chunk_num_projections):
        theta = generate_uniform_unit_sphere_projections(dim=d, num_projection=chunk, dtype=torch.float32, device=mu1s.device)
        prod_mu1s = torch.matmul(mu1s, theta.transpose(0, 1))
        prod_mu2s = torch.matmul(mu2s, theta.transpose(0, 1))
        prod_Sigma1s = torch.sqrt(torch.matmul(Sigma1s, (theta**2).transpose(0, 1)))
        prod_Sigma2s = torch.sqrt(torch.matmul(Sigma2s, (theta**2).transpose(0, 1)))
        X = torch.stack([prod_mu1s, torch.log(prod_Sigma1s)], dim=-1)
        Y = torch.stack([prod_mu2s, torch.log(prod_Sigma2s)], dim=-1)
        psi = generate_uniform_unit_sphere_projections(dim=2, num_projection=chunk, dtype=torch.float32, device=mu1s.device)
        X_projection = torch.sum(X * psi.unsqueeze(0), dim=-1) # (batch_size, chunk)
        Y_projection = torch.sum(Y * psi.unsqueeze(0), dim=-1) # (batch_size, chunk)
        w_1d = Wasserstein_One_Dimension(X=X_projection, Y=Y_projection, p=p) # (chunk)
        sw_chunk += torch.sum(torch.pow(w_1d, p))
    return torch.pow(sw_chunk / num_projections, 1/p)



def inter_batch_loss_multilevel_gaussian(mu1s, Sigma1s, mu2s, Sigma2s, num_projections=10000, num_inner_projections=1000, p=2):
    """
        Compute Sliced Wasserstein between two Mixture of Gaussians
        mu1s: R^(batch_size, dims, num_gaussians)
        Sigma1s: R^(batch_size, dims, num_gaussians)
        mu2s: R^(batch_size, dims, num_gaussians)
        Sigma2s: R^(batch_size, dims, num_gaussians)
    """
    chunk = 1000
    if num_projections < chunk:
        chunk = num_projections
        chunk_num_projections = 1
    else:
        chunk_num_projections = num_projections // chunk

    sw_chunk = 0
    batch_size = mu1s.shape[0]
    d = mu1s.shape[1]
    num_gaussians = mu1s.shape[2]

    mu1s = mu1s.permute(0, 2, 1) # [batch_size, num_gaussians, dims]
    mu2s = mu2s.permute(0, 2, 1) # [batch_size, num_gaussians, dims]
    Sigma1s = Sigma1s.permute(0, 2, 1) # [batch_size, num_gaussians, dims]
    Sigma2s = Sigma2s.permute(0, 2, 1) # [batch_size, num_gaussians, dims]

    for i in range(chunk_num_projections):
        theta = generate_uniform_unit_sphere_projections(dim=d, num_projection=chunk, dtype=torch.float32, device=mu1s.device) # [chunk, dims]
        prod_mu1s = torch.matmul(mu1s, theta.transpose(0, 1)) # [batch_size, num_gaussians, chunk]
        prod_mu2s = torch.matmul(mu2s, theta.transpose(0, 1)) # [batch_size, num_gaussians, chunk]
        prod_Sigma1s = torch.sqrt(torch.matmul(Sigma1s, (theta**2).transpose(0, 1))) # [batch_size, num_gaussians, chunk]
        prod_Sigma2s = torch.sqrt(torch.matmul(Sigma2s, (theta**2).transpose(0, 1))) # [batch_size, num_gaussians, chunk]

        del theta 

        X = torch.stack([prod_mu1s, torch.log(prod_Sigma1s)], dim=-1) # [batch_size, num_gaussians, chunk, 2]
        Y = torch.stack([prod_mu2s, torch.log(prod_Sigma2s)], dim=-1) # [batch_size, num_gaussians, chunk, 2]
        psi = generate_uniform_unit_sphere_projections(dim=2, num_projection=chunk, dtype=torch.float32, device=mu1s.device)
        X_projection = torch.sum(X * psi.unsqueeze(0), dim=-1) # (batch_size, num_gaussians, chunk)
        Y_projection = torch.sum(Y * psi.unsqueeze(0), dim=-1) # (batch_size, num_gaussians, chunk)
        X_projection = X_projection.permute(2, 0, 1) # (chunk, batch_size, num_gaussians)
        Y_projection = Y_projection.permute(2, 0, 1) # (chunk, batch_size, num_gaussians)

        psi = generate_uniform_unit_sphere_projections(dim=num_gaussians, num_projection=num_inner_projections, dtype=torch.float32, device=mu1s.device) # [num_inner_projections, num_gaussians]
        prod_X_projection = torch.matmul(X_projection, psi.transpose(0, 1)) # [chunk, batch_size, num_inner_projections]
        prod_Y_projection = torch.matmul(Y_projection, psi.transpose(0, 1)) # [chunk, batch_size, num_inner_projections]

        del psi

        X_sorted, _ = torch.sort(prod_X_projection, dim=1) # [chunk, batch_size, num_inner_projections]
        Y_sorted, _ = torch.sort(prod_Y_projection, dim=1) # [chunk, batch_size, num_inner_projections]
        diff_quantiles = torch.abs(X_sorted - Y_sorted) # [chunk, batch_size, num_inner_projections]

        w_1d = torch.pow(torch.sum(torch.pow(diff_quantiles, p), dim=1) / batch_size, 1/p)  # [chunk, num_inner_projections]
        w_1d = torch.pow(torch.sum(torch.pow(w_1d, p), dim=1) / num_inner_projections, 1/p) # [chunk]

        sw_chunk += torch.sum(torch.pow(w_1d, p))
    return torch.pow(sw_chunk / num_projections, 1/p)



if __name__ == "__main__":

    num_mogs = 5
    num_gaussians = 4
    dims = 2
    # Generate group: mỗi MoG có center riêng, các Gaussian quanh center
    def generate_group(mean_shift):
        mog_centers = torch.randn((num_mogs, dims)) + torch.tensor(mean_shift)
        mu = mog_centers[:, None, :] + 0.3 * torch.randn((num_mogs, num_gaussians, dims))  # Gaussian gần nhau
        sigma = torch.abs(torch.randn((num_mogs, num_gaussians, dims))) * 0.2 + 0.2
        return mu, sigma

    mu1, sigma1 = generate_group([0.0, 0.0])
    mu2, sigma2 = generate_group([8.0, 8.0])
    mu3, sigma3 = generate_group([-4.0, 4.0])

    # Plotting helper
    def plot_ellipse(ax, mean, sigma, color, alpha=0.4):
        ellipse = Ellipse(xy=mean, width=2*sigma[0], height=2*sigma[1],
                        edgecolor=color, facecolor=color, alpha=alpha)
        ax.add_patch(ellipse)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    cmap1 = cm.get_cmap('Blues', num_mogs + 2)
    cmap2 = cm.get_cmap('Reds', num_mogs + 2)
    cmap3 = cm.get_cmap('Greens', num_mogs + 2)

    for group_idx, (mu_group, sigma_group, cmap, group_label) in enumerate([
        (mu1, sigma1, cmap1, "Group 1"),
        (mu2, sigma2, cmap2, "Group 2"),
        (mu3, sigma3, cmap3, "Group 3")
    ]):
        for mog_idx in range(num_mogs):
            color = cmap(mog_idx + 2)
            for g_idx in range(num_gaussians):
                mu = mu_group[mog_idx, g_idx]
                sigma = sigma_group[mog_idx, g_idx]
                ax.scatter(mu[0], mu[1], color=color, label=f"{group_label} - MoG {mog_idx+1}" if g_idx == 0 else "", s=40)
                plot_ellipse(ax, mu, sigma, color)

    ax.set_title("Groups with Mixtures of Gaussians (Clustered Components)")
    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.grid(True)
    ax.legend(loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.savefig("GM.png")

    sw_12 = inter_batch_loss_multilevel_gaussian(mu1s=mu1, Sigma1s=sigma1, mu2s=mu2, Sigma2s=sigma2, num_projections=10000, num_inner_projections=1000, p=2)
    sw_23 = inter_batch_loss_multilevel_gaussian(mu1s=mu2, Sigma1s=sigma2, mu2s=mu3, Sigma2s=sigma3, num_projections=10000, num_inner_projections=1000, p=2)
    sw_13 = inter_batch_loss_multilevel_gaussian(mu1s=mu1, Sigma1s=sigma1, mu2s=mu3, Sigma2s=sigma3, num_projections=10000, num_inner_projections=1000, p=2)

    print(f"1-2: {sw_12}")
    print(f"2-3: {sw_23}")
    print(f"3-1: {sw_13}")