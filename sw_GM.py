import torch 
from ot import generate_uniform_unit_sphere_projections, Wasserstein_One_Dimension



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

        X_sorted, _ = torch.sort(prod_X_projection, dim=1) # [chunk, batch_size, num_inner_projections]
        Y_sorted, _ = torch.sort(prod_Y_projection, dim=1) # [chunk, batch_size, num_inner_projections]
        diff_quantiles = torch.abs(X_sorted - Y_sorted) # [chunk, batch_size, num_inner_projections]

        w_1d = torch.pow(torch.sum(torch.pow(diff_quantiles, p), dim=1) / batch_size, 1/p)  # [chunk, num_inner_projections]
        w_1d = torch.pow(torch.sum(torch.pow(w_1d, p), dim=1) / num_inner_projections, 1/p) # [chunk]

        sw_chunk += torch.sum(torch.pow(w_1d, p))
    return torch.pow(sw_chunk / num_projections, 1/p)


if __name__ == "__main__":

    batch_size = 16
    dims = 8
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Generate random means and (diagonal) covariances for two batches of Gaussians
    mu1s = torch.randn((batch_size, dims), device=device)
    mu2s = torch.randn((batch_size, dims), device=device)

    # Ensure positive variances (diagonal covariances)
    Sigma1s = torch.abs(torch.randn((batch_size, dims), device=device)) + 1e-3
    Sigma2s = torch.abs(torch.randn((batch_size, dims), device=device)) + 1e-3

    sw_dist = inter_batch_loss_gaussian(mu1s, Sigma1s, mu2s, Sigma2s, num_projections=1000, p=2)
    print("Sliced Wasserstein distance:", sw_dist.item())


    batch_size = 8
    dims = 16
    num_gaussians = 5
    num_projections = 500
    num_inner_projections = 50
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Generate random multilevel means and diagonal covariances
    mu1s = torch.randn((batch_size, dims, num_gaussians), device=device)
    mu2s = torch.randn((batch_size, dims, num_gaussians), device=device)

    # Positive diagonal variances
    Sigma1s = torch.abs(torch.randn((batch_size, dims, num_gaussians), device=device)) + 1e-3
    Sigma2s = torch.abs(torch.randn((batch_size, dims, num_gaussians), device=device)) + 1e-3

    # Call your function
    sw_multi = inter_batch_loss_multilevel_gaussian(
        mu1s, Sigma1s, mu2s, Sigma2s,
        num_projections=num_projections,
        num_inner_projections=num_inner_projections,
        p=2
    )

    print("Sliced Wasserstein distance (multi-level):", sw_multi.item())
