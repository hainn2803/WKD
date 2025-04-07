import torch 
from ot import generate_uniform_unit_sphere_projections, Wasserstein_One_Dimension



def SOT_GMs(mu1s, Sigma1s, mu2s, Sigma2s, num_projections=10000, p=2):

    chunk = 1000

    if num_projections < chunk:
        chunk = num_projections
        chunk_num_projections = 1
    else:
        chunk_num_projections = num_projections // chunk

    sum_w_p = 0

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

        w_1d = Wasserstein_One_Dimension(X=X_projection, Y=Y_projection, p=p) # (batch_size, chunk)

        sw_chunk += torch.sum(torch.pow(w_1d, p))
    
    return torch.pow(sw_chunk / num_projections, 1/p)



mu1s = torch.randn(64, 256)
Sigma1s = torch.randn(64, 256)

mu2s = torch.randn(64, 256)
Sigma2s = torch.randn(64, 256)

SOT_GMs(mu1s, Sigma1s, mu2s, Sigma2s, num_projections=10000, p=2)