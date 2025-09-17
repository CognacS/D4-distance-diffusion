from typing import Optional
import torch


def compute_gramm_matrix(edge_dist: torch.Tensor, edge_mask: Optional[torch.Tensor]=None) -> torch.Tensor:
    """
    Compute the Gram matrix from the edge distances.
    :param edge_dist: Tensor of shape (n, n) containing pairwise distances.
    :param edge_mask: Optional tensor of shape (n, n) indicating which edges to consider.
                      If provided, only the distances where edge_mask is True will be used.
    :return: Gram matrix of shape (n, n).
    """
    d_squared = edge_dist ** 2
    n = d_squared.shape[-1]
    if edge_mask is not None:
        true_n = edge_mask[..., :1].sum(dim=-2, keepdim=True)
    else:
        true_n = n
    H = torch.eye(n).unsqueeze(0).to(edge_dist.device) - torch.ones(n, n).unsqueeze(0).to(edge_dist.device) / true_n
    if d_squared.ndim <= 2:
        H = H.squeeze(0)
    #H = H.masked_fill_(~edge_mask, 0) if edge_mask is not None else H
    B = -0.5 * H @ d_squared @ H
    B.masked_fill_(~edge_mask, 0) if edge_mask is not None else B
    return B


def get_ignored_eigenvalues(edge_dist: torch.Tensor, n_components: int = 3, edge_mask: Optional[torch.Tensor]=None) -> torch.Tensor:
    """
    Get the eigenvalues that are ignored in MDS.
    :param edge_dist: Tensor of shape (n, n) containing pairwise distances.
    :param n_components: Number of dimensions to reduce to.
    :return: Tensor of ignored eigenvalues.
    """
    if edge_mask is not None:
        edge_dist = edge_dist * edge_mask
    B = compute_gramm_matrix(edge_dist, edge_mask)
    eigvals = torch.linalg.eigvalsh(B.to(torch.float64))
    
    # Sort eigenvalues in descending order and ignore the last n_components
    return eigvals[..., :-n_components]


def mds(edge_dist: torch.Tensor, n_components: int = 3, edge_mask: Optional[torch.Tensor]=None) -> torch.Tensor:
    """
    Perform Multidimensional Scaling (MDS) to recover positions from edge distances.
    :param edge_dist: Tensor of shape (n, n) containing pairwise distances.
    :param n_components: Number of dimensions to reduce to.
    :return: Tensor of shape (n, n_components) containing the positions.
    """
    if edge_mask is not None:
        edge_dist = edge_dist * edge_mask
    B = compute_gramm_matrix(edge_dist, edge_mask)
    # with eigh, eigenvalues are already in ascending order
    eigvals, eigvecs = torch.linalg.eigh(B.to(torch.float64))
    
    eigvals_nd = eigvals.flip(dims=(-1,))[..., :n_components]  # Sort eigenvalues in descending order
    eigvecs_nd = eigvecs.flip(dims=(-1,))[..., :n_components]  # Corresponding eigenvectors
    
    ret = eigvecs_nd * torch.sqrt(eigvals_nd).unsqueeze(-2)
    
    return ret.to(edge_dist.dtype)  # Scale by the square root of eigenvalues


def compute_affine(
        A: torch.Tensor,
        B: torch.Tensor,
        mask: torch.Tensor,
        eps: float = 1e-6
    ):
    """
    Closed-form linear map M[b] = (A^T A + eps I)^(-1) A^T B,
    ignoring padded rows (mask==0), all batched with @.
    
    Args:
        A, B : (B, N, p) embeddings
        mask : (B, N) with 1 for real points, 0 for padding
        eps  : float, ridge for numerical stability
        
    Returns:
        M : (B, p, p) projection matrices
    """
    Bbatch, N, p = A.shape
    
    #A = A - A.mean(dim=1, keepdim=True)  # center A
    
    # add 1 padding
    A = torch.cat([A, torch.ones(Bbatch, N, 1, device=A.device)], dim=-1)  # (B, N, p+1)
    B = torch.cat([B, torch.ones(Bbatch, N, 1, device=B.device)], dim=-1)  # (B, N, p+1)
    p = p + 1
    
    # 1) zero out padded rows
    A_mask = A * mask.unsqueeze(-1)    # (B, N, p)
    B_mask = B * mask.unsqueeze(-1)    # (B, N, p)
    
    # 2) form AtA and AtB via batched @
    A_t = A_mask.transpose(1, 2)       # (B, p, N)
    AtA = A_t @ A_mask                 # (B, p, p)
    AtB = A_t @ B_mask                 # (B, p, p)
    
    # 3) invert (AtA + eps I) and multiply
    I = torch.eye(p, device=A.device).unsqueeze(0)  # (1, p, p)
    AtA_reg = AtA + eps * I
    AtA_inv = torch.linalg.inv(AtA_reg)             # (B, p, p)
    M = AtA_inv @ AtB                               # (B, p, p)
    M = M[..., :-1]
    
    # translation
    T = M[..., -1:, :]  # (B, 1, p)
    M = M[..., :-1, :]  # (B, p, p)
    
    return M, T

def transform_vectors(A, projection, translation, mask):
    return (A @ projection + translation) * mask.unsqueeze(-1)



import torch
import torch.nn as nn
import torch.optim as optim


class ExternalMDS(nn.Module):
    """
    Perform metric MDS on distance matrices with missing entries.

    Args:
        n_points: int, maximum number of points (padded dimension n).
        dim: int, target embedding dimension.
    """
    def __init__(self, batch_size, n_points: int, dim: int):
        super().__init__()
        # Initialize embeddings for all batches; will be broadcasted per batch
        # Shape: (1, n_points, dim)
        self.embeddings = nn.Parameter(torch.randn(batch_size, n_points, dim))

    def forward(
        self,
        edge_dist: torch.Tensor,
        edge_mask: torch.Tensor,
        ext_X: torch.Tensor = None,
        ext_edge_dist: torch.Tensor = None,
        ext_edge_mask: torch.Tensor = None
    ):
        """
        Compute stress given batched distances and mask.

        edge_dist: (b, n, n) tensor of observed distances (zeros or arbitrary at missing positions)
        edge_mask: (b, n, n) binary mask: 1 indicates valid distance, 0 missing/padding
        ext_edge_dist: (b, m, m) tensor of external distances (optional)
        ext_edge_mask: (b, m, m) binary mask for external distances (optional)

        Returns:
            stress: scalar tensor
        """
        # expand embeddings to batch
        X = self.embeddings  # (B, N, dim)
        # compute pairwise Euclidean distances between embeddings
        # ||xi - xj||_2
        diff = X.unsqueeze(2) - X.unsqueeze(1)
        dists = torch.norm(diff, dim=-1)  # (B, N,
        # compute stress only on observed entries
        stress_matrix = edge_mask * (dists - edge_dist) ** 2
        stress = stress_matrix.sum(dim=-1)
        tot = edge_mask.sum(dim=-1)
        
        if ext_X is not None:
            diff_ext = X.unsqueeze(2) - ext_X.unsqueeze(1)
            ext_dists = torch.norm(diff_ext, dim=-1)
            stress_matrix = ext_edge_mask * (ext_dists - ext_edge_dist) ** 2
            stress = stress + stress_matrix.sum(dim=-1)
            tot = tot + ext_edge_mask.sum(dim=-1)
            
        # average stress per observed entry
        stress = (stress / tot.clamp(min=1)).sum()
        return stress


def fit_dist_ext(
            edge_dist: torch.Tensor,
            edge_mask: torch.Tensor,
            node_mask: torch.Tensor = None,
            ext_X: Optional[torch.Tensor] = None,
            ext_edge_dist: Optional[torch.Tensor] = None,
            ext_edge_mask: Optional[torch.Tensor] = None,
            dim: int = 2,
            lr: float = 0.1,
            n_epochs: int = 500,
            verbose: bool = False) -> torch.Tensor:
    """
    Fit batched coordinates via gradient descent.

    Args:
        edge_dist: (b, n, n) tensor of distances (padded entries can be zero)
        edge_mask: (b, n, n) tensor of 0/1 mask, 1=valid distance
        node_mask: optional (b, n) mask for valid nodes (1 valid, 0 padded)
        dim: target embedding dimension
        lr: learning rate
        n_epochs: number of optimization steps
        verbose: print progress every 50 epochs

    Returns:
        embeddings: (b, n, dim) learned coordinates
    """
    B, N, _ = edge_dist.shape
    device = edge_dist.device

    # If node_mask not given, infer from mask: a row with any valid distance
    if node_mask is None:
        node_mask = edge_mask.sum(dim=-1) > 0  # (B, N)
        node_mask = node_mask.float()

    # Initialize model
    model = ExternalMDS(batch_size=B, n_points=N, dim=dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min', factor=0.5, patience=100, threshold=0.1,
        verbose=verbose
    )
    

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        stress = model(edge_dist, edge_mask, ext_X, ext_edge_dist, ext_edge_mask)
        stress.backward()
        optimizer.step()
        scheduler.step(stress)
        if (epoch % 50 == 0 or epoch == n_epochs - 1) and verbose:
            print(f"Epoch {epoch+1}/{n_epochs}, stress={stress.item():.4f}, lr={optimizer.param_groups[0]['lr']:.6f}")

    # Return the optimized embeddings, masked padded nodes to zero
    embeddings = model.embeddings.detach()
    if node_mask is not None:
        embeddings = embeddings * node_mask.unsqueeze(-1)
    return embeddings
