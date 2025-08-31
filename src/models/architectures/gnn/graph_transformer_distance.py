from typing import Optional, Dict, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from src.datatypes.dense import DenseGraph, DenseEdges, get_bipartite_edge_mask_dense, get_edge_mask_dense
from src.datatypes.features.posenc import SinusoidalPosEmb

from src.models.architectures.gnn.graph_transformer import (
    DIM_X, DIM_E, DIM_Y,
    Etoy, Xtoy
)

from typing import Optional, Dict, Tuple

import math

import torch
import torch.nn as nn
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm
from torch.nn import functional as F
from torch import Tensor

from torch.nn.modules.linear import Linear

from src.datatypes.dense import DenseGraph, DenseEdges, get_bipartite_edge_mask_dense, get_edge_mask_dense



class XEyTransformerLayer(nn.Module):
    """ Transformer that updates node, edge and global features
        d_x: node features
        d_e: edge features
        dz : global features
        n_head: the number of heads in the multi_head_attention
        dim_feedforward: the dimension of the feedforward network model after self-attention
        dropout: dropout probablility. 0 to disable
        layer_norm_eps: eps value in layer normalizations.
    """
    def __init__(
            self,
            dx: int,
            de: int,
            dy: int,
            heads: int = 8,
            dim_ffX: int = 2048,
            dim_ffE: int = 128,
            dim_ffy: int = 2048,
            dropout: float = 0.1,
            layer_norm_eps: float = 1e-5,
            act_fn = nn.ReLU
        ):
        """Builds graph transformer layer
        Parameters
        ----------
        dx : int
            number of node features
        de : int
            number of edge features
        dy : Optional[int]
            number of global features. Optional for allowing the absence of global features
        n_head : int
            number of attention heads. Must be a divisor of dx
        dim_ffX : int, optional
            size of intermediate features in nodes FFN, by default 2048
        dim_ffE : int, optional
            size of intermediate features in edges FFN, by default 128
        dim_ffy : int, optional
            size of intermediate features in global FFN, by default 2048
        dropout : float, optional
            dropout probability, by default 0.1
        layer_norm_eps : float, optional
            layer normalization parameter epsilon, by default 1e-5
        """
        
        super().__init__()

        self.self_attn = XEyBlockAttention(dx, de, dy, heads)

        self.normX1 = LayerNorm(dx, eps=layer_norm_eps)
        self.normE1 = LayerNorm(de, eps=layer_norm_eps)
        self.norm_y1 = LayerNorm(dy, eps=layer_norm_eps)
        self.dropoutX1 = Dropout(dropout)
        self.dropoutE1 = Dropout(dropout)
        self.dropout_y1 = Dropout(dropout)

        # nodes FFN
        self.ffnX = nn.Sequential(
            Linear(dx, dim_ffX),
            act_fn(),
            Dropout(dropout),
            Linear(dim_ffX, dx),
            Dropout(dropout)
        )
        self.normX2 = LayerNorm(dx, eps=layer_norm_eps)

        # edges FFN
        self.ffnE = nn.Sequential(
            Linear(de, dim_ffE),
            act_fn(),
            Dropout(dropout),
            Linear(dim_ffE, de),
            Dropout(dropout)
        )
        self.normE2 = LayerNorm(de, eps=layer_norm_eps)

        # global FFN
        self.ffny = nn.Sequential(
            Linear(dy, dim_ffy),
            act_fn(),
            Dropout(dropout),
            Linear(dim_ffy, dy),
            Dropout(dropout)
        )
        self.norm_y2 = LayerNorm(dy, eps=layer_norm_eps)


    def forward(
            self,
            X: Tensor,
            E: Tensor,
            y: Tensor,
            node_mask: Tensor,
            edge_mask: Optional[Tensor]=None
        ) -> tuple[Tensor, Tensor, Tensor]:


        ####################  SELF-ATTENTION-RESIDUAL BLOCK  ###################
        # self-attention + cross-attention
        newX, newE, new_y = self.self_attn(
            Xq=X,
            Xk=X,
            node_mask_q=node_mask,
            node_mask_k=node_mask,
            E=E,
            y=y,
            edge_mask=edge_mask
        )

        #TODO: might implement pre-norm later
        # residual on nodes
        newX = self.dropoutX1(newX)
        X = self.normX1(X + newX)

        # residual on edges
        newE = self.dropoutE1(newE)
        E = self.normE1(E + newE)

        # residual on global
        new_y = self.dropout_y1(new_y)
        y = self.norm_y1(y + new_y)

        #########################  FFN-RESIDUAL BLOCK  #########################
        # X = norm(X + FFN(X))
        newX = self.ffnX(X)
        X = self.normX2(X + newX)

        # E = norm(E + FFN(E))
        newE = self.ffnE(E)
        E = self.normE2(E + newE)

        # y = norm(y + FFN(y))
        new_y = self.ffny(y)
        y = self.norm_y2(y + new_y)

        return X, E, y


class MaskedSoftmax(nn.Module):
    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, x, mask):
        x = x.masked_fill(~mask, -float('inf'))
        x = torch.softmax(x, dim=self.dim)
        return x.masked_fill(~mask, 0.0)



class GraphAttention(nn.Module):

    def __init__(
            self,
            dx: int,
            de: int,
            n_head: int
        ):

        super().__init__()

        assert dx % n_head == 0, f"Cannot divide nodes features size by number of heads: dx: {dx} -- nhead: {n_head}"

        self.dx = dx
        self.de = de

        self.df = int(dx / n_head)
        self.n_head = n_head

        # Attention
        self.q_proj = Linear(dx, dx)
        self.k_proj = Linear(dx, dx)
        self.v_proj = Linear(dx, dx)

        # FiLM E to X
        self.e_add = Linear(de, dx)
        self.e_mul = Linear(de, dx)

        self.masked_softmax = MaskedSoftmax(dim=2)


    def forward(
            self,
            Xq: Tensor,
            Xk: Tensor,
            E: Tensor,
            node_mask_q: Tensor,
            node_mask_k: Tensor,
            edge_mask: Optional[Tensor]=None
        ) -> tuple[Tensor, Tensor, Tensor]:

        bs, nq, nk, _ = E.shape

        #######################  FAKE NODES MASKS SETUP  #######################

        # unsqueeze to enable masking (dot product only if same number of dims)
        xq_mask = node_mask_q                   # (bs, nq, 1)
        xk_mask = node_mask_k                   # (bs, nk, 1)
        eq_mask = xq_mask.unsqueeze(2)          # (bs, nq, 1, 1)
        ek_mask = xk_mask.unsqueeze(1)          # (bs, 1, nk, 1)
        if edge_mask is None:
            edge_mask = eq_mask * ek_mask

        ####################  QUERIES, KEYS, VALUES SETUP  #####################

        Q = self.q_proj(Xq)			# (bs, nq, dx)
        K = self.k_proj(Xk)			# (bs, nk, dx)
        V = self.v_proj(Xk)			# (bs, nk, dx)


        # Reshape to (bs, n, n_head, df) with dx = n_head * df
        Q = Q.reshape((*Q.shape[:2], self.n_head, self.df))
        K = K.reshape((*K.shape[:2], self.n_head, self.df))
        V = V.reshape((*V.shape[:2], self.n_head, self.df))

        # setup dimensions for outer product
        Q = Q.unsqueeze(2)			# (bs, nq, 1, n_head, df)
        K = K.unsqueeze(1)			# (bs, 1, nk, n head, df)
        V = V.unsqueeze(1)			# (bs, 1, nk, n_head, df)

        ####################  OUTER PRODUCT SELF-ATTENTION  ####################

        # Compute unnormalized attentions. A is (bs, nq, nk, n_head, df)
        A = Q * K					# outer product
        A = A / math.sqrt(self.df)	# scaling by sqrt(df)

        # ----> node informed attention matrix A of shape (bs, nq, nk, n_head, df)

        #################  ATTENTION = FILM(EDGES, ATTENTION)  #################
        E1 = self.e_add(E)				# bs, nq, nk, dx
        E1 = E1.reshape((*E.shape[:3], self.n_head, self.df))

        E2 = self.e_mul(E)				# bs, nq, nk, dx
        E2 = E2.reshape((*E.shape[:3], self.n_head, self.df))

        # Incorporate edge features to the self attention scores.
        A = E1 + (E2 + 1) * A                   # (bs, nq, nk, n_head, df)

        # ----> node/edge informed attention matrix A of shape (bs, n, n, n_head, df)

        ####################  ATTENTION: AGGREGATE VALUES  #####################

        # Compute attentions. attn is still (bs, n, n, n_head, df)
        # use masked softmax to avoid attention on non-existing nodes
        softmax_mask = ek_mask.expand(-1, nq, -1, self.n_head)	# bs, nq, nk, n_head
        attn_weights = A.sum(-1)                                # bs, nq, nk, n_head
        attn = self.masked_softmax(attn_weights, softmax_mask)  # bs, nq, nk, n_head

        # Compute weighted values
        weighted_V = attn.unsqueeze(-1) * V				# (bs, nq, nk, n_head, df)
        weighted_V = weighted_V.sum(dim=2)				# (bs, nq, n_head, df)

        # Send output to input dim
        weighted_V = weighted_V.flatten(start_dim=-2)	# (bs, nq, dx)

        # ----> self-attention output, aggregated node values (bs, nq, dx)

        return weighted_V, A.flatten(start_dim=-2)



class XEyBlockAttention(nn.Module):
    """ Self attention layer that also updates the representations on the edges. """

    def __init__(
            self,
            dx: int,
            de: int,
            dy: int,
            n_head: int
        ):
        """Self-attention block for computing new edges, nodes and global features
        Parameters
        ----------
        dx : int
            number of node features
        de : int
            number of edge features
        dy : int
            number of global features
        n_head : int
            number of attention heads. Must be a divisor of dx
        """
        super().__init__()

        assert dx % n_head == 0, f"Cannot divide nodes features size by number of heads: dx: {dx} -- nhead: {n_head}"

        self.dx = dx
        self.de = de
        self.dy = dy

        self.df = int(dx / n_head)
        self.n_head = n_head

        # Attention
        self.attn = GraphAttention(dx, de, n_head) # self-attention

        # FiLM y to E
        self.y_e_mul = Linear(dy, dx)           # Warning: here it's dx and not de
        self.y_e_add = Linear(dy, dx)

        # FiLM y to X
        self.y_x_mul = Linear(dy, dx)
        self.y_x_add = Linear(dy, dx)

        # Process y
        self.y_proj = Linear(dy, dy)
        self.reduce_x = Xtoy(dx, dy)	# projection of (mean, std, min, max)
        self.reduce_e = Etoy(de, dy)	# projection of (mean, std, min, max)

        # Output layers
        self.x_out = Linear(dx, dx)
        self.e_out = Linear(dx, de)
        self.y_out = nn.Sequential(nn.Linear(3 * dy, dy), nn.ReLU(), nn.Linear(dy, dy))


    def forward(
            self,
            Xq: Tensor,
            Xk: Tensor,
            node_mask_q: Tensor,
            node_mask_k: Tensor,
            E: Tensor,
            y: Tensor,
            edge_mask: Optional[Tensor]=None
        ) -> tuple[Tensor, Tensor, Tensor]:
        """Updates the nodes, edges and global representations
        Parameters
        ----------
        X : Tensor
            node features of shape (bs, n, d)
        E : Tensor
            edge features of shape (bs, nq, nk, d)
        y : Tensor
            global features of shape (bs, dy)
        node_mask : Tensor
            node masks for non-existing nodes (due to padding in dense representation) of shape (bs, n)
        Returns
        -------
        newX : Tensor
            new node features of shape (bs, n, d)
        newE : Tensor
            new edge features of shape (bs, n, n, d)
        new_y : Tensor
            new global features of shape (bs, dy)
        """

        bs, nq, nk, _ = E.shape

        #######################  FAKE NODES MASKS SETUP  #######################

        V: Tensor     # (bs, nq, dx)
        A: Tensor  # (bs, nq, nk, de)
        V, A = self.attn(
            Xq=Xq, Xk=Xk, E=E,
            node_mask_q=node_mask_q, node_mask_k=node_mask_k, edge_mask=edge_mask
        )

        ###############  OUT_EDGES = LIN(FILM(GLOBAL, EDGES))  #################

        # Incorporate y to E
        ye1 = self.y_e_add(y).unsqueeze(1).unsqueeze(1)     # (bs, 1, 1, dx)
        ye2 = self.y_e_mul(y).unsqueeze(1).unsqueeze(1)     # (bs, 1, 1, dx)
        newE = ye1 + (ye2 + 1) * A

        # Output E
        newE = self.e_out(newE) * edge_mask					# (bs, nq, nk, de)

        # ----> END OF EDGES BRANCH

        ###############  OUT_NODES = LIN(FILM(GLOBAL, NODES))  #################

        # Incorporate y to X
        yx1 = self.y_x_add(y).unsqueeze(1)      # (bs, 1, dx)
        yx2 = self.y_x_mul(y).unsqueeze(1)      # (bs, 1, dx)
        newX = yx1 + (yx2 + 1) * V

        # Output X
        newX = self.x_out(newX) * node_mask_q   # (bs, nq, dx)

        # ----> END OF NODES BRANCH, new nodes (bs, n, dx)

        ###############  GLOBAL BRANCH  #################

        # Process y based on X and E
        y = self.y_proj(y)			# (bs, dy)
        e_y = self.reduce_e(E, edge_mask)		# (bs, dy)
        x_y = self.reduce_x(Xq, node_mask_q)		# (bs, dy)

        new_y = torch.cat([y, x_y, e_y], dim=-1)	# concat everything
        new_y = self.y_out(new_y)   # (bs, dy)

        return newX, newE, new_y


from src.models import reg_architectures

@reg_architectures.register()
class GraphTransformerDistance(nn.Module):
    """
    n_layers : int -- number of layers
    dims : dict -- contains dimensions for each feature type
    """
    def __init__(
            self,
            input_dims: Dict,
            output_dims: Dict,
            num_layers: int,
            encdec_hidden_dims: Dict,
            transf_inout_dims: Dict,
            transf_ffn_dims: Dict,
            transf_hparams: Dict,
            distance_dim: int = 16,
            use_residuals_inout: bool = True,
            act_fn = 'silu',
            **kwargs
        ):

        super().__init__()

        if act_fn == 'relu':
            self.act_fn = nn.ReLU
        elif act_fn == 'silu':
            self.act_fn = nn.SiLU
        else:
            raise ValueError(f"Activation function {act_fn} not recognized")

        self.num_layers = num_layers
        self.use_residuals_inout = use_residuals_inout
        
        self.distance_enc = SinusoidalPosEmb(distance_dim, scale=100.0)

        self.in_dim_x = input_dims[DIM_X]
        self.in_dim_e = input_dims[DIM_E] + distance_dim  # distance is added to edges
        self.in_dim_y = input_dims[DIM_Y]

        self.encdec_hidden_dims = encdec_hidden_dims
        self.transf_inout_dims = transf_inout_dims
        self.transf_ffn_dims = transf_ffn_dims

        self.using_y = self.in_dim_y is not None

        self.out_dim_x = output_dims[DIM_X]
        self.out_dim_e = output_dims[DIM_E]
        self.out_dim_y = output_dims[DIM_Y]

        ###########################  INPUT ENCODERS  ###########################
        # nodes encoder
        self.mlp_in_X = nn.Sequential(
            nn.Linear(self.in_dim_x, encdec_hidden_dims[DIM_X]),
            self.act_fn(),
            nn.Linear(encdec_hidden_dims[DIM_X], transf_inout_dims[DIM_X]),
            nn.LayerNorm(transf_inout_dims[DIM_X])
        )

        # edges encoder
        self.mlp_in_E = nn.Sequential(
            nn.Linear(self.in_dim_e, encdec_hidden_dims[DIM_E]),
            self.act_fn(),
            nn.Linear(encdec_hidden_dims[DIM_E], transf_inout_dims[DIM_E])
        )

        if self.using_y:
            # global encoder
            self.mlp_in_y = nn.Sequential(
                nn.Linear(self.in_dim_y, encdec_hidden_dims[DIM_Y]),
                self.act_fn(),
                nn.Linear(encdec_hidden_dims[DIM_Y], transf_inout_dims[DIM_Y])
            )
        else:
            self.fixed_y = nn.Parameter(torch.randn(transf_inout_dims[DIM_Y]))


        #######################  MAIN BODY: TRANSFORMER  #######################

        self.tf_layers = nn.ModuleList([
            XEyTransformerLayer(
                dx=transf_inout_dims[DIM_X],
                de=transf_inout_dims[DIM_E],
                dy=transf_inout_dims[DIM_Y],
                dim_ffX=transf_ffn_dims[DIM_X],
                dim_ffE=transf_ffn_dims[DIM_E],
                dim_ffy=transf_ffn_dims[DIM_Y],
                act_fn=self.act_fn,
                **transf_hparams
            )
            for _ in range(num_layers)
        ])

        ##########################  OUTPUT DECODERS  ###########################

        # nodes decoder
        self.mlp_out_X = nn.Sequential(
            nn.Linear(transf_inout_dims[DIM_X], encdec_hidden_dims[DIM_X]),
            self.act_fn(),
            nn.Linear(encdec_hidden_dims[DIM_X], self.out_dim_x)
        )

        # edges decoder
        self.mlp_out_E = nn.Sequential(
            nn.Linear(transf_inout_dims[DIM_E], encdec_hidden_dims[DIM_E]),
            self.act_fn(),
            nn.Linear(encdec_hidden_dims[DIM_E], self.out_dim_e)
        )
        self.mlp_out_D = nn.Sequential(
            nn.Linear(transf_inout_dims[DIM_E], encdec_hidden_dims[DIM_E]),
            self.act_fn(),
            nn.Linear(encdec_hidden_dims[DIM_E], 1)
        )

        if self.using_y:
            # global decoder
            self.mlp_out_y = nn.Sequential(
                nn.Linear(transf_inout_dims[DIM_Y], encdec_hidden_dims[DIM_Y]),
                self.act_fn(),
                nn.Linear(encdec_hidden_dims[DIM_Y], self.out_dim_y)
            )


    def forward(
            self,
            graph: DenseGraph
        ) -> DenseGraph:

        ########################  ASSERTIONS ON INPUT  #########################
        X, E, y, D = graph.x, graph.edge_adjmat, graph.y, graph.edge_dist

        assert X.shape[-1] == self.in_dim_x, \
            f"X.shape[-1] = {X.shape[-1]}, self.in_dim_x = {self.in_dim_x}"
        assert E.shape[-1] == self.in_dim_e - self.distance_enc.dim, \
            f"E.shape[-1] = {E.shape[-1]}, self.in_dim_e = {self.in_dim_e}"
        assert y is None or y.shape[-1] == self.in_dim_y, \
            f"y.shape[-1] = {y.shape[-1]}, self.in_dim_y = {self.in_dim_y}"

        bs, nq = X.shape[0], X.shape[1]

        ###############  SETUP SELFLOOP REMOVAL (DIAGONAL) MASK  ###############
        
        node_mask = graph.node_mask.unsqueeze(-1)
        edge_mask = graph.edge_mask.unsqueeze(-1)
        triang_mask = get_edge_mask_dense(edge_mask=graph.edge_mask, only_triangular=True).unsqueeze(-1)

        def mask_everything(X, E):

            X = X * node_mask
            E = E * edge_mask
            
            return X, E

        ######################  SAVE RESIDUAL FOR LATER  #######################
        if self.use_residuals_inout:
            X_to_out = X[..., :self.out_dim_x]
            E_to_out = E[..., :self.out_dim_e]
            D_to_out = D  # distance is always 1-dimensional
            if self.using_y:
                y_to_out = y[..., :self.out_dim_y]

        ###########################  ENCODE INPUTS  ############################
        # special treatment for edges (to make it symmetric (shouldn't this already be?))
        X = self.mlp_in_X(X)
        D = self.distance_enc(D)
        E = self.mlp_in_E(torch.cat([E, D], dim=-1))  # concatenate distance to edges
        E = (E + E.transpose(1, 2)) / 2   # new_E should already be symmetric if E is symmetric!!!

        if self.using_y:
            y = self.mlp_in_y(y)
        else:
            y = self.fixed_y.clone().expand(bs, -1)

        # mask everything before feeding to transformer
        X, E = mask_everything(X, E)

        #######################  MAIN BODY: TRANSFORMER  #######################

        for layer in self.tf_layers:
            X, E, y = layer(X, E, y, node_mask, edge_mask)


        ###########################  DECODE OUTPUT  ############################
        X = self.mlp_out_X(X)
        D = self.mlp_out_D(E).squeeze(-1)  # distance is always 1-dimensional
        E = self.mlp_out_E(E)
        if self.using_y:
            y = self.mlp_out_y(y)

        ###########################  FINAL RESIDUAL  ###########################
        if self.use_residuals_inout:
            X = X + X_to_out
            E = E + E_to_out
            D = D + D_to_out
            
        # remove selfloop and make symmetric
        E = E * triang_mask
        #E = (E + torch.transpose(E, 1, 2)) / 2 # here it's ok!
        E = (E + torch.transpose(E, 1, 2))
        
        D = D * triang_mask.squeeze(-1)
        D = D + torch.transpose(D, 1, 2)
        
        if self.use_residuals_inout:
            if self.using_y:
                y = y + y_to_out
        
        # mask everything before returning
        out_graph = DenseGraph(X, E, y, graph.node_mask, graph.edge_mask, edge_dist=D).apply_mask()

        ###############################  RETURN  ###############################

        return out_graph
