from typing import Optional, Dict, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from src.datatypes.dense import DenseGraph, DenseEdges, get_bipartite_edge_mask_dense, get_edge_mask_dense
from src.datatypes.features.posenc import SinusoidalPosEmb

from src.models.architectures.gnn.graph_transformer import (
    DIM_X, DIM_E, DIM_Y,
    Etoy, Xtoy, EtoX
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
    def __init__(self, dx: int, de: int, dy: int, heads: int, dim_ffX: int = 2048, dd=128,
                 dim_ffE: int = 128, dim_ffy: int = 2048, dim_ffD: int = 128 ,dropout: float = 0.1,
                 layer_norm_eps: float = 1e-5, device=None, dtype=None, last_layer=False,
                 extended_D_ffn=True) -> None:
        kw = {'device': device, 'dtype': dtype}
        super().__init__()

        self.extended_D_ffn = extended_D_ffn
        #self.self_attn = NodeEdgeBlock(dx, de, dy, n_head, last_layer=last_layer)
        #self.self_attn = XEySelfAttention(dx, de, dy, dd ,n_head)
        self.self_attn = GraphSelfAttention(dx, de, dy, dd, last_layer=last_layer, n_head=heads)

        self.linX1 = Linear(dx, dim_ffX, **kw)
        self.linX2 = Linear(dim_ffX, dx, **kw)
        # self.normX1 = SetNorm(feature_dim=dx, eps=layer_norm_eps, **kw)
        # self.normX2 = SetNorm(feature_dim=dx, eps=layer_norm_eps, **kw)
        self.normX1 = LayerNorm(dx, eps=layer_norm_eps, **kw)
        self.normX2 = LayerNorm(dx, eps=layer_norm_eps, **kw)
        self.dropoutX1 = Dropout(dropout)
        self.dropoutX2 = Dropout(dropout)
        self.dropoutX3 = Dropout(dropout)

        self.linE1 = Linear(de, dim_ffE, **kw)
        self.linE2 = Linear(dim_ffE, de, **kw)
        # self.normE1 = GraphNorm(feature_dim=de, eps=layer_norm_eps, **kw)
        # self.normE2 = GraphNorm(feature_dim=de, eps=layer_norm_eps, **kw)
        self.normE1 = LayerNorm(de, eps=layer_norm_eps, **kw)
        self.normE2 = LayerNorm(de, eps=layer_norm_eps, **kw)
        self.dropoutE1 = Dropout(dropout)
        self.dropoutE2 = Dropout(dropout)
        self.dropoutE3 = Dropout(dropout)

        self.linD1 = Linear(dd, dim_ffD)
        self.linD2 = Linear(dim_ffD, dd)
        self.normD1 = LayerNorm(dd, eps=layer_norm_eps, **kw)
        self.normD2 = LayerNorm(dd, eps=layer_norm_eps, **kw)
        self.dropoutD1 = Dropout(dropout)
        self.dropoutD2 = Dropout(dropout)
        self.dropoutD3 = Dropout(dropout)
        if self.extended_D_ffn:
            self.linD2_bis = Linear(dim_ffD, dim_ffD)
            self.linD3_bis = Linear(dim_ffD, dim_ffD)
        self.dropout_dbis = Dropout(dropout)
        self.dropout_dtris = Dropout(dropout)

        self.last_layer = last_layer
        if not last_layer:
            self.lin_y1 = Linear(dy, dim_ffy, **kw)
            self.lin_y2 = Linear(dim_ffy, dy, **kw)
            self.norm_y1 = LayerNorm(dy, eps=layer_norm_eps, **kw)
            self.norm_y2 = LayerNorm(dy, eps=layer_norm_eps, **kw)
            self.dropout_y1 = Dropout(dropout)
            self.dropout_y2 = Dropout(dropout)
            self.dropout_y3 = Dropout(dropout)

        self.activation = F.silu

    def forward(
            self,
            X: Tensor,
            E: Tensor,
            y: Tensor,
            D: Tensor,
            node_mask: Tensor,
            edge_mask: Tensor,
        ) -> tuple[Tensor, Tensor, Tensor]:
        x_mask = node_mask.unsqueeze(-1)        # bs, n, 1
        e_mask1 = x_mask.unsqueeze(2)           # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(1)           # bs, 1, n, 1
        #newX, newE, new_y, vel = self.self_attn(X=X, ,E=E, y=y, node_mask=node_mask, dist=pos, edge_mask_triangular=edge_mask_triangular)

        newX, newE, new_y, vel = self.self_attn(X=X, E=E, y=y, node_mask=node_mask,
                                                dist=D, edge_mask=edge_mask)

        newX_d = self.dropoutX1(newX)
        # X = self.normX1(X + newX_d, x_mask)
        X = self.normX1(X + newX_d)

        newD_d = self.dropoutD1(vel)
        D = self.normD1(D + newD_d)

        newE_d = self.dropoutE1(newE)
        # E = self.normE1(E + newE_d, e_mask1, e_mask2)
        E = self.normE1(E + newE_d)

        if not self.last_layer:
            new_y_d = self.dropout_y1(new_y)
            y = self.norm_y1(y + new_y_d)

        ff_outputX = self.linX2(self.dropoutX2(self.activation(self.linX1(X))))
        ff_outputX = self.dropoutX3(ff_outputX)
        # X = self.normX2(X + ff_outputX, x_mask)
        X = self.normX2(X + ff_outputX)

        ff_outputE = self.linE2(self.dropoutE2(self.activation(self.linE1(E))))
        ff_outputE = self.dropoutE3(ff_outputE)

        E = self.normE2(E + ff_outputE)
        E = 0.5 * (E + torch.transpose(E, 1, 2))

        if self.extended_D_ffn:
            D_1 = (((self.activation(self.linD1(D)))))
            D_2 = (((self.activation(self.linD2_bis(D_1)))))
            D_3 = (((self.activation(self.linD3_bis(D_2)))))

            ff_outputD = self.linD2(D_3)
        else:

            ff_outputD = self.linD2((self.dropoutD2(self.activation(self.linD1(D)))))
            ff_outputD = self.dropoutD3(ff_outputD)

        D = self.normD2(D + ff_outputD)
        D = 0.5 * (D + torch.transpose(D,1,2))


        if not self.last_layer:
            ff_output_y = self.lin_y2(self.dropout_y2(self.activation(self.lin_y1(y))))
            ff_output_y = self.dropout_y3(ff_output_y)
            y = self.norm_y2(y + ff_output_y)

        return X, E, y, D



class MaskedSoftmax(nn.Module):
    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, x, mask):
        x = x.masked_fill(~mask, -float('inf'))
        x = torch.softmax(x, dim=self.dim)
        return x.masked_fill(~mask, 0.0)
    

class GraphSelfAttention(nn.Module):
    def __init__(
        self,
        dx: int,
        de: int,
        dy: int,
        dd: int,
        n_head: int,
        last_layer: bool
    ):
        super().__init__()
        assert dx % n_head == 0, f"dx: {dx} -- nhead: {n_head}"
        self.dx = dx
        self.de = de
        self.dy = dy
        self.df = int(dx / n_head)
        self.n_head = n_head
        self.last_layer = last_layer

        self.in_E = Linear(de, de)

        self.k_proj_dist = Linear(dx, dx)
        self.v_proj_dist = Linear(dx, dx)
        self.q_proj_dist = Linear(dx, dx)

        # FILM LAYER X TO E
        self.x_e_mul1 = Linear(dx, de)
        self.x_e_mul2 = Linear(dx, de)

        # ATTENTIO  LAYER
        self.k = Linear(dx, dx)
        self.q = Linear(dx, dx)
        self.v = Linear(dx, dx)
        self.a = Linear(dx, n_head, bias=False)
        self.out = Linear(dx * n_head, dx)

        # INCOPRPORATE E TO X

        self.e_att_mul = Linear(de, n_head)

        self.pos_att_mul = Linear(de, n_head)
        self.e_x_mul = EtoX(de, dx)
        self.pos_x_mul = EtoX(de, dx)

        # FILM Y TO E

        self.y_e_mul = Linear(dy, de)           # Warning: here it's dx and not de
        self.y_e_add = Linear(dy, de)

        self.pre_softmax = Linear(de, dx)       # Unused, but needed to load old checkpoints

        # FILM Y TO X
        self.y_x_mul = Linear(dy, dx)
        self.y_x_add = Linear(dy, dx)

        # FILM DIST  to X

        self.d_add = Linear(dd, n_head)
        self.d_mul = Linear(dd, n_head)

        # FILM DIST TO E

        self.d_add_e = Linear(de, dx)
        self.d_mul_e = Linear(de, dx)
        
        #self.dist_add_e = Linear(dd,de)
        #self.dist_mul_e = Linear(dd, de)

        # FILM DIST TO Y

        self.y_e_add_d = Linear(dy, n_head)
        self.y_e_mul_d = Linear(dy, n_head)

        # Process y
        self.last_layer = last_layer
        if not last_layer:
            self.y_y = Linear(dy, dy)
            self.x_y = Xtoy(dx, dy)
            self.e_y = Etoy(de, dy)
            self.dist_y = Etoy(de, dy)

        # Output layers
        self.x_out = Linear(dx, dx)
        self.e_out = Linear(de, de)
        if not last_layer:
            self.y_out = nn.Sequential(nn.Linear(dy, dy), nn.ReLU(), nn.Linear(dy, dy))

        self.d_out = Linear(n_head, dd)
        
        self.masked_softmax = MaskedSoftmax(dim=2)

    def forward(
        self,
        X: Tensor,
        E: Tensor,
        y: Tensor,
        node_mask: Tensor,
        edge_mask:Tensor,
        dist: Tensor
    ):
        bs, n, _ = X.shape
        x_mask = node_mask       # bs, n, 1
        e_mask1 = x_mask.unsqueeze(2)           # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(1)           # bs, 1, n, 1

        Y = self.in_E(E)

        # 1.1 Incorporate x
        x_e_mul1 = self.x_e_mul1(X) * x_mask
        x_e_mul2 = self.x_e_mul2(X) * x_mask
        Y = Y * x_e_mul1.unsqueeze(1) * x_e_mul2.unsqueeze(2) * edge_mask

        # aggiungi la E

        # * riscrivi non con element wise !!!!!!!
       
        # added inside last 
        #dist_add = self.dist_add_e(dist)
        #dist_mul = self.dist_mul_e(dist)
        #Y = (Y + dist_add + Y * dist_mul) * e_mask1 * e_mask2   # bs, n, n, dx

        # 1.3 Incorporate y to E
        y_e_add = self.y_e_add(y).unsqueeze(1).unsqueeze(1)  # bs, 1, 1, de
        y_e_mul = self.y_e_mul(y).unsqueeze(1).unsqueeze(1)
        E = (Y + y_e_add + Y * y_e_mul) * e_mask1 * e_mask2

        # Output E
        Eout = self.e_out(E) * e_mask1 * e_mask2      # bs, n, n, de

        # 2. Process the node features
        Q = (self.q(X) * x_mask).unsqueeze(2)          # bs, 1, n, dx
        K = (self.k(X) * x_mask).unsqueeze(1)          # bs, n, 1, dx
        prod = Q * K / math.sqrt(Y.size(-1))   # bs, n, n, dx
        a = self.a(prod) * e_mask1 * e_mask2   # bs, n, n, n_head

        # FILM LAYER with distance feature ATTENTION = FILM(EDGE,ATTENTION)
        
        D1 = self.d_add(dist)
        D1 = D1.reshape((*E.shape[:3], self.n_head))
        
        D2 = self.d_mul(dist)				# bs, nq, nk
        D2 = D2.reshape((*dist.shape[:3], self.n_head))

        a = D1 + (D2 + 1) * a
        
        # 2.1 Incorporate edge features
        e_x_mul = self.e_att_mul(E)
        a = a + e_x_mul * a
        
        # OUT_DIST = LIN(FILM(GLOB_Y, ATTENTION))
        
        ye1 = self.y_e_add_d(y).unsqueeze(1).unsqueeze(1)              
        ye2 = self.y_e_mul_d(y).unsqueeze(1).unsqueeze(1)   
        
        newD = ye1 + (ye2 + 1) * a
        
        newD = self.d_out(newD) * e_mask1 * e_mask2
        
        # 2.3 Self-attention
        softmax_mask = e_mask2.expand(-1, n, -1, self.n_head)
        alpha = self.masked_softmax(a, softmax_mask).unsqueeze(-1)  # bs, n, n, n_head
        V = (self.v(X) * x_mask).unsqueeze(1).unsqueeze(3)      # bs, 1, n, 1, dx
        weighted_V = alpha * V                                  # bs, n, n, n_heads, dx
        weighted_V = weighted_V.sum(dim=2)                      # bs, n, n_head, dx
        weighted_V = weighted_V.flatten(start_dim=2)            # bs, n, n_head x dx
        weighted_V = self.out(weighted_V) * x_mask              # bs, n, dx

        # Incorporate E to X
        e_x_mul = self.e_x_mul(E, e_mask2)
        weighted_V = weighted_V + e_x_mul * weighted_V
        

        # Incorporate y to X
        yx1 = self.y_x_add(y).unsqueeze(1)                     # bs, 1, dx
        yx2 = self.y_x_mul(y).unsqueeze(1)
        newX = weighted_V * (yx2 + 1) + yx1

        # Output X
        Xout = self.x_out(newX) * x_mask
        #diffusion_utils.assert_correctly_masked(Xout, x_mask)
          # bs, dy

                # Process y based on X and E
        if self.last_layer:
            y_out = None
        else:
            y = self.y_y(y)
            e_y = self.e_y(Y, edge_mask)
            x_y = self.x_y(newX, x_mask)
            new_y = y + x_y + e_y
            y_out = self.y_out(new_y)     

        return Xout, Eout, y_out, newD.squeeze(-1)


from src.models import reg_architectures

DIM_C = 'node_charges'
DIM_D = 'edge_dist'

@reg_architectures.register()
class GraphTransformerDistanceOriginal(nn.Module):
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
            encode_distances: bool = False,
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
        
        self.input_dims = input_dims
        self.output_dims = output_dims

        self.num_layers = num_layers
        self.use_residuals_inout = use_residuals_inout
        self.encode_distances = encode_distances

        self.in_dim_x = input_dims[DIM_X] + input_dims[DIM_C]
        self.in_dim_e = input_dims[DIM_E]
        self.in_dim_y = input_dims[DIM_Y]
        
        
        if self.encode_distances:
            self.distance_enc = SinusoidalPosEmb(distance_dim, scale=100.0)
            self.in_dim_d = distance_dim - 1 + input_dims[DIM_D]
        else:
            self.in_dim_d = input_dims[DIM_D] 

        self.encdec_hidden_dims = encdec_hidden_dims
        self.transf_inout_dims = transf_inout_dims
        self.transf_ffn_dims = transf_ffn_dims

        self.using_y = self.in_dim_y is not None

        self.out_dim_x = output_dims[DIM_X]
        self.out_dim_e = output_dims[DIM_E]
        self.out_dim_y = output_dims[DIM_Y]
        self.out_dim_c = output_dims[DIM_C]
        self.out_dim_d = output_dims[DIM_D] # can be > 1, e.g., with periscopic or conditional distances

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
        
        # edges encoder
        # self.mlp_in_D = nn.Sequential(
        #     nn.Linear(distance_dim, encdec_hidden_dims[DIM_D]),
        #     self.act_fn(),
        #     nn.Linear(encdec_hidden_dims[DIM_D], transf_inout_dims[DIM_D])
        # )
        self.mlp_in_D = nn.Sequential(
            nn.Linear(self.in_dim_d, encdec_hidden_dims[DIM_D]),
            self.act_fn(),
            nn.Linear(encdec_hidden_dims[DIM_D], transf_inout_dims[DIM_D])
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
                dd=transf_inout_dims[DIM_D], # distance features have same dim as edges
                dim_ffX=transf_ffn_dims[DIM_X],
                dim_ffE=transf_ffn_dims[DIM_E],
                dim_ffy=transf_ffn_dims[DIM_Y],
                dim_ffD=transf_ffn_dims[DIM_D], # distance features have same dim as edges
                last_layer=(i == num_layers - 1),
                **transf_hparams
            )
            for i in range(num_layers)
        ])

        ##########################  OUTPUT DECODERS  ###########################

        # nodes decoder
        self.mlp_out_X = nn.Sequential(
            nn.Linear(transf_inout_dims[DIM_X], encdec_hidden_dims[DIM_X]),
            self.act_fn(),
            nn.Linear(encdec_hidden_dims[DIM_X], self.out_dim_x)
        )
        self.mlp_out_C = nn.Sequential(
            nn.Linear(transf_inout_dims[DIM_X], encdec_hidden_dims[DIM_X]),
            self.act_fn(),
            nn.Linear(encdec_hidden_dims[DIM_X], self.out_dim_c)
        )

        # edges decoder
        self.mlp_out_E = nn.Sequential(
            nn.Linear(transf_inout_dims[DIM_E], encdec_hidden_dims[DIM_E]),
            self.act_fn(),
            nn.Linear(encdec_hidden_dims[DIM_E], self.out_dim_e)
        )
        self.mlp_out_D = nn.Sequential(
            nn.Linear(transf_inout_dims[DIM_D], encdec_hidden_dims[DIM_D]),
            self.act_fn(),
            nn.Linear(encdec_hidden_dims[DIM_D], self.out_dim_d),
            nn.SiLU()
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
        X, E, y, D, C = graph.x, graph.edge_adjmat, graph.y, graph.edge_dist, graph.node_charges

        assert X.shape[-1] == self.input_dims[DIM_X]
        assert E.shape[-1] == self.input_dims[DIM_E]
        assert y is None or y.shape[-1] == self.input_dims[DIM_Y]
        assert C.shape[-1] == self.input_dims[DIM_C]

        bs, n = X.shape[0], X.shape[1]

        ###############  SETUP SELFLOOP REMOVAL (DIAGONAL) MASK  ###############
        
        node_mask = graph.node_mask.unsqueeze(-1)
        edge_mask = graph.edge_mask.unsqueeze(-1)
        triang_mask = get_edge_mask_dense(edge_mask=graph.edge_mask, only_triangular=True).unsqueeze(-1)
        diag_mask = ~torch.eye(n, device=graph.x.device, dtype=torch.bool)
        diag_mask = diag_mask.unsqueeze(0).unsqueeze(-1).expand(bs, -1, -1, -1)

        def mask_everything(X, E, D):

            X = X * node_mask
            E = E * edge_mask
            D = D * edge_mask
            
            return X, E, D

        ######################  SAVE RESIDUAL FOR LATER  #######################
        if self.use_residuals_inout:
            X_to_out = X[..., :self.out_dim_x]
            E_to_out = E[..., :self.out_dim_e]
            D_to_out = D  # distance is always 1-dimensional
            C_to_out = C[..., :self.out_dim_c]
            if self.using_y:
                y_to_out = y[..., :self.out_dim_y]

        ###########################  ENCODE INPUTS  ############################
        # special treatment for edges (to make it symmetric (shouldn't this already be?))
        X = self.mlp_in_X(torch.cat([X, C], dim=-1)) # concatenate nodes with charges
        if self.encode_distances:
            if D.ndim == 4:
                others = D[..., 1:]
                D = D[..., 0]
            else:
                others = None
            D = self.distance_enc(D)
            if others is not None:
                D = torch.cat([D, others], dim=-1)
        if D.ndim == 3:
            D = D.unsqueeze(-1)
        D = self.mlp_in_D(D)
        D = (D + D.transpose(1, 2)) / 2
        E = self.mlp_in_E(E)  # concatenate distance to edges
        E = (E + E.transpose(1, 2)) / 2

        if self.using_y:
            y = self.mlp_in_y(y)
        else:
            y = self.fixed_y.clone().expand(bs, -1)

        # mask everything before feeding to transformer
        X, E, D = mask_everything(X, E, D)

        #######################  MAIN BODY: TRANSFORMER  #######################

        for layer in self.tf_layers:
            X, E, y, D = layer(X, E, y, D, node_mask, edge_mask)


        ###########################  DECODE OUTPUT  ############################
        C = self.mlp_out_C(X)
        X = self.mlp_out_X(X)
        D = self.mlp_out_D(D).squeeze(-1)  # distance is always 1-dimensional
        E = self.mlp_out_E(E)
        if self.using_y:
            y = self.mlp_out_y(y)

        ###########################  FINAL RESIDUAL  ###########################
        if self.use_residuals_inout:
            X = X + X_to_out
            C = C + C_to_out
            E = E + E_to_out
            #D = D + D_to_out
            
        # remove selfloop and make symmetric
        #E = E * triang_mask
        E = E * diag_mask
        E = (E + E.transpose(1, 2)) / 2
        #E = (E + torch.transpose(E, 1, 2))
        
        #D = D * triang_mask.squeeze(-1)
        sq_diag_mask = diag_mask.squeeze(-1) if D.ndim == 3 else diag_mask
        D = D * sq_diag_mask
        D = (D + D.transpose(1, 2)) / 2
        #D = D + torch.transpose(D, 1, 2)
        
        if self.use_residuals_inout:
            if self.using_y:
                y = y + y_to_out
        
        # mask everything before returning
        out_graph = DenseGraph(
            x=X,
            edge_adjmat=E,
            y=y,
            node_mask=graph.node_mask,
            edge_mask=graph.edge_mask,
            edge_dist=D,
            node_charges=C
        ).apply_mask()

        ###############################  RETURN  ###############################

        return out_graph
