from typing import Optional, Dict, Tuple

import math

import torch
import torch.nn as nn
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm
from torch.nn import functional as F
from torch import Tensor
from torch.nn import init


from torch.nn.modules.linear import Linear

from src.datatypes.dense import DenseGraph, DenseEdges, get_bipartite_edge_mask_dense, get_edge_mask_dense

class Xtoy(nn.Module):
    def __init__(self, dx, dy):
        """ Map node features to global features """
        super().__init__()
        self.lin = nn.Linear(4 * dx, dy)

    def forward(self, X, x_mask):
        """ X: bs, n, dx. """
        x_mask = x_mask.expand(-1, -1, X.shape[-1])
        float_imask = 1 - x_mask.float()
        m = X.sum(dim=1) / torch.sum(x_mask, dim=1)
        mi = (X + 1e5 * float_imask).min(dim=1)[0]
        ma = (X - 1e5 * float_imask).max(dim=1)[0]
        std = torch.sum(((X - m[:, None, :]) ** 2) * x_mask, dim=1) / torch.sum(x_mask, dim=1)
        z = torch.hstack((m, mi, ma, std))
        out = self.lin(z)
        return out


class Etoy(nn.Module):
    def __init__(self, d, dy):
        """ Map edge features to global features. """
        super().__init__()
        self.lin = nn.Linear(4 * d, dy)

    def forward(self, E, e_mask1, e_mask2):
        """ E: bs, n, n, de
            Features relative to the diagonal of E could potentially be added.
        """
        mask = (e_mask1 * e_mask2).expand(-1, -1, -1, E.shape[-1])
        float_imask = 1 - mask.float()
        divide = torch.sum(mask, dim=(1, 2))
        m = E.sum(dim=(1, 2)) / divide
        mi = (E + 1e5 * float_imask).min(dim=2)[0].min(dim=1)[0]
        ma = (E - 1e5 * float_imask).max(dim=2)[0].max(dim=1)[0]
        std = torch.sum(((E - m[:, None, None, :]) ** 2) * mask, dim=(1, 2)) / divide
        z = torch.hstack((m, mi, ma, std))
        out = self.lin(z)
        return out


def masked_softmax(x, mask, **kwargs):
    if torch.sum(mask) == 0:
        return x
    x_masked = x.clone()
    x_masked[mask == 0] = -float("inf")
    return torch.softmax(x_masked, **kwargs)




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
    def __init__(self, dx: int, de: int, dy: int, n_head: int, dim_ffX: int = 2048, dd=128,
                 dim_ffE: int = 128, dim_ffy: int = 2048, dim_ffD: int = 128 ,dropout: float = 0.1,
                 layer_norm_eps: float = 1e-5, device=None, dtype=None, last_layer=False) -> None:
        kw = {'device': device, 'dtype': dtype}
        super().__init__()

        #self.self_attn = NodeEdgeBlock(dx, de, dy, n_head, last_layer=last_layer)
        #self.self_attn = XEySelfAttention(dx, de, dy, dd ,n_head)
        self.self_attn = self_attention_mod(dx, de, dy, dd ,n_head, last_layer=last_layer)

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
        self.linD2_bis = Linear(dim_ffD, 256)
        self.linD3_bis = Linear(256, dim_ffD)
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

        self.activation = F.selu

    def forward(self,
            X: Tensor,
            E: Tensor,
            y: Tensor,
            node_mask: Tensor,
            pos: Tensor,
            edge_mask_triangular: Tensor,
            ext_X: Optional[Tensor]=None,
            ext_E: Optional[Tensor]=None,
            ext_node_mask: Optional[Tensor]=None,
        ) -> tuple[Tensor, Tensor, Tensor]:
        """ Pass the input through the encoder layer.
            X: (bs, n, d)
            E: (bs, n, n, d)
            y: (bs, dy)
            node_mask: (bs, n) Mask for the src keys per batch (optional)
            Output: newX, newE, new_y with the same shape.
        """
        x_mask = node_mask.unsqueeze(-1)        # bs, n, 1
        e_mask1 = x_mask.unsqueeze(2)           # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(1)           # bs, 1, n, 1
        #newX, newE, new_y, vel = self.self_attn(X=X, ,E=E, y=y, node_mask=node_mask, dist=pos, edge_mask_triangular=edge_mask_triangular)

        newX, newE, new_y, vel = self.self_attn(X=X, E=E, y=y, node_mask=node_mask,
                                                dist=pos, edge_mask_triangular=edge_mask_triangular)

        newX_d = self.dropoutX1(newX)
        # X = self.normX1(X + newX_d, x_mask)
        X = self.normX1(X + newX_d)

        newD_d = self.dropoutD1(vel)
        D = self.normD1(vel)

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

        D_1 = (((self.activation(self.linD1(D)))))
        D_2 = (((self.activation(self.linD2_bis(D_1)))))
        D_3 = (((self.activation(self.linD3_bis(D_2)))))

        ff_outputD = self.linD2(D_3)

        #ff_outputD = self.linD2((self.dropoutD2(self.activation(self.linD1(D)))))
        ff_outputD = (ff_outputD)

        D = self.normD2(D + ff_outputD)
        D = 0.5 * (D + torch.transpose(D,1,2))


        if not self.last_layer:
            ff_output_y = self.lin_y2(self.dropout_y2(self.activation(self.lin_y1(y))))
            ff_output_y = self.dropout_y3(ff_output_y)
            y = self.norm_y2(y + ff_output_y)

        return X, E, y, D, node_mask


class MaskedSoftmax(nn.Module):
    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, x, mask):
        x = x.masked_fill(~mask, -float('inf'))
        x = torch.softmax(x, dim=self.dim)
        return x.masked_fill(~mask, 0.0)

class self_attention_mod(nn.Module):
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
        edge_mask_triangular:Tensor,
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
        Y = Y * x_e_mul1.unsqueeze(1) * x_e_mul2.unsqueeze(2) * e_mask1 * e_mask2

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
        alpha = masked_softmax(a, softmax_mask, dim=2).unsqueeze(-1)  # bs, n, n, n_head
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
            e_y = self.e_y(Y, e_mask1, e_mask2)
            x_y = self.x_y(newX, x_mask)
            new_y = y + x_y + e_y
            y_out = self.y_out(new_y)     

        return Xout, Eout, y_out, newD.squeeze(-1)




class EtoX(nn.Module):
    def __init__(self, de, dx):
        super().__init__()
        self.lin = nn.Linear(4 * de, dx)

    def forward(self, E, e_mask2):
        """ E: bs, n, n, de"""
        bs, n, _, de = E.shape
        e_mask2 = e_mask2.expand(-1, n, -1, de)
        float_imask = 1 - e_mask2.float()
        m = E.sum(dim=2) / torch.sum(e_mask2, dim=2)
        mi = (E + 1e5 * float_imask).min(dim=2)[0]
        ma = (E - 1e5 * float_imask).max(dim=2)[0]
        std = torch.sum(((E - m[:, :, None, :]) ** 2) * e_mask2, dim=2) / torch.sum(e_mask2, dim=2)
        z = torch.cat((m, mi, ma, std), dim=2)
        out = self.lin(z)
        return out



#############  TRANSFORMER OPTIONS  ##############

from src.models.architectures.gnn.graph_transformer import DIM_X, DIM_E, DIM_Y
from src.models.architectures.gnn.graph_transformer_distance_split import DIM_C, DIM_D

if False:
    POS_INF = 1e9
    NEG_INF = -1e9
else:
    POS_INF = float('inf')
    NEG_INF = float('-inf')

from src.models import reg_architectures

@reg_architectures.register()
class distance_GraphTransformer_qm9(nn.Module):
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
            use_residuals_inout: bool = True,
            act_fn_in = nn.ReLU,
            act_fn_out = nn.ReLU,
            simpler: bool = False,
            **kwargs
        ):

        super().__init__()

        self.num_layers = num_layers
        self.use_residuals_inout = use_residuals_inout
        self.simpler = simpler

        self.in_dim_x = input_dims[DIM_X]
        self.in_dim_e = input_dims[DIM_E]
        self.in_dim_y = input_dims[DIM_Y]
        self.in_dim_c = input_dims[DIM_C]
        self.in_DIM_D = input_dims[DIM_D]

        self.encdec_hidden_dims = encdec_hidden_dims
        self.transf_inout_dims = transf_inout_dims
        self.transf_ffn_dims = transf_ffn_dims

        self.using_y = self.in_dim_y is not None

        self.out_dim_x = output_dims[DIM_X]
        self.out_dim_e = output_dims[DIM_E]
        self.out_dim_y = output_dims[DIM_Y]
        self.out_dim_c = output_dims[DIM_C]
        self.out_dim_dist = output_dims[DIM_D]

        ###########################  INPUT ENCODERS  ###########################
        # nodes encoder
        self.mlp_in_X = nn.Sequential(
            nn.Linear(self.in_dim_x + self.in_dim_c, encdec_hidden_dims[DIM_X]),
            act_fn_in(),
            nn.Linear(encdec_hidden_dims[DIM_X], transf_inout_dims[DIM_X]),
            act_fn_in()
        )

        # edges encoder
        self.mlp_in_E = nn.Sequential(
            nn.Linear(self.in_dim_e, encdec_hidden_dims[DIM_E]),
            act_fn_in(),
            nn.Linear(encdec_hidden_dims[DIM_E], transf_inout_dims[DIM_E]),
            act_fn_in()
        )

        self.mlp_dist = nn.Sequential(
            nn.Linear(1, encdec_hidden_dims[DIM_D]),
            act_fn_in(),
            nn.Linear(encdec_hidden_dims[DIM_D], 128),
            nn.Tanh()
        )
        
        self.mlp_eigen  = nn.Sequential(
            nn.Linear(9, encdec_hidden_dims[DIM_D]),
            act_fn_in(),
            nn.Linear(encdec_hidden_dims[DIM_D], transf_inout_dims[DIM_X]),
            act_fn_in()
        )


        if self.using_y:
            # global encoder
            self.mlp_in_y = nn.Sequential(
                nn.Linear(self.in_dim_y, encdec_hidden_dims[DIM_Y]),
                act_fn_in(),
                nn.Linear(encdec_hidden_dims[DIM_Y], transf_inout_dims[DIM_Y]),
                act_fn_in()
            )
        else:
            self.fixed_y = nn.Parameter(torch.randn(transf_inout_dims[DIM_Y]))

        if self.simpler:
            self.mlp_in_ext_E = self.mlp_in_E
        else:
            self.mlp_in_ext_E = nn.Sequential(
                nn.Linear(self.in_dim_e, encdec_hidden_dims[DIM_E]),
                act_fn_in(),
                nn.Linear(encdec_hidden_dims[DIM_E], transf_inout_dims[DIM_E])
            )


        #######################  MAIN BODY: TRANSFORMER  #######################

        tlayer_class = XEyTransformerLayer

        self.tf_layers = nn.ModuleList([
            tlayer_class(
                dx=transf_inout_dims[DIM_X],
                de=transf_inout_dims[DIM_E],
                dy=transf_inout_dims[DIM_Y],
                n_head=transf_hparams['heads'],
                dim_ffX=transf_ffn_dims[DIM_X],
                dim_ffE=transf_ffn_dims[DIM_E],
                dim_ffy=transf_ffn_dims[DIM_Y]
            )
            for _ in range(num_layers)
        ])

        ##########################  OUTPUT DECODERS  ###########################

        # nodes decoder
        self.mlp_out_X = nn.Sequential(
            nn.Linear(transf_inout_dims[DIM_X], encdec_hidden_dims[DIM_X]),
            act_fn_out(),
            nn.Linear(encdec_hidden_dims[DIM_X], self.out_dim_c + self.out_dim_x)
        )

        # edges decoder
        self.mlp_out_E = nn.Sequential(
            nn.Linear(transf_inout_dims[DIM_E], encdec_hidden_dims[DIM_E]),
            act_fn_out(),
            nn.Linear(encdec_hidden_dims[DIM_E], self.out_dim_e)
        )

        self.mlp_out_dist_new = nn.Sequential(
                nn.Linear(128,64),
                act_fn_out(),
                nn.Linear(64, 1),
                nn.SiLU()
            )

        if self.using_y:
            # global decoder
            self.mlp_out_y = nn.Sequential(
                nn.Linear(transf_inout_dims[DIM_Y], encdec_hidden_dims[DIM_Y]),
                act_fn_out(),
                nn.Linear(encdec_hidden_dims[DIM_Y], self.out_dim_y)
            )

        if self.simpler:
            pass
        else:
            self.mlp_out_ext_E = nn.Sequential(
                nn.Linear(transf_inout_dims[DIM_E], encdec_hidden_dims[DIM_E]),
                act_fn_out(),
                nn.Linear(encdec_hidden_dims[DIM_E], self.out_dim_e)
            )


    def get_external_nodes_dim(self):
        return self.transf_inout_dims[DIM_X]

    def forward(
            self,
            graph: DenseGraph,
            ext_X: Optional[Tensor]=None,
            ext_node_mask: Optional[Tensor]=None,
            ext_edges: Optional[DenseEdges]=None
        ) -> Tuple[DenseGraph, Optional[DenseEdges]]:

        ########################  ASSERTIONS ON INPUT  #########################
        X, E, y, dist_1, c = graph.x, graph.edge_adjmat, graph.y, graph.edge_dist, graph.node_charges

        using_ext = ext_X is not None

        bs, nq = X.shape[0], X.shape[1]

        ###############  SETUP SELFLOOP REMOVAL (DIAGONAL) MASK  ###############

        node_mask = graph.node_mask.unsqueeze(-1)
        edge_mask = graph.edge_mask.unsqueeze(-1)
        triang_mask = get_edge_mask_dense(edge_mask=graph.edge_mask, only_triangular=True).unsqueeze(-1)

        def mask_everything(X, E, ext_E=None):

            X = X * node_mask
            E = E * edge_mask
            if ext_E is not None:
                ext_E = ext_E * ext_edges.edge_mask.unsqueeze(-1)

            return X, E, ext_E

        X = torch.cat((X, c), dim=-1)
        
        dist = dist_1[:,:,:,0].unsqueeze(-1)
        
        eigen = dist_1[:,:,:,1]

        ######################  SAVE RESIDUAL FOR LATER  #######################
        if self.use_residuals_inout:
            X_to_out = X[..., :self.out_dim_x+self.out_dim_c]
            E_to_out = E[..., :self.out_dim_e]
            if self.using_y:
                y_to_out = y[..., :self.out_dim_y]
            if using_ext:
                ext_E_to_out = ext_E[..., :self.out_dim_e]

        ###########################  ENCODE INPUTS  ############################

        eigen = self.mlp_eigen(eigen)
        X = self.mlp_in_X(X)
        E = self.mlp_in_E(E) * triang_mask
        E = (E + E.transpose(1, 2))

        dist = self.mlp_dist(dist) * triang_mask
        dist = (dist+dist.transpose(1,2))
        X=X+eigen
        if self.using_y:
            y = self.mlp_in_y(y)
        else:
            y = self.fixed_y.clone().expand(bs, -1)


        # mask everything before feeding to transformer
        X, E, ext_E = mask_everything(X, E)

        #######################  MAIN BODY: TRANSFORMER  #######################

        for layer in self.tf_layers:
                X, E, y, new_dist, node_mask = layer(X=X, E=E, y=y, edge_mask_triangular=edge_mask, pos=dist, node_mask=node_mask)

        ###########################  DECODE OUTPUT  ############################
        X = self.mlp_out_X(X)
        E = self.mlp_out_E(E)

        new_dist = self.mlp_out_dist_new(new_dist) * triang_mask

        ###########################  FINAL RESIDUAL  ###########################

        # remove selfloop and make symmetric
        E = E * triang_mask

        E = (E + torch.transpose(E, 1, 2))

        new_dist = new_dist + torch.transpose(new_dist, 1,2)

        X = (X + X_to_out)

        final_X = X[..., :self.out_dim_x]
        c = X[..., self.out_dim_x:]

        # mask everything before returning
        out_graph = DenseGraph(x=final_X, edge_adjmat=E, y=y_to_out, node_mask=graph.node_mask, edge_mask=graph.edge_mask, edge_dist=new_dist.squeeze(-1),
                               node_charges=c).apply_mask()

        ###############################  RETURN  ###############################

        return out_graph
