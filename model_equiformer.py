import torch
import torch.nn as nn
import torch_geometric.nn as gnn
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from tools import *
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import torch_geometric
from torch_geometric.data import Batch
import torch.nn.init as init
from torch_cluster import radius_graph
from radial_func import RadialProfile
import e3nn
from e3nn import o3
from e3nn.util.jit import compile_mode
from e3nn.nn.models.v2106.gate_points_message_passing import tp_path_exists
from tensor_product_rescale import (TensorProductRescale, LinearRS,
    FullyConnectedTensorProductRescale, irreps2gate, sort_irreps_even_first)

from fast_activation import Activation, Gate
from gaussian_rbf import GaussianRadialBasisLayer


import torch
import torch.nn as nn
import torch.nn.functional as F

_RESCALE = True
_USE_BIAS = True

# QM9
_MAX_ATOM_TYPE = 5
# Statistics of QM9 with cutoff radius = 5
_AVG_NUM_NODES = 18.03065905448718
_AVG_DEGREE = 15.57930850982666
       

# # QM7
# _MAX_ATOM_TYPE = 5
# # Statistics of QM9 with cutoff radius = 5
# _AVG_NUM_NODES = 6.78797
# _AVG_DEGREE = 1.89648
       


class SeparableFCTP(torch.nn.Module):
    '''
        Use separable FCTP for spatial convolution.
    '''
    def __init__(self, irreps_node_input, irreps_edge_attr, irreps_node_output, 
        fc_neurons, use_activation=False, norm_layer='graph', 
        internal_weights=False):
        
        super().__init__()
        self.irreps_node_input = o3.Irreps(irreps_node_input)
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)
        self.irreps_node_output = o3.Irreps(irreps_node_output)
        norm = get_norm_layer(norm_layer)
        
        self.dtp = DepthwiseTensorProduct(self.irreps_node_input, self.irreps_edge_attr, 
            self.irreps_node_output, bias=False, internal_weights=internal_weights)
        
        self.dtp_rad = None
        if fc_neurons is not None:
            self.dtp_rad = RadialProfile(fc_neurons + [self.dtp.tp.weight_numel])
            for (slice, slice_sqrt_k) in self.dtp.slices_sqrt_k.values():
                self.dtp_rad.net[-1].weight.data[slice, :] *= slice_sqrt_k
                self.dtp_rad.offset.data[slice] *= slice_sqrt_k
                
        irreps_lin_output = self.irreps_node_output
        irreps_scalars, irreps_gates, irreps_gated = irreps2gate(self.irreps_node_output)
        if use_activation:
            irreps_lin_output = irreps_scalars + irreps_gates + irreps_gated
            irreps_lin_output = irreps_lin_output.simplify()
        self.lin = LinearRS(self.dtp.irreps_out.simplify(), irreps_lin_output)
        
        self.norm = None
        if norm_layer is not None:
            self.norm = norm(self.lin.irreps_out)
        
        self.gate = None
        if use_activation:
            if irreps_gated.num_irreps == 0:
                gate = Activation(self.irreps_node_output, acts=[torch.nn.SiLU()])
            else:
                gate = Gate(
                    irreps_scalars, [torch.nn.SiLU() for _, ir in irreps_scalars],  # scalar
                    irreps_gates, [torch.sigmoid for _, ir in irreps_gates],  # gates (scalars)
                    irreps_gated  # gated tensors
                )
            self.gate = gate
    
    
    def forward(self, node_input, edge_attr, edge_scalars, batch=None, **kwargs):
        '''
            Depthwise TP: `node_input` TP `edge_attr`, with TP parametrized by 
            self.dtp_rad(`edge_scalars`).
        '''
        weight = None
        if self.dtp_rad is not None and edge_scalars is not None:    
            weight = self.dtp_rad(edge_scalars)
        out = self.dtp(node_input, edge_attr, weight)
        out = self.lin(out)
        if self.norm is not None:
            out = self.norm(out, batch=batch)
        if self.gate is not None:
            out = self.gate(out)
        return out
        

@compile_mode('script')
class Vec2AttnHeads(torch.nn.Module):
    '''
        Reshape vectors of shape [N, irreps_mid] to vectors of shape
        [N, num_heads, irreps_head].
    '''
    def __init__(self, irreps_head, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.irreps_head = irreps_head
        self.irreps_mid_in = []
        for mul, ir in irreps_head:
            self.irreps_mid_in.append((mul * num_heads, ir))
        self.irreps_mid_in = o3.Irreps(self.irreps_mid_in)
        self.mid_in_indices = []
        start_idx = 0
        for mul, ir in self.irreps_mid_in:
            self.mid_in_indices.append((start_idx, start_idx + mul * ir.dim))
            start_idx = start_idx + mul * ir.dim
    
    
    def forward(self, x):
        N, _ = x.shape
        out = []
        for ir_idx, (start_idx, end_idx) in enumerate(self.mid_in_indices):
            temp = x.narrow(1, start_idx, end_idx - start_idx)
            temp = temp.reshape(N, self.num_heads, -1)
            out.append(temp)
        out = torch.cat(out, dim=2)
        return out
    
    
    def __repr__(self):
        return '{}(irreps_head={}, num_heads={})'.format(
            self.__class__.__name__, self.irreps_head, self.num_heads)
    
    
@compile_mode('script')
class AttnHeads2Vec(torch.nn.Module):
    '''
        Convert vectors of shape [N, num_heads, irreps_head] into
        vectors of shape [N, irreps_head * num_heads].
    '''
    def __init__(self, irreps_head):
        super().__init__()
        self.irreps_head = irreps_head
        self.head_indices = []
        start_idx = 0
        for mul, ir in self.irreps_head:
            self.head_indices.append((start_idx, start_idx + mul * ir.dim))
            start_idx = start_idx + mul * ir.dim
    
    
    def forward(self, x):
        N, _, _ = x.shape
        out = []
        for ir_idx, (start_idx, end_idx) in enumerate(self.head_indices):
            temp = x.narrow(2, start_idx, end_idx - start_idx)
            temp = temp.reshape(N, -1)
            out.append(temp)
        out = torch.cat(out, dim=1)
        return out
    
    
    def __repr__(self):
        return '{}(irreps_head={})'.format(self.__class__.__name__, self.irreps_head)


class ConcatIrrepsTensor(torch.nn.Module):
    
    def __init__(self, irreps_1, irreps_2):
        super().__init__()
        assert irreps_1 == irreps_1.simplify()
        self.check_sorted(irreps_1)
        assert irreps_2 == irreps_2.simplify()
        self.check_sorted(irreps_2)
        
        self.irreps_1 = irreps_1
        self.irreps_2 = irreps_2
        self.irreps_out = irreps_1 + irreps_2
        self.irreps_out, _, _ = sort_irreps_even_first(self.irreps_out) #self.irreps_out.sort()
        self.irreps_out = self.irreps_out.simplify()
        
        self.ir_mul_list = []
        lmax = max(irreps_1.lmax, irreps_2.lmax)
        irreps_max = []
        for i in range(lmax + 1):
            irreps_max.append((1, (i, -1)))
            irreps_max.append((1, (i,  1)))
        irreps_max = o3.Irreps(irreps_max)
        
        start_idx_1, start_idx_2 = 0, 0
        dim_1_list, dim_2_list = self.get_irreps_dim(irreps_1), self.get_irreps_dim(irreps_2)
        for _, ir in irreps_max:
            dim_1, dim_2 = None, None
            index_1 = self.get_ir_index(ir, irreps_1)
            index_2 = self.get_ir_index(ir, irreps_2)
            if index_1 != -1:
                dim_1 = dim_1_list[index_1]
            if index_2 != -1:
                dim_2 = dim_2_list[index_2]
            self.ir_mul_list.append((start_idx_1, dim_1, start_idx_2, dim_2))
            start_idx_1 = start_idx_1 + dim_1 if dim_1 is not None else start_idx_1
            start_idx_2 = start_idx_2 + dim_2 if dim_2 is not None else start_idx_2
          
            
    def get_irreps_dim(self, irreps):
        muls = []
        for mul, ir in irreps:
            muls.append(mul * ir.dim)
        return muls
    
    
    def check_sorted(self, irreps):
        lmax = None
        p = None
        for _, ir in irreps:
            if p is None and lmax is None:
                p = ir.p
                lmax = ir.l
                continue
            if ir.l == lmax:
                assert p < ir.p, 'Parity order error: {}'.format(irreps)
            assert lmax <= ir.l                
        
    
    def get_ir_index(self, ir, irreps):
        for index, (_, irrep) in enumerate(irreps):
            if irrep == ir:
                return index
        return -1
    
    
    def forward(self, feature_1, feature_2):
        
        output = []
        for i in range(len(self.ir_mul_list)):
            start_idx_1, mul_1, start_idx_2, mul_2 = self.ir_mul_list[i]
            if mul_1 is not None:
                output.append(feature_1.narrow(-1, start_idx_1, mul_1))
            if mul_2 is not None:
                output.append(feature_2.narrow(-1, start_idx_2, mul_2))
        output = torch.cat(output, dim=-1)
        return output
    
    
    def __repr__(self):
        return '{}(irreps_1={}, irreps_2={})'.format(self.__class__.__name__, 
            self.irreps_1, self.irreps_2)

        

class NodeEmbeddingNetwork(torch.nn.Module):
    
    def __init__(self, irreps_node_embedding, max_atom_type=_MAX_ATOM_TYPE, bias=True):
        
        super().__init__()
        self.max_atom_type = max_atom_type
        self.irreps_node_embedding = o3.Irreps(irreps_node_embedding)
        self.atom_type_lin = LinearRS(o3.Irreps('{}x0e'.format(self.max_atom_type)), 
            self.irreps_node_embedding, bias=bias)
        self.atom_type_lin.tp.weight.data.mul_(self.max_atom_type ** 0.5)
        
        
    def forward(self, node_atom):
        '''
            `node_atom` is a LongTensor.
        '''
        
        node_atom_onehot = torch.nn.functional.one_hot(node_atom, self.max_atom_type).float()
        node_attr = node_atom_onehot
        
        node_embedding = self.atom_type_lin(node_atom_onehot)
        
        return node_embedding, node_attr, node_atom_onehot


class ScaledScatter(torch.nn.Module):
    def __init__(self, avg_aggregate_num):
        super().__init__()
        self.avg_aggregate_num = avg_aggregate_num + 0.0


    def forward(self, x, index, **kwargs):
        out = scatter(x, index, **kwargs)
        out = out.div(self.avg_aggregate_num ** 0.5)
        return out
    
    
    def extra_repr(self):
        return 'avg_aggregate_num={}'.format(self.avg_aggregate_num)
    
  

class FullyConnectedTensorProductRescaleSwishGate(FullyConnectedTensorProductRescale):
    
    def __init__(self, irreps_in1, irreps_in2, irreps_out,
        bias=True, rescale=True,
        internal_weights=None, shared_weights=None,
        normalization=None):
        
        irreps_scalars, irreps_gates, irreps_gated = irreps2gate(irreps_out)
        if irreps_gated.num_irreps == 0:
            gate = Activation(irreps_out, acts=[torch.nn.SiLU()])
        else:
            gate = Gate(
                irreps_scalars, [torch.nn.SiLU() for _, ir in irreps_scalars],  # scalar
                irreps_gates, [torch.sigmoid for _, ir in irreps_gates],  # gates (scalars)
                irreps_gated  # gated tensors
            )
        super().__init__(irreps_in1, irreps_in2, gate.irreps_in,
            bias=bias, rescale=rescale,
            internal_weights=internal_weights, shared_weights=shared_weights,
            normalization=normalization)
        self.gate = gate
        
        
    def forward(self, x, y, weight=None):
        out = self.forward_tp_rescale_bias(x, y, weight)
        out = self.gate(out)
        return out
    


def DepthwiseTensorProduct(irreps_node_input, irreps_edge_attr, irreps_node_output, 
    internal_weights=False, bias=True):
    '''
        The irreps of output is pre-determined. 
        `irreps_node_output` is used to get certain types of vectors.
    '''
    irreps_output = []
    instructions = []
    
    for i, (mul, ir_in) in enumerate(irreps_node_input):
        for j, (_, ir_edge) in enumerate(irreps_edge_attr):
            for ir_out in ir_in * ir_edge:
                if ir_out in irreps_node_output or ir_out == o3.Irrep(0, 1):
                    k = len(irreps_output)
                    irreps_output.append((mul, ir_out))
                    instructions.append((i, j, k, 'uvu', True))
        
    irreps_output = o3.Irreps(irreps_output)
    irreps_output, p, _ = sort_irreps_even_first(irreps_output) #irreps_output.sort()
    instructions = [(i_1, i_2, p[i_out], mode, train)
        for i_1, i_2, i_out, mode, train in instructions]
    tp = TensorProductRescale(irreps_node_input, irreps_edge_attr,
            irreps_output, instructions,
            internal_weights=internal_weights,
            shared_weights=internal_weights,
            bias=bias, rescale=_RESCALE)
    return tp    


class EdgeDegreeEmbeddingNetwork(torch.nn.Module):
    def __init__(self, irreps_node_embedding, irreps_edge_attr, fc_neurons, avg_aggregate_num):
        super().__init__()
        self.exp = LinearRS(o3.Irreps('1x0e'), irreps_node_embedding, 
            bias=_USE_BIAS, rescale=_RESCALE)
        self.dw = DepthwiseTensorProduct(irreps_node_embedding, 
            irreps_edge_attr, irreps_node_embedding, 
            internal_weights=False, bias=False)
        self.rad = RadialProfile(fc_neurons + [self.dw.tp.weight_numel])
        for (slice, slice_sqrt_k) in self.dw.slices_sqrt_k.values():
            self.rad.net[-1].weight.data[slice, :] *= slice_sqrt_k
            self.rad.offset.data[slice] *= slice_sqrt_k
        self.proj = LinearRS(self.dw.irreps_out.simplify(), irreps_node_embedding)
        self.scale_scatter = ScaledScatter(avg_aggregate_num)
        
    
    def forward(self, node_input, edge_attr, edge_scalars, edge_src, edge_dst, batch):
        node_features = torch.ones_like(node_input.narrow(1, 0, 1))
        node_features = self.exp(node_features)
        weight = self.rad(edge_scalars)
        edge_features = self.dw(node_features[edge_src], edge_attr, weight)
        edge_features = self.proj(edge_features)
        node_features = self.scale_scatter(edge_features, edge_dst, dim=0, 
            dim_size=node_features.shape[0])
        return node_features
    


class PathTransformer(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers):
        super(PathTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, model_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, 9, model_dim))  # 假设最大序列长度为 100
        encoder_layers = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.fc_out = nn.Linear(model_dim, model_dim)  # 假设输出是一个标量（属性）

    def forward(self, paths):
        # paths: (batch_size, max_seq_len, feature_dim)
        batch_size, max_seq_len, _ = paths.size()

        # 嵌入
        embedded = self.embedding(paths)  # (batch_size, max_seq_len, model_dim)
        embedded += self.positional_encoding[:, :max_seq_len, :]  # 添加位置编码

        # 转换为 (max_seq_len, batch_size, model_dim)
        embedded = embedded.permute(1, 0, 2)

        # 使用 Transformer Encoder
        transformer_out = self.transformer_encoder(embedded)  # (max_seq_len, batch_size, model_dim)

        # 取最后一个时间步的输出
        output = transformer_out[-1, :, :]  # (batch_size, model_dim)

        # 输出层
        final_output = self.fc_out(output)  # (batch_size, 1)
        return final_output
    

class LabelSeqTransformer(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers):
        super(LabelSeqTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, model_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, 9, model_dim))  # 假设最大序列长度为 100
        encoder_layers = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.fc_out = nn.Linear(model_dim, model_dim)  # 假设输出是一个标量（属性）

    def forward(self, paths, mask):
        
        batch_size, max_seq_len, _ = paths.size()

        # 找到有效的行：至少有一个位置为 True
        valid_rows = mask.any(dim=1)

        # 提取有效的 paths 和 mask
        valid_paths = paths[valid_rows]  # 仅保留有效的路径
        valid_mask = mask[valid_rows]   # 仅保留对应的有效掩码

        # 替换 NaN 值为 0（如需要）
        # valid_paths = torch.nan_to_num(valid_paths, nan=0.0)

        # 嵌入
        embedded = self.embedding(valid_paths)  
        embedded += self.positional_encoding[:, :max_seq_len, :]

        # 转换为 (max_seq_len, batch_size, model_dim)
        embedded = embedded.permute(1, 0, 2)
        
        # 使用 Transformer Encoder，应用掩码
        transformer_out = self.transformer_encoder(embedded, src_key_padding_mask=~valid_mask)

        # 提取最后一层的输出
        output = transformer_out[-1, :, :]
        final_output = self.fc_out(output)

        # 将结果插回到原来的形状
        result = torch.zeros(batch_size, final_output.size(1), device=final_output.device)
        result[valid_rows] = final_output

        return result





def Equiformer(irreps_in='5x0e', radius=5.0, num_basis=128, 
    atomref=None, task_mean=None, task_std=None, **kwargs):
    
    model = GraphAttentionTransformer(
        irreps_in=irreps_in,
        irreps_node_embedding='128x0e+64x1e+32x2e', num_layers=6,
        irreps_node_attr='1x0e', irreps_sh='1x0e+1x1e+1x2e',
        max_radius=radius,
        number_of_basis=num_basis, fc_neurons=[64, 64], 
        irreps_feature='512x0e',
        irreps_head='32x0e+16x1e+8x2e', num_heads=4, irreps_pre_attn=None,
        rescale_degree=False, nonlinear_message=True,
        irreps_mlp_mid='384x0e+192x1e+96x2e',
        norm_layer='layer',
        device = kwargs['device'],
        alpha_drop=0.2, proj_drop=0.0, out_drop=0.0, drop_path_rate=0.0,
        mean=task_mean, std=task_std, scale=None, atomref=atomref)
    return model

def get_mul_0(irreps):
    mul_0 = 0
    for mul, ir in irreps:
        if ir.l == 0 and ir.p == 1:
            mul_0 += mul
    return mul_0

class SmoothLeakyReLU(torch.nn.Module):
    def __init__(self, negative_slope=0.2):
        super().__init__()
        self.alpha = negative_slope
        
    
    def forward(self, x):
        x1 = ((1 + self.alpha) / 2) * x
        x2 = ((1 - self.alpha) / 2) * x * (2 * torch.sigmoid(x) - 1)
        return x1 + x2
    
    
    def extra_repr(self):
        return 'negative_slope={}'.format(self.alpha)
            



@compile_mode('script')
class GraphAttention(torch.nn.Module):
    '''
        1. Message = Alpha * Value
        2. Two Linear to merge src and dst -> Separable FCTP -> 0e + (0e+1e+...)
        3. 0e -> Activation -> Inner Product -> (Alpha)
        4. (0e+1e+...) -> (Value)
    '''
    def __init__(self,
        irreps_node_input, irreps_node_attr,
        irreps_edge_attr, irreps_node_output,
        fc_neurons,
        irreps_head, num_heads, irreps_pre_attn=None, 
        rescale_degree=False, nonlinear_message=False,
        alpha_drop=0.1, proj_drop=0.1):
        
        super().__init__()
        self.irreps_node_input = o3.Irreps(irreps_node_input)
        self.irreps_node_attr = o3.Irreps(irreps_node_attr)
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)
        self.irreps_node_output = o3.Irreps(irreps_node_output)
        self.irreps_pre_attn = self.irreps_node_input if irreps_pre_attn is None \
            else o3.Irreps(irreps_pre_attn)
        self.irreps_head = o3.Irreps(irreps_head)
        self.num_heads = num_heads
        self.rescale_degree = rescale_degree
        self.nonlinear_message = nonlinear_message
        
        # Merge src and dst
        self.merge_src = LinearRS(self.irreps_node_input, self.irreps_pre_attn, bias=True)
        self.merge_dst = LinearRS(self.irreps_node_input, self.irreps_pre_attn, bias=False)
        
        irreps_attn_heads = irreps_head * num_heads
        irreps_attn_heads, _, _ = sort_irreps_even_first(irreps_attn_heads) #irreps_attn_heads.sort()
        irreps_attn_heads = irreps_attn_heads.simplify() 
        mul_alpha = get_mul_0(irreps_attn_heads)
        mul_alpha_head = mul_alpha // num_heads
        irreps_alpha = o3.Irreps('{}x0e'.format(mul_alpha)) # for attention score
        irreps_attn_all = (irreps_alpha + irreps_attn_heads).simplify()
        
        self.sep_act = None
        if self.nonlinear_message:
            # Use an extra separable FCTP and Swish Gate for value
            self.sep_act = SeparableFCTP(self.irreps_pre_attn, 
                self.irreps_edge_attr, self.irreps_pre_attn, fc_neurons, 
                use_activation=True, norm_layer=None, internal_weights=False)
            self.sep_alpha = LinearRS(self.sep_act.dtp.irreps_out, irreps_alpha)
            self.sep_value = SeparableFCTP(self.irreps_pre_attn, 
                self.irreps_edge_attr, irreps_attn_heads, fc_neurons=None, 
                use_activation=False, norm_layer=None, internal_weights=True)
            self.vec2heads_alpha = Vec2AttnHeads(o3.Irreps('{}x0e'.format(mul_alpha_head)), 
                num_heads)
            self.vec2heads_value = Vec2AttnHeads(self.irreps_head, num_heads)
        else:
            self.sep = SeparableFCTP(self.irreps_pre_attn, 
                self.irreps_edge_attr, irreps_attn_all, fc_neurons, 
                use_activation=False, norm_layer=None)
            self.vec2heads = Vec2AttnHeads(
                (o3.Irreps('{}x0e'.format(mul_alpha_head)) + irreps_head).simplify(), 
                num_heads)
        
        self.alpha_act = Activation(o3.Irreps('{}x0e'.format(mul_alpha_head)), 
            [SmoothLeakyReLU(0.2)])
        self.heads2vec = AttnHeads2Vec(irreps_head)
        
        self.mul_alpha_head = mul_alpha_head
        self.alpha_dot = torch.nn.Parameter(torch.randn(1, num_heads, mul_alpha_head))
        torch_geometric.nn.inits.glorot(self.alpha_dot) # Following GATv2
        
        self.alpha_dropout = None
        if alpha_drop != 0.0:
            self.alpha_dropout = torch.nn.Dropout(alpha_drop)
        
        self.proj = LinearRS(irreps_attn_heads, self.irreps_node_output)
        self.proj_drop = None
        if proj_drop != 0.0:
            self.proj_drop = EquivariantDropout(self.irreps_node_input, 
                drop_prob=proj_drop)
        
        
    def forward(self, node_input, node_attr, edge_src, edge_dst, edge_attr, edge_scalars, 
        batch, **kwargs):
        
        message_src = self.merge_src(node_input)
        message_dst = self.merge_dst(node_input)
        message = message_src[edge_src] + message_dst[edge_dst]
        
        if self.nonlinear_message:          
            weight = self.sep_act.dtp_rad(edge_scalars)
            message = self.sep_act.dtp(message, edge_attr, weight)
            alpha = self.sep_alpha(message)
            alpha = self.vec2heads_alpha(alpha)
            value = self.sep_act.lin(message)
            value = self.sep_act.gate(value)
            value = self.sep_value(value, edge_attr=edge_attr, edge_scalars=edge_scalars)
            value = self.vec2heads_value(value)
        else:
            message = self.sep(message, edge_attr=edge_attr, edge_scalars=edge_scalars)
            message = self.vec2heads(message)
            head_dim_size = message.shape[-1]
            alpha = message.narrow(2, 0, self.mul_alpha_head)
            value = message.narrow(2, self.mul_alpha_head, (head_dim_size - self.mul_alpha_head))
        
        # inner product
        alpha = self.alpha_act(alpha)
        alpha = torch.einsum('bik, aik -> bi', alpha, self.alpha_dot)
        alpha = torch_geometric.utils.softmax(alpha, edge_dst)
        alpha = alpha.unsqueeze(-1)
        if self.alpha_dropout is not None:
            alpha = self.alpha_dropout(alpha)
        attn = value * alpha
        attn = scatter(attn, index=edge_dst, dim=0, dim_size=node_input.shape[0])
        attn = self.heads2vec(attn)
        
        if self.rescale_degree:
            degree = torch_geometric.utils.degree(edge_dst, 
                num_nodes=node_input.shape[0], dtype=node_input.dtype)
            degree = degree.view(-1, 1)
            attn = attn * degree
            
        node_output = self.proj(attn)
        
        if self.proj_drop is not None:
            node_output = self.proj_drop(node_output)
        
        return node_output
    
    
    def extra_repr(self):
        output_str = super(GraphAttention, self).extra_repr()
        output_str = output_str + 'rescale_degree={}, '.format(self.rescale_degree)
        return output_str
               
               
from instance_norm import EquivariantInstanceNorm
from graph_norm import EquivariantGraphNorm
from layer_norm import EquivariantLayerNormV2
from fast_layer_norm import EquivariantLayerNormFast

def get_norm_layer(norm_type):
    if norm_type == 'graph':
        return EquivariantGraphNorm
    elif norm_type == 'instance':
        return EquivariantInstanceNorm
    elif norm_type == 'layer':
        return EquivariantLayerNormV2
    elif norm_type == 'fast_layer':
        return EquivariantLayerNormFast
    elif norm_type is None:
        return None
    else:
        raise ValueError('Norm type {} not supported.'.format(norm_type))


@compile_mode('script')
class FeedForwardNetwork(torch.nn.Module):
    '''
        Use two (FCTP + Gate)
    '''
    def __init__(self,
        irreps_node_input, irreps_node_attr,
        irreps_node_output, irreps_mlp_mid=None,
        proj_drop=0.1):
        
        super().__init__()
        self.irreps_node_input = o3.Irreps(irreps_node_input)
        self.irreps_node_attr = o3.Irreps(irreps_node_attr)
        self.irreps_mlp_mid = o3.Irreps(irreps_mlp_mid) if irreps_mlp_mid is not None \
            else self.irreps_node_input
        self.irreps_node_output = o3.Irreps(irreps_node_output)
        
        self.fctp_1 = FullyConnectedTensorProductRescaleSwishGate(
            self.irreps_node_input, self.irreps_node_attr, self.irreps_mlp_mid, 
            bias=True, rescale=_RESCALE)
        self.fctp_2 = FullyConnectedTensorProductRescale(
            self.irreps_mlp_mid, self.irreps_node_attr, self.irreps_node_output, 
            bias=True, rescale=_RESCALE)
        
        self.proj_drop = None
        if proj_drop != 0.0:
            self.proj_drop = EquivariantDropout(self.irreps_node_output, 
                drop_prob=proj_drop)
            
        
    def forward(self, node_input, node_attr, **kwargs):
        node_output = self.fctp_1(node_input, node_attr)
        node_output = self.fctp_2(node_output, node_attr)
        if self.proj_drop is not None:
            node_output = self.proj_drop(node_output)
        return node_output
    

@compile_mode('script')
class TransBlock(torch.nn.Module):
    '''
        1. Layer Norm 1 -> GraphAttention -> Layer Norm 2 -> FeedForwardNetwork
        2. Use pre-norm architecture
    '''
    
    def __init__(self,
        irreps_node_input, irreps_node_attr,
        irreps_edge_attr, irreps_node_output,
        fc_neurons,
        irreps_head, num_heads, irreps_pre_attn=None, 
        rescale_degree=False, nonlinear_message=False,
        alpha_drop=0.1, proj_drop=0.1,
        drop_path_rate=0.0,
        irreps_mlp_mid=None,
        norm_layer='layer'):
        
        super().__init__()
        self.irreps_node_input = o3.Irreps(irreps_node_input)
        self.irreps_node_attr = o3.Irreps(irreps_node_attr)
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)
        self.irreps_node_output = o3.Irreps(irreps_node_output)
        self.irreps_pre_attn = self.irreps_node_input if irreps_pre_attn is None \
            else o3.Irreps(irreps_pre_attn)
        self.irreps_head = o3.Irreps(irreps_head)
        self.num_heads = num_heads
        self.rescale_degree = rescale_degree
        self.nonlinear_message = nonlinear_message
        self.irreps_mlp_mid = o3.Irreps(irreps_mlp_mid) if irreps_mlp_mid is not None \
            else self.irreps_node_input
        
        self.norm_1 = get_norm_layer(norm_layer)(self.irreps_node_input)
        self.ga = GraphAttention(irreps_node_input=self.irreps_node_input, 
            irreps_node_attr=self.irreps_node_attr,
            irreps_edge_attr=self.irreps_edge_attr, 
            irreps_node_output=self.irreps_node_input,
            fc_neurons=fc_neurons,
            irreps_head=self.irreps_head, 
            num_heads=self.num_heads, 
            irreps_pre_attn=self.irreps_pre_attn, 
            rescale_degree=self.rescale_degree, 
            nonlinear_message=self.nonlinear_message,
            alpha_drop=alpha_drop, 
            proj_drop=proj_drop)
        
        self.drop_path = GraphDropPath(drop_path_rate) if drop_path_rate > 0. else None
        
        self.norm_2 = get_norm_layer(norm_layer)(self.irreps_node_input)
        #self.concat_norm_output = ConcatIrrepsTensor(self.irreps_node_input, 
        #    self.irreps_node_input)
        self.ffn = FeedForwardNetwork(
            irreps_node_input=self.irreps_node_input, #self.concat_norm_output.irreps_out, 
            irreps_node_attr=self.irreps_node_attr,
            irreps_node_output=self.irreps_node_output, 
            irreps_mlp_mid=self.irreps_mlp_mid,
            proj_drop=proj_drop)
        self.ffn_shortcut = None
        if self.irreps_node_input != self.irreps_node_output:
            self.ffn_shortcut = FullyConnectedTensorProductRescale(
                self.irreps_node_input, self.irreps_node_attr, 
                self.irreps_node_output, 
                bias=True, rescale=_RESCALE)
            
            
    def forward(self, node_input, node_attr, edge_src, edge_dst, edge_attr, edge_scalars, 
        batch, **kwargs):
        
        node_output = node_input
        node_features = node_input
        node_features = self.norm_1(node_features, batch=batch)
        #norm_1_output = node_features
        node_features = self.ga(node_input=node_features, 
            node_attr=node_attr, 
            edge_src=edge_src, edge_dst=edge_dst, 
            edge_attr=edge_attr, edge_scalars=edge_scalars,
            batch=batch)
        
        if self.drop_path is not None:
            node_features = self.drop_path(node_features, batch)
        node_output = node_output + node_features
        
        node_features = node_output
        node_features = self.norm_2(node_features, batch=batch)
        #node_features = self.concat_norm_output(norm_1_output, node_features)
        node_features = self.ffn(node_features, node_attr)
        if self.ffn_shortcut is not None:
            node_output = self.ffn_shortcut(node_output, node_attr)
        
        if self.drop_path is not None:
            node_features = self.drop_path(node_features, batch)
        node_output = node_output + node_features
        
        return node_output
    


class GraphAttentionTransformer(torch.nn.Module):
    def __init__(self,
        irreps_in='5x0e',
        irreps_node_embedding='128x0e+64x1e+32x2e', num_layers=6,
        irreps_node_attr='1x0e', irreps_sh='1x0e+1x1e+1x2e',
        max_radius=5.0,
        number_of_basis=128, basis_type='gaussian', fc_neurons=[64, 64], 
        irreps_feature='512x0e',
        irreps_head='32x0e+16x1o+8x2e', num_heads=4, irreps_pre_attn=None,
        rescale_degree=False, nonlinear_message=False,
        irreps_mlp_mid='128x0e+64x1e+32x2e',
        norm_layer='layer',
        alpha_drop=0.2, proj_drop=0.0, out_drop=0.0,
        drop_path_rate=0.0, device=None,
        mean=None, std=None, scale=None, atomref=None):
        super().__init__()

        # import pdb;pdb.set_trace()
        self.device = device
        
        self.max_radius = max_radius
        self.number_of_basis = number_of_basis
        self.alpha_drop = alpha_drop
        self.proj_drop = proj_drop
        self.out_drop = out_drop
        self.drop_path_rate = drop_path_rate
        self.norm_layer = norm_layer
        self.task_mean = mean
        self.task_std = std
        self.scale = scale
        self.register_buffer('atomref', atomref)

        self.irreps_node_attr = o3.Irreps(irreps_node_attr)
        self.irreps_node_input = o3.Irreps(irreps_in)
        self.irreps_node_embedding = o3.Irreps(irreps_node_embedding)
        self.lmax = self.irreps_node_embedding.lmax
        self.irreps_feature = o3.Irreps(irreps_feature)
        self.num_layers = num_layers
        self.irreps_edge_attr = o3.Irreps(irreps_sh) if irreps_sh is not None \
            else o3.Irreps.spherical_harmonics(self.lmax)
        self.fc_neurons = [self.number_of_basis] + fc_neurons
        self.irreps_head = o3.Irreps(irreps_head)
        self.num_heads = num_heads
        self.irreps_pre_attn = irreps_pre_attn
        self.rescale_degree = rescale_degree
        self.nonlinear_message = nonlinear_message
        self.irreps_mlp_mid = o3.Irreps(irreps_mlp_mid)
        
        self.atom_embed = NodeEmbeddingNetwork(self.irreps_node_embedding, _MAX_ATOM_TYPE)
        self.basis_type = basis_type
        if self.basis_type == 'gaussian':
            self.rbf = GaussianRadialBasisLayer(self.number_of_basis, cutoff=self.max_radius)
        elif self.basis_type == 'bessel':
            self.rbf = RadialBasis(self.number_of_basis, cutoff=self.max_radius, 
                rbf={'name': 'spherical_bessel'})
        else:
            raise ValueError
        self.edge_deg_embed = EdgeDegreeEmbeddingNetwork(self.irreps_node_embedding, 
            self.irreps_edge_attr, self.fc_neurons, _AVG_DEGREE)
        
        self.blocks = torch.nn.ModuleList()
        self.build_blocks()
        self.layer_norm_ = nn.LayerNorm(512)
        self.norm = get_norm_layer(self.norm_layer)(self.irreps_feature)
        self.out_dropout = None
        if self.out_drop != 0.0:
            self.out_dropout = EquivariantDropout(self.irreps_feature, self.out_drop)
        
        self.output_feature_dim = 512*3+256+64*5
        # self.output_feature_dim = 512
        self.irreps_output_feature_dim = o3.Irreps(f'{self.output_feature_dim}'+'x0e') 
        # import pdb;pdb.set_trace()
        self.head1 = torch.nn.Sequential(
            LinearRS(self.irreps_output_feature_dim, self.irreps_feature, rescale=_RESCALE), 
            Activation(self.irreps_feature, acts=[torch.nn.SiLU()]),
            LinearRS(self.irreps_feature, o3.Irreps('1x0e'), rescale=_RESCALE)) 
        self.head2 = torch.nn.Sequential(
            LinearRS(self.irreps_feature, self.irreps_feature, rescale=_RESCALE), 
            Activation(self.irreps_feature, acts=[torch.nn.SiLU()]),
            LinearRS(self.irreps_feature, o3.Irreps('1x0e'), rescale=_RESCALE)) 
        self.scale_scatter = ScaledScatter(_AVG_NUM_NODES)
        
        self.apply(self._init_weights)
        # import pdb;pdb.set_trace()
        
        import pandas as pd
        self.smiles_to_graph_cache = {}
        qm9_csv_path = "/home/data1/lk/project/mol_generate/GDSS/data/qm9.csv"
        qm9_df = pd.read_csv(qm9_csv_path)
        self.smiles_to_index = {row['SMILES1']: row['Unnamed: 0'] for _, row in qm9_df.iterrows()}
        self.suppl = Chem.SDMolSupplier('/home/data1/lk/project/mol_property/ViSNet/dataset/raw/gdb9.sdf', removeHs=False,
                            sanitize=False)

        self.path_encoder = PathTransformer(input_dim=512, model_dim=256, num_heads=4, num_layers=2)
        self.path_encoder = self.path_encoder.to(device)  # Move model to GPU
        self.label_encoder = LabelSeqTransformer(input_dim=1, model_dim=64, num_heads=4, num_layers=2)
        self.label_encoder = self.label_encoder.to(device)  # Move model to GPU
        
            
    def build_blocks(self):
        for i in range(self.num_layers):
            if i != (self.num_layers - 1):
                irreps_block_output = self.irreps_node_embedding
            else:
                irreps_block_output = self.irreps_feature
            blk = TransBlock(irreps_node_input=self.irreps_node_embedding, 
                irreps_node_attr=self.irreps_node_attr,
                irreps_edge_attr=self.irreps_edge_attr, 
                irreps_node_output=irreps_block_output,
                fc_neurons=self.fc_neurons, 
                irreps_head=self.irreps_head, 
                num_heads=self.num_heads, 
                irreps_pre_attn=self.irreps_pre_attn, 
                rescale_degree=self.rescale_degree,
                nonlinear_message=self.nonlinear_message,
                alpha_drop=self.alpha_drop, 
                proj_drop=self.proj_drop,
                drop_path_rate=self.drop_path_rate,
                irreps_mlp_mid=self.irreps_mlp_mid,
                norm_layer=self.norm_layer)
            self.blocks.append(blk)
                      
    def _init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)
                                
    @torch.jit.ignore
    def no_weight_decay(self):
        no_wd_list = []
        named_parameters_list = [name for name, _ in self.named_parameters()]
        for module_name, module in self.named_modules():
            if (isinstance(module, torch.nn.Linear) 
                or isinstance(module, torch.nn.LayerNorm)):
                for parameter_name, _ in module.named_parameters():
                    if isinstance(module, torch.nn.Linear) and 'weight' in parameter_name:
                        continue
                    global_parameter_name = module_name + '.' + parameter_name
                    assert global_parameter_name in named_parameters_list
                    no_wd_list.append(global_parameter_name)
                    
        return set(no_wd_list)

    def forward_with_path(self, graph1, graph2, paths, paths_labels, paths_labels_masks):
        
        # for temp in graph1.smiles:
        #     graph_ = smiles_to_graph_xyz_sch(self.smiles_to_index[temp], self.suppl)  # 将 SMILES 转换为图
        #     import pdb;pdb.set_trace()
        #     self.smiles_to_graph_cache[temp] = graph_  # 缓存转换后的图
        
        
        # import pdb; pdb.set_trace()
        # 对第一个图进行卷积
        x1 = self.drug_encoder(f_in = graph1.x, pos = graph1.pos, batch = graph1.batch, node_atom = graph1.z)
    
        # 对第二个图进行卷积
        x2 = self.drug_encoder(f_in = graph2.x, pos = graph2.pos, batch = graph2.batch, node_atom = graph2.z)

        # 融合两个图的特征
        x12 = x1 - x2  # 使用差值进行融合
        # x21 = -x1 + x2  # 使用差值进行融合
        # 将路径信息编码为特征向量
        
        path_features, path_labels_features = self.encode_paths(paths, paths_labels, paths_labels_masks)
        # path_features, path_labels_features = self.encode_paths(paths_slice, paths_labels_slice, paths_labels_masks_slice)

       
        # x = torch.cat([x12,x21],dim=1)
        x_one = torch.cat([x12, x1, x2, path_features, path_labels_features], dim=1)  # 将路径特征拼接
        # x_two = torch.cat([x1, path_features, path_labels_features], dim=1)
        # x_two = x1
        # import pdb; pdb.set_trace()
        # x_one = self.fc1_1(x_one)
        # x_one = torch.tanh(x_one)
        # x_one = self.dropout(x_one)
        # x_one = self.fc2_1(x_one)
        # x_one = torch.tanh(x_one)
        # x_one = self.fc3_1(x_one)
        
        # x_two = self.fc1_2(x_two)
        # x_two = torch.relu(x_two)
        # x_two = self.dropout(x_two)
        # x_two = self.fc2_2(x_two)
        # x_two = torch.relu(x_two)
        # x_two = self.fc3_2(x_two)
        # return torch.cat([x_one,x_two],dim=1)

        
        outputs_one = self.head1(x_one)
        outputs_two = self.head2(x1)
        # x_one = self.fc1_1(x_one)
        # x_one = torch.tanh(x_one)
        # x_one = self.dropout(x_one)
        # x_one = self.fc2_1(x_one)
        # x_one = torch.tanh(x_one)
        # x_one = self.fc3_1(x_one)
        
        # x_two = self.fc1_2(x_two)
        # x_two = torch.tanh(x_two)
        # x_two = self.dropout(x_two)
        # x_two = self.fc2_2(x_two)
        # x_two = torch.tanh(x_two)
        # x_two = self.fc3_2(x_two)
        return torch.cat([outputs_one,outputs_two],dim=1)
    
    def forward(self, graph1, graph2, paths, paths_labels, paths_labels_masks):
        
        # for temp in graph1.smiles:
        #     graph_ = smiles_to_graph_xyz_sch(self.smiles_to_index[temp], self.suppl)  # 将 SMILES 转换为图
        #     import pdb;pdb.set_trace()
        #     self.smiles_to_graph_cache[temp] = graph_  # 缓存转换后的图
        
        
        # import pdb; pdb.set_trace()
        # 对第一个图进行卷积
        x1 = self.drug_encoder(f_in = graph1.x, pos = graph1.pos, batch = graph1.batch, node_atom = graph1.z)
    
        # 对第二个图进行卷积
        # x2 = self.drug_encoder(f_in = graph2.x, pos = graph2.pos, batch = graph2.batch, node_atom = graph2.z)

        # 融合两个图的特征
        # x12 = x1 - x2  # 使用差值进行融合
        # x21 = -x1 + x2  # 使用差值进行融合
        # 将路径信息编码为特征向量
        
        # path_features, path_labels_features = self.encode_paths(paths, paths_labels, paths_labels_masks)
        # path_features, path_labels_features = self.encode_paths(paths_slice, paths_labels_slice, paths_labels_masks_slice)

       
        # x = torch.cat([x12,x21],dim=1)
        # x_one = torch.cat([x12, x1, x2, path_features, path_labels_features], dim=1)  # 将路径特征拼接
        # x_two = torch.cat([x1, path_features, path_labels_features], dim=1)
        # x_two = x1
        # import pdb; pdb.set_trace()
        # x_one = self.fc1_1(x_one)
        # x_one = torch.tanh(x_one)
        # x_one = self.dropout(x_one)
        # x_one = self.fc2_1(x_one)
        # x_one = torch.tanh(x_one)
        # x_one = self.fc3_1(x_one)
        
        # x_two = self.fc1_2(x_two)
        # x_two = torch.relu(x_two)
        # x_two = self.dropout(x_two)
        # x_two = self.fc2_2(x_two)
        # x_two = torch.relu(x_two)
        # x_two = self.fc3_2(x_two)
        # return torch.cat([x_one,x_two],dim=1)
        
        # outputs_one = self.head1(x_one)
        x1 = self.head2(x1)
        # x_one = self.fc1_1(x_one)
        # x_one = torch.tanh(x_one)
        # x_one = self.dropout(x_one)
        # x_one = self.fc2_1(x_one)
        # x_one = torch.tanh(x_one)
        # x_one = self.fc3_1(x_one)
        
        # x_two = self.fc1_2(x_two)
        # x_two = torch.tanh(x_two)
        # x_two = self.dropout(x_two)
        # x_two = self.fc2_2(x_two)
        # x_two = torch.tanh(x_two)
        # x_two = self.fc3_2(x_two)
        return x1
    
    def encode_paths(self, paths, paths_labels, paths_labels_masks):

        # paths = torch.nan_to_num(paths, nan=0.0)
        # paths_labels = torch.nan_to_num(paths_labels, nan=0.0)
        
        # 编码路径信息为特征向量
        encoded_features = []
        
        i_length, j_length, k_length = paths_labels.size()
        
        # 用于缓存已转换的图
        
        all_graphs = []  # 用于存储所有图以进行批处理
        all_indices = []  # 存储每个图对应的路径索引
        paths_list = []
        
        for i_idx in range(i_length):
        
            
            for j_idx in range(j_length):
                if paths[j_idx][0][i_idx] == 'PAD':
                    break
                
                for k_idx in range(k_length):
                    temp = paths[j_idx][k_idx][i_idx]
                    if temp == 'PAD':  # 提前结束条件
                        break
                    
                    # 检查缓存字典，避免重复计算
                    if temp in self.smiles_to_graph_cache:
                        graph = self.smiles_to_graph_cache[temp]
                    else:
                        # qm7
                        # graph = smiles_to_graph_xyz_equiformer_qm7(temp)  # 将 SMILES 转换为图
                        
                        # qm9
                        graph = smiles_to_graph_xyz_equiformer(self.smiles_to_index[temp], self.suppl)  # 将 SMILES 转换为图
                        
                        self.smiles_to_graph_cache[temp] = graph  # 缓存转换后的图
                    
                    all_graphs.append(graph)
                    all_indices.append((i_idx, j_idx, k_idx))  # 记录索引

        # 处理所有图的批次编码
        if all_graphs:
            batch = Batch.from_data_list(all_graphs).to(self.device)
            drug_features_list = []
            batch_size = 64  # 设置批次大小
            # self.drug_encoder(f_in = graph1.x, pos = graph1.pos, batch = graph1.batch, node_atom = graph1.z)
            # import pdb;pdb.set_trace()
            # 分批处理图数据
            for start_idx in range(0, len(all_graphs), batch_size):
                sub_graphs = all_graphs[start_idx:start_idx + batch_size]

                # 创建小批次并移动到指定设备
                batch = Batch.from_data_list(sub_graphs).to(self.device)

                with torch.no_grad():
                    drug_features_batch = self.drug_encoder(f_in = batch.x, pos = batch.pos, batch = batch.batch, node_atom = batch.z)

                # 存储当前批次结果
                drug_features_list.append(drug_features_batch)
                
                
            drug_features = torch.cat(drug_features_list, dim=0)
            # drug_features = self.drug_encoder(f_in = batch.x, pos = batch.pos, batch = batch.batch, node_atom = batch.z)
            drug_features = self.layer_norm_(drug_features)
 
        
            # 为每个路径特征分配相应的药物特征
            path_features = [[] for _ in range(i_length)]  # 创建一个长度为 i_length 的列表，用于存储每个药物的特征

            for index, feature in zip(all_indices, drug_features):
                # index[0] 是 i_idx，代表当前药物的索引
                i_idx = index[0]  # 当前药物索引
                j_idx = index[1]  # 当前路径索引

                # 确保 path_features 的双重嵌套结构
                if len(path_features[i_idx]) <= j_idx:
                    # 如果当前 i_idx 的列表长度小于 j_idx，进行填充
                    path_features[i_idx].extend([[]] * (j_idx + 1 - len(path_features[i_idx])))

                # 将特征添加到对应的 i_idx 和 j_idx
                path_features[i_idx][j_idx].append(feature)  # 将当前特征添加到对应的路径中
                

            # 计算每个药物的平均路径特征
            for path_feature in path_features:
                path_feature = [ pad_sequence(vec, batch_first=True) for vec in path_feature]
                path_vectors_for_one_drug = pad_sequence(path_feature, batch_first=True)
                # import pdb;pdb.set_trace()
                path_embedding = self.path_encoder(path_vectors_for_one_drug)
                avg_path_feature = torch.mean(path_embedding, dim=0)

                # if avg_path_feature.dim()==1:
                #     avg_path_feature = avg_path_feature.expand(1,1,512)
                paths_list.append(avg_path_feature)
        

        # paths_labels_ = paths_labels.view(paths_labels.size(0)*paths_labels.size(1), -1, 1)
        # paths_labels_masks = paths_labels_masks.view(paths_labels.size(0)*paths_labels.size(1),-1).bool()
        paths_labels_ = paths_labels.reshape(paths_labels.size(0)*paths_labels.size(1), -1, 1)
        paths_labels_masks = paths_labels_masks.reshape(paths_labels.size(0)*paths_labels.size(1),-1).bool()

        paths_labels_features =  self.label_encoder(paths_labels_.to(self.device),paths_labels_masks.to(self.device))
     
        # paths_labels_features = torch.mean(paths_labels_features.reshape(paths_labels.size(0), paths_labels.size(1), -1),dim=1)
        paths_labels_features = paths_labels_features.reshape(paths_labels.size(0), -1)
 
        return torch.stack(paths_list).to(self.device), paths_labels_features



    def drug_encoder(self, f_in, pos, batch, node_atom, **kwargs) -> torch.Tensor:


        
        
        
        edge_src, edge_dst = radius_graph(pos, r=self.max_radius, batch=batch,
            max_num_neighbors=1000)
        edge_vec = pos.index_select(0, edge_src) - pos.index_select(0, edge_dst)
        edge_sh = o3.spherical_harmonics(l=self.irreps_edge_attr,
            x=edge_vec, normalize=True, normalization='component')

        
        # Qm9
        node_atom = node_atom.new_tensor([-1, 0, -1, -1, -1, -1, 1, 2, 3, 4])[node_atom]
        
        # Qm7
        # node_atom = node_atom.new_tensor([-1, 0, -1, -1, -1, -1, 1, 2, 3, -1, -1, -1, -1, -1, -1, -1, 4])[node_atom]
        
        atom_embedding, atom_attr, atom_onehot = self.atom_embed(node_atom)
        edge_length = edge_vec.norm(dim=1)
        #edge_length_embedding = sin_pos_embedding(x=edge_length, 
        #    start=0.0, end=self.max_radius, number=self.number_of_basis, 
        #    cutoff=False)
        edge_length_embedding = self.rbf(edge_length)
        edge_degree_embedding = self.edge_deg_embed(atom_embedding, edge_sh, 
            edge_length_embedding, edge_src, edge_dst, batch)
        node_features = atom_embedding + edge_degree_embedding
        node_attr = torch.ones_like(node_features.narrow(1, 0, 1))
        
        for blk in self.blocks:
            node_features = blk(node_input=node_features, node_attr=node_attr, 
                edge_src=edge_src, edge_dst=edge_dst, edge_attr=edge_sh, 
                edge_scalars=edge_length_embedding, 
                batch=batch)
        
   
        
        outputs = self.norm(node_features, batch=batch)
        if self.out_dropout is not None:
            outputs = self.out_dropout(outputs)
        # outputs = self.head(node_features)
        outputs = self.scale_scatter(outputs, batch, dim=0)
        
        if self.scale is not None:
            outputs = self.scale * outputs

        return outputs
