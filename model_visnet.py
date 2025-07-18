import torch
import torch.nn as nn
import torch_geometric.nn as gnn
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from tools import *
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from torch_geometric.data import Batch
import torch.nn.init as init
import pandas as pd
from visnet_block import ViSNetBlock
import visnet_output_modules

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


class ViSNet(nn.Module):
    def __init__(
        self,
        representation_model,
        output_model,
        prior_model=None,
        reduce_op="add",
        mean=None,
        std=None,
        derivative=False,
    ):
        super(ViSNet, self).__init__()
        self.representation_model = representation_model
        self.output_model = output_model

        self.prior_model = prior_model
        if not output_model.allow_prior_model and prior_model is not None:
            self.prior_model = None
            rank_zero_warn(
                "Prior model was given but the output model does "
                "not allow prior models. Dropping the prior model."
            )

        self.reduce_op = reduce_op
        self.derivative = derivative

        mean = torch.scalar_tensor(0) if mean is None else mean
        self.register_buffer("mean", mean)
        std = torch.scalar_tensor(1) if std is None else std
        self.register_buffer("std", std)

        self.reset_parameters()

    def reset_parameters(self):
        self.representation_model.reset_parameters()
        self.output_model.reset_parameters()
        if self.prior_model is not None:
            self.prior_model.reset_parameters()

    def forward(self, data: Data):
        
        if self.derivative:
            data.pos.requires_grad_(True)

        x, v = self.representation_model(data)
        x = self.output_model.pre_reduce(x, v, data.z, data.pos, data.batch)
        x = x * self.std

        if self.prior_model is not None:
            x = self.prior_model(x, data.z)

        out = scatter(x, data.batch, dim=0, reduce=self.reduce_op)
        out = self.output_model.post_reduce(out)
        
        out = out + self.mean

        # compute gradients with respect to coordinates
        if self.derivative:
            grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(out)]
            dy = grad(
                [out],
                [data.pos],
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True,
            )[0]
            if dy is None:
                raise RuntimeError("Autograd returned None for the force prediction.")
            return out, -dy
        return out, None

class RegressionHead(nn.Module):
    """
    A regression head with 3 linear layers and SiLU activation function.
    """
    def __init__(self, input_dim=835, hidden_dim=512, output_dim=1, activation="silu"):
        super(RegressionHead, self).__init__()
        
        # 获取激活函数类
        act_class = nn.SiLU if activation == "silu" else nn.ReLU
        
        # 定义 3 层 Linear 网络
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            act_class(),
            nn.Linear(hidden_dim, hidden_dim),
            act_class(),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def reset_parameters(self):
        """
        Initialize weights of linear layers using Xavier initialization.
        """
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        """
        Forward pass through the regression head.
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, input_dim]
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, output_dim]
        """
        return self.net(x)
    
class GNNModel(nn.Module):
    def __init__(self, device):
        super(GNNModel, self).__init__()

        self.dropout = nn.Dropout(0.3)
        
        self.path_encoder = PathTransformer(input_dim=512, model_dim=512, num_heads=4, num_layers=2)
        self.path_encoder = self.path_encoder.to(device)  # Move model to GPU
        self.label_encoder = LabelSeqTransformer(input_dim=1, model_dim=64, num_heads=4, num_layers=2)
        self.label_encoder = self.label_encoder.to(device)  # Move model to GPU
        
        self.device = device

        self.smiles_to_graph_cache = {}
        qm9_csv_path = "/home/data1/lk/project/mol_generate/GDSS/data/qm9.csv"
        qm9_df = pd.read_csv(qm9_csv_path)
        self.smiles_to_index = {row['SMILES1']: row['Unnamed: 0'] for _, row in qm9_df.iterrows()}
        self.suppl = Chem.SDMolSupplier('/home/data1/lk/project/mol_property/ViSNet/dataset/raw/gdb9.sdf', removeHs=False,
                            sanitize=False)
        
        # visnet_args = dict(
        #     lmax=args["lmax"],
        #     vecnorm_type=args["vecnorm_type"],
        #     trainable_vecnorm=args["trainable_vecnorm"],
        #     num_heads=args["num_heads"],
        #     num_layers=args["num_layers"],
        #     hidden_channels=args["embedding_dimension"],
        #     num_rbf=args["num_rbf"],
        #     rbf_type=args["rbf_type"],
        #     trainable_rbf=args["trainable_rbf"],
        #     activation=args["activation"],
        #     attn_activation=args["attn_activation"],
        #     max_z=args["max_z"],
        #     cutoff=args["cutoff"],
        #     max_num_neighbors=args["max_num_neighbors"],
        #     vertex_type=args["vertex_type"],
        # )
        visnet_args = dict(
            lmax=2,
            vecnorm_type="max_min",
            trainable_vecnorm=False,
            num_heads=8,
            num_layers=9,
            hidden_channels=512,
            num_rbf=64,
            rbf_type="expnorm",
            trainable_rbf=False,
            activation="silu",
            attn_activation="silu",
            max_z=100,
            cutoff=5,
            max_num_neighbors=32,
            vertex_type=None,
        )
        # self.representation_model = ViSNetBlock(**visnet_args)
        self.representation_model = ViSNetBlock(**visnet_args)
        output_prefix = "Equivariant"
        # self.output_model = getattr(visnet_output_modules, output_prefix + args["output_model"])(args["embedding_dimension"], args["activation"])
        self.output_model = getattr(visnet_output_modules, output_prefix + 'Scalar')(512, 'silu')
        # self.output_model_ = getattr(visnet_output_modules, output_prefix + 'Scalar')(512, 'silu')
        self.evo_reg_head = RegressionHead(input_dim=2368, hidden_dim=512, output_dim=1)

        self.reset_parameters()
        
        # create output network


    def reset_parameters(self):
        self.representation_model.reset_parameters()
        self.output_model.reset_parameters()
        # self.output_model_.reset_parameters()
        self.evo_reg_head.reset_parameters()
    
    def freeze(self):
        # 冻结 drug_encoder 的所有参数
        for param in self.representation_model.parameters():
            param.requires_grad = False

    
    def drug_encoder(self, graph):
        x, v = self.representation_model(graph)
        return x, v
    
    
    def forward_with_path(self, graph1, graph2, paths, paths_labels, paths_labels_masks):
        
        x1, v1 = self.drug_encoder(graph1)
        x2, v2 = self.drug_encoder(graph2)
        
        # import pdb;pdb.set_trace()
        
        
        scatter_x1 = scatter(x1, graph1.batch, dim=0, reduce="add")

        scatter_x2 = scatter(x2, graph2.batch, dim=0, reduce="add")
        

        x12 = scatter_x1 - scatter_x2  # 使用差值进行融合
        path_features, path_labels_features = self.encode_paths(paths, paths_labels, paths_labels_masks)


        # 融合两个图的特征
        
        # x21 = -x1 + x2  # 使用差值进行融合
        # 将路径信息编码为特征向量
        

       
        # x = torch.cat([x12,x21],dim=1)
        x_one = torch.cat([x12, scatter_x1, scatter_x2, path_features, path_labels_features], dim=1)  # 将路径特征拼接
        
        x_two = self.output_model.pre_reduce(x1, v1, graph1.z, graph1.pos, graph1.batch)
        x_two = scatter(x_two, graph1.batch, dim=0, reduce="add")
        x_two = self.output_model.post_reduce(x_two)
        # x_2 = self.output_model.pre_reduce(x2, v2, graph2.z, graph2.pos, graph2.batch)
        # scatter_x2_dim1 = scatter(x_2, graph2.batch, dim=0, reduce="add")
        
        
        
        
        # x_two = torch.cat([x1, path_features, path_labels_features], dim=1)
        x_one = self.evo_reg_head(x_one)
        
        
        return torch.cat([x_one,x_two],dim=1)
    
    
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
                        graph = smiles_to_graph_xyz(self.smiles_to_index[temp], self.suppl)  # 将 SMILES 转换为图
                        self.smiles_to_graph_cache[temp] = graph  # 缓存转换后的图
                    
                    all_graphs.append(graph)
                    all_indices.append((i_idx, j_idx, k_idx))  # 记录索引

        # 处理所有图的批次编码
        if all_graphs:
            drug_features_list = []
            batch_size = 512  # 设置批次大小

            # 分批处理图数据
            for start_idx in range(0, len(all_graphs), batch_size):
                sub_graphs = all_graphs[start_idx:start_idx + batch_size]

                # 创建小批次并移动到指定设备
                batch = Batch.from_data_list(sub_graphs).to(self.device)

                with torch.no_grad():
                    drug_features_batch = self.drug_encoder(batch)[0]

                # 存储当前批次结果
                drug_features_list.append(drug_features_batch)

                # # 显式释放临时变量以优化显存
                # del batch, drug_features_batch
                # torch.cuda.empty_cache()  # 手动释放未使用的显存

            # 合并所有批次的结果
            drug_features = torch.cat(drug_features_list, dim=0)
                                
            
            
            
            
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
                path_embedding = self.path_encoder(path_vectors_for_one_drug.to(self.device))
                avg_path_feature = torch.mean(path_embedding, dim=0)

                # if avg_path_feature.dim()==1:
                #     avg_path_feature = avg_path_feature.expand(1,1,512)
                paths_list.append(avg_path_feature)
        
     
        paths_labels_ = paths_labels.view(paths_labels.size(0)*paths_labels.size(1), -1, 1)
        paths_labels_masks = paths_labels_masks.view(paths_labels.size(0)*paths_labels.size(1),-1).bool()
        paths_labels_features =  self.label_encoder(paths_labels_.to(self.device),paths_labels_masks.to(self.device))
     
        # paths_labels_features = torch.mean(paths_labels_features.reshape(paths_labels.size(0), paths_labels.size(1), -1),dim=1)
        paths_labels_features = paths_labels_features.reshape(paths_labels.size(0), -1)
 
        return torch.stack(paths_list).to(self.device), paths_labels_features



    def forward(self, graph1):
        
        x1, v1 = self.drug_encoder(graph1)
        # x2, v2 = self.drug_encoder(graph2)
        
        # import pdb;pdb.set_trace()
        
        
        scatter_x1 = scatter(x1, graph1.batch, dim=0, reduce="add")

        # scatter_x2 = scatter(x2, graph2.batch, dim=0, reduce="add")
        

        # x12 = scatter_x1 - scatter_x2  # 使用差值进行融合
        # path_features, path_labels_features = self.encode_paths(paths, paths_labels, paths_labels_masks)


        # 融合两个图的特征
        
        # x21 = -x1 + x2  # 使用差值进行融合
        # 将路径信息编码为特征向量
        

       
        # x = torch.cat([x12,x21],dim=1)
        # x_one = torch.cat([x12, scatter_x1, scatter_x2, path_features, path_labels_features], dim=1)  # 将路径特征拼接
        
        x_two = self.output_model.pre_reduce(x1, v1, graph1.z, graph1.pos, graph1.batch)
        x_two = scatter(x_two, graph1.batch, dim=0, reduce="add")
        x_two = self.output_model.post_reduce(x_two)
        # x_2 = self.output_model.pre_reduce(x2, v2, graph2.z, graph2.pos, graph2.batch)
        # scatter_x2_dim1 = scatter(x_2, graph2.batch, dim=0, reduce="add")
        
        
        
        
        # x_two = torch.cat([x1, path_features, path_labels_features], dim=1)
        # x_one = self.evo_reg_head(x_one)
        
        return x_two
        # return torch.cat([x_one,x_two],dim=1)
    