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
        transformer_out = self.transformer_encoder(embedded) 

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

# --------------------------------graphsage-----------------------------
from torch_geometric.nn import GraphSAGE
# --------------------------------graphsage-----------------------------
    
class GNNModel(nn.Module):
    def __init__(self, device, is_r2=False):
        super(GNNModel, self).__init__()
        self.drug_model = GraphSAGE(
            in_channels=90,
            hidden_channels=256,
            num_layers=3,
            out_channels=256, 
            dropout=0.3,
        )

        self.fc1_1 = nn.Linear(512*4+64+256, 512)
        
        self.fc2_1 = nn.Linear(512, 256)
        self.fc3_1 = nn.Linear(256, 1)
        
        self.fc1_2 = nn.Linear(512, 512)
        self.fc2_2 = nn.Linear(512, 256)
        self.fc3_2 = nn.Linear(256, 1)
        
        self.dropout = nn.Dropout(0.3)
        
        self.path_encoder = PathTransformer(input_dim=512, model_dim=512, num_heads=4, num_layers=2)
        self.path_encoder = self.path_encoder.to(device)  # Move model to GPU
        self.label_encoder = LabelSeqTransformer(input_dim=1, model_dim=64, num_heads=4, num_layers=2)
        self.label_encoder = self.label_encoder.to(device)  # Move model to GPU
        
        self.device = device
        
        import pandas as pd
        self.device = device
        self.smiles_to_graph_cache = {}
        self.softplus = nn.Softplus()

        if is_r2:
            self.layer_norm = nn.LayerNorm(512)
            self.is_r2 = True
        else:
            self.is_r2 = False
        
    def drug_encoder(self, graph):
        # 对第一个图进行卷积
        x = self.conv1(graph.x, graph.edge_index)
        x = torch.relu(x)
        x = self.conv2(x, graph.edge_index)
        x = torch.relu(x)
        x = self.conv3(x, graph.edge_index)
        x = torch.relu(x)
        x = self.conv4(x, graph.edge_index)
        x = torch.relu(x)
        x = torch.cat([gmp(x, graph.batch), gap(x, graph.batch)], dim=1)
        return x
    
    # def forward_origin(self, graph):
    #     return self.drug_model(graph)
    
    def forward_origin(self, graph1):
        # 对第一个图进行卷积
        x1 = self.conv1(graph1.x, graph1.edge_index)
        x1 = torch.relu(x1)
        x1 = self.conv2(x1, graph1.edge_index)
        x1 = torch.relu(x1)
        x1 = self.conv3(x1, graph1.edge_index)
        x1 = torch.relu(x1)
        x1 = self.conv4(x1, graph1.edge_index)
        x1 = torch.relu(x1)
        x1 = torch.cat([gmp(x1, graph1.batch), gap(x1, graph1.batch)], dim=1)

        x_two = self.fc1_2(x1)
        x_two = torch.relu(x_two)
        x_two = self.dropout(x_two)
        x_two = self.fc2_2(x_two)
        x_two = torch.relu(x_two)
        x_two = self.fc3_2(x_two)
        return x_two
        
        
    def forward(self, graph1, graph2):
        x1 = self.drug_encoder(graph1)
        x2 = self.drug_encoder(graph2)

        # 融合两个图的特征
        x12 = x1 - x2  # 使用差值进行融合
        x21 = -x1 + x2  # 使用差值进行融合
        x = torch.cat([x12,x21],dim=1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        x = torch.tanh(x)
        return x
    
    def forward_with_path(self, graph1, graph2, paths, paths_labels, paths_labels_masks):

        
        # 对第一个图进行卷积
        x1 = self.drug_encoder(graph1)
    
        # 对第二个图进行卷积
        x2 = self.drug_encoder(graph2)

        # 融合两个图的特征
        x12 = x1 - x2  # 使用差值进行融合
        # x21 = -x1 + x2  # 使用差值进行融合
        # 将路径信息编码为特征向量
        
        path_features, path_labels_features = self.encode_paths(paths, paths_labels, paths_labels_masks)
        # path_features, path_labels_features = self.encode_paths(paths_slice, paths_labels_slice, paths_labels_masks_slice)

       
        # x = torch.cat([x12,x21],dim=1)
        x_one = torch.cat([x12, x1, x2, path_features, path_labels_features], dim=1)  # 将路径特征拼接
        # x_two = torch.cat([x1, path_features, path_labels_features], dim=1)
        x_two = x1

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
        
        x_one = self.fc1_1(x_one)
        x_one = torch.tanh(x_one)
        x_one = self.dropout(x_one)
        x_one = self.fc2_1(x_one)
        x_one = torch.tanh(x_one)
        x_one = self.fc3_1(x_one)
        
        x_two = self.fc1_2(x_two)
        x_two = torch.tanh(x_two)
        x_two = self.dropout(x_two)
        x_two = self.fc2_2(x_two)
        x_two = torch.tanh(x_two)
        x_two = self.fc3_2(x_two)
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
                        graph = smiles_to_graph(temp)  # 将 SMILES 转换为图
                        self.smiles_to_graph_cache[temp] = graph  # 缓存转换后的图
                    
                    all_graphs.append(graph)
                    all_indices.append((i_idx, j_idx, k_idx))  # 记录索引

        # 处理所有图的批次编码
        if all_graphs:
            batch = Batch.from_data_list(all_graphs).to(self.device)
            
            
            drug_features = self.drug_encoder(batch)
            if self.is_r2:
                drug_features = self.layer_norm(drug_features)
 
        
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


    def get_features(self, graph1, graph2, paths, paths_labels, paths_labels_masks):
        
        
        # 对第一个图进行卷积
        x1 = self.conv1(graph1.x, graph1.edge_index)
        x1 = torch.relu(x1)
        x1 = self.conv2(x1, graph1.edge_index)
        x1 = torch.relu(x1)
        x1 = self.conv3(x1, graph1.edge_index)
        x1 = torch.relu(x1)
        x1 = self.conv4(x1, graph1.edge_index)
        x1 = torch.relu(x1)
        x1 = torch.cat([gmp(x1, graph1.batch), gap(x1, graph1.batch)], dim=1)
       
        # 对第二个图进行卷积
        x2 = self.conv1(graph2.x, graph2.edge_index)
        x2 = torch.relu(x2)
        x2 = self.conv2(x2, graph2.edge_index)
        x2 = torch.relu(x2)
        x2 = self.conv3(x2, graph2.edge_index)
        x2 = torch.relu(x2)
        x2 = self.conv4(x2, graph2.edge_index)
        x2 = torch.relu(x2)
        x2 = torch.cat([gmp(x2, graph2.batch), gap(x2, graph2.batch)], dim=1)

        # 融合两个图的特征
        x12 = x1 - x2  # 使用差值进行融合
        # x21 = -x1 + x2  # 使用差值进行融合
        # 将路径信息编码为特征向量
        
        path_features, path_labels_features = self.encode_paths(paths, paths_labels, paths_labels_masks)
        # path_features, path_labels_features = self.encode_paths(paths_slice, paths_labels_slice, paths_labels_masks_slice)

       
        # x = torch.cat([x12,x21],dim=1)
        x_one = torch.cat([x12, x1, x2, path_features, path_labels_features], dim=1)  # 将路径特征拼接
        # x_two = torch.cat([x1, path_features, path_labels_features], dim=1)
        x_one = self.fc1_1(x_one)
        x_one = torch.tanh(x_one)
        x_one = self.dropout(x_one)
        x_one = self.fc2_1(x_one)
        return x_one
    
    
    # def encode_paths(self, paths, paths_labels):
    #     # 编码路径信息为特征向量
    #     encoded_features = []
        
    #     i_length, j_length, k_length = paths_labels.size()
        
    #     # 用于缓存已转换的图
    #     smiles_to_graph_cache = {}

    #     for i_idx in tqdm(range(i_length)):
    #         path_for_one_drug = []
            
    #         all_path_features = []
    #         graphs = []  # 用于存储所有图以进行批处理

    #         for j_idx in range(j_length):
    #             if paths[j_idx][0][i_idx] == 'PAD':
    #                 break
                
    #             path_features = []
    #             for k_idx in range(k_length):
    #                 temp = paths[j_idx][k_idx][i_idx]
    #                 if temp == 'C':  # 提前结束条件
    #                     break
                    
    #                 # 检查缓存字典，避免重复计算
    #                 if temp in smiles_to_graph_cache:
    #                     graph = smiles_to_graph_cache[temp]
    #                 else:
    #                     graph = smiles_to_graph(temp)  # 将 SMILES 转换为图
    #                     smiles_to_graph_cache[temp] = graph  # 缓存转换后的图
                    
    #                 graphs.append(graph)

    #             if graphs:
    #                 # 使用 Batch 将所有图组合为一个批次
    #                 batch = Batch.from_data_list(graphs).to(self.device)
    #                 drug_features = self.drug_encoder(batch)

    #                 # 将药物特征分配给路径特征
    #                 path_features = drug_features.split(1)  # 将 drug_features 分割为单独的特征
    #                 all_path_features.append(path_features)
                
    #         # 将每个子列表转换为张量，并进行填充
    #         padded_vectors = []

    #         for vectors in all_path_features:
    #             # 将1x512的向量列表转换为一个张量
    #             tensor_list = [vec.squeeze(0) for vec in vectors]  # 移除多余的维度
    #             padded_vectors.append(pad_sequence(tensor_list, batch_first=True))

    #         # 合并所有填充后的子列表
    #         padded_batch = pad_sequence(padded_vectors, batch_first=True)
    #         path_embedding = self.path_encoder(padded_batch)
    #         avg_path_feature_for_one_drug = torch.mean(path_embedding, dim=0)
    #         encoded_features.append(avg_path_feature_for_one_drug)

    #     # 最终返回编码的特征
    #     return torch.stack(encoded_features).to(self.device)

    
    # def encode_paths(self, paths, paths_labels):
    #     # 编码路径信息为特征向量
    #     encoded_features = []
        
        
    #     i_length, j_length, k_length = paths_labels.size()
        
    #     i_idx, j_idx, k_idx = 0, 0, 0
        
    #     for i_idx  in tqdm(range(i_length)):
    #         path_for_one_drug = []
    #         for j_idx in range(j_length):
    #             path_for_one_path = []
    #             path_features = []
                
    #             if paths[j_idx][0][i_idx] == 'PAD':
    #                 break
                
    #             for k_idx in range(k_length):
                    
    #                 temp = paths[j_idx][k_idx][i_idx]
    #                 path_for_one_path.append(temp)
                 
    #                 graph = smiles_to_graph(temp)  # 将 SMILES 转换为图
                    
    #                 drug_feature = self.drug_encoder(graph.to(self.device))
    #                 path_features.append(drug_feature)
                    
    #                 if temp == 'C':
    #                     break
                    
    #             path_embedding = self.path_encoder(torch.stack(path_features).permute(1,0,2).to(self.device))
    #             path_for_one_drug.append(path_embedding)
                
    #         avg_path_feature_for_one_drug = torch.mean(torch.stack(path_for_one_drug).to(self.device), dim=0).to(self.device)
    #         encoded_features.append(avg_path_feature_for_one_drug)

    #     return torch.stack(encoded_features).to(self.device)  # 返回的特征张量