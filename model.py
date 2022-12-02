import torch
import torch.nn as nn
from torch.nn import functional as F
from scipy.spatial.distance import pdist, squareform
import numpy as np
import math

class AsymmetricConvolution(nn.Module):

    def __init__(self, in_cha, out_cha):
        super(AsymmetricConvolution, self).__init__()

        self.conv1 = nn.Conv2d(in_cha, out_cha, kernel_size=(3, 1), padding=(1, 0), bias=False)
        self.conv2 = nn.Conv2d(in_cha, out_cha, kernel_size=(1, 3), padding=(0, 1))

        self.shortcut = lambda x: x

        if in_cha != out_cha:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_cha, out_cha, 1, bias=False)
            )

        self.activation = nn.PReLU()

    def forward(self, x):

        shortcut = self.shortcut(x)

        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x2 = self.activation(x2 + x1)

        return x2 + shortcut


class InteractionMask(nn.Module):

    def __init__(self, number_asymmetric_conv_layer=7, spatial_channels=4, temporal_channels=4):
        super(InteractionMask, self).__init__()

        self.number_asymmetric_conv_layer = number_asymmetric_conv_layer

        self.spatial_asymmetric_convolutions = nn.ModuleList()
        self.temporal_asymmetric_convolutions = nn.ModuleList()

        for i in range(self.number_asymmetric_conv_layer):
            self.spatial_asymmetric_convolutions.append(
                AsymmetricConvolution(spatial_channels, spatial_channels)
            )
            self.temporal_asymmetric_convolutions.append(
                AsymmetricConvolution(temporal_channels, temporal_channels)
            )

        self.spatial_output = nn.Sigmoid()
        self.temporal_output = nn.Sigmoid()

    def forward(self, dense_spatial_interaction, dense_temporal_interaction, threshold=0.5):

        assert len(dense_temporal_interaction.shape) == 4
        assert len(dense_spatial_interaction.shape) == 4

        for j in range(self.number_asymmetric_conv_layer):
            dense_spatial_interaction = self.spatial_asymmetric_convolutions[j](dense_spatial_interaction)
            dense_temporal_interaction = self.temporal_asymmetric_convolutions[j](dense_temporal_interaction)

        spatial_interaction_mask = self.spatial_output(dense_spatial_interaction)
        temporal_interaction_mask = self.temporal_output(dense_temporal_interaction)

        spatial_zero = torch.zeros_like(spatial_interaction_mask, device='cuda')
        temporal_zero = torch.zeros_like(temporal_interaction_mask, device='cuda')

        spatial_interaction_mask = torch.where(spatial_interaction_mask > threshold, spatial_interaction_mask,
                                               spatial_zero)

        temporal_interaction_mask = torch.where(temporal_interaction_mask > threshold, temporal_interaction_mask,
                                               temporal_zero)

        return spatial_interaction_mask, temporal_interaction_mask


class ZeroSoftmax(nn.Module):

    def __init__(self):
        super(ZeroSoftmax, self).__init__()

    def forward(self, x, dim=0, eps=1e-5):
        x_exp = torch.pow(torch.exp(x) - 1, exponent=2)
        x_exp_sum = torch.sum(x_exp, dim=dim, keepdim=True)
        x = x_exp / (x_exp_sum + eps)
        return x


class SelfAttention(nn.Module):

    def __init__(self, in_dims=2, d_model=64, num_heads=4):
        super(SelfAttention, self).__init__()

        self.embedding = nn.Linear(in_dims, d_model)
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)

        self.scaled_factor = torch.sqrt(torch.Tensor([d_model])).cuda()
        self.softmax = nn.Softmax(dim=-1)

        self.num_heads = num_heads

    def split_heads(self, x):

        # x [batch_size seq_len d_model]
        x = x.reshape(x.shape[0], -1, self.num_heads, x.shape[-1] // self.num_heads).contiguous()

        return x.permute(0, 2, 1, 3)  # [batch_size nun_heads seq_len depth]

    def forward(self, x, mask=False, multi_head=False):

        # batch_size seq_len 2

        assert len(x.shape) == 3
        embeddings = self.embedding(x)  # batch_size seq_len d_model
        query = self.query(embeddings)  # batch_size seq_len d_model
        key = self.key(embeddings)      # batch_size seq_len d_model

        if multi_head:
            query = self.split_heads(query)  # B num_heads seq_len d_model
            key = self.split_heads(key)  # B num_heads seq_len d_model
            attention = torch.matmul(query, key.permute(0, 1, 3, 2))  # (batch_size, num_heads, seq_len, seq_len)
        else:
            attention = torch.matmul(query, key.permute(0, 2, 1))  # (batch_size, seq_len, seq_len)

        attention = self.softmax(attention / self.scaled_factor)

        if mask is True:

            mask = torch.ones_like(attention)
            attention = attention * torch.tril(mask)
        return attention, embeddings


class SpatialTemporalFusion(nn.Module):

    def __init__(self, obs_len=8):
        super(SpatialTemporalFusion, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(obs_len, obs_len, 1),
            nn.PReLU()
        )

        self.shortcut = nn.Sequential()

    def forward(self, x):

        x = self.conv(x) + self.shortcut(x)
        return x.squeeze()


class SparseWeightedAdjacency(nn.Module):

    def __init__(self, spa_in_dims=2, tem_in_dims=3, embedding_dims=64, obs_len=8, dropout=0,
                 number_asymmetric_conv_layer=7):
        super(SparseWeightedAdjacency, self).__init__()

        # dense interaction
        self.spatial_attention = SelfAttention(spa_in_dims, embedding_dims)
        self.temporal_attention = SelfAttention(tem_in_dims, embedding_dims)

        # attention fusion
        self.spa_fusion = SpatialTemporalFusion(obs_len=obs_len)

        # interaction mask
        self.interaction_mask = InteractionMask(
            number_asymmetric_conv_layer=number_asymmetric_conv_layer
        )

        self.dropout = dropout
        self.zero_softmax = ZeroSoftmax()
        self.softmax = nn.Softmax(dim=-1)

    def dot_product_angle(self, v1,v2):
        if np.linalg.norm(v1)==0 or np.linalg.norm(v2)==0:
            print('Zero magnitude vector!')
            return 0
        else:
            vector_dot_product=np.dot(v1, v2)
            arccos = np.arccos(vector_dot_product/(np.linalg.norm(v1)*np.linalg.norm(v2)))
            angle = np.degrees(arccos)
            if v1[1] < 0:
                angle = 360 - angle
            if v1[1] == 0:
                if v1[0] > 0:
                    angle = 0
                if v1[0] < 0:
                    angle = 180
            return angle
        # return 0


    def collision_risk(self, dense_spatial_interaction, obs_traj, category):

        _, mulhead, _, _ = dense_spatial_interaction.size()
        category = category.squeeze()

        obs_traj = obs_traj.permute(2, 0, 1)  # (T N 2)
        T, N, D = obs_traj.size()

        #the S, V, Direction of each agent
        current_state = torch.zeros((T, N, D + 3))  # stored the S, V, Direction
        size = torch.Tensor([5, 1, 2, 6])
        num_c = len(size) #the number of categories
        for i in range(T):
            for j in range(N):
                for z in range(D):
                    current_state[i, j, z] = obs_traj[i, j, z]
                current_state[i, j, D] = size[int(category[j] - 1)]  # size
        #the calculation of velocity
        for n in range(N):
            obs_i = obs_traj[:, n, :]
            for t in range(T - 1):
                difx = obs_i[t + 1, 0].item() - obs_i[t, 0].item()
                dify = obs_i[t + 1, 1].item() - obs_i[t, 1].item()
                current_state[t, n, D + 1] = math.sqrt(
                    math.pow(difx, 2) + math.pow(dify, 2))  # velocity
                #the calculation of deriction
                v1_agent = np.array([difx, dify])
                v2_agent = np.array([1, 0])
                dire = self.dot_product_angle(v1_agent, v2_agent)
                current_state[t, n, D + 2] = dire
                # if difx != 0:
                #     # dire = 180*math.atan(dify/difx)/(math.pi)
                #     dire = 180 * math.atan2(dify, difx) / (math.pi)
                #     if dire < 0:
                #         dire = 360 + dire
                #     current_state[t, n, D + 2] = dire
                # else:
                #     if dify > 0:
                #         dire = 90
                #     elif dify < 0:
                #         dire = 270
                #     else:
                #         dire = 0
                #     current_state[t, n, D + 2] = dire
            current_state[T - 1, n, D + 2] = current_state[T - 2, n, D + 2]  # 角度
            current_state[T - 1, n, D + 1] = current_state[T - 2, n, D + 1]  # 速度

        #the collision risk
        dense = dense_spatial_interaction[:, 0, :, :]  # 8*N*N
        angleffect = torch.zeros_like(dense)

        for t in range(T):

            for n in range(N):
                for interobs in range(N):
                    if n!=interobs:
                        diffangle = current_state[t, interobs, D + 2] - current_state[t, n, D + 2]  # 角度差
                        if diffangle < 0:
                            diffangle = 360 + diffangle
                        cos_diffangle = math.fabs(math.cos(diffangle))
                        diffy = current_state[t, interobs, 1].item() - current_state[t, n, 1].item()
                        diffx = current_state[t, interobs, 0].item() - current_state[t, n, 0].item()

                        v1 = np.array([diffx, diffy])
                        v2 = np.array([1, 0])
                        angle = self.dot_product_angle(v1, v2)

                        angle_n = current_state[t, n, D + 2]
                        if ((0 <= angle_n <= 298) and (angle_n < angle <= angle_n + 62)):
                            # k>0
                            if 180 < diffangle < 360:
                                angleffect[t, n, interobs] = 1 - cos_diffangle
                        elif (angle_n > 298) and ((angle_n < angle < 360) or (0 <= angle <= (angle_n - 298))):
                            # k>0
                            if 180 < diffangle < 360:
                                angleffect[t, n, interobs] = 1 - cos_diffangle
                        elif angle_n == angle:
                            # k=0
                            if diffangle == 0 or diffangle == 180:
                                angleffect[t, n, interobs] = cos_diffangle
                        elif ((62 <= angle_n) and (angle_n - 62 <= angle < angle_n)):
                            # k<0
                            if 0 < diffangle < 180:
                                angleffect[t, n, interobs] = 1 - cos_diffangle
                        elif ((0 <= angle_n < 62) and ((angle_n + 298 < angle < 360) or (0 < angle < angle_n))):
                            # k<0
                            if 0 < diffangle < 180:
                                angleffect[t, n, interobs] = 1 - cos_diffangle
                angleffect[t, n, n] = 0


        collision_risk = torch.zeros_like(angleffect)  # 8*N*N
        for t in range(T):
            for n in range(N):
                for objid in range(N):

                    collision_risk[t, n, objid] = 0.1* current_state[t, objid, D] + 0.4 * current_state[
                        t, objid, D+1] + 0.5 * angleffect[t, n, objid]
        sumCR = torch.zeros((num_c, num_c))
        num = torch.zeros((num_c, num_c))
        CR = torch.zeros((num_c, num_c))
        for t in range(T):
            for n in range(N):
                for obj in range(N):
                    if collision_risk[t, n, obj] != 0:
                        sumCR[int(category[n]) - 1, int(category[obj]) - 1] = sumCR[int(category[n]) - 1, int(
                            category[obj]) - 1] + collision_risk[t, n, obj]
                        num[int(category[n]) - 1, int(category[obj]) - 1] = num[int(category[n]) - 1, int(
                            category[obj]) - 1] + 1

        for nc in range(sumCR.size()[0]):
            for ncj in range(sumCR.size()[0]):
                if num[nc, ncj] != 0:
                    CR[nc, ncj] = sumCR[nc, ncj] / num[nc, ncj]

        return CR



    def behavior_weight(self, dense_spatial_interaction, obs_traj, category, multi_head=True):
        r"""trick： in one dataset,since the category of agents is not change, the behavior response among agents
        from fixed categoris is common respectively, such as response of A categories of agent to A categories of agent,
        response of A categories of agent to B categories of agent, and response of B categories of agent to A categories
         of agent. Moreover, all category of agents may be not appeared in the each frame. Therefore, we use a trick:
         we print the output of collision_risk.py, and select the corresponding this output when all category are appeared,
         then treat it as the beahvior response among all categories of agents.

                See :the variant category_beh_response`
                """

        _, mulhead, _, _ = dense_spatial_interaction.size()
        category = category.squeeze()

        # category_beh_response = self.collision_risk(dense_spatial_interaction, obs_traj, category)

        category_beh_response = torch.tensor([[5, 1, 3, 6],
                                              [5, 1, 3, 6],
                                              [5, 1, 3, 6],
                                              [5, 1, 3, 6]])

        obs_traj = obs_traj.permute(2, 0, 1)  # (T N 2)
        T, N, D = obs_traj.size()
        agent_behsponse_cube = torch.zeros(T, N, N)  # resort agent-agent behavior response
        for numped in range(N):
            for col in range(N):
                agent_behsponse_cube[:, numped, col] = category_beh_response[int(category[numped].item()) - 1, int(category[col].item()) - 1]
            agent_behsponse_cube[:, numped, numped] = 15
        agent_behsponse_cube = self.softmax(agent_behsponse_cube)

        if multi_head:
            agent_behsponse_weight = agent_behsponse_cube.unsqueeze(1).repeat(1, mulhead, 1, 1).cuda()

        return agent_behsponse_weight


    def diatance_weight(self, dense_spatial_interaction, obs_traj, multi_head=True):
        obs_traj = obs_traj.permute(2, 0, 1)  # (T N 2)
        T, N, D = obs_traj.size()
        dis_cube = np.zeros((T, N, N))
        _, mulhead, _, _ = dense_spatial_interaction.size()

        obs_traj = obs_traj.cuda().data.cpu().numpy()
        for t in range(T):
            dis_cube[t] = squareform(pdist(obs_traj[t], metric='euclidean'))
        dis_weight = torch.tensor(
            np.divide(1., dis_cube, out=np.zeros_like(dis_cube), where=(dis_cube > 0.)) + np.identity(N))

        dis_weight = self.softmax(dis_weight)

        if multi_head:
            dis_weight = dis_weight.unsqueeze(1).repeat(1, mulhead, 1, 1).cuda()

        return dis_weight

    def fusion_weight(self, attention, dis_weight, beh_weight):

        # ######### dot product: consider attention, distance, behavior
        att_dis = torch.mul(attention, dis_weight.float())
        att_dis = self.softmax(att_dis)
        fusion_weig = torch.mul(att_dis, beh_weight)
        fusion_weig = self.softmax(fusion_weig)

        return fusion_weig

    def forward(self, graph, identity, obs_traj, category):

        assert len(graph.shape) == 3

        spatial_graph = graph[:, :, 1:]  # (T N 2)
        temporal_graph = graph.permute(1, 0, 2)  # (N T 3)

        dense_spatial_interaction, spatial_embeddings = self.spatial_attention(spatial_graph, multi_head=True)

        #bi-behavior
        dis_weight = self.diatance_weight(dense_spatial_interaction, obs_traj, multi_head=True)
        beh_weight = self.behavior_weight(dense_spatial_interaction, obs_traj, category, multi_head=True)
        dense_spatial_interaction = self.fusion_weight(dense_spatial_interaction, dis_weight, beh_weight)

        # (N num_heads T T)   (N T d_model)
        dense_temporal_interaction, temporal_embeddings = self.temporal_attention(temporal_graph, multi_head=True)

        # attention fusion
        st_interaction = self.spa_fusion(dense_spatial_interaction.permute(1, 0, 2, 3)).permute(1, 0, 2, 3)
        ts_interaction = dense_temporal_interaction

        spatial_mask, temporal_mask = self.interaction_mask(st_interaction, ts_interaction)

        # self-connected
        spatial_mask = spatial_mask + identity[0].unsqueeze(1)
        temporal_mask = temporal_mask + identity[1].unsqueeze(1)

        #sparse adjacency_matrix of spatial and temporal
        normalized_spatial_adjacency_matrix = self.zero_softmax(dense_spatial_interaction * spatial_mask, dim=-1)
        normalized_temporal_adjacency_matrix = self.zero_softmax(dense_temporal_interaction * temporal_mask, dim=-1)

        return normalized_spatial_adjacency_matrix, normalized_temporal_adjacency_matrix,\
               spatial_embeddings, temporal_embeddings


class GraphConvolution(nn.Module):

    def __init__(self, in_dims=2, embedding_dims=16, dropout=0):
        super(GraphConvolution, self).__init__()

        self.embedding = nn.Linear(in_dims, embedding_dims, bias=False)
        self.activation = nn.PReLU()

        self.dropout = dropout

    def forward(self, graph, adjacency):

        gcn_features = self.embedding(torch.matmul(adjacency, graph))
        gcn_features = F.dropout(self.activation(gcn_features), p=self.dropout)

        return gcn_features  # [batch_size num_heads seq_len hidden_size]


class Treedeliver_spatial(nn.Module):

    def __init__(self, in_dims=2, embedding_dims=16, dropout=0):
        super(Treedeliver_spatial, self).__init__()

        self.embedding = nn.Linear(in_dims, embedding_dims, bias=False)
        self.activation = nn.PReLU()

        self.dropout = dropout

    def Treeinteraction(self, adjacency, graph, pednum):
        updata = torch.zeros_like(torch.matmul(adjacency, graph))
        for i in range(pednum):
            updatahidden = torch.matmul(adjacency, graph)
            updatahidden[:, :, i, :] = graph[:, :, i, :]
            updatahidden = F.dropout(self.activation(updatahidden), p=self.dropout)
            updata[:, :, i, :] = torch.matmul(adjacency, updatahidden)[:, :, i, :]
        return updata

    def forward(self, graph, adjacency):

        _, _, pednum, _ = graph.size()
        if pednum <= 2:
            gcn_features = self.embedding(torch.matmul(adjacency, graph))
        else:
            gcn_features = self.embedding(self.Treeinteraction(adjacency, graph, pednum))

        gcn_features = F.dropout(self.activation(gcn_features), p=self.dropout)

        return gcn_features  # [batch_size num_heads seq_len hidden_size]



class SparseGraphConvolution(nn.Module):

    def __init__(self, in_dims=16, embedding_dims=16, dropout=0):
        super(SparseGraphConvolution, self).__init__()

        self.dropout = dropout

        self.spatial_temporal_sparse_gcn = nn.ModuleList()
        self.temporal_spatial_sparse_gcn = nn.ModuleList()

        # tree
        self.spatial_temporal_sparse_gcn.append(Treedeliver_spatial(in_dims, embedding_dims))
        self.spatial_temporal_sparse_gcn.append(GraphConvolution(embedding_dims, embedding_dims))

        self.temporal_spatial_sparse_gcn.append(GraphConvolution(in_dims, embedding_dims))
        self.temporal_spatial_sparse_gcn.append(Treedeliver_spatial(embedding_dims, embedding_dims))

    def forward(self, graph, normalized_spatial_adjacency_matrix, normalized_temporal_adjacency_matrix):

        # graph [1 seq_len num_pedestrians  3]
        # _matrix [batch num_heads seq_len seq_len]

        graph = graph[:, :, :, 1:]
        spa_graph = graph.permute(1, 0, 2, 3)  # (seq_len 1 num_p 2)
        tem_graph = spa_graph.permute(2, 1, 0, 3)  # (num_p 1 seq_len 2) _spatial

        gcn_spatial_features = self.spatial_temporal_sparse_gcn[0](spa_graph, normalized_spatial_adjacency_matrix)
        gcn_spatial_features = gcn_spatial_features.permute(2, 1, 0, 3)
        # [num_p num_heads seq_len d]
        gcn_spatial_temporal_features = self.spatial_temporal_sparse_gcn[1](gcn_spatial_features, normalized_temporal_adjacency_matrix)

        gcn_temporal_features = self.temporal_spatial_sparse_gcn[0](tem_graph, normalized_temporal_adjacency_matrix)
        gcn_temporal_features = gcn_temporal_features.permute(2, 1, 0, 3)
        gcn_temporal_spatial_features = self.temporal_spatial_sparse_gcn[1](gcn_temporal_features,
                                                                            normalized_spatial_adjacency_matrix)

        return gcn_spatial_temporal_features, gcn_temporal_spatial_features.permute(2, 1, 0, 3)


class TrajectoryModel(nn.Module):

    def __init__(self,
                 number_asymmetric_conv_layer=7, embedding_dims=64, number_gcn_layers=1, dropout=0,
                 obs_len=8, pred_len=12, n_tcn=5,
                 out_dims=5, num_heads=4):
        super(TrajectoryModel, self).__init__()

        self.number_gcn_layers = number_gcn_layers
        self.n_tcn = n_tcn
        self.dropout = dropout

        # sparse graph learning
        self.sparse_weighted_adjacency_matrices = SparseWeightedAdjacency(
            number_asymmetric_conv_layer=number_asymmetric_conv_layer
        )

        # graph convolution
        self.stsgcn = SparseGraphConvolution(
            in_dims=2, embedding_dims=embedding_dims // num_heads, dropout=dropout
        )

        self.fusion_ = nn.Conv2d(num_heads, num_heads, kernel_size=1, bias=False)

        self.tcns = nn.ModuleList()

        self.tcns.append(nn.Sequential(
            nn.Conv2d(obs_len, pred_len, 3, padding=1),
            nn.PReLU()
        ))

        for j in range(1, self.n_tcn):
            self.tcns.append(nn.Sequential(
                nn.Conv2d(pred_len, pred_len, 3, padding=1),
                nn.PReLU()
        ))

        self.output = nn.Linear(embedding_dims // num_heads, out_dims)
        self.normlize_graph = nn.BatchNorm2d(3)
        self.normlize_traj = nn.BatchNorm2d(2)

    def forward(self, graph, identity, obs_traj, category):

        # graph 1 obs_len N 3 bi-behavior
        normalized_spatial_adjacency_matrix, normalized_temporal_adjacency_matrix, spatial_embeddings, temporal_embeddings = \
            self.sparse_weighted_adjacency_matrices(graph.squeeze(), identity, obs_traj.squeeze(), category)

        #tree
        gcn_temporal_spatial_features, gcn_spatial_temporal_features = self.stsgcn(
            graph, normalized_spatial_adjacency_matrix, normalized_temporal_adjacency_matrix
        )

        gcn_representation = self.fusion_(gcn_temporal_spatial_features) + gcn_spatial_temporal_features

        gcn_representation = gcn_representation.permute(0, 2, 1, 3)
        features = self.tcns[0](gcn_representation)
        for k in range(1, self.n_tcn):
            features = F.dropout(self.tcns[k](features) + features, p=self.dropout)
        prediction = torch.mean(self.output(features), dim=-2)

        return prediction.permute(1, 0, 2).contiguous()
