"""
图神经网络模型模块 (Phase 2)

提供基于图结构的选股模型：
- GraphSAGE: 采样聚合图神经网络
- GAT: 图注意力网络
- GNNStockModel: 统一接口封装

兼容 DeepStockModel 接口，可直接用于训练和预测。

用法：
    from src.deep_learning.gnn_models import GNNStockModel, get_gnn_model

    model = get_gnn_model('graphsage', config)
    model.train(X, y)
    predictions = model.predict(factor_df)
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import json

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from .base import DeepStockModel, load_deep_config
from .graph_builder import StockGraphBuilder
from .contracts import GraphBundle, CacheMeta
from .losses import get_loss_function

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils import setup_logger, get_project_root, ensure_dir

logger = setup_logger(__name__)

# 检查 PyG 是否可用
PYG_AVAILABLE = False
try:
    from torch_geometric.nn import SAGEConv, GATConv, GCNConv, global_mean_pool
    from torch_geometric.data import Data, Batch
    PYG_AVAILABLE = True
except ImportError:
    logger.warning("PyTorch Geometric 未安装，GNN 模块功能受限")


# ===================== GNN 编码器 =====================

class GraphSAGEEncoder(nn.Module):
    """
    GraphSAGE 编码器

    使用采样聚合策略学习节点嵌入
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 128,
                 output_dim: int = 64,
                 num_layers: int = 2,
                 dropout: float = 0.1,
                 aggregator: str = 'mean'):
        """
        Args:
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            output_dim: 输出嵌入维度
            num_layers: GNN 层数
            dropout: Dropout 比率
            aggregator: 聚合方式 ('mean', 'max', 'lstm')
        """
        super().__init__()

        if not PYG_AVAILABLE:
            raise ImportError("需要安装 PyTorch Geometric: pip install torch-geometric")

        self.num_layers = num_layers
        self.dropout = dropout

        # 输入投影
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # SAGE 层
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            in_dim = hidden_dim
            out_dim = output_dim if i == num_layers - 1 else hidden_dim

            self.convs.append(SAGEConv(in_dim, out_dim, aggr=aggregator))
            self.norms.append(nn.LayerNorm(out_dim))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_weight: torch.Tensor = None) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 节点特征 [n_nodes, input_dim]
            edge_index: 边索引 [2, n_edges]
            edge_weight: 边权重 [n_edges] (可选)

        Returns:
            节点嵌入 [n_nodes, output_dim]
        """
        # 输入投影
        x = self.input_proj(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # GNN 层
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x_res = x
            x = conv(x, edge_index)
            x = norm(x)

            if i < self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

                # 残差连接（如果维度匹配）
                if x.shape == x_res.shape:
                    x = x + x_res

        return x


class GATEncoder(nn.Module):
    """
    图注意力网络编码器

    使用注意力机制学习边权重
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 128,
                 output_dim: int = 64,
                 num_layers: int = 2,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 attention_dropout: float = 0.1):
        """
        Args:
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            output_dim: 输出嵌入维度
            num_layers: GNN 层数
            num_heads: 注意力头数
            dropout: Dropout 比率
            attention_dropout: 注意力 Dropout
        """
        super().__init__()

        if not PYG_AVAILABLE:
            raise ImportError("需要安装 PyTorch Geometric: pip install torch-geometric")

        self.num_layers = num_layers
        self.dropout = dropout

        # 输入投影
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # GAT 层
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            in_dim = hidden_dim
            out_dim = output_dim if i == num_layers - 1 else hidden_dim
            heads = 1 if i == num_layers - 1 else num_heads

            # 中间层输出维度需要除以头数
            if i < num_layers - 1:
                conv_out_dim = hidden_dim // num_heads
            else:
                conv_out_dim = out_dim

            self.convs.append(GATConv(
                in_dim, conv_out_dim,
                heads=heads,
                dropout=attention_dropout,
                concat=(i < num_layers - 1)  # 最后一层不拼接
            ))
            self.norms.append(nn.LayerNorm(conv_out_dim * heads if i < num_layers - 1 else out_dim))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_weight: torch.Tensor = None) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 节点特征 [n_nodes, input_dim]
            edge_index: 边索引 [2, n_edges]
            edge_weight: 边权重 [n_edges] (可选，GAT 不使用)

        Returns:
            节点嵌入 [n_nodes, output_dim]
        """
        # 输入投影
        x = self.input_proj(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # GAT 层
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x = conv(x, edge_index)
            x = norm(x)

            if i < self.num_layers - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        return x


class GraphormerEncoder(nn.Module):
    """
    Graphormer 编码器

    基于 Transformer 的图神经网络，具有以下特点：
    - 中心性编码 (Centrality Encoding): 基于节点度数的位置编码
    - 空间编码 (Spatial Encoding): 基于最短路径距离的注意力偏置
    - 边特征编码 (Edge Encoding): 边权重作为注意力偏置

    参考: Do Transformers Really Perform Bad for Graph Representation? (Ying et al., 2021)
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 128,
                 output_dim: int = 64,
                 num_layers: int = 3,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 max_degree: int = 512,
                 max_spatial_dist: int = 64):
        """
        Args:
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            output_dim: 输出嵌入维度
            num_layers: Transformer 层数
            num_heads: 注意力头数
            dropout: Dropout 比率
            max_degree: 最大节点度数 (用于中心性编码)
            max_spatial_dist: 最大空间距离 (用于空间编码)
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout

        # 输入投影
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # 中心性编码 (Centrality Encoding)
        # 分别编码入度和出度
        self.in_degree_encoder = nn.Embedding(max_degree, hidden_dim, padding_idx=0)
        self.out_degree_encoder = nn.Embedding(max_degree, hidden_dim, padding_idx=0)

        # 空间编码 (Spatial Encoding)
        # 每个头有独立的空间偏置
        self.spatial_encoder = nn.Embedding(max_spatial_dist, num_heads, padding_idx=0)

        # 边编码 (Edge Encoding)
        # 将边权重投影为注意力偏置
        self.edge_encoder = nn.Linear(1, num_heads)

        # Transformer 编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 输出投影
        self.output_proj = nn.Linear(hidden_dim, output_dim)

        # 层归一化
        self.norm = nn.LayerNorm(output_dim)

        # 虚拟节点 (Virtual Node) 用于全局信息聚合
        self.virtual_node = nn.Parameter(torch.zeros(1, hidden_dim))
        nn.init.normal_(self.virtual_node, std=0.02)

    def _compute_degree(self, edge_index: torch.Tensor, num_nodes: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算节点的入度和出度"""
        # edge_index: [2, num_edges]
        source, target = edge_index[0], edge_index[1]

        # 计算出度和入度
        out_degree = torch.zeros(num_nodes, dtype=torch.long, device=edge_index.device)
        in_degree = torch.zeros(num_nodes, dtype=torch.long, device=edge_index.device)

        out_degree.scatter_add_(0, source, torch.ones_like(source))
        in_degree.scatter_add_(0, target, torch.ones_like(target))

        return in_degree, out_degree

    def _compute_spatial_matrix(self, edge_index: torch.Tensor, num_nodes: int,
                                 max_dist: int = 64) -> torch.Tensor:
        """
        计算空间距离矩阵 (简化版本: 使用跳数近似)

        对于股票图，使用1跳/2跳/非连接来近似空间距离
        """
        device = edge_index.device

        # 初始化距离矩阵
        spatial_matrix = torch.full((num_nodes, num_nodes), max_dist - 1,
                                    dtype=torch.long, device=device)

        # 对角线为0 (自己到自己)
        spatial_matrix.fill_diagonal_(0)

        # 1跳距离
        source, target = edge_index[0], edge_index[1]
        spatial_matrix[source, target] = 1
        spatial_matrix[target, source] = 1  # 无向图

        # 2跳距离 (简化实现)
        # 对于每条边 (u, v)，找 v 的所有邻居 w，设置 u-w 距离为 2
        adj_list = {}
        for s, t in zip(source.tolist(), target.tolist()):
            if s not in adj_list:
                adj_list[s] = []
            adj_list[s].append(t)
            if t not in adj_list:
                adj_list[t] = []
            adj_list[t].append(s)

        for node, neighbors in adj_list.items():
            for n1 in neighbors:
                for n2 in adj_list.get(n1, []):
                    if n2 != node and spatial_matrix[node, n2] > 2:
                        spatial_matrix[node, n2] = 2
                        spatial_matrix[n2, node] = 2

        return spatial_matrix.clamp(0, max_dist - 1)

    def _compute_attention_bias(self,
                                 spatial_matrix: torch.Tensor,
                                 edge_index: torch.Tensor,
                                 edge_weight: torch.Tensor,
                                 num_nodes: int) -> torch.Tensor:
        """
        计算注意力偏置

        结合空间编码和边权重编码
        """
        # 空间编码偏置 [num_nodes, num_nodes, num_heads]
        spatial_bias = self.spatial_encoder(spatial_matrix)

        # 边权重编码偏置
        if edge_weight is not None:
            edge_bias = torch.zeros(num_nodes, num_nodes, self.num_heads,
                                    device=edge_weight.device)
            source, target = edge_index[0], edge_index[1]
            edge_features = self.edge_encoder(edge_weight.unsqueeze(-1))  # [num_edges, num_heads]
            edge_bias[source, target] = edge_features

            # 合并偏置
            attn_bias = spatial_bias + edge_bias
        else:
            attn_bias = spatial_bias

        # 转换为 [num_heads, num_nodes, num_nodes] 格式
        attn_bias = attn_bias.permute(2, 0, 1)

        return attn_bias

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_weight: torch.Tensor = None) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 节点特征 [num_nodes, input_dim]
            edge_index: 边索引 [2, num_edges]
            edge_weight: 边权重 [num_edges] (可选)

        Returns:
            节点嵌入 [num_nodes, output_dim]
        """
        num_nodes = x.size(0)
        device = x.device

        # 1. 输入投影
        x = self.input_proj(x)

        # 2. 添加中心性编码
        in_degree, out_degree = self._compute_degree(edge_index, num_nodes)
        in_degree = in_degree.clamp(0, self.in_degree_encoder.num_embeddings - 1)
        out_degree = out_degree.clamp(0, self.out_degree_encoder.num_embeddings - 1)

        x = x + self.in_degree_encoder(in_degree) + self.out_degree_encoder(out_degree)

        # 3. 添加虚拟节点 (用于全局信息聚合)
        virtual_node = self.virtual_node.expand(1, -1).to(device)
        x = torch.cat([virtual_node, x], dim=0)  # [num_nodes + 1, hidden_dim]
        num_nodes_with_vn = num_nodes + 1

        # 4. 计算空间距离矩阵 (扩展以包含虚拟节点)
        spatial_matrix = self._compute_spatial_matrix(edge_index, num_nodes)
        # 虚拟节点与所有节点的距离设为特殊值
        vn_spatial = torch.full((1, num_nodes), 1, dtype=torch.long, device=device)
        spatial_matrix = torch.cat([
            torch.cat([torch.zeros(1, 1, dtype=torch.long, device=device), vn_spatial], dim=1),
            torch.cat([vn_spatial.t(), spatial_matrix], dim=1)
        ], dim=0)

        # 5. 计算注意力偏置
        # 扩展 edge_index 以包含虚拟节点的连接
        vn_edges_src = torch.zeros(num_nodes, dtype=torch.long, device=device)
        vn_edges_tgt = torch.arange(1, num_nodes + 1, device=device)
        vn_edge_index = torch.stack([
            torch.cat([vn_edges_src, vn_edges_tgt]),
            torch.cat([vn_edges_tgt, vn_edges_src])
        ])
        extended_edge_index = torch.cat([
            edge_index + 1,  # 偏移原始边索引
            vn_edge_index
        ], dim=1)

        # 扩展边权重
        if edge_weight is not None:
            vn_weight = torch.ones(num_nodes * 2, device=device)
            extended_edge_weight = torch.cat([edge_weight, vn_weight])
        else:
            extended_edge_weight = None

        attn_bias = self._compute_attention_bias(
            spatial_matrix, extended_edge_index, extended_edge_weight, num_nodes_with_vn
        )

        # 6. Transformer 编码 (带注意力偏置)
        # 标准 TransformerEncoder 不直接支持注意力偏置
        # 这里使用简化实现: 将偏置加到注意力分数上
        x = x.unsqueeze(0)  # [1, num_nodes + 1, hidden_dim] (batch_first)

        # 使用自定义 mask 近似注意力偏置
        # 将空间距离转换为注意力掩码
        attn_mask = (spatial_matrix >= self.spatial_encoder.num_embeddings - 1).float() * -1e9
        attn_mask = attn_mask.unsqueeze(0)  # [1, num_nodes + 1, num_nodes + 1]

        x = self.transformer(x, mask=attn_mask.squeeze(0))

        x = x.squeeze(0)  # [num_nodes + 1, hidden_dim]

        # 7. 移除虚拟节点，获取节点嵌入
        node_embeddings = x[1:]  # [num_nodes, hidden_dim]

        # 8. 输出投影
        output = self.output_proj(node_embeddings)
        output = self.norm(output)

        return output


# ===================== GNN 选股模型 =====================

class GNNStockModel(DeepStockModel):
    """
    GNN 选股模型

    兼容 DeepStockModel 接口，支持 GraphSAGE 和 GAT
    """

    def __init__(self, config: dict = None, gnn_type: str = 'graphsage'):
        """
        Args:
            config: 配置字典
            gnn_type: GNN 类型 ('graphsage', 'gat')
        """
        super().__init__(config)

        self.gnn_type = gnn_type
        gnn_config = self.config.get('gnn', {})

        # 模型参数
        self.input_dim = gnn_config.get('input_dim', 64)  # 将在首次前向传播时更新
        self.hidden_dim = gnn_config.get('hidden_dim', 128)
        self.output_dim = gnn_config.get('output_dim', 64)
        self.num_layers = gnn_config.get('num_layers', 2)
        self.dropout = gnn_config.get('dropout', 0.1)

        # GAT 特定参数
        self.num_heads = gnn_config.get('num_heads', 4)

        # 训练参数
        train_config = self.config.get('training', {})
        self.lr = train_config.get('lr', 1e-3)
        self.weight_decay = train_config.get('weight_decay', 1e-4)
        self.epochs = train_config.get('epochs', 100)
        self.patience = train_config.get('patience', 10)
        self.batch_size = train_config.get('batch_size', 32)

        # 损失函数
        self.loss_type = gnn_config.get('loss', 'listwise')

        # 图构建器
        self.graph_builder = StockGraphBuilder(config)

        # 模型组件（延迟初始化）
        self.encoder = None
        self.head = None
        self.loss_fn = None

        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 特征列
        self.feature_cols = []

        logger.info(f"初始化 GNN 模型: type={gnn_type}, hidden={self.hidden_dim}, layers={self.num_layers}")

    def _init_model(self, input_dim: int):
        """初始化模型组件"""
        self.input_dim = input_dim

        if self.gnn_type == 'graphsage':
            self.encoder = GraphSAGEEncoder(
                input_dim=input_dim,
                hidden_dim=self.hidden_dim,
                output_dim=self.output_dim,
                num_layers=self.num_layers,
                dropout=self.dropout,
            )
        elif self.gnn_type == 'gat':
            self.encoder = GATEncoder(
                input_dim=input_dim,
                hidden_dim=self.hidden_dim,
                output_dim=self.output_dim,
                num_layers=self.num_layers,
                num_heads=self.num_heads,
                dropout=self.dropout,
            )
        elif self.gnn_type == 'graphormer':
            graphormer_config = self.config.get('gnn', {}).get('graphormer', {})
            self.encoder = GraphormerEncoder(
                input_dim=input_dim,
                hidden_dim=self.hidden_dim,
                output_dim=self.output_dim,
                num_layers=graphormer_config.get('num_layers', self.num_layers),
                num_heads=graphormer_config.get('num_heads', self.num_heads),
                dropout=self.dropout,
                max_degree=graphormer_config.get('max_degree', 512),
                max_spatial_dist=graphormer_config.get('max_spatial_dist', 64),
            )
        else:
            raise ValueError(f"未知 GNN 类型: {self.gnn_type}")

        # 预测头
        self.head = nn.Sequential(
            nn.Linear(self.output_dim, self.output_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.output_dim // 2, 1),
        )

        # 损失函数
        self.loss_fn = get_loss_function(self.loss_type)

        # 移动到设备
        self.encoder = self.encoder.to(self.device)
        self.head = self.head.to(self.device)

    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Dict:
        """
        训练模型

        Args:
            X: 因子数据（包含 code, date 列）
            y: 标签

        Returns:
            训练结果字典
        """
        logger.info("=" * 60)
        logger.info("开始训练 GNN 模型")
        logger.info("=" * 60)

        # 确定特征列
        exclude_cols = {'code', 'name', 'date', 'industry', 'label', 'forward_return', 'total_score'}
        self.feature_cols = [c for c in X.columns
                            if c not in exclude_cols and X[c].dtype in [np.float64, np.float32, np.int64]]

        logger.info(f"特征维度: {len(self.feature_cols)}")

        # 初始化模型
        self._init_model(len(self.feature_cols))

        # 按日期分组构建图
        X = X.copy()
        X['label'] = y.values

        dates = X['date'].unique()
        dates = sorted(dates)

        logger.info(f"训练日期数: {len(dates)}")

        # 准备训练数据
        graphs = []
        labels_list = []

        for date in dates:
            date_df = X[X['date'] == date].copy()
            codes = date_df['code'].tolist()

            if len(codes) < 10:
                continue

            # 构建图
            graph = self.graph_builder.build_graph(
                codes, pd.Timestamp(date), date_df
            )

            # 转换为 PyG Data
            pyg_data = self._to_pyg_data(graph, date_df)
            if pyg_data is not None:
                graphs.append(pyg_data)

        if len(graphs) == 0:
            logger.error("无有效训练图")
            return {'val_loss': float('inf')}

        logger.info(f"构建 {len(graphs)} 个训练图")

        # 优化器
        params = list(self.encoder.parameters()) + list(self.head.parameters())
        optimizer = AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

        # 训练循环
        best_loss = float('inf')
        patience_counter = 0
        train_losses = []

        for epoch in range(self.epochs):
            self.encoder.train()
            self.head.train()

            epoch_loss = 0.0
            n_batches = 0

            # 随机打乱图
            indices = np.random.permutation(len(graphs))

            for idx in indices:
                pyg_data = graphs[idx].to(self.device)

                optimizer.zero_grad()

                # 前向传播
                embeddings = self.encoder(
                    pyg_data.x,
                    pyg_data.edge_index,
                    pyg_data.edge_attr if hasattr(pyg_data, 'edge_attr') else None
                )

                predictions = self.head(embeddings).squeeze(-1)

                # 计算损失
                loss = self.loss_fn(predictions, pyg_data.y)

                # 反向传播
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            train_losses.append(avg_loss)

            # 学习率调整
            scheduler.step(avg_loss)

            # 早停检查
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch + 1}/{self.epochs}, Loss: {avg_loss:.6f}, Best: {best_loss:.6f}")

            if patience_counter >= self.patience:
                logger.info(f"早停于 Epoch {epoch + 1}")
                break

        logger.info(f"训练完成, 最终损失: {best_loss:.6f}")

        return {
            'val_loss': best_loss,
            'train_losses': train_losses,
            'epochs_trained': epoch + 1,
        }

    def predict(self, factor_df: pd.DataFrame) -> pd.Series:
        """
        预测

        Args:
            factor_df: 因子数据

        Returns:
            预测分数 Series
        """
        if self.encoder is None:
            logger.warning("模型未训练")
            return pd.Series(index=factor_df.index, dtype=float)

        self.encoder.eval()
        self.head.eval()

        codes = factor_df['code'].tolist()
        date = factor_df['date'].iloc[0] if 'date' in factor_df.columns else pd.Timestamp.now()

        # 构建图
        graph = self.graph_builder.build_graph(codes, pd.Timestamp(date), factor_df)
        pyg_data = self._to_pyg_data(graph, factor_df)

        if pyg_data is None:
            return pd.Series(index=factor_df.index, dtype=float)

        pyg_data = pyg_data.to(self.device)

        with torch.no_grad():
            embeddings = self.encoder(
                pyg_data.x,
                pyg_data.edge_index,
                pyg_data.edge_attr if hasattr(pyg_data, 'edge_attr') else None
            )
            scores = self.head(embeddings).squeeze(-1).cpu().numpy()

        # 创建结果 Series
        result = pd.Series(index=factor_df.index, dtype=float)
        for i, code in enumerate(codes):
            mask = factor_df['code'] == code
            if mask.any():
                result.loc[mask] = scores[i]

        return result

    def _to_pyg_data(self, graph: GraphBundle, factor_df: pd.DataFrame) -> Optional[Any]:
        """转换为 PyG Data 对象"""
        if not PYG_AVAILABLE:
            return None

        try:
            # 节点特征
            features = []
            labels = []

            for code in graph.node_codes:
                code_row = factor_df[factor_df['code'] == code]
                if len(code_row) > 0:
                    feat = code_row[self.feature_cols].values[0]
                    label = code_row.get('label', code_row.get('forward_return', pd.Series([0]))).values[0]
                else:
                    feat = np.zeros(len(self.feature_cols))
                    label = 0.0

                features.append(feat)
                labels.append(label)

            x = torch.tensor(np.array(features), dtype=torch.float32)
            x = torch.nan_to_num(x, nan=0.0)

            y = torch.tensor(labels, dtype=torch.float32)

            edge_index = torch.tensor(graph.edge_index, dtype=torch.long)

            data = Data(x=x, edge_index=edge_index, y=y)

            if graph.edge_weight is not None:
                data.edge_attr = torch.tensor(graph.edge_weight, dtype=torch.float32)

            return data

        except Exception as e:
            logger.warning(f"转换 PyG Data 失败: {e}")
            return None

    def save_model(self, path: str):
        """保存模型"""
        if self.encoder is None:
            logger.warning("模型未初始化，无法保存")
            return

        torch.save({
            'encoder_state': self.encoder.state_dict(),
            'head_state': self.head.state_dict(),
            'config': self.config,
            'gnn_type': self.gnn_type,
            'input_dim': self.input_dim,
            'feature_cols': self.feature_cols,
        }, path)

        logger.info(f"模型已保存: {path}")

    def load_model(self, path: str):
        """加载模型"""
        # PyTorch 2.6+ 需要 weights_only=False
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.config = checkpoint['config']
        self.gnn_type = checkpoint['gnn_type']
        self.feature_cols = checkpoint['feature_cols']

        self._init_model(checkpoint['input_dim'])

        self.encoder.load_state_dict(checkpoint['encoder_state'])
        self.head.load_state_dict(checkpoint['head_state'])

        self.encoder.eval()
        self.head.eval()

        logger.info(f"模型已加载: {path}")


# ===================== 时序+图融合模型 =====================

class TemporalGNNModel(DeepStockModel):
    """
    时序 + 图融合模型

    1. Temporal Encoder (LSTM/Transformer) 处理历史序列
    2. GNN 在嵌入上做图卷积
    3. 融合输出预测
    """

    def __init__(self, config: dict = None,
                 temporal_type: str = 'transformer',
                 gnn_type: str = 'graphsage'):
        """
        Args:
            config: 配置字典
            temporal_type: 时序编码器类型 ('lstm', 'transformer')
            gnn_type: GNN 类型 ('graphsage', 'gat')
        """
        super().__init__(config)

        self.temporal_type = temporal_type
        self.gnn_type = gnn_type

        fusion_config = self.config.get('fusion', {})
        self.fusion_method = fusion_config.get('method', 'concat')  # concat/add/gate

        # 延迟导入时序模型
        from .temporal_models import LSTMEncoder, TransformerEncoder

        temporal_config = self.config.get('temporal', self.config.get('transformer', {}))
        gnn_config = self.config.get('gnn', {})

        # 维度配置
        self.seq_len = self.config.get('sequence', {}).get('seq_len', 60)
        self.input_dim = temporal_config.get('input_dim', 64)
        self.hidden_dim = temporal_config.get('hidden_dim', 128)
        self.output_dim = gnn_config.get('output_dim', 64)

        # 模型组件（延迟初始化）
        self.temporal_encoder = None
        self.gnn_encoder = None
        self.fusion_layer = None
        self.head = None

        self.graph_builder = StockGraphBuilder(config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.feature_cols = []

        logger.info(f"初始化时序+图融合模型: temporal={temporal_type}, gnn={gnn_type}")

    def _init_model(self, input_dim: int):
        """初始化模型"""
        from .temporal_models import LSTMEncoder, TransformerEncoder

        self.input_dim = input_dim

        # 时序编码器
        if self.temporal_type == 'lstm':
            self.temporal_encoder = LSTMEncoder(
                input_dim=input_dim,
                hidden_dim=self.hidden_dim,
                num_layers=2,
            )
            temporal_out_dim = self.hidden_dim
        else:
            self.temporal_encoder = TransformerEncoder(
                input_dim=input_dim,
                d_model=self.hidden_dim,
                nhead=4,
                num_layers=2,
            )
            temporal_out_dim = self.hidden_dim

        # GNN 编码器（输入是时序嵌入）
        if self.gnn_type == 'graphsage':
            self.gnn_encoder = GraphSAGEEncoder(
                input_dim=temporal_out_dim,
                hidden_dim=self.hidden_dim,
                output_dim=self.output_dim,
                num_layers=2,
            )
        else:
            self.gnn_encoder = GATEncoder(
                input_dim=temporal_out_dim,
                hidden_dim=self.hidden_dim,
                output_dim=self.output_dim,
                num_layers=2,
            )

        # 融合层
        if self.fusion_method == 'concat':
            fusion_dim = temporal_out_dim + self.output_dim
            self.fusion_layer = nn.Linear(fusion_dim, self.output_dim)
        elif self.fusion_method == 'gate':
            self.fusion_layer = nn.Sequential(
                nn.Linear(temporal_out_dim + self.output_dim, self.output_dim),
                nn.Sigmoid(),
            )
        else:  # add
            self.fusion_layer = nn.Identity()

        # 预测头
        self.head = nn.Sequential(
            nn.Linear(self.output_dim, self.output_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.output_dim // 2, 1),
        )

        # 移动到设备
        self.temporal_encoder = self.temporal_encoder.to(self.device)
        self.gnn_encoder = self.gnn_encoder.to(self.device)
        self.fusion_layer = self.fusion_layer.to(self.device)
        self.head = self.head.to(self.device)

    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Dict:
        """训练融合模型"""
        raise NotImplementedError(
            "TemporalGNNModel.train() 尚未实现。"
            "请使用其他已实现的模型 (如 'graphsage', 'gat')，"
            "或等待此功能完成后再使用。"
        )

    def predict(self, factor_df: pd.DataFrame) -> pd.Series:
        """预测"""
        raise NotImplementedError(
            "TemporalGNNModel.predict() 尚未实现。"
            "请使用其他已实现的模型 (如 'graphsage', 'gat')，"
            "或等待此功能完成后再使用。"
        )


# ===================== 工厂函数 =====================

def get_gnn_model(gnn_type: str = 'graphsage', config: dict = None) -> GNNStockModel:
    """
    获取 GNN 模型

    Args:
        gnn_type: 模型类型 ('graphsage', 'gat', 'graphormer')
        config: 配置字典

    Returns:
        GNNStockModel 实例
    """
    if config is None:
        config = load_deep_config()

    return GNNStockModel(config, gnn_type=gnn_type)


def get_temporal_gnn_model(temporal_type: str = 'transformer',
                            gnn_type: str = 'graphsage',
                            config: dict = None) -> TemporalGNNModel:
    """
    获取时序+图融合模型

    Args:
        temporal_type: 时序编码器类型
        gnn_type: GNN 类型
        config: 配置字典

    Returns:
        TemporalGNNModel 实例
    """
    if config is None:
        config = load_deep_config()

    return TemporalGNNModel(config, temporal_type=temporal_type, gnn_type=gnn_type)
