'''
热身赛：基于 GCN 的 Cora 节点分类任务
参赛选手需要填充注释为 TODO 的部分完成该赛题。
'''

import os.path as osp
import json
import pickle

import jittor as jt
from jittor import nn
import numpy as np
from jittor_geometric.nn import GCNConv
from jittor_geometric.ops import cootocsr, cootocsc
from jittor_geometric.nn.conv.gcn_conv import gcn_norm


# ============================================================
# 基本配置
# ============================================================
jt.flags.use_cuda = 1
jt.misc.set_global_seed(42)

# ============================================================
# 第一步：加载数据集
# ============================================================
data_path = osp.join('data', 'cora.pkl')

with open(data_path, 'rb') as f:
    raw = pickle.load(f)

# 将 numpy 数据转为 jittor 张量，构造 data 对象
class GraphData:
    pass

data = GraphData()
data.x = jt.array(raw['x'].astype(np.float32))
data.y = jt.array(raw['y'].astype(np.int64))
data.edge_index = jt.array(raw['edge_index'].astype(np.int64))
data.train_mask = jt.array(raw['train_mask'])
data.val_mask = jt.array(raw['val_mask'])
data.test_mask = jt.array(raw['test_mask'])
num_features = raw['num_features']
num_classes = raw['num_classes']

# 对特征做行归一化（等同于 T.NormalizeFeatures()）
row_sum = data.x.sum(dim=1, keepdims=True)
row_sum = jt.clamp(row_sum, min_v=1e-12)
data.x = data.x / row_sum

# ============================================================
# 第二步：图的边归一化 + 稀疏格式转换
# ============================================================
v_num = data.x.shape[0]
edge_index, edge_weight = data.edge_index, None

edge_index, edge_weight = gcn_norm(
    edge_index, edge_weight, v_num,
    improved=False, add_self_loops=True
)

# 将 COO 格式转换为 CSC 和 CSR 稀疏矩阵格式
with jt.no_grad():
    data.csc = cootocsc(edge_index, edge_weight, v_num)
    data.csr = cootocsr(edge_index, edge_weight, v_num)

# ============================================================
# 第三步：定义 GCN 模型
# ============================================================
class GCNNet(nn.Module):
    def __init__(self, num_features, num_classes, hidden_dim=256, dropout=0.8):
        super(GCNNet, self).__init__()
        self.dropout = dropout

        # 定义两层 GCN 卷积层
        # 提示：GCNConv(in_channels, out_channels)
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, num_classes)

    def execute(self):
        x, csc, csr = data.x, data.csc, data.csr

        # 实现前向传播
        # 1. 第一层 GCN 卷积 + ReLU 激活
        # 2. Dropout
        # 3. 第二层 GCN 卷积
        x = nn.relu(self.conv1(x, csc, csr))
        x = nn.dropout(x, self.dropout, is_train=self.training)
        x = self.conv2(x, csc, csr)

        return x

# 初始化模型和优化器
model = GCNNet(
    num_features=num_features,
    num_classes=num_classes,
    hidden_dim=256,
    dropout=0.8
)
optimizer = nn.Adam(params=model.parameters(), lr=0.01, weight_decay=5e-4)

# ============================================================
# 第四步：定义训练函数
# ============================================================
def train():
    model.train()

    # 1. 前向传播
    pred = model()[data.train_mask]

    # 2. 获取训练集标签
    label = data.y[data.train_mask]
    # 3. 计算交叉熵损失
    loss = nn.cross_entropy_loss(pred, label)
    # 4. 反向传播更新参数
    optimizer.step(loss)

# ============================================================
# 第五步：定义测试函数
# ============================================================
def test():
    model.eval()
    logits = model()
    accs = []

    for mask in [data.train_mask, data.val_mask]:
        # 实现预测和准确率计算
        # 1. 使用 jt.argmax 获取预测类别
        pred, _ = jt.argmax(logits[mask], dim=1)
        # 2. 计算预测准确率
        acc = (pred == data.y[mask]).float().mean()
        accs.append(acc)

    return accs

# ============================================================
# 第六步：训练模型
# ============================================================
best_val_acc = 0

for epoch in range(1, 201):
    train()
    train_acc, val_acc = test()

    if val_acc > best_val_acc:
        best_val_acc = val_acc

    if epoch % 20 == 0:
        log = 'Epoch: {:03d}, Train Acc: {:.4f}, Val Acc: {:.4f}'
        print(log.format(epoch, train_acc, best_val_acc))

print(f'\n最终结果: Val Acc: {best_val_acc:.4f}')

# ============================================================
# 第七步：生成并保存预测结果
# ============================================================
model.eval()

# 1. 使用训练好的模型对所有节点进行预测
logits = model()
pred, _ = jt.argmax(logits, dim=1)

# 2. 提取测试集节点的预测类别
test_indices = jt.where(data.test_mask > 0)[0]

# 3. 构建字典 {节点编号: 预测类别}
result = {}
for idx in test_indices:
    result[str(int(idx))] = int(pred[int(idx)])

# 4. 保存为 result.json
output_path = 'result.json'
with open(output_path, 'w') as f:
    json.dump(result, f, indent=2)

print(f"预测结果已保存到 {output_path}")
print(f"共预测 {len(result)} 个测试节点")
