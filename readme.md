# 实验过程笔记

## 12月10日-12月11日

### 完成内容 
- [x] 完成数据获取代码 **get_data.py**
- [x] 完成embedding的实现 **embedding.py**
- [x] 完成position encoding的实现 **position_encoding.py**

### 实验笔记

#### 一、 PyTorch 核心机制与模型管理
1. 参数与状态管理 (nn.Parameter vs Buffer)
这是构建自定义层（如 PositionalEncoding）时的核心区分。
| nn.Parameter | self.register_buffer |
| ------------ | -------------------- |
| 用于需要梯度更新的参数（如权重矩阵）， | 用于不需要梯度更新的状态（如缓存）， |
| 在模型保存和加载时会自动处理。 | 在模型保存和加载时会自动处理。 |
| 需要梯度更新的参数通常是模型的核心权重。 | 缓存状态通常是一些中间计算结果或配置参数。 |

2. 梯度控制与计算图
- detach():

作用：截断梯度流。返回一个共享内存的新 Tensor，但永远切断与历史计算图的联系。  

场景：Teacher Forcing, GAN 的判别器训练, RNN 截断反向传播 (TBPTT)。  

安全修改：若需修改 detach 后的值且不影响原值，必须用 .detach().clone()。

- with torch.no_grad()::

作用：上下文管理器。暂时关闭 Autograd 引擎的录制功能，不建立计算图。

场景：模型推理 (Inference), 验证集评估。

- data (禁区)：

警告：严禁使用 tensor.data 修改值，因为 Autograd 无法追踪，会导致梯度计算错误且不报错。

3. 模型自省 (Introspection)
- model.parameters(): 返回无名张量生成器，专供优化器 (Adam, SGD) 使用。

- model.named_parameters(): 返回 (name, tensor)，用于调试、冻结特定层或差分学习率。

- model.state_dict(): 返回包含 Parameter 和 Buffer 的字典，用于模型保存/加载。  
> 注意前两个使用时是作为迭代器使用，要使用两个参数和for解包元组，  
> 而 state_dict() 返回的是一个字典。

#### 二、 Tensor 运算与维度处理
1. 广播机制 (Broadcasting)
PyTorch 的核心魔法，允许不同形状的张量进行算术运算。
规则：从后向前对齐，如果维度是 1，则自动扩展以匹配对方。

2. 矩阵乘法符号
- @：矩阵乘法，等价于 torch.matmul()。
>它可以忽略前两维（Batch, Head），只对最后两维做矩阵乘法。 这是 Transformer 并行计算的基石。
- *：逐元素乘法，等价于 torch.mul()。

#### 三、 模块化架构设计 (Architecture)

1. nn.Sequential: 傻瓜式流水线。适合单输入单输出、顺序执行的简单块（如 FFN）。

2. nn.ModuleList: 注册过的列表。适合需要循环遍历、传递多参数（如 mask）的复杂层堆叠（如 Transformer Encoder Layers）。

3. 模块嵌套设计，可定义复杂层级结构，如 Transformer Encoder Layer 中的多头注意力
和前馈网络。可单独定义，使用print(model)查看结构。然后使用model.EncoderLayer作为属性访问下属结构。
“如果是 nn.ModuleList 或 nn.Sequential 定义的层堆叠，可以使用 model.layers[0] 访问。如果是普通的子模块属性，必须通过属性名 model.layer_name 访问。”

| 容器        | nn.Sequential     | nn.ModuleList                   |nn.Module|
|-----------|-------------------|---------------------------------| --- |
| Forward机制 | 自动执行,数据自动从第一层流向最后一层 | 手动执行,必须在 forward 中写 for 循环或索引调用 |手动编写。必须显式定义数据的流向|
|灵活性|低|高|最高|
|访问方式| model.net[0]| model.layers[0] | model.layer_name|

<details>
<summary>示例代码</summary>

```python
import torch
import torch.nn as nn

# --- 1. 定义子组件 (普通零件) ---
class AttentionBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # 这是一个普通的子模块属性 -> 必须用点号访问
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        return self.proj(self.qkv(x))

# --- 2. 定义层级结构 (Layer) ---
class EncoderLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # [A] 普通属性访问
        self.self_attn = AttentionBlock(dim)
        
        # [B] Sequential: 内部是一个简单的流水线 -> 索引访问
        self.feed_forward = nn.Sequential(
            nn.Linear(dim, dim * 4),  # index 0
            nn.ReLU(),                # index 1
            nn.Linear(dim * 4, dim)   # index 2
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # 手动编排数据流 (ResNet结构)
        x = x + self.self_attn(x)
        x = x + self.feed_forward(self.norm(x))
        return x

# --- 3. 定义整体模型 (Main Model) ---
class MiniTransformer(nn.Module):
    def __init__(self, layer_count=2, dim=128):
        super().__init__()
        
        # [C] ModuleList: 用于堆叠 N 层 -> 索引访问
        # 注意：这里仅仅是把层存起来，PyTorch 知道它们有参数
        self.layers = nn.ModuleList([
            EncoderLayer(dim) for _ in range(layer_count)
        ])
        
        # [A] 普通属性
        self.final_norm = nn.LayerNorm(dim)

    def forward(self, x):
        # ModuleList 必须自己写循环！
        for layer in self.layers:
            x = layer(x)
        return self.final_norm(x)

# ==========================================
#              架构操作实验室
# ==========================================

# 1. 实例化模型
model = MiniTransformer(layer_count=2, dim=10)
print("--- 原始模型结构 ---")
# print(model) # 打印整体结构，太长可以不看

# --- 场景一：深层访问 (Deep Access) ---
# 目标：访问第 0 层 EncoderLayer 里的 self_attention 里的 proj 层
# 路径解析：
# model           (MiniTransformer)
# .layers         (ModuleList) -> 用 [0] 访问
# .self_attn      (AttentionBlock) -> 普通属性，用 .名字 访问
# .proj           (Linear) -> 普通属性，用 .名字 访问

print("\n--- 1. 访问深层权重 ---")
target_layer = model.layers[0].self_attn.proj
print(f"目标层类型: {type(target_layer)}")
print(f"权重形状: {target_layer.weight.shape}")

# --- 场景二：访问 Sequential 内部 ---
# 目标：访问第 1 层 EncoderLayer 的 FFN 里的 ReLU
relu_layer = model.layers[1].feed_forward[1]
print(f"Sequential内部层: {relu_layer}")

# --- 场景三：外科手术 (修改模型) ---
print("\n--- 2. 修改模型 (魔改) ---")

# 手术 A: 替换 ModuleList 中的整层
# 比如：我想把第 1 层直接删掉，换成直通 (Identity)，相当于做了一次“层剪枝”
model.layers[1] = nn.Identity()
print("已将 layers[1] 替换为 Identity (直通层)")

# 手术 B: 修改 Sequential 内部
# 比如：我觉得 ReLU 不好，想换成 GeLU
# 注意：Sequential 是可变的 (Mutable)，可以直接赋值
model.layers[0].feed_forward[1] = nn.GELU()
print("已将 layers[0] 的激活函数换为 GELU")

# 手术 C: 冻结特定层的参数 (Fine-tuning 常用)
# 只冻结第 0 层的 Attention，不冻结 FFN
for param in model.layers[0].self_attn.parameters():
    param.requires_grad = False
print("已冻结 layers[0].self_attn 的梯度")

# --- 验证修改结果 ---
print("\n--- 修改后的结构片段 ---")
print(f"Layer 0 FFN: {model.layers[0].feed_forward}")
print(f"Layer 1: {model.layers[1]}")
```

</details>