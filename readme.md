# transfomer实验过程笔记

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

# 12月16日

## 完成内容

- [x] 完成注意力计算机制分头及多头合并 attention.py
- [x] 完成前馈网络层 feed_forward.py

- [x] 完成注意力机制与前馈网络层结合的堆叠编码层 encoder_layer

## 实验笔记

### 一、 维度变换与内存机制分头 (Split Heads) 的最佳实践

- **反模式**：使用 `torch.split`或者`torch.chunk`。会返回 Tuple，导致无法进行并行矩阵运算，且涉及内存拷贝。
- **最佳实践**：`view` + `transpose`。
  - 逻辑：先 `view` 增加维度，再 `transpose` 交换维度。
  - 公式：`[Batch, Seq, D_model]` -> `[Batch, Seq, Head, D_k]` -> `[Batch, Head, Seq, D_k]`。
  - 目的：将 `Head` 维度移至 `Seq` 之前，使每个头在逻辑上独立并行。

**contiguous() 的必要性**

- **现象**：`transpose`、`permute` 等操作只修改 Tensor 的元数据 (Stride)，不改变物理内存顺序。
- **冲突**：`.view()` 操作强制要求物理内存连续。若在 `transpose` 后直接调用 `view`，会报错 `RuntimeError: input is not contiguous`。
- **解决方案**：在重塑之前显式调用 `.contiguous()`，强制进行一次内存搬运和对齐。
  - 常用连招：`x.transpose(1, 2).contiguous().view(...)`

###  二、 PyTorch API 避坑指南** **torch.tensor vs torch.Tensor**

- **torch.tensor (小写)**：工厂函数 (Factory Function)。
  - 行为：智能推断数据类型 (如输入整数列表则生成 Int64)，总是深拷贝。
  - 推荐：**始终使用此方式**。
- **torch.Tensor (大写)**：类构造函数 (Constructor)。
  - 行为：是 `torch.FloatTensor` 的别名。无论输入什么都强转 Float。
  - **危险行为**：若传入单个整数 `torch.Tensor(5)`，不会创建标量，而是创建长度为 5 的**未初始化内存** (包含随机垃圾数据)。
  - Transformer 架构细节:Post-Norm 结构中的 Dropout 位置
    - **原则**：Dropout 应作用于残差相加**之前**，而非 LayerNorm 之后。
    - **流程**：`x = Norm(x_old + Dropout(Sublayer(x_old)))`。
    - **影响**：若放在 Norm 之后，会破坏归一化后的数据分布 (均值0方差1)，导致训练不稳定。

### 三、 Tensor 切分与维度重塑 API 详解

**1. 切分：`torch.chunk` vs `torch.split`** 这两个 API 的核心区别在于：**你是想指定“切几刀”（数量），还是指定“每块多大”（长度）。**

- **`torch.chunk(input, chunks, dim)`**：

  - **语义**：**“均分”**。尝试将 Tensor 在指定维度上平均切成 `chunks` 份。
  - **行为**：如果不能整除，最后一份会变小。
  - **场景**：LSTM/GRU 中将合并的门控权重拆分为 Input Gate, Forget Gate 等（数量固定）。
  - **返回**：Tuple。

- **`torch.split(tensor, split_size_or_sections, dim)`**：

  - **语义**：**“定制”**。按指定的长度切分。
  - **模式 A (传入 int)**：每一份的长度都是 `split_size`。
  - **模式 B (传入 list)**：如 `[10, 20, 5]`，精确指定每一份的长度。
  - **场景**：处理不等长的特征拼接，或者多头注意力中手动拆分（虽然常用 `view` 代替）。
  - **返回**：Tuple。

  代码对比

  ```python
  x = torch.randn(10, 6) # [Batch, Feature]
  
  # 1. Chunk: 我要切成 3 份 (6/3=2, 每份长2)
  a, b, c = torch.chunk(x, chunks=3, dim=1) 
  # a.shape -> (10, 2)
  
  # 2. Split (模式A): 我要每份长 2 (结果切出3份)
  parts = torch.split(x, split_size=2, dim=1) 
  # len(parts) -> 3
  
  # 3. Split (模式B): 我要一份长1，一份长5
  p1, p2 = torch.split(x, [1, 5], dim=1)
  # p1 -> (10, 1), p2 -> (10, 5)
  
  ```

  **2. 维度重塑：`view` vs `reshape`** Transformer 中用到最多的操作，用于实现多头注意力的“分头”与“合并”。

  - **`Tensor.view(\*shape)`**：
    - **特性**：**零拷贝 (Zero-copy)**。它仅仅改变 Tensor 的元数据 (Stride/Shape)，不改变底层数据。
    - **硬性要求**：Tensor 在内存中必须是**连续的 (Contiguous)**。
    - **报错**：如果 Tensor 不连续（如刚做过 `transpose`），会报 `RuntimeError`。
    - **推荐**：作为架构师，首选 `view`，因为它能帮你检查内存问题，避免隐式拷贝带来的性能损耗。
  - **`Tensor.reshape(\*shape)`**：
    - **特性**：**“老好人”**。如果 Tensor 连续，等价于 `view`；如果不连续，它会自动拷贝数据（相当于 `contiguous().view()`）。
    - **缺点**：因为行为不确定（可能拷贝也可能不拷贝），在追求极致性能的模型中（如大模型推理），不如 `view` 透明。

  ------

  **3. 维度交换与内存：`transpose` vs `permute`**

  - **`Tensor.transpose(dim0, dim1)`**：
    - **功能**：**只交换两个维度**。
    - **场景**：Attention 中交换 Seq_len 和 Head (`transpose(1, 2)`)。
    - **副作用**：操作后 Tensor 变为 **非连续 (Non-contiguous)**。
  - **`Tensor.permute(\*dims)`**：
    - **功能**：**一次性重新排列所有维度**。
    - **语法**：`x.permute(0, 2, 1, 3)`。
    - **对比**：比 `transpose` 更灵活，但底层原理一样，也会导致内存不连续。

  ------

  **4. 内存修复：`Tensor.contiguous()`**

  - **定义**：强制将 Tensor 的数据在内存中重新排列，使其物理顺序与逻辑顺序一致。
  - **代价**：涉及**内存申请和数据搬运**（深拷贝），有计算开销。
  - **Transformer 必背连招**：

  ```python
  # 错误写法 (RuntimeError)
  # x = x.transpose(1, 2).view(batch, -1) 
  
  # 正确写法
  x = x.transpose(1, 2).contiguous().view(batch, -1)
  ```

  

## 12月17日

### 完成内容 

- [x] 完成输入嵌入到编码的完整编码块编写 encoder.py
- [x] 完成解码堆叠层编写 decoder_layer.py
- [x] 完成输出嵌入到解码的完整解码块编写，包含掩码生成 decoder.py
- [x] 完成完整模型组装，从编码到解码到最后线性输出层完全实现,至此模型搭建完成 transfomer.py

### 实验笔记

####  一. 矩阵几何系列：`torch.tril` 及其家族

这是今天实现“因果掩码 (Subsequent Mask)”的核心。

- **`torch.tril(input, diagonal=0)`**

  - **全称**: **Tri**angle **L**ower (下三角矩阵)。

  - **作用**: 将矩阵的上三角部分全部置为 0，保留下三角部分。

  - **参数 `diagonal`**: 控制对角线的位置。默认为 0 (主对角线)。如果设为 -1，则主对角线也会被清零。

  - **场景**: Transformer Decoder 中用于屏蔽未来时刻（让 $t$ 时刻只能看 $0...t$）。

  - **代码实战**:

    Python

    ```
    # 生成一个全 1 的方阵，然后只保留下三角
    mask = torch.tril(torch.ones(seq_len, seq_len))
    ```

- **延伸：`torch.triu(input, diagonal=0)`**

  - **全称**: **Tri**angle **U**pper (上三角矩阵)。
  - **作用**: 与 `tril` 相反，保留右上角，将左下角置为 0。
  - **场景**: 虽然标准 Transformer 用不到，但在某些双向语言模型（如 XLNet）或计算矩阵的协方差时会用到。

### 二. 维度扩展系列：`unsqueeze` (广播之源)

这是实现 Padding Mask 能够自动适配 Multi-Head Attention 的关键。

- **`Tensor.unsqueeze(dim)`**

  - **作用**: 在指定位置插入一个**长度为 1** 的新维度。

  - **物理意义**: 并没有增加数据，只是改变了看待数据的“视角”，为广播机制 (Broadcasting) 做准备。

  - **实战解析**:

    ```python
    # src: [Batch, Seq]
    mask = (src != 0)          # [Batch, Seq]
    
    # 第一次 unsqueeze(1): [Batch, 1, Seq]  -> 对应 Head 维度
    # 第二次 unsqueeze(2): [Batch, 1, 1, Seq] -> 对应 Query 维度
    mask = mask.unsqueeze(1).unsqueeze(2)
    ```

  - **核心逻辑**: PyTorch 看到维度为 `1` 时，会自动把数据复制 N 份以匹配其他张量的形状。
  
    

# 1月1日

## 完成内容

- [x] **实现图像切块与线性投影模块 `patch_embedding.py`**

- [x] **实现 [CLS] Token 与 Position Embedding 参数定义及初始化 `vit.py`**

- [x] **实现 Encoder 模块对 ViT 向量输入的兼容性重构 `encoder.py`**

- [x] **实现 ViT 完整模型组装与前向传播逻辑 `vit.py`**

- [x] **实现模型集成测试脚本与维度验证 `test_vit.py`**

- [x] **实现项目 `.gitignore` 过滤规则配置与 Git 存档tag `v1.0-model-core`**

## 实验笔记

#### 一、 维度重塑系列：`transpose` vs `permute`

这是操作多维张量（尤其是 Transformer 中的多头注意力）时最频繁使用的两个“空间变换”工具。

- **`Tensor.transpose(dim0, dim1)`**

  - **作用**: 交换且**只能交换两个**指定的维度。
  - **特点**: 它是 `permute` 的特例。在处理 2D 矩阵转置（`dim0=0, dim1=1`）时语义最清晰。
  - **场景**: 常见于单次维度交换，如在 Patch Embedding 后交换序列维度和通道维度。

- **`Tensor.permute(\*dims)`**

  - **作用**: 一次性对**所有维度**进行重新排列。

  - **特点**: 必须传入一个包含所有维度索引的列表。它比 `transpose` 更灵活，逻辑更直观。

  - **代码实战**:

    Python

    ```
    # x 形状: [Batch, Channel, H, W]
    # 换成 ViT 需要的格式: [Batch, H, W, Channel]
    x = x.permute(0, 2, 3, 1) 
    ```

- **核心区别**: `transpose` 只能两两交换；`permute` 是全局重排。**注意**：两者返回的张量通常与原张量共享内存，若后续接 `view` 操作，通常需要调用 `.contiguous()`。

#### 二、 参数定义系列：`torch.Tensor`, `torch.tensor` 与 `nn.Parameter`

这三者决定了“数据”能否进化为“权重”。

- **`torch.Tensor` (类)**
  - **本质**: PyTorch 张量的基础类。调用 `torch.Tensor(2, 3)` 会创建一个形状为 2x3 的张量，但其内存是**未初始化的（数值随机）**。
  - **风险**: 容易引入 NaN 或极大值，除非紧接着进行手动初始化。
- **`torch.tensor` (工厂函数)**
  - **本质**: 从现有数据（list, numpy）**深拷贝**并创建张量的首选方法。它会自动推断数据类型。
  - **场景**: 创建辅助张量、常量。
- **`nn.Parameter` (参数包裹器)**
  - **本质**: 一种特殊的 `Tensor`。当它被赋值给 `nn.Module` 的属性时，会自动被加入到 `model.parameters()` 中。
  - **物理意义**: 告诉优化器（Optimizer）：“**这块数据是模型权重，请在反向传播时更新它。**”
  - **场景**: ViT 中的 `cls_token` 和 `pos_embed` 必须使用它包裹，否则模型训练时这些位置信息不会进化。

#### 三、 维度扩张系列：`expand` vs `repeat`

这是实现 `[CLS] Token` 从单向量适配到 Batch 维度的关键。

- **`Tensor.expand(\*sizes)`**

  - **作用**: 返回张量的一个新视图，将维度为 1 的轴“拉伸”到更大的尺寸。

  - **物理意义**: **内存零开销**。它并不真正复制数据，而是通过改变步长（stride）逻辑让同一个数据在不同位置反复出现。

  - **约束**: 只能扩展维度为 1 的轴。

  - **代码实战**:

    Python

    ```
    # cls_token: [1, 1, 768]
    # 适配 Batch=32: [32, 1, 768]
    cls_token = self.cls_token.expand(32, -1, -1) # -1 表示该维度不改变
    ```

- **`Tensor.repeat(\*sizes)`**

  - **作用**: 沿着指定维度**真正地复制**数据。
  - **物理意义**: 会开辟新的内存空间。
  - **区别**: 优先使用 `expand` 以节省显存；只有当需要真正独立的数据副本时才用 `repeat`。
