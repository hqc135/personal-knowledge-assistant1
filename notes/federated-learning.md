

# 联邦学习与隐私计算：FedAvg、差分隐私、同态加密

> 这块偏系统/安全方向，面试中如果聊到数据隐私、多方协作训练就会涉及。参考了 McMahan 2017 原始论文和一些综述。

## 为什么需要联邦学习

核心矛盾：**数据孤岛 vs 模型需要大数据**

- 医院有病历数据但不能共享（HIPAA）
- 银行有交易数据但不能出域（监管）
- 手机有用户行为数据但不能上传（隐私）

联邦学习的解决思路：**数据不动，模型动**——各方在本地训练，只共享模型参数/梯度。

## FedAvg：最基础的联邦学习算法

### 算法流程

```python
# 服务器端
def fedavg_server(global_model, clients, rounds, C, E, B):
    for t in range(rounds):
        # 1. 随机选择一部分客户端
        selected = random.sample(clients, max(1, int(C * len(clients))))
        
        # 2. 下发全局模型
        local_models = []
        for client in selected:
            local_model = client.local_train(global_model, E, B)
            local_models.append((client.data_size, local_model))
        
        # 3. 加权平均聚合
        total_size = sum(size for size, _ in local_models)
        global_model = weighted_average(local_models, total_size)
    
    return global_model

# 客户端
def local_train(self, global_model, E, B):
    model = copy(global_model)
    for epoch in range(E):  # E: 本地训练轮数
        for batch in DataLoader(self.data, batch_size=B):
            loss = compute_loss(model, batch)
            loss.backward()
            optimizer.step()
    return model
```

### 关键超参数

- **C**：每轮参与的客户端比例（通常 0.1-0.3）
- **E**：本地训练 epoch 数（太大会导致 client drift）
- **B**：本地 batch size

**面试高频考点**：FedAvg 和分布式 SGD 的区别？

答：分布式 SGD 每个 step 都同步梯度，FedAvg 允许多步本地更新后再聚合。FedAvg 通信效率高但会有 client drift 问题。

### Client Drift 问题

**这里我一开始理解错了**：以为本地多训练几轮总是好的（减少通信），实际上当各客户端数据分布差异大（Non-IID）时，本地训练越多，各客户端模型偏离全局最优越远。

```
IID 数据：各客户端数据分布相同 → FedAvg 收敛好
Non-IID 数据：各客户端数据分布不同 → 本地最优 ≠ 全局最优 → drift
```

解决方案：
- FedProx：加正则项约束本地模型不要偏离全局模型太远
- SCAFFOLD：用控制变量修正梯度方向
- FedNova：归一化本地更新步数

## 差分隐私（Differential Privacy）

### 核心定义

一个随机算法 M 满足 (ε, δ)-差分隐私，如果对任意相邻数据集 D 和 D'（只差一条记录）：

```
P[M(D) ∈ S] ≤ e^ε · P[M(D') ∈ S] + δ
```

**直觉理解**：不管你的数据在不在数据集里，算法的输出分布几乎不变。ε 越小隐私保护越强。

**和信息论的类比**：ε 可以理解为"信息泄露量"的上界——攻击者从输出中能推断出关于任何个体的信息量被 ε 限制住了。

### 在联邦学习中的应用（DP-FedAvg）

```python
def dp_local_train(self, global_model, E, B, clip_norm, noise_scale):
    model = copy(global_model)
    for epoch in range(E):
        for batch in DataLoader(self.data, batch_size=B):
            loss = compute_loss(model, batch)
            loss.backward()
            
            # 1. 梯度裁剪（限制单个样本的影响）
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            
            # 2. 加高斯噪声
            for param in model.parameters():
                noise = torch.randn_like(param.grad) * noise_scale * clip_norm
                param.grad += noise
            
            optimizer.step()
    return model
```

### 隐私预算的组合

多次查询会消耗隐私预算。关键定理：

- **基本组合**：k 次 ε-DP 查询 → kε-DP（线性增长，太悲观）
- **高级组合**：k 次 ε-DP 查询 → O(ε√k)-DP（亚线性，更紧）
- **Rényi DP / zCDP**：更紧的组合界，实际系统常用

**面试考点**：隐私和模型效用的 trade-off 怎么平衡？→ 没有免费午餐，ε 越小噪声越大模型越差。实践中 ε=1~10 是常见范围。

## 同态加密（Homomorphic Encryption）

### 核心思想

在密文上直接计算，解密后得到和明文计算相同的结果：

```
Enc(a) ⊕ Enc(b) = Enc(a + b)  // 加法同态
Enc(a) ⊗ Enc(b) = Enc(a × b)  // 乘法同态
```

### 分类

| 类型 | 支持操作 | 代表方案 | 性能 |
|------|----------|----------|------|
| 半同态（PHE） | 仅加法或仅乘法 | Paillier（加法）、RSA（乘法） | 快，实用 |
| 有限全同态（SHE） | 有限次加法+乘法 | BGV、BFV | 中等 |
| 全同态（FHE） | 任意次加法+乘法 | CKKS、TFHE | 慢，但通用 |

### 在联邦学习中的应用

```python
# 用 Paillier 加密梯度聚合
from phe import paillier

# 各客户端加密自己的梯度
public_key, private_key = paillier.generate_paillier_keypair()

# 客户端侧
encrypted_gradients = [public_key.encrypt(g) for g in local_gradients]

# 服务器侧（只能看到密文，但可以做加法聚合！）
aggregated = encrypted_gradients[0]
for eg in encrypted_gradients[1:]:
    aggregated = aggregated + eg  # 密文加法 = 明文加法

# 解密得到聚合梯度（由可信第三方或联合解密）
result = private_key.decrypt(aggregated)
```

**这里的关键洞察**：FedAvg 的聚合操作本质上就是加权平均（加法+标量乘法），而 Paillier 恰好支持这两个操作。所以半同态加密就够用了，不需要昂贵的全同态。

### 性能问题

同态加密的主要瓶颈：
- 密文膨胀：一个 32-bit 浮点数加密后可能变成 2048-bit
- 计算开销：比明文计算慢 3-6 个数量级
- CKKS 方案支持近似计算，对 ML 场景更友好（允许一定精度损失）

**这里我一开始理解错了**：以为同态加密能完美保护隐私且无性能损失，实际上对于大模型（比如 LLM 的几十亿参数），全同态加密目前完全不可行。实际系统通常只加密敏感的梯度子集或用更轻量的方案。

## 安全聚合（Secure Aggregation）

比同态加密更轻量的方案，核心思想是**掩码抵消**：

```
客户端 A 和 B 协商一个随机掩码 mask_AB
A 上传: gradient_A + mask_AB
B 上传: gradient_B - mask_AB

服务器聚合: (gradient_A + mask_AB) + (gradient_B - mask_AB) = gradient_A + gradient_B
```

服务器只能看到聚合结果，看不到单个客户端的梯度。Google 的 Gboard 键盘预测就用了这个方案。

## 三种隐私技术对比

| 维度 | 差分隐私 | 同态加密 | 安全聚合 |
|------|----------|----------|----------|
| 保护对象 | 单条数据的影响 | 原始数据/梯度 | 单个客户端的梯度 |
| 计算开销 | 低（只加噪声） | 高（密文计算） | 中（掩码生成） |
| 通信开销 | 无额外 | 高（密文膨胀） | 中（掩码协商） |
| 精度影响 | 有（噪声降低精度） | 无/极小 | 无 |
| 可组合性 | 隐私预算会消耗 | 无限次使用 | 无限次使用 |
| 防御能力 | 防推断攻击 | 防服务器窥探 | 防服务器窥探 |

**实际系统通常组合使用**：安全聚合保护传输过程 + 差分隐私保护聚合结果。

## 联邦学习的攻击与防御

### 攻击类型

1. **梯度反演攻击**：从共享的梯度反推原始训练数据
   ```
   已知: ∇L(x, y; θ)  # 梯度
   求解: argmin_x' ||∇L(x', y'; θ) - ∇L(x, y; θ)||²  # 优化找原始输入
   ```
   这说明"只共享梯度"并不安全！

2. **投毒攻击**：恶意客户端上传有毒的模型更新
3. **搭便车攻击**：客户端不真正训练，只享受全局模型

### 防御

- 梯度裁剪 + 差分隐私噪声 → 防梯度反演
- Byzantine-robust 聚合（如 Krum、Trimmed Mean）→ 防投毒
- 贡献度评估 → 防搭便车

## 和其他主题的交叉

1. **和 LLM 微调的关系**：联邦学习 + LoRA 是当前热点——各客户端只训练和共享 LoRA adapter（参数量小），大幅降低通信成本。这和参数高效微调（PEFT）的思想结合得很好。
2. **和知识图谱的关系**：跨机构构建知识图谱时，各方不愿共享原始数据，可以用联邦学习训练 KG embedding（FedE），只共享实体/关系的向量表示。
3. **和 Agent 框架的关系**：多 Agent 协作场景中，如果各 Agent 持有私有知识库，联邦学习提供了一种在不暴露私有数据的前提下协作学习的范式。

## 面试准备 checklist

- [ ] 能手写 FedAvg 伪代码
- [ ] 能解释 Non-IID 数据为什么导致 client drift
- [ ] 能写出差分隐私的数学定义并解释直觉
- [ ] 能对比三种隐私技术的适用场景
- [ ] 能解释为什么"只共享梯度"不够安全（梯度反演攻击）
- [ ] 知道 FedAvg + DP + Secure Aggregation 的组合方案

## 遗留疑问

- 联邦学习在 LLM 预训练阶段有意义吗？通信成本和模型规模的矛盾怎么解决？
- 差分隐私的 ε 在实际部署中怎么选？有没有自动化的方法？
- 异构设备（手机 vs 服务器）的联邦学习，怎么处理算力差异导致的 straggler 问题？
