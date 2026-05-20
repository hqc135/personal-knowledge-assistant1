
# Agent 框架：ReAct、AutoGPT、LangGraph 设计模式

> 这块是最近面试被问最多的方向之一，整理自论文 + LangChain 文档 + 自己跑 demo 的经验

## 什么是 LLM Agent

一句话：**LLM + 工具调用 + 规划能力 + 记忆 = Agent**

和普通的 prompt engineering 区别在于：Agent 有一个循环（loop），能根据环境反馈动态决定下一步行动，而不是一次性生成答案。

## ReAct：最经典的 Agent 范式

### 核心思想

ReAct = Reasoning + Acting，交替进行"思考"和"行动"：

```
循环:
  Thought: 我需要查找XXX的信息
  Action: search("XXX")
  Observation: [搜索结果]
  Thought: 根据结果，我现在知道了...还需要...
  Action: lookup("YYY")
  Observation: [查询结果]
  ...
  Thought: 我现在有足够信息了
  Action: finish(answer)
```

### 为什么有效

- Thought 步骤让 LLM 做 chain-of-thought 推理，减少幻觉
- Action 步骤让 LLM 获取外部信息，弥补知识截断
- 交替进行比纯 CoT 或纯 Action 都好（论文有消融实验）

**和 Transformer 的 attention 机制类比**：ReAct 的 Thought 类似于 self-attention（模型内部推理），Action 类似于 cross-attention（从外部获取信息）。

### 面试高频考点

Q: ReAct 和 CoT 的区别？
A: CoT 只有推理没有行动，容易幻觉；ReAct 能通过工具调用 ground truth 来验证推理。

## AutoGPT：自主 Agent 的早期探索

### 架构

```python
while not task_complete:
    # 1. 规划
    plan = llm.think(goal, memory, feedback)
    
    # 2. 选择工具并执行
    action = llm.select_action(plan, available_tools)
    result = execute(action)
    
    # 3. 更新记忆
    memory.add(action, result)
    
    # 4. 自我反思
    feedback = llm.reflect(result, goal)
```

### 问题（踩坑记录）

**这里我一开始理解错了**：以为 AutoGPT 很强大能解决复杂任务，实际跑起来发现：

1. **循环陷阱**：经常在几个 action 之间死循环，没有有效的终止条件
2. **成本爆炸**：每一步都要调用 LLM，复杂任务可能调用几十上百次
3. **规划能力弱**：LLM 的长期规划能力其实很差，经常走偏

这也是为什么后来社区转向了更结构化的框架（如 LangGraph）。

## LangGraph：状态机 + Agent

### 核心设计模式

LangGraph 把 Agent 建模为一个**有向图（DAG 或有环图）**：

```python
from langgraph.graph import StateGraph

# 定义状态
class AgentState(TypedDict):
    messages: list
    next_action: str

# 定义节点（每个节点是一个处理函数）
graph = StateGraph(AgentState)
graph.add_node("reason", reasoning_node)
graph.add_node("act", action_node)
graph.add_node("reflect", reflection_node)

# 定义边（转移条件）
graph.add_conditional_edges("reason", decide_next_step, {
    "need_action": "act",
    "done": END
})
graph.add_edge("act", "reflect")
graph.add_edge("reflect", "reason")
```

### 关键设计模式

1. **Router 模式**：一个节点根据输入分发到不同子图
2. **Plan-and-Execute**：先生成完整计划，再逐步执行，执行中可以修改计划
3. **Multi-Agent**：多个 Agent 协作，每个有不同角色（类似微服务架构）
4. **Human-in-the-loop**：在关键节点暂停等待人类确认

### 和传统软件工程的对比

| 概念 | 传统软件 | Agent 框架 |
|------|----------|-----------|
| 控制流 | if/else, 循环 | LLM 决策 + 条件边 |
| 状态管理 | 数据库/变量 | State + Memory |
| 错误处理 | try/catch | 反思 + 重试 |
| 模块化 | 函数/类 | 节点/子图 |

## 记忆系统设计

Agent 的记忆分三层（**和计算机存储层次结构类比**）：

- **工作记忆**（类似寄存器）：当前对话上下文，在 context window 里
- **短期记忆**（类似内存）：最近几轮交互的摘要，用 buffer/summary 存
- **长期记忆**（类似磁盘）：向量数据库存储的历史经验，需要检索

```python
# 典型的记忆检索流程
relevant_memories = vector_store.similarity_search(current_query, k=5)
context = format_memories(relevant_memories)
response = llm(system_prompt + context + current_query)
```

**这和 RAG 的检索增强生成本质上是同一个模式**——都是"检索相关信息 → 注入上下文 → 生成"。

## 工具调用的实现

### Function Calling 范式

```json
{
  "name": "search_web",
  "description": "搜索互联网获取最新信息",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {"type": "string", "description": "搜索关键词"}
    },
    "required": ["query"]
  }
}
```

LLM 输出结构化的工具调用请求，框架负责执行并把结果返回。**面试考点：如何设计工具描述让 LLM 准确选择工具？**

答：描述要具体、有边界条件、给 few-shot 示例。

## 和其他主题的交叉

1. **和知识图谱的关系**：Agent 可以把知识图谱作为工具——用 SPARQL 查询结构化知识，比纯向量检索更精确
2. **和多模态的关系**：多模态 Agent 需要视觉感知能力（如 LLaVA）作为输入模块，才能处理图像/视频相关任务
3. **和联邦学习的关系**：多 Agent 协作场景下，如果各 Agent 持有私有数据，可以用联邦学习的思想做隐私保护的协作推理

## 面试准备 checklist

- [ ] 能手写 ReAct 的 prompt template
- [ ] 能解释 LangGraph 的状态机模型
- [ ] 能分析 Agent 循环不收敛的原因和解决方案
- [ ] 能设计一个 multi-agent 系统的架构图