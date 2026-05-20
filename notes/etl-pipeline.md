
# ETL Pipeline 设计：Airflow、数据质量监控

## 什么是 ETL

Extract-Transform-Load，数据从源系统到目标系统的搬运和加工过程。听起来简单，但工程复杂度极高——因为要处理的是"脏活累活"：数据延迟、格式不一致、上游 schema 变更、任务失败重试……

现在也有人说 ELT（先 Load 到数据湖再 Transform），本质区别是 Transform 发生在哪里。云数据仓库（BigQuery、Snowflake）算力便宜后，ELT 越来越流行。

## Airflow 核心概念

Apache Airflow 是目前最主流的工作流编排工具（虽然 Dagster、Prefect 在追赶）。

### 核心抽象

```python
# 一个最简单的 DAG
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'data_team',
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
    'email_on_failure': True,
}

with DAG(
    dag_id='feature_pipeline',
    default_args=default_args,
    schedule_interval='0 2 * * *',  # 每天凌晨2点
    start_date=datetime(2024, 1, 1),
    catchup=False,  # 不回填历史
) as dag:
    
    extract = PythonOperator(
        task_id='extract_user_events',
        python_callable=extract_from_kafka,
    )
    
    transform = PythonOperator(
        task_id='compute_features',
        python_callable=compute_user_features,
    )
    
    load = PythonOperator(
        task_id='load_to_feature_store',
        python_callable=write_to_redis,
    )
    
    extract >> transform >> load
```

关键概念：
- **DAG**：有向无环图，定义任务依赖关系
- **Operator**：任务的执行单元（PythonOperator、BashOperator、SparkSubmitOperator 等）
- **Task Instance**：某次运行中的具体任务实例
- **XCom**：任务间传递小数据的机制（注意：不适合传大数据）
- **Connection/Variable**：管理外部系统连接和配置

> **面试高频考点**：Airflow 的 executor 类型。LocalExecutor（单机多进程）、CeleryExecutor（分布式）、KubernetesExecutor（每个 task 一个 pod）。大规模生产环境一般用 Kubernetes。

### 我踩过的坑

1. **catchup=True 的坑**：默认 Airflow 会回填 start_date 到现在的所有调度。如果 start_date 设得很早，一上线就疯狂跑历史任务。一定要显式设 `catchup=False`
2. **XCom 传大数据**：XCom 存在 metadata DB 里，传大 DataFrame 会把 DB 撑爆。大数据应该写到 S3/GCS，XCom 只传路径
3. **时区问题**：Airflow 内部用 UTC，但 `schedule_interval` 的 cron 表达式也是 UTC。和业务方沟通时要注意时区转换
4. **DAG 解析性能**：DAG 文件里不要有重计算逻辑（比如读数据库），因为 scheduler 会频繁解析所有 DAG 文件

### Airflow vs 其他编排工具

| 工具 | 优势 | 劣势 |
|------|------|------|
| Airflow | 生态成熟、社区大 | 本地开发体验差、DAG 解析慢 |
| Dagster | 类型系统好、本地测试友好 | 社区相对小 |
| Prefect | Python-native、动态 DAG | 商业化倾向 |
| dbt | SQL Transform 专精 | 只做 T，不做 E 和 L |

实际项目中经常是 Airflow 编排 + dbt 做 Transform 的组合。

## 数据质量监控

数据质量问题是 ML 系统最常见的 silent failure。模型效果突然下降，80% 的原因是上游数据出了问题。

### 数据质量的维度

```
完整性（Completeness）：是否有缺失值、数据是否到齐
准确性（Accuracy）：数值是否在合理范围
一致性（Consistency）：同一实体在不同表中是否一致
时效性（Timeliness）：数据是否按时到达
唯一性（Uniqueness）：是否有重复记录
```

### 实现方案

**方案一：自定义检查（适合小团队）**

```python
# 在 Airflow DAG 里加数据质量检查 task
def check_data_quality(**context):
    df = read_today_data()
    
    # 完整性检查
    null_ratio = df.isnull().sum() / len(df)
    assert null_ratio['user_id'] == 0, "user_id 不应有空值"
    assert null_ratio['amount'] < 0.05, f"amount 空值率 {null_ratio['amount']:.2%} 超过阈值"
    
    # 范围检查
    assert df['age'].between(0, 150).all(), "age 存在异常值"
    assert df['amount'].min() >= 0, "amount 不应为负"
    
    # 数量级检查（和历史对比）
    yesterday_count = get_yesterday_count()
    today_count = len(df)
    change_ratio = abs(today_count - yesterday_count) / yesterday_count
    assert change_ratio < 0.3, f"数据量波动 {change_ratio:.2%} 超过 30%"
    
    # 分布漂移检查
    # 这里可以用 KS 检验或 PSI
    
quality_check = PythonOperator(
    task_id='data_quality_check',
    python_callable=check_data_quality,
)

extract >> transform >> quality_check >> load
```

**方案二：Great Expectations（业界标准）**

```python
import great_expectations as gx

context = gx.get_context()

# 定义 expectation suite
suite = context.add_expectation_suite("user_features_suite")

# 声明式定义数据质量规则
validator.expect_column_values_to_not_be_null("user_id")
validator.expect_column_values_to_be_between("age", min_value=0, max_value=150)
validator.expect_column_mean_to_be_between("purchase_amount", min_value=10, max_value=1000)
validator.expect_table_row_count_to_be_between(min_value=100000, max_value=500000)
```

**方案三：数据可观测性平台**

Monte Carlo、Bigeye 这类 SaaS 产品，自动学习数据的正常模式，异常时告警。适合数据量大、表多的场景。

### 数据漂移检测

> 和特征工程的关系：特征分布漂移（data drift）是模型效果衰减的主要原因。监控特征分布是 MLOps 的核心环节。

常用方法：
- **PSI (Population Stability Index)**：衡量两个分布的差异

$$PSI = \sum_{i=1}^n (A_i - E_i) \times \ln\frac{A_i}{E_i}$$

其中 $A_i$ 是实际分布在第 i 个 bin 的占比，$E_i$ 是期望分布的占比。

PSI 解读：
- < 0.1：稳定
- 0.1-0.25：需要关注
- > 0.25：显著漂移，需要干预

- **KS 检验**：非参数检验，判断两个分布是否相同
- **JS 散度**：对称版的 KL 散度，数值更稳定

```python
import numpy as np

def compute_psi(expected, actual, bins=10):
    """计算 PSI"""
    breakpoints = np.linspace(0, 1, bins + 1)
    
    expected_percents = np.histogram(expected, breakpoints)[0] / len(expected)
    actual_percents = np.histogram(actual, breakpoints)[0] / len(actual)
    
    # 避免除零
    expected_percents = np.clip(expected_percents, 1e-4, None)
    actual_percents = np.clip(actual_percents, 1e-4, None)
    
    psi = np.sum((actual_percents - expected_percents) * 
                  np.log(actual_percents / expected_percents))
    return psi
```

## 生产环境的 Pipeline 设计原则

1. **幂等性（Idempotency）**：同一个任务重跑结果一样。用 `INSERT OVERWRITE` 而不是 `INSERT INTO`
2. **可回溯性**：数据分区按日期组织，方便回填和排查
3. **失败隔离**：一个 task 失败不应该影响无依赖的其他 task
4. **监控告警**：SLA 监控（任务是否按时完成）+ 数据质量告警
5. **文档化**：每个 DAG 写清楚 owner、数据源、下游依赖

> 和数据标注的关系：标注数据的产出也是一个 ETL 过程——从标注平台 Extract 标注结果，Transform（清洗、一致性检验、格式转换），Load 到训练数据存储。可以用同样的 Airflow 框架来编排。

## 面试常见问题

- "数据延迟了怎么办？" → Sensor 等待 + 超时告警 + 降级策略（用昨天的数据）
- "如何保证 exactly-once？" → 幂等设计 + checkpoint + 事务
- "Airflow 的 scheduler 是怎么工作的？" → 定期扫描 DAG 文件，根据 schedule_interval 和依赖关系创建 task instance，放入 executor 队列
- "线上特征和离线特征不一致怎么排查？" → Feature Store 统一管理 + 线上线下用同一份计算逻辑 + 定期 diff 检测

## 个人总结

ETL 看起来不 sexy，但它是整个数据/ML 系统的地基。面试中如果能把 "我设计了一个 robust 的 pipeline，包含质量监控和自动告警" 讲清楚，比单纯说 "我用了 XXX 模型" 更能体现工程能力。
