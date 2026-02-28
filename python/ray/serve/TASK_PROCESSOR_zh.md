# Ray Serve 异步任务处理 API

## 一、背景与动机

Ray Serve 非常擅长处理低延迟的同步推理请求，但对于以下类型的长耗时任务缺乏原生支持：

- **批量推理**（Batch Inference）
- **模型微调**（Model Fine-tuning）
- **大规模 ETL 数据处理**

如果在标准 Deployment 中直接执行上述任务，会导致：

- 请求超时（Request Timeout）
- Actor 被长期阻塞
- 缺乏容错机制（无法持久化、无法重试）

本模块（`ray/serve/task_processor.py`）提出了一套**生产者-消费者**架构来解决上述问题。

---

## 二、架构概览

```
生产者 Deployment          消息代理（Broker）        消费者 Deployment
（Task Producer）    ─────►  Redis / SQS / Kafka  ◄─────  （Task Consumer）
    │                                                           │
    │  enqueue_task(name, kwargs)                              │  @task_handler
    │  ─────────────────────────►  task_id                    │  def process(...)
    │                                                           │
    └── 立即返回 task_id ◄─────────────────────────────────────┘
```

- **生产者**：普通的 Serve Deployment，通过 `adapter.enqueue_task()` 将任务投递到消息代理，立刻返回 `task_id`。
- **消费者**：使用 `@serve.task_consumer` 装饰的 Deployment，在后台持续拉取并执行任务，内置重试和死信队列（DLQ）支持。

---

## 三、公开 API

### 3.1 `TaskProcessorConfig`

任务队列适配器的 Pydantic 配置模型。

```python
from ray.serve.task_processor import TaskProcessorConfig, _InMemorySettings

config = TaskProcessorConfig(
    queue_name="document_indexing_queue",   # 主队列名称
    adapter_config=_InMemorySettings(),     # 适配器后端配置
    max_retry=3,                            # 最大应用层重试次数（默认 3）
    failed_task_queue_name="dlq_failed",    # 耗尽重试次数后转移到的队列（可选）
    unprocessable_task_queue_name="dlq",    # 无法反序列化或找不到处理器时的 DLQ（可选）
)
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `queue_name` | `str` | 主工作队列名称 |
| `adapter_config` | `Any` | 后端适配器配置（例如 `_InMemorySettings`、`CelerySettings`） |
| `max_retry` | `int`（默认 3） | 最大应用层重试次数，超出后任务标记为 FAILURE |
| `failed_task_queue_name` | `str \| None` | 接收永久失败任务的队列名称 |
| `unprocessable_task_queue_name` | `str \| None` | 死信队列（DLQ）名称 |

---

### 3.2 `TaskResult`

标准化的任务状态/结果封装对象。

```python
from ray.serve.task_processor import TaskResult

result: TaskResult = adapter.enqueue_task("my_task", kwargs={"x": 1})
print(result.id)      # 任务 ID（Ray Serve 层）
print(result.status)  # PENDING / STARTED / SUCCESS / FAILURE
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `id` | `str` | Ray Serve 层面的任务唯一标识符 |
| `status` | `str` | 当前状态：`PENDING`、`STARTED`、`SUCCESS`、`FAILURE` |
| `backend_task_id` | `str` | 底层后端（例如 Celery）分配的内部 ID |
| `created_at` | `float` | 任务入队时的 Unix 时间戳 |
| `result` | `Any \| None` | 任务完成后的结果数据（失败或未完成时为 `None`） |

---

### 3.3 `TaskProcessorAdapter`（抽象基类）

所有后端适配器都必须继承此类并实现以下方法：

| 方法 | 说明 |
|---|---|
| `initialize(**kwargs)` | 初始化适配器（建立连接、配置工作者） |
| `register_task_handler(func, name)` | 将函数注册为指定名称的任务处理器 |
| `enqueue_task(task_name, args, kwargs, **options)` | 将任务提交到队列，返回 `TaskResult` |
| `get_task_status(task_id)` | 查询任务当前状态及结果 |
| `start_consumer(**kwargs)` | 启动工作者线程/进程，开始轮询队列 |
| `stop_consumer()` | 停止消费者轮询（可选） |
| `shutdown()` | 释放所有资源（可选） |
| `health_check()` | 健康检查，返回 `bool`（可选，默认 `True`） |
| `cancel_task(task_id)` | 取消任务（可选，默认 `False`） |
| `get_metrics()` | 获取适配器指标字典（可选） |

---

### 3.4 `@serve.task_handler(name)`

**方法装饰器**，将 Deployment 类中的某个方法注册为指定任务类型的处理入口。

```python
@serve.task_handler(name="index_document")
def index_document(self, document_id: str, url: str):
    # 实际的任务处理逻辑
    content = self.indexer.download(url)
    return self.indexer.process(content)
```

- `name`：任务的逻辑名称（字符串），需与生产者调用 `enqueue_task` 时使用的名称一致。
- 省略 `name` 时，默认使用方法的 `__name__`。

---

### 3.5 `@serve.task_consumer(config)`

**类装饰器**，将 Deployment 类转换为后台任务消费者。

装饰器会自动完成以下工作：

1. 扫描类中所有被 `@task_handler` 装饰的方法并注册到适配器。
2. 修改 `__init__`，在用户自定义构造函数完成后自动调用 `adapter.initialize()` 和 `adapter.start_consumer()`。
3. 修改（或创建）`__del__`，在副本销毁时自动调用 `adapter.shutdown()`。

---

## 四、完整使用示例

### 4.1 定义消费者 Deployment

```python
from ray import serve
from ray.serve.task_processor import TaskProcessorConfig, _InMemorySettings

task_config = TaskProcessorConfig(
    queue_name="document_indexing_queue",
    adapter_config=_InMemorySettings(),  # 本地测试用；生产环境替换为 CelerySettings 等
    max_retry=3,
    unprocessable_task_queue_name="dlq_document_indexing",
)

@serve.deployment
@serve.task_consumer(task_config)
class DocumentIndexingConsumer:
    def __init__(self):
        # 初始化一次昂贵的资源（例如模型、客户端）
        self.indexer = DocumentIndexingEngine()

    @serve.task_handler(name="index_document")
    def index_document(self, document_id: str, document_url: str):
        """由队列中每条任务触发执行。"""
        content = self.indexer.download(document_url)
        result = self.indexer.process(content)
        return {"document_id": document_id, "status": "indexed", "metadata": result}
```

### 4.2 定义生产者 Deployment

```python
from fastapi import FastAPI, Request
from ray import serve
from ray.serve.task_processor import _build_adapter

app = FastAPI()

@serve.deployment
@serve.ingress(app)
class APIProducer:
    def __init__(self, task_config, consumer_app):
        self.adapter = _build_adapter(task_config)
        self.adapter.initialize()

    @app.post("/index")
    async def submit_task(self, request: Request):
        data = await request.json()
        # 将任务投递到队列，立即返回 task_id
        task = self.adapter.enqueue_task("index_document", kwargs=data)
        return {"task_id": task.id}

    @app.get("/status/{task_id}")
    async def get_task_status(self, task_id: str):
        return self.adapter.get_task_status(task_id).dict()
```

### 4.3 绑定并部署应用

```python
# 1. 绑定消费者
consumer_app = DocumentIndexingConsumer.bind()

# 2. 绑定生产者（传入配置和消费者 handle）
producer_app = APIProducer.bind(task_config, consumer_app)

# 3. 部署（Ray Serve 会同时启动消费者 Deployment）
serve.run(producer_app)
```

---

## 五、升级与回滚注意事项

### 5.1 不兼容来源

| 变更类型 | 风险 |
|---|---|
| 任务名称变更 | 已入队的旧任务找不到处理器，进入 DLQ |
| 处理器方法删除 | 同上 |
| 参数结构变更 | 反序列化失败，进入 DLQ |

### 5.2 Ray Serve 的处理策略

- **死信队列（DLQ）**：无法反序列化或找不到处理器的任务自动转移至 `unprocessable_task_queue_name` 指定的队列，携带失败元数据（异常堆栈、载荷、任务名称）。
- **可观测性指标**：发出处理器未找到、反序列化失败的指标和日志。
- **手动恢复路径**：用户可检查 DLQ 并选择重新入队、转换迁移或主动丢弃。
- **关闭时停止轮询**：副本进入关闭流程后，停止接受新任务。

### 5.3 Ray Serve 无法保证的事项

- 无法强制校验不同版本间的任务 Schema 兼容性。
- 无法阻止外部生产者发送不符合预期格式的任务。
- 无法自动完成旧格式任务到新格式的迁移（需用户负责）。

---

## 六、术语表

| 术语 | 说明 |
|---|---|
| **Task（任务）** | 用户定义的函数及其调用参数的组合 |
| **Task Processor（任务处理器）** | 例如 Celery、ARQ、RQ 等 |
| **Broker（代理/消息队列）** | 例如 Redis、SQS、Kafka |
| **Task Processor Adapter** | Ray Serve 对不同任务处理器的可插拔抽象接口 |
| **Task Consumer Deployment** | 负责从队列拉取并执行后台任务的 Serve Deployment |
| **Task Handler** | Task Consumer Deployment 中被 `@task_handler` 标记的方法 |
| **Task Result** | 包含任务 ID、状态和可选结果的标准化响应对象 |
| **Dead Letter Queue (DLQ)** | 死信队列，存放因反序列化失败或处理器缺失而无法处理的任务 |

---

## 七、典型使用场景

- **视频推理**：单个视频推理任务耗时超过标准请求超时时长，使用异步任务处理可充分发挥 Ray Serve 的弹性伸缩能力。
- **文档索引**：将文档内容下载、解析、写入向量数据库等流程异步化，生产者立即返回任务 ID，前端轮询状态。
- **模型微调**：将训练任务投递到队列，由 GPU 节点上的消费者 Deployment 按需执行。
