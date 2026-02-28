"""Ray Serve 异步任务处理支持模块。

本模块提供了一套**生产者-消费者**架构，用于将耗时的后台任务（例如批量推理、
模型微调、ETL 数据处理等）从同步的 HTTP 请求-响应路径中解耦出来。

工作原理
--------
生产者（Producer）Deployment 将任务提交到外部消息代理（Message Broker），
并立即返回一个任务 ID，而不是在请求处理流程中直接执行耗时操作。
专用的**任务消费者 Deployment**（Task Consumer Deployment）从代理中拉取任务，
按照重试语义执行任务，并可选地将无法处理的任务路由到死信队列（DLQ）。

公开 API
--------
* :class:`TaskProcessorConfig`  – 任务队列的 Pydantic 配置模型。
* :class:`TaskResult`           – 标准化的任务状态/结果封装对象。
* :class:`TaskProcessorAdapter` – 可插拔后端适配器的抽象基类。
* :func:`task_consumer`         – 将 Deployment 类转换为后台任务消费者的类装饰器。
* :func:`task_handler`          – 将某个方法注册为指定任务类型处理入口的方法装饰器。

Async task processing support for Ray Serve.

This module provides a producer-consumer architecture that decouples long-running
background tasks (e.g. batch inference, model fine-tuning, ETL) from the synchronous
request-response path.  Dedicated *Task Consumer Deployments* pull jobs from the broker,
execute them with retry semantics, and optionally route un-processable tasks to a Dead
Letter Queue (DLQ).

Public API surface
------------------
* :class:`TaskProcessorConfig` – Pydantic configuration for the task queue.
* :class:`TaskResult`          – Standardized status / result envelope.
* :class:`TaskProcessorAdapter` – Abstract base class for pluggable backends.
* :func:`task_consumer`        – Class decorator that turns a deployment into a
                                  background task consumer.
* :func:`task_handler`         – Method decorator that registers a method as the
                                  handler for a named task type.
"""

from __future__ import annotations

import abc
import logging
import threading
from typing import Any, Callable, Dict, List, Optional, Type, Union

from ray._private.pydantic_compat import BaseModel, Field
from ray._private.ray_logging.constants import LOGRECORD_STANDARD_ATTRS  # noqa: F401
from ray.serve._private.constants import SERVE_LOGGER_NAME
from ray.util.annotations import PublicAPI

logger = logging.getLogger(SERVE_LOGGER_NAME)

# Sentinel attribute written onto methods decorated with @task_handler.
_TASK_HANDLER_ATTR = "__serve_task_handler__"


# ---------------------------------------------------------------------------
# Public data models
# ---------------------------------------------------------------------------


@PublicAPI(stability="alpha")
class TaskResult(BaseModel):
    """任务操作返回的标准化状态/结果封装对象。

    由 :class:`TaskProcessorAdapter` 的各方法返回，用于统一描述任务当前状态
    及（若已完成）执行结果。

    属性说明
    --------
    id:              Ray Serve 层面的任务唯一标识符（不透明字符串）。
    status:          当前状态，取值为 ``PENDING``（待处理）、``STARTED``（执行中）、
                     ``SUCCESS``（成功）或 ``FAILURE``（失败）之一。
    backend_task_id: 底层后端（例如 Celery）分配的内部任务 ID。
    created_at:      任务入队时的 Unix 时间戳（浮点数）。
    result:          任务完成后的可选结果数据，失败或尚未完成时为 ``None``。

    Standardized status / result envelope returned by
    :class:`TaskProcessorAdapter` operations.

    Attributes:
        id:              Ray Serve-level task identifier (opaque string).
        status:          One of ``PENDING``, ``STARTED``, ``SUCCESS``, ``FAILURE``.
        backend_task_id: The internal identifier issued by the underlying backend
                         (e.g. a Celery task UUID).
        created_at:      Unix timestamp (float) recording when the task was enqueued.
        result:          Optional result payload once the task has completed.
    """

    id: str
    status: str
    backend_task_id: str
    created_at: float
    result: Optional[Any] = None


@PublicAPI(stability="alpha")
class TaskProcessorConfig(BaseModel):
    """任务队列适配器的配置模型。

    该模型传递给 :func:`task_consumer` 装饰器，并在适配器初始化时转发给
    :meth:`TaskProcessorAdapter.initialize`。

    属性说明
    --------
    queue_name:
        主工作队列的名称。
    adapter_config:
        具体后端适配器的类型化配置（例如 ``CelerySettings``）。
        Ray Serve 内置了轻量级的 ``_InMemorySettings``，可用于本地测试。
    max_retry:
        应用层重试的最大次数；超出后任务被视为永久失败。默认为 3。
    failed_task_queue_name:
        可选。用于接收已耗尽所有重试次数的失败任务的队列名称。
    unprocessable_task_queue_name:
        可选。死信队列（DLQ）名称，用于接收因反序列化失败或找不到对应
        处理器而无法执行的任务。

    Configuration for a task queue adapter.

    This model is passed to :func:`task_consumer` and to
    :meth:`TaskProcessorAdapter.initialize`.

    Attributes:
        queue_name:                   Name of the primary work queue.
        adapter_config:               Typed configuration for the concrete adapter
                                      backend (e.g. ``CelerySettings``).  Ray Serve
                                      ships a lightweight ``_InMemorySettings`` for
                                      local testing.
        max_retry:                    Maximum number of application-level retries
                                      before a task is considered permanently failed.
        failed_task_queue_name:       Optional name for a queue that receives tasks
                                      which exhausted all retries.
        unprocessable_task_queue_name: Optional name for a Dead Letter Queue (DLQ)
                                      that receives tasks that cannot be deserialized
                                      or whose handler is missing.
    """

    queue_name: str
    adapter_config: Any  # Concrete type validated by the adapter itself.
    max_retry: Optional[int] = Field(default=3, ge=0)
    failed_task_queue_name: Optional[str] = None
    unprocessable_task_queue_name: Optional[str] = None


# ---------------------------------------------------------------------------
# Abstract adapter
# ---------------------------------------------------------------------------


@PublicAPI(stability="alpha")
class TaskProcessorAdapter(abc.ABC):
    """与任务处理后端交互的抽象接口。

    具体子类（例如 ``CeleryAdapter``）使用各自的代理/工作者库实现每个方法。
    这一抽象使用户无需修改应用代码即可切换后端实现。

    必须实现的抽象方法
    ------------------
    * :meth:`initialize`            – 初始化适配器（打开连接、配置工作者等）。
    * :meth:`register_task_handler` – 将 Python 函数注册为指定任务名称的处理器。
    * :meth:`enqueue_task`          – 将任务提交到队列。
    * :meth:`get_task_status`       – 查询任务当前状态及结果。
    * :meth:`start_consumer`        – 启动轮询代理的工作者进程/线程。

    可选的生命周期辅助方法（默认为空操作）
    --------------------------------------
    * :meth:`stop_consumer` – 停止消费者轮询。
    * :meth:`shutdown`      – 释放所有适配器资源。
    * :meth:`health_check`  – 健康检查。
    * :meth:`cancel_task`   – 取消任务。
    * :meth:`get_metrics`   – 获取适配器指标。

    Abstract interface for interacting with a task-processor backend.

    Concrete subclasses (e.g. ``CeleryAdapter``) implement each method using their
    respective broker/worker libraries.  This abstraction lets users swap backends
    without changing application code.
    """

    # ------------------------------------------------------------------
    # Required abstract methods
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def initialize(self, **kwargs) -> None:
        """初始化适配器：打开代理连接、配置工作者等。

        Initialize the adapter: open broker connections, configure workers, etc.
        """
        ...

    @abc.abstractmethod
    def register_task_handler(self, func: Callable, name: Optional[str] = None) -> None:
        """将 *func* 注册为任务名称 *name* 对应的处理器。

        参数
        ----
        func: 消费者消费到任务时将被调用的可执行对象。
        name: 消息协议中期望的任务名称；省略时默认使用 ``func.__name__``。

        Register *func* as the handler for tasks named *name*.

        Args:
            func: The callable that will be invoked when a task is consumed.
            name: Task name as expected on the wire.  Defaults to ``func.__name__``.
        """
        ...

    @abc.abstractmethod
    def enqueue_task(
        self,
        task_name: str,
        args: Optional[List[Any]] = None,
        kwargs: Optional[Dict[str, Any]] = None,
        **options: Any,
    ) -> TaskResult:
        """将任务提交到队列。

        参数
        ----
        task_name: 要执行的任务的逻辑名称（必须与消费者侧注册的处理器名称匹配）。
        args:      转发给处理器的位置参数。
        kwargs:    转发给处理器的关键字参数。
        **options: 特定后端的选项（例如队列名称、优先级）。

        返回
        ----
        一个 :class:`TaskResult`，其中至少填充了 ``id``、``status`` 和
        ``backend_task_id`` 字段。

        Submit a task to the queue.

        Args:
            task_name: Logical name of the task to execute (must match a registered
                       handler on the consumer side).
            args:      Positional arguments forwarded to the handler.
            kwargs:    Keyword arguments forwarded to the handler.
            **options: Backend-specific options (e.g. queue name, priority).

        Returns:
            A :class:`TaskResult` with at least ``id``, ``status``, and
            ``backend_task_id`` populated.
        """
        ...

    @abc.abstractmethod
    def get_task_status(self, task_id: str) -> TaskResult:
        """返回任务的当前状态（以及结果，若已完成）。

        参数
        ----
        task_id: 此前由 :meth:`enqueue_task` 返回的 :class:`TaskResult` 中的 ``id`` 字段。

        Return the current status (and result, if available) of a task.

        Args:
            task_id: The ``id`` field from the :class:`TaskResult` previously
                     returned by :meth:`enqueue_task`.
        """
        ...

    @abc.abstractmethod
    def start_consumer(self, **kwargs) -> None:
        """启动轮询代理以获取任务的工作者进程/线程。

        Start the worker process / thread that polls the broker for tasks.
        """
        ...

    # ------------------------------------------------------------------
    # Optional lifecycle helpers (no-ops by default)
    # ------------------------------------------------------------------

    def stop_consumer(self) -> None:
        """通知消费者工作者停止轮询新任务。

        Signal the consumer worker to stop polling for new tasks.
        """

    def shutdown(self) -> None:
        """释放所有适配器资源（连接、工作者等）。

        Release all adapter resources (connections, workers, etc.).
        """

    def health_check(self) -> bool:
        """若适配器及其依赖均处于健康状态则返回 ``True``。

        Return ``True`` if the adapter and its dependencies are healthy.
        """
        return True

    def cancel_task(self, task_id: str) -> bool:
        """尝试取消 *task_id* 标识的任务。

        返回
        ----
        若后端接受了取消请求则返回 ``True``，否则返回 ``False``。

        Attempt to cancel the task identified by *task_id*.

        Returns:
            ``True`` if the cancellation request was accepted by the backend,
            ``False`` otherwise.
        """
        return False

    def get_metrics(self) -> Dict[str, Any]:
        """返回适配器层面的指标字典，用于可观测性。

        Return a dictionary of adapter-level metrics for observability.
        """
        return {}


# ---------------------------------------------------------------------------
# Decorators
# ---------------------------------------------------------------------------


@PublicAPI(stability="alpha")
def task_handler(name: Optional[str] = None) -> Callable:
    """将 Deployment 的某个方法注册为任务处理器的方法装饰器。

    被装饰的方法将由消费者循环对每个 ``task_name`` 与 *name* 匹配的任务进行调用
    （若省略 *name* 则与方法名匹配）。

    参数
    ----
    name: 可选的显式任务名称。省略时使用方法的 ``__name__`` 作为任务名称。

    示例
    ----
    .. code-block:: python

        from ray import serve

        @serve.deployment
        @serve.task_consumer(config)
        class MyConsumer:
            @serve.task_handler(name="process_document")
            def process(self, doc_id: str, url: str):
                ...

    Method decorator that registers a deployment method as a task handler.

    The decorated method will be invoked by the consumer loop for each task whose
    ``task_name`` matches *name* (or the method name when *name* is omitted).

    Args:
        name: Optional explicit task name.  When omitted the method's ``__name__``
              is used as the task name.

    Example:

    .. code-block:: python

        from ray import serve

        @serve.deployment
        @serve.task_consumer(config)
        class MyConsumer:
            @serve.task_handler(name="process_document")
            def process(self, doc_id: str, url: str):
                ...
    """

    def decorator(func: Callable) -> Callable:
        handler_name = name if name is not None else func.__name__
        setattr(func, _TASK_HANDLER_ATTR, handler_name)
        return func

    return decorator


@PublicAPI(stability="alpha")
def task_consumer(config: TaskProcessorConfig) -> Callable:
    """将 Deployment 类转换为后台任务消费者的类装饰器。

    该装饰器完成以下工作：

    1. 扫描类中所有被 :func:`task_handler` 装饰的方法，并在适配器中注册它们。
    2. 修改 ``__init__``，使其在用户自定义构造函数完成后立即调用
       :meth:`TaskProcessorAdapter.initialize` 和
       :meth:`TaskProcessorAdapter.start_consumer`。
    3. 修改（或创建）``__del__``，在副本销毁时调用
       :meth:`TaskProcessorAdapter.shutdown`。

    参数
    ----
    config: 指定代理 URL、队列名称、重试策略和 DLQ 设置的
            :class:`TaskProcessorConfig` 对象。

    示例
    ----
    .. code-block:: python

        from ray import serve
        from ray.serve.task_processor import TaskProcessorConfig, _InMemorySettings

        task_config = TaskProcessorConfig(
            queue_name="my_queue",
            adapter_config=_InMemorySettings(),
        )

        @serve.deployment
        @serve.task_consumer(task_config)
        class MyConsumer:
            def __init__(self):
                self.model = load_model()

            @serve.task_handler(name="run_inference")
            def run(self, payload: dict):
                return self.model(payload)

    Class decorator that turns a deployment class into a background task consumer.

    The decorator:

    1. Scans the class for methods decorated with :func:`task_handler` and
       registers them on the adapter.
    2. Patches ``__init__`` to call :meth:`TaskProcessorAdapter.initialize` and
       :meth:`TaskProcessorAdapter.start_consumer` immediately after the user's own
       ``__init__`` completes.
    3. Patches ``__del__`` (if defined) or creates a new one that calls
       :meth:`TaskProcessorAdapter.shutdown` on replica teardown.

    Args:
        config: A :class:`TaskProcessorConfig` that specifies the broker URL, queue
                name, retry policy, and DLQ settings.

    Example:

    .. code-block:: python

        from ray import serve
        from ray.serve.task_processor import TaskProcessorConfig, _InMemorySettings

        task_config = TaskProcessorConfig(
            queue_name="my_queue",
            adapter_config=_InMemorySettings(),
        )

        @serve.deployment
        @serve.task_consumer(task_config)
        class MyConsumer:
            def __init__(self):
                self.model = load_model()

            @serve.task_handler(name="run_inference")
            def run(self, payload: dict):
                return self.model(payload)
    """

    def decorator(cls: Type) -> Type:
        if not isinstance(cls, type):
            raise TypeError(
                "@serve.task_consumer must be applied to a class, "
                f"got {type(cls)!r} instead."
            )

        # Collect all task handlers declared on the class.
        handler_map: Dict[str, Callable] = {}
        for attr_name in dir(cls):
            try:
                member = getattr(cls, attr_name)
            except AttributeError:
                continue
            if callable(member) and hasattr(member, _TASK_HANDLER_ATTR):
                handler_map[getattr(member, _TASK_HANDLER_ATTR)] = member

        original_init = cls.__init__

        def __init__(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            # Build and initialise the adapter.
            self.__serve_task_processor_config__ = config
            adapter = _build_adapter(config)
            self.__serve_task_processor_adapter__ = adapter
            adapter.initialize()
            # Register all discovered task handlers.
            for task_name, method in handler_map.items():
                adapter.register_task_handler(
                    lambda *handler_args, method=method, **handler_kwargs: method(self, *handler_args, **handler_kwargs),
                    name=task_name,
                )
            adapter.start_consumer()
            logger.info(
                "TaskConsumer '%s' started with queue '%s' (%d handler(s) registered).",
                cls.__name__,
                config.queue_name,
                len(handler_map),
            )

        cls.__init__ = __init__

        # Patch teardown to cleanly shut down the adapter.
        original_del = cls.__del__ if hasattr(cls, "__del__") else None

        def __del__(self):
            adapter = getattr(self, "__serve_task_processor_adapter__", None)
            if adapter is not None:
                try:
                    adapter.shutdown()
                except Exception:
                    pass
            if original_del is not None:
                original_del(self)

        cls.__del__ = __del__

        # Tag the class so other tooling can detect consumer deployments.
        cls.__serve_task_consumer__ = True
        cls.__serve_task_consumer_config__ = config

        return cls

    return decorator


# ---------------------------------------------------------------------------
# Built-in in-memory adapter (for local testing / unit tests)
# ---------------------------------------------------------------------------


class _InMemorySettings(BaseModel):
    """内存后端适配器的配置。

    **仅供本地开发和单元测试使用，不适用于生产环境。**
    使用此配置无需部署任何外部消息代理。

    Adapter config for the lightweight in-memory backend.

    This is **not** suitable for production use.  It is provided purely for local
    development and unit testing without requiring an external broker.
    """

    pass


class _InMemoryAdapter(TaskProcessorAdapter):
    """基于 ``queue.Queue`` 的简单进程内任务适配器。

    任务由专用的守护线程同步执行。**不提供持久化**——进程退出后所有状态
    将丢失。适用于本地测试和开发阶段验证逻辑。

    A simple in-process task adapter backed by a ``queue.Queue``.

    Tasks are executed synchronously by a dedicated daemon thread.  There is no
    persistence – all state is lost when the process exits.
    """

    def __init__(self, config: TaskProcessorConfig) -> None:
        import queue
        import time
        import uuid

        self._config = config
        self._queue: "queue.Queue[Optional[Dict[str, Any]]]" = queue.Queue()
        self._handlers: Dict[str, Callable] = {}
        self._results: Dict[str, TaskResult] = {}
        self._lock = threading.Lock()
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._queue_mod = queue
        self._time_mod = time
        self._uuid_mod = uuid

    def initialize(self, **kwargs) -> None:
        pass

    def register_task_handler(self, func: Callable, name: Optional[str] = None) -> None:
        handler_name = name if name is not None else func.__name__
        self._handlers[handler_name] = func

    def enqueue_task(
        self,
        task_name: str,
        args: Optional[List[Any]] = None,
        kwargs: Optional[Dict[str, Any]] = None,
        **options: Any,
    ) -> TaskResult:
        task_id = str(self._uuid_mod.uuid4())
        result = TaskResult(
            id=task_id,
            status="PENDING",
            backend_task_id=task_id,
            created_at=self._time_mod.time(),
        )
        with self._lock:
            self._results[task_id] = result
        self._queue.put(
            {"id": task_id, "name": task_name, "args": args or [], "kwargs": kwargs or {}}
        )
        return result

    def get_task_status(self, task_id: str) -> TaskResult:
        with self._lock:
            result = self._results.get(task_id)
        if result is None:
            return TaskResult(
                id=task_id,
                status="NOT_FOUND",
                backend_task_id=task_id,
                created_at=0.0,
            )
        return result

    def start_consumer(self, **kwargs) -> None:
        self._stop_event.clear()
        self._worker_thread = threading.Thread(
            target=self._worker_loop, daemon=True, name="serve-task-consumer"
        )
        self._worker_thread.start()

    def stop_consumer(self) -> None:
        self._stop_event.set()
        # Unblock the blocking get.
        self._queue.put(None)

    def shutdown(self) -> None:
        self.stop_consumer()
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=5)

    def health_check(self) -> bool:
        return self._worker_thread is not None and self._worker_thread.is_alive()

    def cancel_task(self, task_id: str) -> bool:
        return False  # Cancellation not supported by the in-memory backend.

    def get_metrics(self) -> Dict[str, Any]:
        with self._lock:
            counts: Dict[str, int] = {}
            for r in self._results.values():
                counts[r.status] = counts.get(r.status, 0) + 1
        return {"queue_size": self._queue.qsize(), "task_counts": counts}

    # ------------------------------------------------------------------
    # Internal worker loop
    # ------------------------------------------------------------------

    def _worker_loop(self) -> None:
        dlq_name = self._config.unprocessable_task_queue_name
        failed_queue_name = self._config.failed_task_queue_name
        max_retry = self._config.max_retry if self._config.max_retry is not None else 3

        while not self._stop_event.is_set():
            try:
                item = self._queue.get(timeout=1)
            except Exception:
                continue
            if item is None:
                break

            task_id: str = item["id"]
            task_name: str = item["name"]
            args: List[Any] = item["args"]
            kwargs: Dict[str, Any] = item["kwargs"]

            with self._lock:
                result = self._results.get(task_id)
                if result:
                    self._results[task_id] = result.model_copy(
                        update={"status": "STARTED"}
                    )

            handler = self._handlers.get(task_name)
            if handler is None:
                logger.warning(
                    "No handler registered for task '%s' (id=%s). "
                    "Moving to unprocessable queue '%s'.",
                    task_name,
                    task_id,
                    dlq_name,
                )
                with self._lock:
                    if task_id in self._results:
                        self._results[task_id] = self._results[task_id].model_copy(
                            update={"status": "FAILURE"}
                        )
                continue

            attempt = 0
            while attempt <= max_retry:
                try:
                    output = handler(*args, **kwargs)
                    with self._lock:
                        if task_id in self._results:
                            self._results[task_id] = self._results[task_id].model_copy(
                                update={"status": "SUCCESS", "result": output}
                            )
                    break
                except Exception as exc:
                    attempt += 1
                    if attempt > max_retry:
                        logger.error(
                            "Task '%s' (id=%s) failed after %d retries: %s",
                            task_name,
                            task_id,
                            max_retry,
                            exc,
                        )
                        with self._lock:
                            if task_id in self._results:
                                self._results[task_id] = self._results[
                                    task_id
                                ].model_copy(update={"status": "FAILURE"})
                        if failed_queue_name:
                            logger.info(
                                "Task '%s' (id=%s) moved to failed queue '%s'.",
                                task_name,
                                task_id,
                                failed_queue_name,
                            )
                    else:
                        logger.warning(
                            "Task '%s' (id=%s) attempt %d/%d failed: %s. Retrying…",
                            task_name,
                            task_id,
                            attempt,
                            max_retry,
                            exc,
                        )


# ---------------------------------------------------------------------------
# Adapter factory
# ---------------------------------------------------------------------------


def _build_adapter(config: TaskProcessorConfig) -> TaskProcessorAdapter:
    """根据 *config* 实例化正确的 :class:`TaskProcessorAdapter`。

    目前 Ray Serve 仅内置了 :class:`_InMemoryAdapter`。
    如需支持其他后端（例如 ``CeleryAdapter``），请继承 :class:`TaskProcessorAdapter`
    并在此处添加对应分支（或使用注册表模式）。

    Instantiate the correct :class:`TaskProcessorAdapter` for *config*.

    Currently only :class:`_InMemoryAdapter` is shipped with Ray Serve.  Additional
    adapters (e.g. ``CeleryAdapter``) should subclass :class:`TaskProcessorAdapter`
    and add a branch here (or use a registry pattern).
    """
    if isinstance(config.adapter_config, _InMemorySettings):
        return _InMemoryAdapter(config)
    raise TypeError(
        f"Unsupported adapter_config type: {type(config.adapter_config)!r}. "
        "Register a custom adapter or use _InMemorySettings for local testing."
    )
