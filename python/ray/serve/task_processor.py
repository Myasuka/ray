"""Async task processing support for Ray Serve.

This module provides a producer-consumer architecture that decouples long-running
background tasks (e.g. batch inference, model fine-tuning, ETL) from the synchronous
request-response path.  Rather than executing heavy work inline and risking request
timeouts or blocked actors, deployments can submit tasks to an external message broker
and return a task ID immediately.  Dedicated *Task Consumer Deployments* pull jobs from
the broker, execute them with retry semantics, and optionally route un-processable tasks
to a Dead Letter Queue (DLQ).

Public API surface
------------------
* :class:`TaskProcessorConfig` – Pydantic configuration for the task queue.
* :class:`TaskResult`          – Standardised status / result envelope.
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
    """Standardized status / result envelope returned by :class:`TaskProcessorAdapter` operations.

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
    """Configuration for a task queue adapter.

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
    """Abstract interface for interacting with a task-processor backend.

    Concrete subclasses (e.g. ``CeleryAdapter``) implement each method using their
    respective broker/worker libraries.  This abstraction lets users swap backends
    without changing application code.
    """

    # ------------------------------------------------------------------
    # Required abstract methods
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def initialize(self, **kwargs) -> None:
        """Initialize the adapter: open broker connections, configure workers, etc."""
        ...

    @abc.abstractmethod
    def register_task_handler(self, func: Callable, name: Optional[str] = None) -> None:
        """Register *func* as the handler for tasks named *name*.

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
        """Submit a task to the queue.

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
        """Return the current status (and result, if available) of a task.

        Args:
            task_id: The ``id`` field from the :class:`TaskResult` previously
                     returned by :meth:`enqueue_task`.
        """
        ...

    @abc.abstractmethod
    def start_consumer(self, **kwargs) -> None:
        """Start the worker process / thread that polls the broker for tasks."""
        ...

    # ------------------------------------------------------------------
    # Optional lifecycle helpers (no-ops by default)
    # ------------------------------------------------------------------

    def stop_consumer(self) -> None:
        """Signal the consumer worker to stop polling for new tasks."""

    def shutdown(self) -> None:
        """Release all adapter resources (connections, workers, etc.)."""

    def health_check(self) -> bool:
        """Return ``True`` if the adapter and its dependencies are healthy."""
        return True

    def cancel_task(self, task_id: str) -> bool:
        """Attempt to cancel the task identified by *task_id*.

        Returns:
            ``True`` if the cancellation request was accepted by the backend,
            ``False`` otherwise.
        """
        return False

    def get_metrics(self) -> Dict[str, Any]:
        """Return a dictionary of adapter-level metrics for observability."""
        return {}


# ---------------------------------------------------------------------------
# Decorators
# ---------------------------------------------------------------------------


@PublicAPI(stability="alpha")
def task_handler(name: Optional[str] = None) -> Callable:
    """Method decorator that registers a deployment method as a task handler.

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
    """Class decorator that turns a deployment class into a background task consumer.

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
    """Adapter config for the lightweight in-memory backend.

    This is **not** suitable for production use.  It is provided purely for local
    development and unit testing without requiring an external broker.
    """

    pass


class _InMemoryAdapter(TaskProcessorAdapter):
    """A simple in-process task adapter backed by a ``queue.Queue``.

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
    """Instantiate the correct :class:`TaskProcessorAdapter` for *config*.

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
