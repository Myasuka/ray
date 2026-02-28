"""Unit tests for the Ray Serve async task processing API.

These tests exercise the public-API surface (TaskProcessorConfig, TaskResult,
TaskProcessorAdapter, task_consumer, task_handler) and the built-in in-memory
adapter without requiring a live Ray cluster.
"""
import threading
import time

import pytest

from ray.serve.task_processor import (
    TaskProcessorAdapter,
    TaskProcessorConfig,
    TaskResult,
    _InMemoryAdapter,
    _InMemorySettings,
    task_consumer,
    task_handler,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(**kwargs) -> TaskProcessorConfig:
    defaults = dict(queue_name="test_queue", adapter_config=_InMemorySettings())
    defaults.update(kwargs)
    return TaskProcessorConfig(**defaults)


# ---------------------------------------------------------------------------
# TaskProcessorConfig
# ---------------------------------------------------------------------------


class TestTaskProcessorConfig:
    def test_defaults(self):
        cfg = _make_config()
        assert cfg.queue_name == "test_queue"
        assert cfg.max_retry == 3
        assert cfg.failed_task_queue_name is None
        assert cfg.unprocessable_task_queue_name is None

    def test_custom_values(self):
        cfg = _make_config(
            queue_name="q1",
            max_retry=5,
            failed_task_queue_name="dlq_failed",
            unprocessable_task_queue_name="dlq_bad",
        )
        assert cfg.max_retry == 5
        assert cfg.failed_task_queue_name == "dlq_failed"
        assert cfg.unprocessable_task_queue_name == "dlq_bad"

    def test_max_retry_zero_allowed(self):
        cfg = _make_config(max_retry=0)
        assert cfg.max_retry == 0


# ---------------------------------------------------------------------------
# TaskResult
# ---------------------------------------------------------------------------


class TestTaskResult:
    def test_basic_fields(self):
        r = TaskResult(
            id="abc",
            status="PENDING",
            backend_task_id="xyz",
            created_at=1234567890.0,
        )
        assert r.id == "abc"
        assert r.status == "PENDING"
        assert r.result is None

    def test_with_result(self):
        r = TaskResult(
            id="1",
            status="SUCCESS",
            backend_task_id="1",
            created_at=0.0,
            result={"score": 0.99},
        )
        assert r.result == {"score": 0.99}


# ---------------------------------------------------------------------------
# _InMemoryAdapter
# ---------------------------------------------------------------------------


class TestInMemoryAdapter:
    def setup_method(self):
        self.config = _make_config()
        self.adapter = _InMemoryAdapter(self.config)
        self.adapter.initialize()

    def teardown_method(self):
        self.adapter.shutdown()

    def test_enqueue_returns_pending_result(self):
        result = self.adapter.enqueue_task("foo", kwargs={"x": 1})
        assert result.status == "PENDING"
        assert result.id
        assert result.backend_task_id == result.id

    def test_get_status_unknown_task(self):
        r = self.adapter.get_task_status("nonexistent")
        assert r.status == "NOT_FOUND"

    def test_handler_executes_successfully(self):
        results = []

        def my_handler(value):
            results.append(value)
            return value * 2

        self.adapter.register_task_handler(my_handler, name="double")
        self.adapter.start_consumer()

        task = self.adapter.enqueue_task("double", args=[21])
        # Wait for the worker thread to process the task.
        deadline = time.time() + 5
        while time.time() < deadline:
            r = self.adapter.get_task_status(task.id)
            if r.status in ("SUCCESS", "FAILURE"):
                break
            time.sleep(0.05)

        r = self.adapter.get_task_status(task.id)
        assert r.status == "SUCCESS"
        assert r.result == 42
        assert results == [21]

    def test_unknown_task_name_results_in_failure(self):
        self.adapter.start_consumer()
        task = self.adapter.enqueue_task("no_such_handler")
        deadline = time.time() + 5
        while time.time() < deadline:
            r = self.adapter.get_task_status(task.id)
            if r.status in ("SUCCESS", "FAILURE"):
                break
            time.sleep(0.05)

        r = self.adapter.get_task_status(task.id)
        assert r.status == "FAILURE"

    def test_retry_on_failure(self):
        config = _make_config(max_retry=2)
        adapter = _InMemoryAdapter(config)
        adapter.initialize()

        call_count = [0]

        def flaky_handler():
            call_count[0] += 1
            if call_count[0] < 3:
                raise RuntimeError("transient error")
            return "ok"

        adapter.register_task_handler(flaky_handler, name="flaky")
        adapter.start_consumer()

        task = adapter.enqueue_task("flaky")
        deadline = time.time() + 5
        while time.time() < deadline:
            r = adapter.get_task_status(task.id)
            if r.status in ("SUCCESS", "FAILURE"):
                break
            time.sleep(0.05)

        r = adapter.get_task_status(task.id)
        assert r.status == "SUCCESS"
        assert r.result == "ok"
        assert call_count[0] == 3
        adapter.shutdown()

    def test_permanent_failure_after_max_retry(self):
        config = _make_config(max_retry=1)
        adapter = _InMemoryAdapter(config)
        adapter.initialize()

        def always_fail():
            raise RuntimeError("boom")

        adapter.register_task_handler(always_fail, name="fail")
        adapter.start_consumer()

        task = adapter.enqueue_task("fail")
        deadline = time.time() + 5
        while time.time() < deadline:
            r = adapter.get_task_status(task.id)
            if r.status in ("SUCCESS", "FAILURE"):
                break
            time.sleep(0.05)

        r = adapter.get_task_status(task.id)
        assert r.status == "FAILURE"
        adapter.shutdown()

    def test_health_check_after_start(self):
        self.adapter.start_consumer()
        assert self.adapter.health_check() is True

    def test_get_metrics_structure(self):
        self.adapter.start_consumer()
        self.adapter.enqueue_task("noop")
        time.sleep(0.1)
        metrics = self.adapter.get_metrics()
        assert "queue_size" in metrics
        assert "task_counts" in metrics

    def test_cancel_task_returns_false(self):
        # In-memory adapter does not support cancellation.
        assert self.adapter.cancel_task("any") is False


# ---------------------------------------------------------------------------
# task_handler decorator
# ---------------------------------------------------------------------------


class TestTaskHandlerDecorator:
    def test_sets_attribute(self):
        from ray.serve.task_processor import _TASK_HANDLER_ATTR

        class MyClass:
            @task_handler(name="my_task")
            def process(self, x):
                return x

        assert hasattr(MyClass.process, _TASK_HANDLER_ATTR)
        assert getattr(MyClass.process, _TASK_HANDLER_ATTR) == "my_task"

    def test_defaults_to_method_name(self):
        from ray.serve.task_processor import _TASK_HANDLER_ATTR

        class MyClass:
            @task_handler()
            def do_work(self):
                pass

        assert getattr(MyClass.do_work, _TASK_HANDLER_ATTR) == "do_work"

    def test_decorated_method_still_callable(self):
        class MyClass:
            @task_handler(name="t")
            def echo(self, v):
                return v

        obj = MyClass()
        assert obj.echo(7) == 7


# ---------------------------------------------------------------------------
# task_consumer decorator
# ---------------------------------------------------------------------------


class TestTaskConsumerDecorator:
    def test_non_class_raises(self):
        with pytest.raises(TypeError, match="class"):

            @task_consumer(_make_config())
            def not_a_class():
                pass

    def test_marks_class(self):
        cfg = _make_config()

        @task_consumer(cfg)
        class FakeConsumer:
            def __init__(self):
                pass

        assert getattr(FakeConsumer, "__serve_task_consumer__", False) is True
        assert FakeConsumer.__serve_task_consumer_config__ is cfg

    def test_init_starts_consumer(self):
        cfg = _make_config()
        started = []

        class TrackedAdapter(_InMemoryAdapter):
            def start_consumer(self, **kwargs):
                started.append(True)
                super().start_consumer(**kwargs)

        # Monkey-patch the factory for this test only.
        import ray.serve.task_processor as _tp

        original_build = _tp._build_adapter
        _tp._build_adapter = lambda c: TrackedAdapter(c)
        try:
            @task_consumer(cfg)
            class MyConsumer:
                def __init__(self):
                    pass

            obj = MyConsumer()
            assert started == [True]
            obj.__del__()
        finally:
            _tp._build_adapter = original_build

    def test_handler_registered_and_executed(self):
        cfg = _make_config()
        outputs = []

        @task_consumer(cfg)
        class WorkerConsumer:
            def __init__(self):
                pass

            @task_handler(name="add")
            def add(self, a, b):
                result = a + b
                outputs.append(result)
                return result

        consumer = WorkerConsumer()
        adapter: _InMemoryAdapter = consumer.__serve_task_processor_adapter__

        task = adapter.enqueue_task("add", kwargs={"a": 3, "b": 4})
        deadline = time.time() + 5
        while time.time() < deadline:
            r = adapter.get_task_status(task.id)
            if r.status in ("SUCCESS", "FAILURE"):
                break
            time.sleep(0.05)

        r = adapter.get_task_status(task.id)
        assert r.status == "SUCCESS"
        assert r.result == 7
        consumer.__del__()
