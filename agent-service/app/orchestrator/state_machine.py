from enum import Enum


class TaskState(Enum):
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    DEGRADED = "degraded"


TRANSITIONS: set[tuple[TaskState, TaskState]] = {
    (TaskState.PENDING, TaskState.READY),
    (TaskState.READY, TaskState.RUNNING),
    (TaskState.RUNNING, TaskState.SUCCESS),
    (TaskState.RUNNING, TaskState.FAILED),
    (TaskState.FAILED, TaskState.RUNNING),    # retry
    (TaskState.FAILED, TaskState.DEGRADED),   # retries exhausted
}


def can_transition(src: TaskState, dst: TaskState) -> bool:
    return (src, dst) in TRANSITIONS


def is_terminal(state: TaskState) -> bool:
    return state in (TaskState.SUCCESS, TaskState.DEGRADED)
