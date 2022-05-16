from enum import auto, Enum


class TaskType(Enum):
    ACT = auto()
    STEP = auto()
    POISON = auto()


class AgentTask:
    def __init__(self, task_type: TaskType, **kwargs):
        self.task_type = task_type
        self.kwargs = kwargs

    def __str__(self):
        return str(self.task_type) + " with " + str(self.kwargs)
