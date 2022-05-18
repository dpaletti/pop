from enum import auto, Enum


class TaskType(Enum):
    ACT = auto()
    STEP = auto()
    SAVE = auto()
    CHOOSE_ACTION = auto()
    LEARN = auto()
    LOAD = auto()
    POISON = auto()

    def __eq__(self, other):
        # Needed for interprocess Enum comparison
        return str(self) == str(other)


class Task:
    def __init__(self, task_type: TaskType, **kwargs):
        self.task_type = task_type
        self.kwargs = kwargs

    def __str__(self):
        return str(self.task_type) + " with " + str(self.kwargs)
