from abc import ABC, abstractmethod
from typing import Optional, Any, Dict, TypeVar
import torch as th
from pathlib import Path

T = TypeVar("T")


class SerializableModule(ABC):
    def __init__(self, log_dir: Optional[str], name: Optional[str]):
        self.log_file = self._get_log_file(log_dir, name)

    @staticmethod
    def _get_log_file(
        log_dir: Optional[str], file_name: Optional[str]
    ) -> Optional[str]:
        if log_dir is not None:
            if file_name is None:
                raise Exception("Please pass a non-null name to get_log_file")
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            return str(Path(log_dir, file_name + ".pt"))
        return None

    @abstractmethod
    def get_state(self: T) -> Dict[str, Any]:
        ...

    def save(self: T) -> None:
        if self.log_file is None:
            raise Exception("Called save() in " + self.name + " with None log_dir")

        checkpoint = self.get_state()

        th.save(checkpoint, self.log_file)

    @classmethod
    def load(
        cls,
        log_file: Optional[str] = None,
        checkpoint_dict: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> T:
        checkpoint: Dict[str, Any] = SerializableModule._load_checkpoint(
            log_file, checkpoint_dict
        )
        return cls.factory(checkpoint, **kwargs)

    @staticmethod
    @abstractmethod
    def factory(checkpoint: Dict[str, Any], **kwargs) -> T:
        ...

    @staticmethod
    def _load_checkpoint(
        log_file: Optional[str], checkpoint_dict: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        if log_file is None and checkpoint_dict is None:
            raise Exception(
                "Cannot load module: both log_file and checkpoint_dict are None"
            )
        return th.load(log_file) if log_file is not None else checkpoint_dict
