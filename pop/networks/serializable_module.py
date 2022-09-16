import sys
from abc import ABC, abstractmethod
from typing import Optional, Any, Dict, TypeVar
import torch as th
from pathlib import Path

T = TypeVar("T")


class SerializableModule(ABC):
    def __init__(self, log_dir: Optional[str], name: Optional[str]):
        self.log_file = self._get_log_file(log_dir, name)
        self.number_of_saves = 0

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

    @staticmethod
    def _add_counter_to_file_path(log_file: str, counter: int) -> str:
        log_file_path = Path(log_file)
        log_file_path_name_split = log_file_path.name.split(".")
        return str(
            Path(
                log_file_path.parents[0],
                log_file_path_name_split[0]
                + "_"
                + str(counter)
                + "."
                + ".".join(log_file_path_name_split[1:]),
            )
        )

    @staticmethod
    def _get_last_saved_checkpoint(log_file: str) -> int:
        return max(
            [
                int(dir_object.stem.split("_")[-1])
                for dir_object in Path(log_file).parents[0].iterdir()
                if dir_object.is_file()
            ]
        )

    def save(self: T) -> None:
        if self.log_file is None:
            raise Exception("Called save() in " + self.name + " with None log_dir")

        checkpoint = self.get_state()
        if self.number_of_saves == 0 and list(Path(self.log_file).parents[0].iterdir()):
            self.number_of_saves = self._get_last_saved_checkpoint(self.log_file) + 2
        else:
            self.number_of_saves += 1

        th.save(
            checkpoint,
            self._add_counter_to_file_path(self.log_file, self.number_of_saves - 1),
        )

    @classmethod
    def load(
        cls,
        log_file: Optional[str] = None,
        checkpoint: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> T:
        checkpoint: Dict[str, Any] = SerializableModule._load_checkpoint(
            log_file, checkpoint
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
        if log_file is not None:
            last_saved_checkpoint = SerializableModule._get_last_saved_checkpoint(
                log_file
            )
            checkpoint_to_load = SerializableModule._add_counter_to_file_path(
                log_file, last_saved_checkpoint
            )
            print("Loaded Last Checkpoint: " + str(checkpoint_to_load))
            while last_saved_checkpoint >= 0:
                try:
                    return th.load(checkpoint_to_load)
                except Exception as e:
                    print(
                        "Exception encountered when loading checkpoint "
                        + str(last_saved_checkpoint)
                    )
                    print(e)

                last_saved_checkpoint -= 1
                checkpoint_to_load = SerializableModule._add_counter_to_file_path(
                    log_file, last_saved_checkpoint
                )

            print("There is no valid checkpoint left to reload")
            sys.exit(
                0
            )  # We usually run in an until loop in a bash script, this allows to break it

        return checkpoint_dict
