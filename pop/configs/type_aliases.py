from typing import Dict, Union

ParsedTOMLDict = Dict[str, Dict[str, Union[int, float, bool, str]]]
EventuallyNestedDict = Dict[
    str,
    Union[int, float, bool, str, Dict[str, Union[int, float, bool, str]]],
]
