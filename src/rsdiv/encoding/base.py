from abc import ABCMeta, abstractclassmethod, abstractmethod
from typing import Any, Dict, List, Union


class BaseEncoder(metaclass=ABCMeta):
    encode_source: Dict[str, Any]

    @abstractmethod
    def encoding_single(cls, org: Union[List, str]) -> Union[int, str]:
        raise NotImplementedError("embedding_single must be implemented.")
