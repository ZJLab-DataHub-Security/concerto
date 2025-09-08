from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class CPTMessage:
    tokens: List[List[int]]
    channels: Optional[List[str]]

@dataclass
class SFTMessage:
    input_ids: List[List[int]]
    labels: List[List[int]]
    channels: Optional[List[str]]

@dataclass
class CPTConcatedMessage:
    result: List[List[int]]
    channels: Optional[List[List[Tuple[str, int]]]]
    length: int

@dataclass
class SFTConcatedMessage:
    result: List[List[int]]
    channels: Optional[List[List[Tuple[str, int]]]]
    length: int