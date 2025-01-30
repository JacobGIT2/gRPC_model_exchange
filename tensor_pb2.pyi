from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class Tensor(_message.Message):
    __slots__ = ("id", "size", "tensor")
    ID_FIELD_NUMBER: _ClassVar[int]
    SIZE_FIELD_NUMBER: _ClassVar[int]
    TENSOR_FIELD_NUMBER: _ClassVar[int]
    id: int
    size: _containers.RepeatedScalarFieldContainer[int]
    tensor: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, id: _Optional[int] = ..., size: _Optional[_Iterable[int]] = ..., tensor: _Optional[_Iterable[float]] = ...) -> None: ...
