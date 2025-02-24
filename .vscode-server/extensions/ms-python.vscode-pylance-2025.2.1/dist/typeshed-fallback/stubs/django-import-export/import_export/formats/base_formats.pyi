from _typeshed import Incomplete, ReadableBuffer
from typing import IO, Any, ClassVar
from typing_extensions import Self, TypeAlias

Dataset: TypeAlias = Incomplete  # tablib.Dataset

class Format:
    def get_title(self) -> type[Self]: ...
    def create_dataset(self, in_stream: str | bytes | IO[Any]) -> Dataset: ...
    def export_data(self, dataset: Dataset, **kwargs: Any) -> Any: ...
    def is_binary(self) -> bool: ...
    def get_read_mode(self) -> str: ...
    def get_extension(self) -> str: ...
    def get_content_type(self) -> str: ...
    @classmethod
    def is_available(cls) -> bool: ...
    def can_import(self) -> bool: ...
    def can_export(self) -> bool: ...

class TablibFormat(Format):
    TABLIB_MODULE: ClassVar[str]
    CONTENT_TYPE: ClassVar[str]
    encoding: str | None
    def __init__(self, encoding: str | None = None) -> None: ...
    def get_format(self) -> type[Any]: ...
    def get_title(self) -> str: ...  # type: ignore[override]
    def create_dataset(self, in_stream: str | bytes | IO[Any], **kwargs: Any) -> Dataset: ...  # type: ignore[override]

class TextFormat(TablibFormat): ...

class CSV(TextFormat):
    def export_data(self, dataset: Dataset, **kwargs: Any) -> str: ...

class JSON(TextFormat):
    def export_data(self, dataset: Dataset, **kwargs: Any) -> str: ...

class YAML(TextFormat):
    def export_data(self, dataset: Dataset, **kwargs: Any) -> str: ...

class TSV(TextFormat):
    def export_data(self, dataset: Dataset, **kwargs: Any) -> str: ...

class ODS(TextFormat):
    def export_data(self, dataset: Dataset, **kwargs: Any) -> bytes: ...

class HTML(TextFormat):
    def export_data(self, dataset: Dataset, **kwargs: Any) -> str: ...

class XLS(TablibFormat):
    def export_data(self, dataset: Dataset, **kwargs: Any) -> bytes: ...
    def create_dataset(self, in_stream: bytes) -> Dataset: ...  # type: ignore[override]

class XLSX(TablibFormat):
    def export_data(self, dataset: Dataset, **kwargs: Any) -> bytes: ...
    def create_dataset(self, in_stream: ReadableBuffer) -> Dataset: ...  # type: ignore[override]

DEFAULT_FORMATS: list[type[Format]]
