from _typeshed import Incomplete

class LabelsResponse:
    openapi_types: Incomplete
    attribute_map: Incomplete
    discriminator: Incomplete
    def __init__(self, labels: Incomplete | None = None, links: Incomplete | None = None) -> None: ...
    @property
    def labels(self): ...
    @labels.setter
    def labels(self, labels) -> None: ...
    @property
    def links(self): ...
    @links.setter
    def links(self, links) -> None: ...
    def to_dict(self): ...
    def to_str(self): ...
    def __eq__(self, other): ...
    def __ne__(self, other): ...
