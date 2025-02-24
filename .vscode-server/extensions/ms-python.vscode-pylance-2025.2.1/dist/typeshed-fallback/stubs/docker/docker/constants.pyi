from collections.abc import Mapping, Sequence
from typing import Final

DEFAULT_DOCKER_API_VERSION: Final[str]
MINIMUM_DOCKER_API_VERSION: Final[str]
DEFAULT_TIMEOUT_SECONDS: Final[int]
STREAM_HEADER_SIZE_BYTES: Final[int]
CONTAINER_LIMITS_KEYS: Final[Sequence[str]]
DEFAULT_HTTP_HOST: Final[str]
DEFAULT_UNIX_SOCKET: Final[str]
DEFAULT_NPIPE: Final[str]
BYTE_UNITS: Final[Mapping[str, int]]
INSECURE_REGISTRY_DEPRECATION_WARNING: Final[str]
IS_WINDOWS_PLATFORM: Final[bool]
WINDOWS_LONGPATH_PREFIX: Final[str]
DEFAULT_USER_AGENT: Final[str]
DEFAULT_NUM_POOLS: Final[int]
DEFAULT_NUM_POOLS_SSH: Final[int]
DEFAULT_MAX_POOL_SIZE: Final[int]
DEFAULT_DATA_CHUNK_SIZE: Final[int]
DEFAULT_SWARM_ADDR_POOL: Final[Sequence[str]]
DEFAULT_SWARM_SUBNET_SIZE: Final[int]
