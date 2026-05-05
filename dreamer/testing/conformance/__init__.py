"""Per-Protocol conformance suites.

Each suite is an abstract pytest class (no ``test_`` prefix on the file).
Component authors subclass the suite and override the ``make_*`` async
factory; the suite runs every contract test against the supplied
implementation.
"""

from dreamer.testing.conformance.context_store import ContextStoreConformance
from dreamer.testing.conformance.dream_lease import DreamLeaseStoreConformance
from dreamer.testing.conformance.ltm_store import LTMStoreConformance
from dreamer.testing.conformance.stm_serializer import STMSerializerConformance
from dreamer.testing.conformance.stm_store import STMStoreConformance

__all__ = [
    "ContextStoreConformance",
    "DreamLeaseStoreConformance",
    "LTMStoreConformance",
    "STMSerializerConformance",
    "STMStoreConformance",
]
