"""schemarepair: a structured-output agent plus a repair-strategy benchmark.

The agent enforces Pydantic schemas on LLM output and repairs failures with
a pluggable strategy. The benchmark measures — on real outputs from a local
open-weight model — how often each repair strategy works, what it costs in
model calls and tokens, and which failure classes actually occur.
"""

from .agent import AgentResult, StructuredOutputAgent, StructuredOutputError
from .localrepair import RepairOutcome, repair
from .taxonomy import classify, taxonomy_counts
from .validate import Validation, validate

__version__ = "0.1.0"

__all__ = [
    "AgentResult",
    "RepairOutcome",
    "StructuredOutputAgent",
    "StructuredOutputError",
    "Validation",
    "classify",
    "repair",
    "taxonomy_counts",
    "validate",
]
