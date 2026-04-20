"""Compatibility facade for the proposal generator backend modules."""

from .proposal_shared import logger
from .proposal_generator import ProposalGenerator
from .runtime_components import (
    ChartEngine,
    DocumentBuilder,
    FinancialAnalyzer,
    FirmAPIClient,
    InternalDataClient,
    KnowledgeBase,
    LogoManager,
    Researcher,
    SchemaMapper,
    StyleEngine,
)

__all__ = [
    "logger",
    "ChartEngine",
    "DocumentBuilder",
    "FinancialAnalyzer",
    "FirmAPIClient",
    "InternalDataClient",
    "KnowledgeBase",
    "LogoManager",
    "ProposalGenerator",
    "Researcher",
    "SchemaMapper",
    "StyleEngine",
]
