"""Compatibility facade for the proposal generator backend modules."""

from .proposal_shared import logger
from .proposal_generator import ProposalGenerator
from .data_sources import FirmAPIClient, InternalDataClient, SchemaMapper
from .document_rendering import ChartEngine, DocumentBuilder, LogoManager, StyleEngine
from .finance import FinancialAnalyzer
from .knowledge_store import KnowledgeBase
from .research import Researcher

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
