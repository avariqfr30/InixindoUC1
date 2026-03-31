"""Shared imports, config, and module-wide utilities for proposal generation."""

import os
from pathlib import Path
import io
import re
import json
import logging
import sqlite3
import shutil
import threading
import time
import uuid
import requests
import pandas as pd
import chromadb
from chromadb.config import Settings
import concurrent.futures
from datetime import datetime
from functools import lru_cache
from typing import Dict, List, Any, Tuple, Optional, Set
from urllib.parse import urlparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont, ImageStat

from sqlalchemy import create_engine
import markdown
from bs4 import BeautifulSoup

from docx import Document
from docx.image.exceptions import UnrecognizedImageError
from docx.shared import Pt, RGBColor, Inches, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
try:
    from pypdf import PdfReader
except Exception:  # pragma: no cover - optional document extraction dependency
    PdfReader = None

from ollama import Client
from chromadb.utils import embedding_functions

from .config import (
    PROJECT_ROOT,
    SERPER_API_KEY, OLLAMA_HOST, LLM_MODEL, EMBED_MODEL,
    APP_STATE_DB_PATH, APP_ASSET_ROOT, GENERATED_OUTPUT_DIR,
    WRITER_FIRM_NAME, DEFAULT_COLOR, UNIVERSAL_STRUCTURE, KAK_RESPONSE_STRUCTURE,
    PERSONAS, PROPOSAL_SYSTEM_PROMPT,
    PROJECT_DATA_FIELD_ALIASES, PROJECT_STANDARD_FIELD_ALIASES, FIRM_PROFILE_FIELD_ALIASES,
    CLIENT_RELATIONSHIP_FIELD_ALIASES,
    DEMO_MODE, FIRM_API_URL, API_AUTH_TOKEN, MOCK_FIRM_STANDARDS, MOCK_FIRM_PROFILE,
    DATA_ACQUISITION_MODE, GENERATION_PROFILE,
    COMPANY_DNA, VALUE_PLAYBOOK, INDUSTRY_VALUE_DRIVERS, CHAPTER_STANDARD_RULES, SPIRIT_OF_AI_RULES,
    WRITER_FIRM_OFFICE_ADDRESS, WRITER_FIRM_EMAIL, WRITER_FIRM_PHONE, WRITER_FIRM_WHATSAPP,
    WRITER_FIRM_WEBSITE, WRITER_FIRM_LEGAL_NAME, WRITER_FIRM_OPERATING_HOURS,
    WRITER_FIRM_PROFILE_SUMMARY, WRITER_FIRM_CREDENTIAL_HIGHLIGHTS, WRITER_FIRM_SOURCE_URLS,
    WRITER_FIRM_PORTFOLIO,
    RESEARCH_CACHE_TTL_SECONDS,
    MAX_PROPOSAL_PAGES, ESTIMATED_WORDS_PER_PAGE, RESERVED_NON_CONTENT_PAGES, PAGE_SAFETY_BUFFER,
)

logger = logging.getLogger(__name__)
