"""Primary proposal generator composed from focused backend mixins."""

from .proposal_shared import *
from .proposal_engine import ProposalEngineMixin
from .proposal_support import ProposalSupportMixin
from .runtime_components import FirmAPIClient, KnowledgeBase


class ProposalGenerator(ProposalEngineMixin, ProposalSupportMixin):
    DEFAULT_CHAPTER_TARGET_WORDS = 700
    BASE_CHAPTER_FLOOR_WORDS = 220
    CLOSING_CHAPTER_FLOOR_WORDS = 170
    BASE_COMPRESSION_FLOOR_WORDS = 240
    CLOSING_COMPRESSION_FLOOR_WORDS = 180
    CHAPTER_FLOOR_WORDS = {
        "c_1": 420,
        "c_2": 450,
        "c_3": 340,
        "c_4": 360,
        "c_5": 430,
        "c_6": 380,
        "c_7": 300,
        "c_8": 300,
        "c_9": 280,
        "c_10": 300,
        "c_11": 320,
        "c_12": 280,
        "c_closing": 220,
        "k_1": 360,
        "k_2": 500,
        "k_3": 420,
        "k_4": 360,
        "k_5": 360,
        "k_6": 260,
        "k_7": 220,
        "k_8": 220,
    }
    CHAPTER_COMPRESSION_FLOORS = {
        "c_1": 320,
        "c_2": 340,
        "c_3": 260,
        "c_4": 280,
        "c_5": 330,
        "c_6": 300,
        "c_7": 240,
        "c_8": 240,
        "c_9": 230,
        "c_10": 240,
        "c_11": 250,
        "c_12": 230,
        "c_closing": 180,
        "k_1": 280,
        "k_2": 380,
        "k_3": 320,
        "k_4": 280,
        "k_5": 280,
        "k_6": 210,
        "k_7": 180,
        "k_8": 180,
    }
    CHAPTER_COMPRESSION_RANK = {
        "c_12": 0,
        "c_8": 1,
        "c_9": 1,
        "c_10": 1,
        "c_11": 1,
        "c_7": 1,
        "c_closing": 1,
        "c_3": 2,
        "c_4": 2,
        "c_1": 3,
        "c_2": 3,
        "c_5": 4,
        "c_6": 4,
        "k_7": 0,
        "k_8": 0,
        "k_6": 1,
        "k_1": 1,
        "k_5": 2,
        "k_4": 2,
        "k_3": 3,
        "k_2": 4,
    }
    CHAPTER_BUSINESS_RANK = {
        "c_1": 5,
        "c_2": 5,
        "c_3": 4,
        "c_4": 4,
        "c_5": 5,
        "c_6": 5,
        "c_7": 3,
        "c_8": 3,
        "c_9": 3,
        "c_10": 4,
        "c_11": 4,
        "c_12": 3,
        "c_closing": 2,
        "k_1": 4,
        "k_2": 5,
        "k_3": 5,
        "k_4": 4,
        "k_5": 4,
        "k_6": 4,
        "k_7": 2,
        "k_8": 3,
    }
    PROPOSAL_ACCEPTANCE_TARGET = 80
    PROPOSAL_CATEGORY_FLOOR = 70

    def __init__(self, kb_instance: KnowledgeBase) -> None:
        self.ollama = Client(host=OLLAMA_HOST)
        self.kb = kb_instance
        self.io_pool = concurrent.futures.ThreadPoolExecutor(max_workers=10)
        self.firm_api = FirmAPIClient()
        self.generation_profile = GENERATION_PROFILE
        self._research_cache: Dict[str, Dict[str, str]] = {}
        self._research_cache_times: Dict[str, float] = {}
        self._research_inflight: Dict[str, threading.Event] = {}
        self._proposal_contract_cache: Dict[str, str] = {}
        self._chapter_context_cache: Dict[str, Dict[str, str]] = {}
        self._cache_lock = threading.RLock()

    @classmethod
    def _target_words(cls, chapter: Dict[str, Any]) -> int:
        m = re.search(r'Target:\s*(\d+)\s*words', chapter.get('length_intent', ''), re.IGNORECASE)
        return int(m.group(1)) if m else cls.DEFAULT_CHAPTER_TARGET_WORDS

    @staticmethod
    def _word_count(text: str) -> int:
        return len(re.findall(r'\b\w+\b', text))

    @staticmethod
    def _rewrite_length_intent(length_intent: str, target_words: int) -> str:
        if not length_intent:
            return f"Target: {target_words} words."
        if re.search(r'Target:\s*\d+\s*words', length_intent, re.IGNORECASE):
            return re.sub(
                r'Target:\s*\d+\s*words',
                f"Target: {int(target_words)} words",
                length_intent,
                flags=re.IGNORECASE
            )
        return f"{length_intent.rstrip('.')} (Target: {int(target_words)} words)."

    def _content_word_budget(self) -> int:
        usable_pages = max(1, MAX_PROPOSAL_PAGES - RESERVED_NON_CONTENT_PAGES - PAGE_SAFETY_BUFFER)
        return int(usable_pages * ESTIMATED_WORDS_PER_PAGE)

    @classmethod
    def _chapter_floor_words(cls, chapter_id: str, for_compression: bool = False) -> int:
        if for_compression:
            return cls.CHAPTER_COMPRESSION_FLOORS.get(
                chapter_id,
                cls.CLOSING_COMPRESSION_FLOOR_WORDS if chapter_id == "c_closing" else cls.BASE_COMPRESSION_FLOOR_WORDS
            )
        return cls.CHAPTER_FLOOR_WORDS.get(
            chapter_id,
            cls.CLOSING_CHAPTER_FLOOR_WORDS if chapter_id == "c_closing" else cls.BASE_CHAPTER_FLOOR_WORDS
        )

    @classmethod
    def _chapter_compression_rank(cls, chapter_id: str) -> int:
        return cls.CHAPTER_COMPRESSION_RANK.get(chapter_id, 2)

    @classmethod
    def _chapter_business_rank(cls, chapter_id: str) -> int:
        return cls.CHAPTER_BUSINESS_RANK.get(chapter_id, 3)

    def _chapter_word_targets(self, chapters: List[Dict[str, Any]]) -> Dict[str, int]:
        base_targets = {chapter["id"]: self._target_words(chapter) for chapter in chapters}
        total_base = sum(base_targets.values())
        budget = self._content_word_budget()
        if total_base <= 0:
            return {chapter["id"]: 500 for chapter in chapters}
        if total_base <= budget:
            return base_targets

        # Scale all chapters down proportionally while keeping minimum readable length.
        scaled: Dict[str, int] = {}
        for chapter in chapters:
            base = base_targets[chapter["id"]]
            floor = self._chapter_floor_words(chapter["id"], for_compression=False)
            scaled[chapter["id"]] = max(floor, int(base * budget / total_base))

        # If still over budget, cut from the longest chapters first.
        overflow = sum(scaled.values()) - budget
        if overflow > 0:
            ordered_ids = sorted(scaled.keys(), key=lambda cid: scaled[cid], reverse=True)
            for cid in ordered_ids:
                if overflow <= 0:
                    break
                floor = self._chapter_floor_words(cid, for_compression=False)
                reducible = max(0, scaled[cid] - floor)
                cut = min(reducible, overflow)
                scaled[cid] -= cut
                overflow -= cut
        return scaled

    def _estimated_pages(self, total_words: int) -> int:
        content_pages = max(1, (total_words + ESTIMATED_WORDS_PER_PAGE - 1) // ESTIMATED_WORDS_PER_PAGE)
        return RESERVED_NON_CONTENT_PAGES + content_pages

    @staticmethod
    def _cache_key(*parts: Any) -> str:
        return "||".join([str(p).strip().lower() for p in parts])

    @staticmethod
    def _cache_put(cache: Dict[str, Any], key: str, value: Any, max_size: int = 128) -> None:
        if key in cache:
            cache.pop(key, None)
        cache[key] = value
        while len(cache) > max_size:
            cache.pop(next(iter(cache)))

    def _cache_get(self, cache: Dict[str, Any], key: str) -> Any:
        with self._cache_lock:
            return cache.get(key)

    def _cache_store(self, cache: Dict[str, Any], key: str, value: Any, max_size: int = 128) -> None:
        with self._cache_lock:
            self._cache_put(cache, key, value, max_size=max_size)
