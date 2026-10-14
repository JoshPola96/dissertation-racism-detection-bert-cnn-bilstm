# builder/schema.py
"""The record every source maps into, and the vocabularies its fields draw from. A record that breaks a vocabulary
fails when it is created, so a bad mapping cannot reach the resource unnoticed."""
import hashlib
from dataclasses import dataclass

HATE_LEVELS = ("hateful", "offensive", "neither", "not_hateful", "unknown")  # not_hateful: offensive or neither
SOFT_LEVELS = HATE_LEVELS[:3]  # one annotator's possible judgements, over which a soft label spreads
TARGET_CATEGORIES = ("race_ethnicity", "religion", "national_origin", "gender", "sexual_orientation", "disability",
                     "age", "political", "other")
TARGET_GROUPS = {  # every group a source may name, with the categories it always implies; sources may add others
    **dict.fromkeys(("black", "white", "asian", "latinx", "arab", "middle_eastern", "indigenous", "pacific_islander",
                     "east_asian", "non_white", "ethnic_minorities", "mixed_race", "romani"), ("race_ethnicity",)),
    "jewish": ("race_ethnicity", "religion"),
    **dict.fromkeys(("african", "american", "british", "chinese", "ethiopian", "german", "hong_kong", "indian",
                     "japanese", "korean", "mexican", "pakistani", "singaporean", "taiwanese", "immigrants",
                     "undocumented_immigrants", "refugees", "migrant_workers"), ("national_origin",)),
    **dict.fromkeys(("muslim", "christian", "hindu", "buddhist", "mormon", "nonreligious"), ("religion",)),
    **dict.fromkeys(("women", "men", "transgender", "transgender_men", "transgender_women", "non_binary"), ("gender",)),
    **dict.fromkeys(("gay", "lesbian", "bisexual", "heterosexual", "asexual"), ("sexual_orientation",)),
    "lgbtq": ("gender", "sexual_orientation"),
    **dict.fromkeys(("disabled", "cognitive_disability", "mental_illness", "physical_disability", "physical_illness",
                     "neurological_disability", "visually_impaired", "hearing_impaired", "autistic"), ("disability",)),
    **dict.fromkeys(("children", "teenagers", "young_adults", "middle_aged", "seniors"), ("age",)),
    **dict.fromkeys(("liberals", "conservatives", "democrats", "feminists", "antifa"), ("political",)),
    **dict.fromkeys(("poor", "working_class", "fat", "economic", "minority"), ("other",)),
}
TEXT_FORMS = ("raw", "preprocessed")
EXPLICITNESS = ("explicit", "implicit", "unknown")


@dataclass(frozen=True)
class Record:
    source: str
    source_id: str
    platform: str
    text: str
    text_form: str  # "raw" as posted, or "preprocessed" by the source's authors (lower-cased, placeholders)
    hate: str
    label_raw: str  # the source's own label or labels, verbatim
    target_category: tuple[str, ...]
    target_group: tuple[str, ...]
    n_annotators: int | None
    soft_label: dict[str, float] | None  # share of annotators per hate level, where a source releases them
    licence: str
    explicitness: str = "unknown"
    author_id: str | None = None

    def __post_init__(self):
        if not isinstance(self.source_id, str) or not isinstance(self.text, str):
            raise TypeError(f"{self.source}:{self.source_id!r}: source_id and text must be strings")
        if self.hate not in HATE_LEVELS:
            raise ValueError(f"{self.source}:{self.source_id}: hate level {self.hate!r} is not one of {HATE_LEVELS}")
        if unknown := set(self.target_category) - set(TARGET_CATEGORIES):
            raise ValueError(f"{self.source}:{self.source_id}: unknown target categories {sorted(unknown)}")
        if unknown := set(self.target_group) - TARGET_GROUPS.keys():
            raise ValueError(f"{self.source}:{self.source_id}: unknown target groups {sorted(unknown)}")
        if self.text_form not in TEXT_FORMS:
            raise ValueError(f"{self.source}:{self.source_id}: text form {self.text_form!r} is not one of {TEXT_FORMS}")
        if self.explicitness not in EXPLICITNESS:
            raise ValueError(f"{self.source}:{self.source_id}: explicitness {self.explicitness!r} is not one of "
                             f"{EXPLICITNESS}")

    @property
    def id(self) -> str:
        """Stable across builds: the same source row always gets the same id."""
        return hashlib.sha256(f"{self.source}:{self.source_id}".encode()).hexdigest()[:16]
