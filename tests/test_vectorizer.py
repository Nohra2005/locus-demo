"""
Unit tests for vectorizer module — no services required, no model loading.
Tests pure functions and static data structures only.

Run: pytest tests/test_vectorizer.py -v
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "visual_engine"))

from vectorizer import shoe_style_from_label, CATEGORY_ALIASES
from clip_labels import CANONICAL_LABELS


class TestShoeSubtype:
    """shoe_style_from_label maps YOLO labels to coarse shoe buckets."""

    def test_sneaker(self):
        assert shoe_style_from_label("white leather sneaker") == "sneaker"

    def test_trainer(self):
        assert shoe_style_from_label("running trainer") == "sneaker"

    def test_platform(self):
        assert shoe_style_from_label("platform shoe") == "sneaker"

    def test_wedge(self):
        assert shoe_style_from_label("wedge sandal") == "sneaker"

    def test_clog(self):
        assert shoe_style_from_label("clog") == "sneaker"

    def test_boot(self):
        assert shoe_style_from_label("ankle boot") == "boot"

    def test_chelsea_boot(self):
        assert shoe_style_from_label("chelsea boot") == "boot"

    def test_combat_boot(self):
        assert shoe_style_from_label("combat boot") == "boot"

    def test_heel(self):
        assert shoe_style_from_label("stiletto heel") == "heel"

    def test_pump(self):
        assert shoe_style_from_label("leather pump") == "heel"

    def test_sandal(self):
        assert shoe_style_from_label("strappy sandal") == "sandal"

    def test_loafer(self):
        assert shoe_style_from_label("leather loafer") == "sandal"

    def test_mule(self):
        assert shoe_style_from_label("open toe mule") == "sandal"

    def test_flat(self):
        assert shoe_style_from_label("ballet flat") == "sandal"

    def test_espadrille(self):
        assert shoe_style_from_label("espadrille") == "sandal"

    def test_unknown_falls_back_to_other(self):
        assert shoe_style_from_label("shoe") == "other"

    def test_case_insensitive(self):
        assert shoe_style_from_label("RUNNING SNEAKER") == "sneaker"
        assert shoe_style_from_label("Ankle Boot") == "boot"


class TestCategoryAliases:
    """CATEGORY_ALIASES maps each category to acceptable YOLO label values."""

    ALL_CATEGORIES = {
        "top", "sports_bra", "sweater", "jacket", "dress",
        "jumpsuit", "skirt", "pants", "leggings", "shorts",
        "shoes", "hat", "bag",
    }

    def test_all_categories_have_alias_entry(self):
        for cat in self.ALL_CATEGORIES:
            assert cat in CATEGORY_ALIASES, f"'{cat}' missing from CATEGORY_ALIASES"

    def test_no_empty_alias_lists(self):
        for cat, aliases in CATEGORY_ALIASES.items():
            assert len(aliases) > 0, f"'{cat}' has empty alias list"

    def test_each_category_includes_itself(self):
        for cat, aliases in CATEGORY_ALIASES.items():
            assert cat in aliases, f"'{cat}' not in its own alias list"

    def test_dress_and_jumpsuit_cross_alias(self):
        assert "jumpsuit" in CATEGORY_ALIASES["dress"]
        assert "dress" in CATEGORY_ALIASES["jumpsuit"]

    def test_skirt_and_dress_cross_alias(self):
        assert "dress" in CATEGORY_ALIASES["skirt"]

    def test_pants_includes_jeans_and_trousers(self):
        assert "jeans" in CATEGORY_ALIASES["pants"]
        assert "trousers" in CATEGORY_ALIASES["pants"]

    def test_leggings_falls_back_to_pants(self):
        assert "pants" in CATEGORY_ALIASES["leggings"]

    def test_jacket_includes_coat(self):
        assert "coat" in CATEGORY_ALIASES["jacket"]

    def test_shoes_includes_sub_types(self):
        aliases = CATEGORY_ALIASES["shoes"]
        assert "sneakers" in aliases
        assert "boots" in aliases
        assert "sandals" in aliases

    def test_bag_includes_handbag_and_backpack(self):
        assert "handbag" in CATEGORY_ALIASES["bag"]
        assert "backpack" in CATEGORY_ALIASES["bag"]


class TestCanonicalLabels:
    """CANONICAL_LABELS is the complete set of fashion categories (plus not_fashion sentinel)."""

    EXPECTED = {
        "top", "sports_bra", "pants", "leggings", "shorts",
        "skirt", "dress", "sweater", "jacket", "shoes",
        "hat", "bag", "jumpsuit", "not_fashion",
    }

    def test_exactly_14_categories(self):
        assert len(CANONICAL_LABELS) == 14, (
            f"Expected 14 categories, got {len(CANONICAL_LABELS)}: {CANONICAL_LABELS}"
        )

    def test_no_duplicates(self):
        assert len(CANONICAL_LABELS) == len(set(CANONICAL_LABELS)), (
            "CANONICAL_LABELS contains duplicates"
        )

    def test_exact_category_set(self):
        assert set(CANONICAL_LABELS) == self.EXPECTED, (
            f"Label mismatch.\nExpected: {self.EXPECTED}\nGot: {set(CANONICAL_LABELS)}"
        )

    def test_all_lowercase(self):
        for label in CANONICAL_LABELS:
            assert label == label.lower(), f"Label '{label}' is not lowercase"
