from schemarepair.schemas import PersonRecord, ProductItem
from schemarepair.taxonomy import classify, taxonomy_counts

P = PersonRecord
GOOD = ('{"name": "Sara Iqbal", "age": 34, "email": "s@example.com", '
        '"is_member": true}')


def test_valid_output():
    r = classify(GOOD, P)
    assert r["is_valid"]


def test_empty_output():
    r = classify("   ", P)
    assert r["primary"] == "empty_output"


def test_markdown_fence_class():
    r = classify("```json\n" + GOOD + "\n```", P)
    assert "markdown_fence" in r["classes"]


def test_not_json():
    r = classify("The record could not be found.", P)
    assert "not_json" in r["classes"]


def test_trailing_comma_class():
    bad = GOOD[:-1] + ",}"
    r = classify(bad, P)
    assert "trailing_comma" in r["classes"]
    assert not r["is_valid"]


def test_single_quotes_class():
    bad = ("{'name': 'Sara Iqbal', 'age': 34, 'email': 's@example.com', "
           "'is_member': True}")
    r = classify(bad, P)
    assert "single_quotes" in r["classes"]
    assert "python_literals" in r["classes"]


def test_missing_field_class():
    bad = '{"name": "Sara Iqbal", "age": 34, "is_member": true}'
    r = classify(bad, P)
    assert "missing_field" in r["classes"]


def test_enum_violation_class():
    bad = ('{"title": "mat", "price": 10.0, "currency": "BTC", '
           '"in_stock": true, "tags": []}')
    r = classify(bad, ProductItem)
    assert "enum_violation" in r["classes"]


def test_type_mismatch_class():
    bad = ('{"name": "Sara Iqbal", "age": "thirty-four", '
           '"email": "s@example.com", "is_member": true}')
    r = classify(bad, P)
    assert "type_mismatch" in r["classes"]


def test_truncated_class():
    r = classify(GOOD[:40], P)
    assert "truncated_json" in r["classes"]


def test_taxonomy_counts_aggregate():
    counts = taxonomy_counts([GOOD, "", GOOD], P)
    assert counts["n"] == 3
    assert counts["valid"] == 2
    assert counts["primary_counts"].get("empty_output") == 1
