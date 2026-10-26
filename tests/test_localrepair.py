from schemarepair.localrepair import repair
from schemarepair.schemas import PersonRecord, ProductItem
from schemarepair.validate import validate

PERSON = PersonRecord


def _ok(text, schema=PERSON):
    out = repair(text, schema)
    assert out.obj is not None, f"repair failed: {out.fixes}"
    schema.model_validate(out.obj)
    return out


def test_strips_markdown_fence():
    out = _ok('```json\n{"name": "Sara Iqbal", "age": 34, '
              '"email": "sara@example.com", "is_member": true}\n```')
    assert "strip_markdown_fence" in out.fixes


def test_removes_trailing_comma():
    out = _ok('{"name": "Tom Novak", "age": 41, "email": "t@example.com", '
              '"is_member": false,}')
    assert "remove_trailing_commas" in out.fixes


def test_single_quotes_and_python_literals():
    out = _ok("{'name': 'Mei Chen', 'age': 28, 'email': 'm@example.com', "
              "'is_member': True}")
    assert "single_to_double_quotes" in out.fixes
    assert "python_literals_to_json" in out.fixes
    assert out.obj["is_member"] is True


def test_unquoted_keys():
    out = _ok('{name: "Omar Aziz", age: 52, email: "o@example.com", '
              'is_member: false}')
    assert "quote_unquoted_keys" in out.fixes


def test_truncated_output_is_closed():
    # Cleanly truncated (missing final brace): the bracket balancer closes it.
    # (A value truncated mid-token, e.g. "is_member": tru, is NOT
    # deterministically repairable and repair() honestly returns None there.)
    out = repair('{"name": "Nadia Haddad", "age": 47, '
                 '"email": "n@example.com", "is_member": true', PERSON)
    assert out.obj is not None
    assert "close_truncated_brackets" in out.fixes
    out_bad = repair('{"name": "Nadia Haddad", "age": 47, '
                     '"email": "n@example.com", "is_member": tru', PERSON)
    assert out_bad.obj is None


def test_prose_wrapping_extracted():
    out = _ok('Sure! Here is the record: {"name": "Ruth Berg", "age": 63, '
              '"email": "r@example.com", "is_member": true} Hope that helps!')
    assert out.obj["name"] == "Ruth Berg"


def test_string_number_coerced_to_int():
    out = repair('{"name": "Kenji Tanaka", "age": "34", '
                 '"email": "k@example.com", "is_member": true}', PERSON)
    assert out.obj["age"] == 34
    assert "coerce:age" in out.fixes


def test_extra_field_dropped():
    out = repair('{"name": "Aisha Bello", "age": 30, "email": "a@example.com", '
                 '"is_member": true, "nickname": "Ash"}', PERSON)
    assert "nickname" not in out.obj
    assert "drop_extra_field:nickname" in out.fixes


def test_enum_case_fixed():
    out = repair('{"title": "yoga mat", "price": 25.0, "currency": "usd", '
                 '"in_stock": true, "tags": ["new"]}', ProductItem)
    assert out.obj["currency"] == "USD"
    assert "enum_case_fix:currency" in out.fixes


def test_unrepairable_returns_none():
    out = repair("I cannot produce that record.", PERSON)
    assert out.obj is None


def test_validate_rejects_extra_field():
    v = validate('{"name": "x", "age": 30, "email": "e@x.com", '
                 '"is_member": true, "zzz": 1}', PERSON)
    assert not v.ok
    assert any(e["type"] == "extra_forbidden" for e in v.errors)
