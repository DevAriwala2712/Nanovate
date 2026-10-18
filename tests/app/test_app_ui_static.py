"""Static UI checks of frontend/index.html and frontend/selftest.html (run by pytest; no browser needed).

The DOM-level UI unit tests live in frontend/selftest.html (open /selftest.html on the running app); these
checks catch the cheap bugs without a browser: a selector for an element that does not exist (a null
dereference at runtime), a CSS variable that is not defined, or a resource loaded from outside the repo.
"""
import re
from pathlib import Path

INDEX = Path("frontend/index.html").read_text()
SELFTEST = Path("frontend/selftest.html").read_text()
SCRIPT = INDEX.split("<script>", 1)[1]
IDS = set(re.findall(r'\bid="([\w-]+)"', INDEX))


def test_every_id_selector_in_the_script_exists():
    used = set(re.findall(r"""\$\$?\(\s*['"`]#([\w-]+)""", SCRIPT)) | set(re.findall(r"""querySelector(?:All)?\(\s*['"`]#([\w-]+)""", SCRIPT))
    missing = sorted(used - IDS)
    assert not missing, f"selectors for missing elements: {missing}"


def test_every_data_p_control_read_by_the_script_exists():
    read = set(re.findall(r"""v\('([\w]+)'\)""", SCRIPT)) | set(re.findall(r'data-p="(\w+)"\]', SCRIPT))
    present = set(re.findall(r'data-p="(\w+)"', INDEX.split("<script>")[0]))
    assert read <= present, sorted(read - present)


def test_css_variables_used_are_defined_in_root():
    root = INDEX.split(":root {", 1)[1].split("}", 1)[0]
    defined = set(re.findall(r"(--[\w-]+)\s*:", root))
    used = set(re.findall(r"var\((--[\w-]+)\)", INDEX)) | set(re.findall(r"cssVar\('(--[\w-]+)'\)", INDEX))
    assert used <= defined, f"undefined CSS variables: {sorted(used - defined)}"


def test_no_external_resources():
    for name, html in (("index.html", INDEX), ("selftest.html", SELFTEST)):
        assert not re.search(r'(src|href)="(https?:)?//', html), f"{name} loads an external resource"
        assert re.findall(r'<script src="([^"]+)"', html) in ([], ["data/run.js"]), name


def test_api_text_is_escaped_where_inserted():
    """Response strings that reach innerHTML go through esc() (or ro(), which calls it); a guard on the risky fields."""
    for field in ("e.api.message", "e.api.code", "r.provenance", "r.d2d.why", "cnn.domain_note"):
        for m in re.finditer(re.escape(field), SCRIPT):
            ctx = SCRIPT[max(0, m.start() - 6):m.start()]
            if SCRIPT[m.end():m.end() + 2] in (" ?", " &", " |", ").") or SCRIPT[m.end():m.end() + 4] == " ===":
                continue   # used as a condition or read, not inserted
            assert "esc(" in ctx or "ro(" in ctx or "textContent" in SCRIPT[max(0, m.start() - 60):m.start()] or "log(" in SCRIPT[max(0, m.start() - 80):m.start()] \
                or "drawBadge" in SCRIPT[max(0, m.start() - 40):m.start()], f"{field} inserted without esc(): ...{SCRIPT[m.start() - 60:m.end() + 10]}"


def test_selftest_covers_the_required_states():
    for needle in ("no ground truth", "server down", "422 error", "500 error", "cancel", "escaping", "layout "):
        assert needle in SELFTEST, needle
