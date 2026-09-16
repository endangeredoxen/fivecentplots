#!/usr/bin/env python3
"""
Sync CSV-driven kwarg documentation into src/fivecentplots/fcp.py docstrings.

Problem this solves
--------------------
src/fivecentplots/kwargs/keywords.py already builds per-function kwarg docs from
kwargs/csv/*.csv (via make_docstrings/kw_header/kw_print) and writes them to
kwargs/docstrings/*.txt when run as __main__. But nothing takes that generated text
and puts it into fcp.py itself -- that step has been manual, and it's gone stale
(e.g. every doc URL in the installed 0.6.1 package still points at .../0.6.0/...
because the .txt-to-fcp.py copy-paste wasn't redone after the version bump).

What this script does
----------------------
For every function in FUNC_SECTIONS below:
  1. Parses fcp.py with `ast` to get the exact line range of that function's docstring.
  2. Within the existing docstring, finds the boundary between hand-written prose
     (summary line, Args:, a hand-written REQUIRED: subsection, Examples section)
     and the CSV-generated "Keyword Args:" body.
  3. Regenerates ONLY that CSV-driven body from the current kwargs/csv/*.csv files
     (reusing keywords.py's own make_docstrings/kw_header/kw_print -- single source
     of truth, no re-implementation of the formatting logic).
  4. Replaces just that region in fcp.py, leaving hand-written prose, the Examples
     section, and everything else in the file completely untouched.
  5. Preserves the file's existing CRLF line endings so this doesn't produce a
     noisy line-ending diff.

Usage
-----
    python sync_fcp_docstrings.py                 # writes changes to fcp.py in place
    python sync_fcp_docstrings.py --check          # exit 1 if fcp.py is out of sync (for CI)
    python sync_fcp_docstrings.py --diff           # print a unified diff, don't write

Run this any time a kwargs/csv/*.csv file changes, or after a version bump (so the
generated example URLs pick up the new version.txt).
"""
import argparse
import ast
import difflib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # so `import keywords` works standalone
from keywords import make_docstrings, kw_header, kw_print  # noqa: E402

FCP_PY_PATH = Path(__file__).resolve().parents[1] / 'fcp.py'

# Mirrors the structure already hardcoded in keywords.py's `if __name__ == '__main__':`
# block -- this is the single place that structure needs to live once this script
# replaces that block's job. (function_name, has_required_section, [(header_or_None, csv_key), ...])
FUNC_SECTIONS = {
    'bar':      [(None, 'bar')],
    'boxplot':  [('BASIC', 'box'), ('GROUPING_TEXT', 'box_label'),
                 ('STAT_LINES', 'box_stat'), ('DIAMONDS', 'box_diamond'),
                 ('VIOLINS', 'box_violin')],
    'contour':  [('BASIC', 'contour'), ('COLOR_BAR', 'cbar')],
    'gantt':    [(None, 'gantt')],
    'heatmap':  [('BASIC', 'heatmap'), ('COLOR_BAR', 'cbar')],
    'hist':     [(None, 'hist')],
    'imshow':   [(None, 'imshow')],
    'nq':       [('BASIC', 'nq'), ('CALCULATION', 'nq_calc')],
    'pie':      [(None, 'pie')],
    'plot':     [('LINES', 'lines'), ('MARKERS', 'markers'),
                 ('AX_[H|V]LINES', 'ax_lines'), ('CONTROL_LIMITS', 'control_limits'),
                 ('CONFIDENCE_INTERVALS', 'conf_int'), ('FIT', 'fit'),
                 ('REFERENCE_LINES', 'ref_line'), ('STAT_LINES', 'stat_line')],
    # Reference-only dummy functions: no hand-written REQUIRED/imgs-style preamble.
    'axes':        [(None, 'axes')],
    'cbar':        [(None, 'cbar')],
    'figure':      [(None, 'figure')],
    'gridlines':   [(None, 'gridlines')],
    'grouping':    [(None, 'grouping')],
    'labels':      [('AXES_LABELS', 'labels'), ('RC_LABELS', 'labels_rc')],
    'legend':      [(None, 'legend')],
    'lines':       [(None, 'lines')],
    'markers':     [(None, 'markers')],
    'options':     [(None, 'options')],
    'tick_labels': [(None, 'tick_labels')],
    'ticks':       [(None, 'ticks')],
    'titles':      [(None, 'titles')],
    'ws':          [(None, 'ws')],
}

INDENT = ' ' * 8  # matches the existing docstring body indent in fcp.py


def build_generated_body(kw: dict, sections: list) -> str:
    """Reproduce exactly what keywords.py's __main__ block builds per function."""
    out = ''
    for header, csv_key in sections:
        if header is not None:
            out += kw_header(header, indent=INDENT)
        out += kw_print(kw[csv_key])
    return out


def find_docstring_span(source: str, func_name: str):
    """Return (start_offset, end_offset, docstring_text, has_returns) for the
    triple-quoted docstring of the given top-level function, using `ast` so this
    works regardless of exact formatting."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            if not (node.body and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, (ast.Constant, ast.Str))):
                raise ValueError(f'{func_name} has no docstring to sync')
            doc_node = node.body[0]
            return doc_node.value.lineno, doc_node.value.end_lineno, doc_node
    raise ValueError(f'function {func_name!r} not found in fcp.py')


def first_kwarg_name(generated_body: str) -> str:
    """Extract the literal kwarg name token from the first line of generated
    kw_print() output, e.g. 'bar_align|align (str): ...' -> 'bar_align|align'."""
    first_line = next(l for l in generated_body.split('\n') if l.strip())
    return first_line.strip().split(' (')[0]


def replace_generated_region(docstring_text: str, generated_body: str, first_header) -> str:
    """Within one function's docstring text, replace only the CSV-generated
    'Keyword Args:' body, preserving everything else (summary, Args:, any
    hand-written preamble like a REQUIRED: subsection or imshow's custom
    'imgs:' block, and the trailing Examples section) verbatim.

    Rather than guessing where hand-written content ends by its shape (which
    breaks for every function differently -- REQUIRED: blocks, multi-line
    wrapped entries, imshow's totally custom preamble), this anchors on
    something unambiguous:
      - if this function's kwarg categories start with a header (e.g.
        'BASIC:', 'LINES:'), that literal header line IS the start anchor
        (kw_header() output is deterministic, so it's a safe string to
        search for);
      - if there's no header (a flat, single-CSV function), the first kwarg
        NAME that will appear in the freshly generated content is used as
        the anchor instead -- it must already exist somewhere in the current
        docstring (that's exactly the line this script is about to refresh),
        so finding it tells us precisely where hand-written prose ends and
        generated content begins, no matter how much hand-written preamble
        (a REQUIRED: block, wrapped multi-line args, or something entirely
        custom like imshow's 'imgs:' section) comes before it.
    """
    lines = docstring_text.split('\n')

    try:
        kw_idx = next(i for i, l in enumerate(lines) if l.strip() == 'Keyword Args:')
    except StopIteration:
        raise ValueError('no "Keyword Args:" section found')

    if first_header is not None:
        anchor = f'{first_header}:'
        matches = [i for i in range(kw_idx + 1, len(lines)) if lines[i].strip() == anchor]
    else:
        anchor = first_kwarg_name(generated_body)
        matches = [i for i in range(kw_idx + 1, len(lines)) if lines[i].strip().startswith(anchor + ' (')]

    if not matches:
        raise ValueError(
            f'could not find start anchor {anchor!r} in existing docstring -- '
            f'this usually means a brand-new kwarg was added and there is no '
            f'existing line to anchor on; add/update this function\'s docstring '
            f'by hand once, then future syncs will work automatically.'
        )
    start = matches[0]

    # End boundary: the earliest of "Returns:" or "Examples" (with or without
    # a "--------" underline on the next line), or end of docstring.
    end = len(lines)
    for i in range(start, len(lines)):
        stripped = lines[i].strip()
        if stripped == 'Returns:' or stripped.startswith('Examples'):
            end = i
            break

    prefix = '\n'.join(lines[:start])
    suffix = '\n'.join(lines[end:])
    body = generated_body.rstrip('\n')
    return prefix + '\n' + body + '\n' + suffix


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--check', action='store_true', help='exit 1 if out of sync, write nothing')
    parser.add_argument('--diff', action='store_true', help='print a unified diff, write nothing')
    args = parser.parse_args()

    kw = make_docstrings()

    # Preserve the file's exact line endings (fcp.py uses CRLF).
    raw_bytes = FCP_PY_PATH.read_bytes()
    newline = '\r\n' if b'\r\n' in raw_bytes else '\n'
    source = raw_bytes.decode('utf-8')
    working = source.replace('\r\n', '\n')  # normalize for ast/line-splitting; restore at the end

    changed_functions = []

    # Process in reverse line order so earlier replacements don't shift later offsets.
    spans = []
    for func_name, sections in FUNC_SECTIONS.items():
        try:
            start_line, end_line, doc_node = find_docstring_span(working, func_name)
        except ValueError as e:
            print(f'WARNING: skipping {func_name}: {e}', file=sys.stderr)
            continue
        spans.append((func_name, sections, start_line, end_line))

    lines = working.split('\n')
    for func_name, sections, start_line, end_line in sorted(spans, key=lambda s: -s[2]):
        # ast line numbers are 1-indexed and inclusive
        doc_lines = lines[start_line - 1:end_line]
        old_docstring_text = '\n'.join(doc_lines)
        # strip the surrounding triple quotes for easier text surgery, restore after
        quote = '"""' if old_docstring_text.strip().startswith('"""') else "'''"
        body_start = old_docstring_text.index(quote) + 3
        body_end = old_docstring_text.rindex(quote)
        inner = old_docstring_text[body_start:body_end]

        first_header = sections[0][0]
        generated_body = build_generated_body(kw, sections)
        try:
            new_inner = replace_generated_region(inner, generated_body, first_header)
        except ValueError as e:
            print(f'WARNING: skipping {func_name}: {e}', file=sys.stderr)
            continue

        if new_inner != inner:
            changed_functions.append(func_name)

        new_docstring_text = old_docstring_text[:body_start] + new_inner + old_docstring_text[body_end:]
        new_doc_lines = new_docstring_text.split('\n')
        lines[start_line - 1:end_line] = new_doc_lines

    new_working = '\n'.join(lines)
    new_source = new_working.replace('\n', newline)

    if new_source == source:
        print('fcp.py is already in sync with kwargs/csv/*.csv -- nothing to do.')
        return 0

    print(f'Out of sync functions: {", ".join(changed_functions)}')

    if args.diff or args.check:
        diff = difflib.unified_diff(
            source.splitlines(keepends=True), new_source.splitlines(keepends=True),
            fromfile='fcp.py (current)', tofile='fcp.py (generated)',
        )
        sys.stdout.writelines(diff)
        return 1 if args.check else 0

    FCP_PY_PATH.write_bytes(new_source.encode('utf-8'))
    print(f'Updated {FCP_PY_PATH}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
