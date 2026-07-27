# Compatibility Baseline Registration Guide

## Overview

After integrating a new operator, register it in compatibility tracking files to ensure API stability is monitored across versions.

## Run the Generator (extractor + DRAFT)

`scripts/op_adapt.py gen-baseline --op-name <name> [--write]` → JSON. Default dry-run shows a diff; `--write` modifies both files (with `.bak` backup) via **targeted text insertion** — it preserves the file's existing inline-array formatting and only adds the new entry/line, so the diff stays minimal (no whole-file reformat).

**Deterministic** (trust):
- `func:` schema key — copied **verbatim** from the YAML `func:` line (preserves `*` separators and defaults).
- `op_api:` schema key — generated as `torch_npu.<op_name>(*args, **kwargs)` (generic form).
- allowlist entry — appended to `torch_npu.all_version` **only when version is `all_version`/`newest`**; version-specific ops are NOT auto-added (see DRAFT §4).
- version **reading** — read **verbatim** from the YAML `op_api:` field (trust the extraction; whether the value is semantically correct is verified in DRAFT §2).

**DRAFT — LLM must verify**:
1. **`op_api:` form** — the script always emits generic `(*args, **kwargs)`. The real schema uses only two forms: generic `(*args, **kwargs)` and typed `(*args: _P.args, **kwargs: _P.kwargs)`. If `inspect.signature(torch_npu.<op_name>)` shows typed args, replace with the typed-args form.
2. **Version value** — the script read the YAML value correctly (no re-extraction needed), but confirm that value matches the operator's actual support range: a brand-new op should be `["newest"]`/`all_version`, not a past `vX.Y` range; the `newest` fallback (when YAML lacks `op_api:`) is uncertain.
3. **Schema placement** — `--write` appends at end; the schema JSON is not strictly alphabetically ordered in practice, so this is acceptable — just ensure both `op_api:` and `func:` entries are present.
4. **Allowlist placement** — if `allowlist_placement` is `version_specific_review_needed`, the script did NOT write to `all_version`; place the name in the correct version-specific list (`v2.6`/`v2.7`/...) manually.

## File Format Reference

Reference for interpreting the generator's output (the script does the writing).

### `test/core_tests/torch_npu_OpApi_schema_all.json`

Each entry maps a signature string to a supported-version list. Two entry types per operator:

**`func:`** — C++/YAML signature, copied **verbatim** from `op_plugin_functions.yaml` by the script (trust it):

```json
"func: npu_{operator_name}(Tensor x, *, ScalarType? dst_type=None) -> Tensor": {
    "version": ["v2.7", "v2.8", "newest"]
}
```

**`op_api:`** — Python public API. Only two real forms exist; the script emits the generic one (DRAFT — replace with the typed form if `inspect.signature` shows typed args):

```json
"op_api: torch_npu.npu_{operator_name}(*args, **kwargs)": { "version": ["newest"] }
"op_api: torch_npu.npu_{operator_name}(*args: _P.args, **kwargs: _P.kwargs)": { "version": ["newest"] }
```

Version tags: `"v2.1"` etc. = specific version · `"newest"` = after the last numbered tag · `"all_version"` = all versions.

### `test/allowlist_for_publicAPI.json`

The script appends the operator name to `torch_npu.all_version` **only when the schema version is `all_version`/`newest`**. For version-specific operators (any `vX.Y` tag), the script does NOT auto-add to `all_version` — it flags `allowlist_placement: "version_specific_review_needed"`, and the LLM must place the name in the correct version-specific list (`v2.6`/`v2.7`/...). This matches the real allowlist convention: version-specific ops are kept out of `all_version`.

```json
{ "torch_npu": { "all_version": ["existing_op", ..., "npu_{operator_name}"], "v2.7": ["..."] } }
```

## Verification

After `--write`, run:

```bash
python test/core_tests/test_compatibility.py
```

## Checklist

- [ ] Ran `gen-baseline` dry-run; reviewed the diff (existing entries marked "will skip")
- [ ] `op_api:` form confirmed: generic vs typed-args (replace if `inspect.signature` shows typed args)
- [ ] Version tags match the operator's actual support range
- [ ] `test_compatibility.py` passes
- [ ] Results reported to CHECKLIST.md

## Troubleshooting

(Accumulate from cases.)
