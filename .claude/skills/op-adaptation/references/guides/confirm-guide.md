# Confirm Guide (C-confirm)

Detailed instructions for the user confirmation step. Reference this guide when executing C-confirm.

---

## Confirm with User

### Conflict Confirmation (FIRST)

Before asking standard questions, submit all ⚠️ items from CHECKLIST.md "Conflict Confirmation" table:

1. Present the specific discrepancy (e.g., "_def.cpp has `group_index` parameter but .md does not mention it")
2. Do NOT make unilateral decisions — user must explicitly confirm ✅ Include or ❌ Exclude
3. For default value conflicts, ask: "Use _def.cpp value ({X}) or .md value ({Y})?"
4. Update CHECKLIST.md after each resolution

### Version Depth Analysis Presentation

When C-parse found `.Version(...)` markers in `_def.cpp` or versioned aclnn entry points in `.md` (parse-guide § Heuristic DRAFT blind spot #3 "Version depth"), cross-analyze `_def.cpp` ↔ `.md` and present:

```
Version depth analysis (_def.cpp ↔ .md):
  .Version(N) ports (Input/Output/Attr):
    - base (no Version): <ports>          # V1
    - Version(N=2): <ports>              # V2
    - Version(N=3): <ports>              # V3, V4... enumerate all found
  .md aclnn entry points: aclnnXxx / aclnnXxxV2 / aclnnXxxV3 / ...
    (one .md per entry point — a single _def.cpp may pair with multiple .md)
  Provided .md covers: <versions>; still missing: <.md files>

Recommendation: implement up to Vn. Reason: <one line>.
Implement up to which version? (If beyond base, provide the corresponding aclnnXxxVn .md / signature.)
```
- A single `_def.cpp` may pair with multiple `.md`; confirm which the user supplied and which are still missing.

### Standard Questions

Ask whether each item is relevant. Also invite any additional information not listed:

- CANN version dispatch (V2/V3/V5 variants)?
- SoC-specific dispatch (e.g., Ascend950 vs others)?
- Is a `.List` overload needed (a Tensor parameter that also supports `int[]` type)?
- Additional torch_npu function variants (e.g., `_asymmetric`, `_v0`)?
- Is the `internal_format_opapi` field needed?
- Special data format support (mxfp4, etc.)?
- **Graph mode** — two ORTHOGONAL capabilities, ask BOTH questions independently. They are not a subset/parent relationship: an operator can be `torch.compile=YES / ACLGraph update=NO` (the common case), `YES / YES` (IFA), or `NO / YES`. Do NOT let one answer imply the other.
  - **Q1 — torch.compile graph capture** (Dynamo / npugraph_ex). This governs the `.default` shape-propagation registration only.
    - "Does this operator need to be traced into a graph via torch.compile?"
    - **YES** → C-meta must execute — register `.default`'s shape/dtype inference (meta is the graph-capture gate: no meta → Dynamo cannot infer shapes → compile fails; wrong meta → silent wrong results). C-build must remove `--disable_torchair` flag to compile torchair into the .so, otherwise torch.compile silently falls back to eager. C-test must generate graph mode test cases (`torch.compile(backend="npugraph_ex")`). C-docs must include graph mode call example. C-verify must cross-verify meta shape/dtype against C++ _out_size (see verify-guide.md Meta Shape/Dtype Logic Verification).
    - **NO** → Skip C-meta. C-build keeps default `--disable_torchair` (smaller package). C-test skips graph mode test cases. C-docs omits graph mode example. C-verify skips Meta Shape/Dtype Logic Verification.
  - **Q2 — ACLGraph update** (manual task-group update, 2 new interfaces). Independent of Q1. This governs whether the `.out` / `_get_max_workspace` pair is generated — it does NOT change Q1's answer. Paradigm: `references/guides/aclgraph-attention-guide.md`.
    - "Does this operator need ACLGraph update graph support (capture-time output/workspace pre-allocation for cross-call variable host params, e.g. dynamic seq-len)?"
    - Detection signals:
      - Signature contains dynamic seq-len host params: **`SymInt[]?` / CPU `Tensor?` / `int[]?`**, named `actual_seq_*` / `context_lens` / `seq_lengths` / `prefix`;
      - or belongs to the FA / IFA / MLA / paged-attention family.
    - **YES** → run the **gate check** first (本区门禁; hitting any stop condition → stop the integration and report back, never force it). Two tiers — **never ask the user what can be determined automatically**; record all conclusions in CHECKLIST:
      - **Automated pre-check** (parse input files / local CANN; no questions):
        - Two-stage interface existence: search the input aclnn signature doc (or local `/usr/local/Ascend/**/op_api/include/` headers) for `aclnnXxxGetWorkspaceSize`; record evidence (file+line) in CHECKLIST. If not found → ask the user whether CANN provides this capability; confirmed absent → **stop condition hit, stop the integration**. Note the interface must return the maximum workspace covering the full range of dynamic host params — **a plain one-shot `GetWorkspaceSize` must not be passed off as it** (CANN doc cross-check);
        - Whether functional is all-positional (no `*,`): decide directly from the signature parsed in C-parse; if so → apply the recommended form `Tensor? workspace=None, *, Tensor(a!)...` (alias outputs must be keyword-only) and record it in CHECKLIST, no question needed.
      - **Code review** (LLM reads code; no questions):
        - Static upper bound of `.out` internal preprocessing (layout conversion, dropout/random mask, cast, extra NPU tasks): Modify mode — review the existing functional C++; Create mode — an implementation constraint enforced at C-cpp（cpp-guide §Attention ACLGraph Variants Constraints; turned into `TORCH_CHECK`);
        - Implementation-time verification (C-cpp/C-verify code review or CANN doc cross-check): output count/shape/dtype/format and Tensor addresses stay unchanged across updates, and varying params only affect executor/tiling (implied by Q2=YES) — if found untrue → stop condition hit, stop the integration.
    - Once confirmed, follow the main flow directly: per-step behavior is driven by SKILL.md's `[conditional: attention ACLGraph]` notes (C-yaml 2 entries / C-cpp helper + 2 functions / C-meta no new registration / C-verify structural checks, missing meta is expected / C-derivatives no new binding / no `_infer_output`); paradigm details in `aclgraph-attention-guide.md`, not repeated here.
    - **NO** → do not generate the pair; keep the flow unchanged.
- **Autograd requirements** (based on auto-detected operator type from C-parse):
  - If operator is a **backward/grad operator** → "The forward op that this backward serves — does its derivatives.yaml binding need updating? (If yes, confirm the forward operator name.)"
  - If operator is a **forward operator** → "Does this operator need autograd backward propagation? If yes, do you have the backward operator's aclnn signature / _def.cpp? Both forward and backward operators need to be registered in C-yaml, and the binding added in C-derivatives."

Record all answers in CHECKLIST.md "Information Required from User" section.

---


## Troubleshooting

(Accumulate from cases.)
