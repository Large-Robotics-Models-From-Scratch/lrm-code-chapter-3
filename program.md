# Program: How to Implement Chapter 3 Code

This is the **operating manual** for this repo, modeled on Sid's `program.md` in `lrm-code-chapter-2`. It tells Claude Code (or a human collaborator) how to use the four inputs — chapter plan, agent toolkit, Manning style guide, and the Ch 2 hand-off — to go from this scaffold to a finished, validated Chapter 3 codebase.

Read this end-to-end before writing any Python.

---

## 1. The Four Inputs

| Input | Where | Role |
|---|---|---|
| **Chapter plan** | `docs/chapter_3_plan.md` (synced with `../lrm-book/chapter_3/chapter_3_structure_and_plan.md`) | The *what*. 7 listings, 7 figures, 5 concept boxes, the Ch3→Ch4 export contract. |
| **Agent toolkit** | `../lrm-code-agents/` | The *check*. Code agents (style-check, listing-check, chapter-continuity, test-gen, resource-check) + (coming) prose agents. |
| **Style guide** | `docs/MANNING_STYLE.md` + `../lrm-book/STYLEGUIDE.md` | The *how*. Manning conventions — annotations, line widths, banned words, code style. |
| **Ch 2 export contract** | `from ch02 import make_pickplace_dataloader, normalize, denormalize` (Sid's repo) | The *upstream*. Frozen interface; do not modify. |

Implementation is a tight loop between these. The plan tells you the next listing. The style guide tells you how to write it. The agents tell you whether you got it right. The Ch 2 contract tells you what data and shapes you can expect.

---

## 2. One-Time Setup

Before the first implementation pass, link the author toolkit and the reader-facing chapter agent into `.claude/agents/` so Claude Code discovers everything at startup.

```bash
cd ~/Desktop/projects/manning/lrm-code-chapter-3

mkdir -p .claude/agents

# Author toolkit (lrm-code-agents): style-check, listing-check, etc.
for f in ../lrm-code-agents/agents/*; do
    ln -sf "../$f" .claude/agents/
done
ln -s ../../lrm-code-agents/CLAUDE.md .claude/CLAUDE.md

# Reader-facing chapter agent (committed in this repo at agents/)
ln -s ../../agents/chapter-03-guide.md .claude/agents/chapter-03-guide.md
```

Optionally, override defaults:

```bash
cp ../lrm-code-agents/defaults.yml .lrm-agents.yml
# edit .lrm-agents.yml — already present in this repo
```

**Then restart Claude Code from this directory.** Subagents are loaded at session start.

---

## 3. The Build Loop

This repo follows the **Raschka-style hybrid model** used across Manning's "from scratch" series, same pattern as `lrm-code-chapter-2`:

- **`src/ch03/*.py`** — the importable Python package. Holds every function and class that downstream chapters or tests touch.
- **`notebooks/ch03.ipynb`** — the reader's primary artifact. Imports from `src/ch03/`, runs cell-by-cell, mirrors the book's prose.

Every listing has a home in **both** artifacts.

For each of the 7 listings in `docs/chapter_3_plan.md`, in order:

1. **Read the listing spec** in `docs/chapter_3_plan.md`.
2. **Locate the module**:

   | Listings | Module | Reader role |
   |---|---|---|
   | 3.1 | `src/ch03/vision_encoder.py` (SigLIP load + freeze + project) | type-along |
   | 3.2 | `src/ch03/viz_similarity.py` (patch self-similarity) | provided utility |
   | 3.3 | `src/ch03/language_backbone.py` (SmolLM2, no vocabulary expansion) | type-along |
   | 3.4 | `src/ch03/state_encoder.py` (6-dim -> 576 MLP) | type-along |
   | 3.5 | `src/ch03/vla_backbone.py` (`embed_inputs`: concatenate the three streams + attention mask + position IDs) | type-along |
   | 3.6 | `src/ch03/vla_backbone.py` (`contextualize` + `forward`: run SmolLM2 via inputs_embeds) | type-along |
   | 3.7 | `tests/` (definition-of-done verification: shapes, observation-prefix order, no vocabulary change) | verification |

3. **Mirror in the notebook**.
4. **Write the code** matching the plan exactly: same variable names, same annotations, same line ordering.
5. **Validate with agents**: `style-check`, `listing-check`.
6. **Run the notebook end-to-end** after each listing lands.
7. **Generate tests**: `test-gen` produces shape/dtype stubs in `tests/`.
8. **Update `agents/chapter-03-guide.md`** if a listing's role changes.
9. **Run `resource-check` at major milestones** (after a full section).

---

## 4. The Export Contract

`VLABackbone.forward(images, input_ids, state, text_attention_mask=None) -> [B, 392 + L + 1, 576]` is the **frozen interface** between Ch 3 and Ch 4. Signature specified in `docs/chapter_3_plan.md` "Hand-off contract to chapter 4" and in `ARCHITECTURE_LOG.md`.

**Implications:**
- Do not rename `VLABackbone` or change the `embed_inputs` / `contextualize` / `forward` signatures. `input_ids` is HF's text-only tokenizer output; the observation prefix is assembled internally. **Ch 4's repo must adopt the 3-tuple `embed_inputs` contract (2026-08-09 concat migration).**
- `images` is `[B, 2, 3, H, W]` in `[0, 1]` (two cameras, overhead then side; resized internally); `input_ids` is `[B, L]` text IDs (pad = eos); the observation-prefix order is image + text + state, length `N = 392 + L + 1`; `state` is `[B, 6]`.
- The output tensor shape is part of the contract. Ch 4 either reads contextualized hidden states from it or extends the observation prefix between `embed_inputs` and `contextualize`.
- The tokenizer is SmolLM2's native (49,152 vocab) and is never expanded: only the language stream carries vocabulary IDs, the image and state streams enter as vectors, so the embedding table keeps its 49,152 rows and `config.vocab_size` stays 49,152. The revised Ch 4 contract does not expand it either.
- Reverse-check enforced via `chapter-continuity` agent and `tests/test_guardrails.py`: no `add_tokens`, `resize_token_embeddings`, or `masked_scatter` in Ch 3 source.

When implementation reveals a problem with this interface, update the plan in `docs/chapter_3_plan.md` *and* `../lrm-book/chapter_3/chapter_3_structure_and_plan.md` *together*, in the same commit.

Run `chapter-continuity` after any change to the export surface.

---

## 5. Figures

The chapter plan lists 7 figures (3.1-3.7). Figure 3.1 is the generic VLA-backbone map (modality-specific encoders, common-width input embeddings, one multimodal sequence, the language backbone, contextualized hidden states) and Figure 3.7 is the concrete implementation recap; both are diagrams rather than rendered outputs. The data-driven figures are produced inside `notebooks/ch03.ipynb` using helper functions in `src/ch03/viz_similarity.py`, exported to `figures/` for the chapter draft.

For each figure:
- The plotting helper lives in `src/ch03/` so it is importable and testable
- The notebook cell calls the helper, sets a fixed seed, and calls `plt.savefig("figures/figure_3_<n>_<slug>.png", dpi=300)`
- Verify the rendered image matches the description in `docs/chapter_3_plan.md`

`MANNING_STYLE.md` §4 has production rules. `../lrm-book/FIGURE_STYLE_GUIDE.md` has full details.

---

## 6. Definition of Done

A listing is done when:
- [ ] Code matches the plan
- [ ] Annotations are contiguous from `#A` and explanations are 1-2 sentences each
- [ ] `style-check` passes
- [ ] `listing-check` passes
- [ ] Notebook cell imports it and runs cleanly
- [ ] `test-gen` has produced a stub test

A module is done when:
- [ ] All its listings are done
- [ ] Tests in `tests/` cover its public interface
- [ ] `resource-check` flags no critical issues

The chapter is done when:
- [ ] All 7 listings implemented in `src/ch03/` and validated
- [ ] `notebooks/ch03.ipynb` runs top-to-bottom from a fresh kernel
- [ ] All 7 figures rendered and saved to `figures/`
- [ ] `agents/chapter-03-guide.md` listing roadmap matches `docs/chapter_3_plan.md`
- [ ] `chapter-continuity` confirms the Ch 4 export is stable
- [ ] A full check pass (all 5 author agents) reports clean at ERROR severity

---

## 7. Working Notes

- **One listing at a time.** Don't write three files in parallel.
- **Update the plan when reality diverges.** Code-first, plan-after creates silent drift.
- **Cross-reference the export contract before any rename.** Ch 4's plan locks the `VLABackbone` symbol.
- **Don't pre-emptively optimize.** Chapter 3 is the simplest backbone that works on a laptop.
- **Notebook is not a dumping ground.** Anything more than ~10 lines belongs in `src/ch03/`.
- **Clear notebook outputs before committing** (pre-commit hook does this via nbstripout).

---

## 8. Where Things Live

| Need | Path |
|---|---|
| What to build | `docs/chapter_3_plan.md` (synced with `../lrm-book/chapter_3/chapter_3_structure_and_plan.md`) |
| How to format it | `docs/MANNING_STYLE.md` |
| Importable Python (export contract) | `src/ch03/` |
| Reader's canonical walkthrough | `notebooks/ch03.ipynb` (to be created in section PRs) |
| Reader's optional companion agent | `agents/chapter-03-guide.md` |
| Author-tooling agents | `../lrm-code-agents/agents/` (symlinked into `.claude/agents/`) |
| Author agent router / config | `../lrm-code-agents/CLAUDE.md`, `defaults.yml`, `.lrm-agents.yml` |
| Rendered figures | `figures/` |
| Tests (CI infrastructure) | `tests/` |
| Reader sanity scripts | `scripts/check_*.py` |
| Full prose/code style rules | `../lrm-book/STYLEGUIDE.md` |
| Figure rendering rules | `../lrm-book/FIGURE_STYLE_GUIDE.md` |
| Chapter prose draft (future) | `../lrm-book/chapter_3/draft_v*.md` |
