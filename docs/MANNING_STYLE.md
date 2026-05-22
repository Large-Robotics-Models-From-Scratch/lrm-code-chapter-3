# Manning Style: What You Need to Know for Chapter 2 Code

This document captures the Manning Publications conventions that govern *how the code in this repo must be written* — annotations, listings, callouts, figures. It is a distilled subset of the team's full style guide, focused on what a code-writing pass needs.

Four upstream sources are authoritative. When this document and an upstream source disagree, the upstream wins. Copy from upstream into this file only when distilling for code-level decisions.

| Source | Path | What it locks |
|--------|------|---------------|
| Full locked style guide | `../../lrm-book/STYLEGUIDE.md` | Prose, voice, banned language, code conventions (~400 lines) |
| Figure rendering rules | `../../lrm-book/FIGURE_STYLE_GUIDE.md` | Palette, axis labels, captions, DPI |
| Manning hard format rules | `../../lrm-book/writing_instructions/writing_instructions.md` | Heading levels for listings/tables/figures, blockquote callouts, DOCX-converter assumptions |
| Body-chapter craft patterns | `../../lrm-book/book_learnings.md` | Cross-book patterns harvested from Raschka, Lambert, Wang/Szeto. See §2 specifically for coding-heavy chapters (Ch 2–7). |

The body-chapter patterns most relevant to Chapter 2 code:
- **Figure X.1 is always the roadmap recap**, not a content figure. Chapter 2's content figures start at 2.2.
- **Lead-in sentence** introduces every listing — function name, purpose, and any figure cross-reference appear before the code.
- **Listing captions are noun phrases**, not full sentences (`Listing 2.5 Loading the SO-101 expert dataset`, not "Listing 2.5: This listing loads...").
- **Honest scoping disclaimers** are repeated across each section — the single biggest reader-trust device across all three reference books.
- **Each listing has a role** — type-along teaching code, API illustration, or provided utility. The role drives how the listing is written; see §1.5 below.

This repo ships two reader-facing artifacts that govern listing style:
- **`notebooks/ch03.ipynb`** — the canonical, version-pinned walkthrough. Every listing maps to a cell.
- **`agents/chapter-02-guide.md`** — an optional Claude Code companion agent. The agent's listing roadmap must mirror this document and the chapter plan; the system prompt is a chapter deliverable on par with listings and figures.

---

## 1. Code Listings

Every code block that appears in the chapter — and therefore every Python file in `src/ch03/` that produces one — must be a **valid Manning listing**.

### 1.1 Annotations

Use Manning's `#A`, `#B`, `#C`, ... annotation format to mark lines that need explanation in the surrounding chapter text.

```python
import gymnasium as gym                          #A
env = gym.make("PickPlaceCube-v0")               #B
obs, info = env.reset(seed=42)                   #C
```

**Rules:**
- Annotations are contiguous starting from `#A`. No gaps (`#A`, `#B`, `#D` is invalid).
- Each annotation gets exactly one explanation in the prose below the listing.
- Explanation count must equal annotation count.
- Explanations describe **purpose**, not syntax. `#A Loads the simulation API` ✅. `#A Imports gymnasium as gym` ❌.
- Each explanation is **1–2 sentences maximum**.

### 1.2 Line Width

- **Unannotated code:** 76 characters max
- **Annotated code:** 55 characters of code + annotation column starting at column 58–60

Break long lines before they hit the annotation column so annotations align cleanly.

### 1.3 Self-Containedness

Self-containedness depends on the listing's role (see §1.5). In all cases the chapter draft must make the convention explicit in the prose so the reader does not have to guess what state each listing assumes.

- **Type-along listings** must run standalone — the reader is typing them. All required imports appear at the top of the listing. The only exception is when the chapter text explicitly marks a listing as a continuation of the previous one.
- **API illustrations** may rely on objects established in an earlier cell of the notebook (typically `env`, `dataset`, or `stats`). The prose names the objects assumed to exist.
- **Provided utilities** live in `src/ch03/*.py` and import freely from each other. The listing rendered in the book shows the full implementation as it exists in the module file, including imports.

### 1.4 Listing Numbering

Listings are numbered by chapter: `2.1`, `2.2`, ..., `2.11`. The plan document fixes the count at 11. If you find yourself wanting a 12th listing during implementation, either fold it into an existing listing or update `chapter_2_plan.md` first.

### 1.5 Listing Roles

Every listing in the chapter plays exactly one of three roles. The role drives how the listing is written and how the prose around it reads.

| Role | Reader's mode | How to write it |
|------|---------------|-----------------|
| **Type-along teaching code** | Types out, line by line | Minimal, conceptually load-bearing. Strip non-essential helpers. Annotations target the *idea*, not the syntax. The lead-in sentence names the function and what it teaches. |
| **API illustration** | Reads and runs | A short library call that surfaces an interface. Don't add framework detail beyond what the next listing needs. The lead-in sentence frames *what API surface is being shown*, not the underlying implementation. |
| **Provided utility** | Imports from `ch03.*` and calls | Full implementation is shown in the book for transparency but lives in `src/ch03/`. The lead-in sentence explicitly says `from ch03.<module> import <function>` and notes that the listing is included for transparency, not for typing. |

The Listings Summary table in `docs/chapter_2_plan.md` is the canonical record of each listing's role. When you change a listing's role, update that table, the lead-in sentence in the chapter draft, and `agents/chapter-02-guide.md` in the same commit.

---

## 2. Code Style (Python)

These are the binding rules from `STYLEGUIDE.md` Section 5 that `style-check` enforces. A quick summary so you don't have to context-switch:

- **Indentation:** 4 spaces, never tabs.
- **Line length:** 76 (or 55 if annotated). See §1.2 above.
- **Imports:** All at the top of the file, in standard / third-party / local groups separated by blank lines.
- **Quotes:** Straight quotes only in code (`"foo"`, not `"foo"`).
- **Naming:** `snake_case` for functions and variables, `PascalCase` for classes, `UPPER_SNAKE` for module constants.
- **No abbreviations:** Write `vision_encoder`, not `ve`. Write `dataloader`, not `dl`. The reader is learning — abbreviations slow them down.
- **No emojis in code or comments.**

Run `style-check` on every `.py` file after a substantial edit.

---

## 3. Callout Boxes

Callouts appear in chapter prose, not in code files. But code listings sometimes need to align with a callout's claim (for example, the "Z-score vs. min-max" callout in §2.5 must match the actual normalize function in code).

When you implement code referenced by a callout, re-read the callout to make sure the code's behavior matches what the prose promises.

---

## 4. Figures

Figures live in `figures/` and are produced by notebooks in `notebooks/`.

### 4.1 Production Rules

- **Format:** PNG at 300 DPI for print, with a transparent or white background.
- **Naming:** `figure_2_{number}_{slug}.png` — e.g., `figure_2_4_action_distributions.png`.
- **Resolution:** 96×96 sim renders should be upscaled or paired with an inset before export.
- **Fonts:** Matplotlib default; do not embed exotic fonts that may not render in print.

### 4.2 Reproducibility

Every figure must be regenerable from a notebook checked into `notebooks/`. The notebook's first cell should set a fixed random seed. Caption text lives in the chapter draft, not in the figure.

For full requirements (color palette, axis labels, caption format), defer to `../../lrm-book/FIGURE_STYLE_GUIDE.md`.

---

## 5. Banned Language (in code comments and docstrings)

These come from `STYLEGUIDE.md` Section 3 and apply to any text the reader will see:

- **Marketing words:** revolutionary, groundbreaking, cutting-edge, state-of-the-art, game-changing, transformative, exciting, incredible, powerful, novel, elegant.
- **Meta-language:** "in this chapter", "we will see", "as we mentioned earlier", "let's dive in", "before we proceed".
- **Hedges:** "very", "really", "quite", "rather", "somewhat", "basically", "essentially".

Use plain, present-tense, active-voice English. "Normalize the action vector." — not "We will now apply normalization."

---

## 6. Quick Checklist (Per File)

Before considering a `.py` file done:

- [ ] Imports at the top, grouped correctly
- [ ] All listings inside the file are self-contained (or marked as continuations) per their role (§1.5)
- [ ] Annotations are contiguous starting at `#A`
- [ ] No abbreviations, no emojis, no banned words
- [ ] Line widths respected (76 unannotated, 55 annotated)
- [ ] Each listing's role (type-along / API illustration / provided utility) matches the Listings Summary table in `docs/chapter_2_plan.md`
- [ ] The lead-in sentence in the chapter draft reflects the role
- [ ] `agents/chapter-02-guide.md` listing roadmap is in sync if a listing's title, order, or role changed
- [ ] Ran `style-check` — clean
- [ ] Ran `listing-check` — clean
- [ ] Tests exist or `test-gen` has been run
