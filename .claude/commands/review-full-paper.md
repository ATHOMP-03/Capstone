---
model: opus
description: "Full 6-pass editorial pipeline on the complete assembled paper. Run this after all sections have been drafted via /draft-section. Assembles all draft_*.tex files in logical order, reads the complete paper, and runs all editors and reviewers with a focus on cross-section cohesion, narrative consistency, and whether the paper holds up as a unified argument to rigorous refereeing. Produces a final report card saved to output/. Triggers include: 'review the full paper', 'run the full pipeline', 'check the whole paper', 'is the paper ready', 'review everything', 'full review'."
---

# Full Paper Review Pipeline

You are coordinating a complete editorial review of a finished (or near-finished) draft. This command runs after all individual sections have been drafted. Your job is to evaluate the paper as a unified argument — not section by section, but as the thing a referee will actually read.

## Step 0 — Assemble the Paper

1. Read `CLAUDE.md` for project context and conventions.
2. Find all drafted sections in `output/draft_*.tex`. Read them in logical paper order:
   - Abstract → Introduction → Literature Review → Data → Methodology → Results → Discussion → Conclusion → Appendix (if present)
3. Read all tables referenced via `\input{}` from `output/*.tex`.
4. Read `output/results_draft.md` to cross-check that the prose matches the actual results.
5. Read any `.bib` file present to understand the bibliography.
6. Skim `src/python/` and `src/r/` estimation code to verify that the methods section describes what was actually estimated.

If any expected sections are missing from `output/draft_*.tex`, note them at the top of the report and proceed with what exists.

---

## Step 1 — Structural & Narrative Review

Read the full instructions in `.claude/commands/review-paper.md`, then apply them to the **complete assembled paper**.

For this pass, the primary lens is **cross-section cohesion**:
- Does the introduction promise what the results deliver — exactly, not approximately?
- Does the methods section set up every test that appears in the results?
- Does the discussion account for every significant (and insignificant) result, or does it selectively interpret?
- Does the conclusion match the actual findings, or does it soften/overstate?
- Is the reverse causality concern (+2.348 placebo, flat lag decay) treated with the same weight throughout all sections, or does it appear prominently in robustness and quietly disappear in the conclusion?
- Are there any internal contradictions between sections (a claim in the intro that the results don't support, a method described that doesn't appear in the tables)?

Record findings under: `## PASS 1: Structural & Narrative`

---

## Step 2 — Econometrics Audit

Read the full instructions in `.claude/commands/econometrics-editor.md`, then audit the **complete paper**.

For this pass, additionally check:
- That the identification strategy stated in the introduction is the one actually implemented in the methodology section and the code.
- That every robustness check described in the methods is reported in the results.
- That the discussion's causal language is consistent with what the design actually supports.
- That coefficient magnitudes cited in the text match the table values exactly.

Record findings under: `## PASS 2: Econometrics Audit`

---

## Step 3 — Content & Logic Edit

Read the full instructions in `.claude/commands/content-editor.md`, then apply a deep content edit to the **complete paper**.

For this pass, pay particular attention to:
- Whether the paper's contribution is clearly differentiated from Gu & Kurov (2020) and Teti et al. (2019) — not just stated but demonstrated.
- Whether the DML–FE divergence (null in FE, significant in DML) is explained with a coherent mechanism or simply reported.
- Whether the `px_high` result (positive sign vs. negative `return` sign) is integrated into the argument or treated as an isolated curiosity.
- Whether the paper earns the word "causal" anywhere it uses it.

Record findings under: `## PASS 3: Content & Logic`

---

## Step 4 — Style Edit

Read the full instructions in `.claude/commands/style-editor.md`, then flag AI-style tells and academic filler across the **complete paper**.

Additionally check for:
- Tone inconsistencies across sections (one section sounds like a consulting report, another like a dissertation).
- Repetition of key phrases across sections that should be varied.
- Any section that reads noticeably weaker or more AI-generated than the others.

Record findings under: `## PASS 4: Style`

---

## Step 5 — LaTeX Copy Edit

Read the full instructions in `.claude/commands/latex-copy-editor.md`, then apply the full copy editing checklist to all `.tex` files.

For this pass, additionally check:
- That all `\input{}` references resolve to actual files in `output/`.
- That table and figure numbering is sequential across the assembled paper.
- That `\label{}` and `\ref{}` are consistent (no `??` would appear in the compiled PDF).
- That the bibliography style is applied consistently across all sections.

Record findings under: `## PASS 5: LaTeX Copy Edit`

---

## Step 6 — Devil's Advocate

Read the full instructions in `.claude/commands/devil.md`, then apply all four questions to the **complete paper as a whole**.

For this pass, the question is not whether any single section has a problem — it is whether the paper, taken as a complete argument, survives adversarial scrutiny. Write the rejection letter for the complete submission.

Record findings under: `## PASS 6: Devil's Advocate`

---

## Step 7 — Final Report Card

After all six passes, produce two outputs:

### Output 1: Annotated Draft

For each substantive issue identified across any pass, add a `% [REVIEW-PASS-N]: issue description` comment at the relevant location in the corresponding `output/draft_*.tex` file. This lets the author find issues in context while editing.

Use these tags:
- `% [R1]:` Structural/narrative issue
- `% [R2]:` Econometrics issue
- `% [R3]:` Content/logic issue
- `% [R4]:` Style flag
- `% [R5]:` LaTeX/copy issue
- `% [R6]:` Devil's advocate flag

### Output 2: Report Card

Save to `output/report_card_full_paper.md` with the following structure:

```
# Full Paper Report Card
Date: [today]
Style: [Academic / Business]
Sections reviewed: [list]

---

## Overall Grade
[A / B / C / D — one letter, one sentence justification]

---

## Cross-Cutting Issues
Issues flagged by 2 or more passes. Highest priority.

| Issue | Passes | Priority |
|-------|--------|----------|
| [specific description] | R1, R2 | Critical |

---

## The Central Argument: Does It Hold?
[1–2 paragraphs answering: does the paper, as assembled, make a coherent and defensible causal argument? What is the single thing that most weakens it?]

---

## Pass-by-Pass Summary

| Pass | Top Finding | Action Required |
|------|-------------|----------------|
| 1. Structural | | |
| 2. Econometrics | | |
| 3. Content | | |
| 4. Style | | |
| 5. LaTeX | | |
| 6. Devil | | |

---

## What the Rejection Letter Would Say
[From Pass 6 — the hostile referee's rejection, written out in full.]

---

## Revision Priority List
Ordered by urgency. Be specific: section, paragraph, issue, fix.

1. [Critical — must fix before advisor review]
2.
3.
...

---

## Submission Readiness Checklist
- [ ] Central argument defensible despite placebo test result
- [ ] Reverse causality treated consistently across all sections
- [ ] All causal claims match what the design supports
- [ ] DML–FE divergence explained, not just reported
- [ ] Cross-section internal consistency confirmed
- [ ] All table values match in-text citations
- [ ] Style consistent across sections
- [ ] LaTeX compiles clean (no ?? references)
- [ ] Devil's objection addressed or acknowledged
```

$ARGUMENTS
