# AGENTS.md

Guidelines for agents working in this repository.

The goal is to turn travel photos and videos into a smooth, watchable video flow.

`README.md` explains how to run the project.
`AGENTS.md` explains how to make decisions while changing it.

## Codex Workflow

- Use `AGENTS.md` as the persistent instruction layer for this repository.
- Use `PLANS.md` only for multi-hour or complex work that needs staged delivery.
- Prefer minimal constraints and iterative execution over upfront design.
- Start with the smallest working implementation, then refine only as needed.
- Update or create documents only when needed to resolve repeated ambiguity or repeated errors.
- Prioritize working code and usable outputs over perfect structure.
- Treat docs other than `AGENTS.md` and `PLANS.md` as reference material, not as instruction layers.

## Delivery Workflow

- Default to `Ask -> Code -> Review` for changes that cross step boundaries, change data contracts, or affect preview quality.
- For small tasks, implement directly, run checks, and self-review instead of adding process overhead.
- For medium tasks, use one coding agent for implementation and a separate review pass or review agent before commit.
- For large tasks, use `PLANS.md`, split work into staged changes, and review both data contracts and generated outputs.
- In this repository, review should check both code quality and whether the output improves watchable scene flow.
- Render-related changes should include output review, not only code review, because preview quality is part of the product.

## Highest Priorities

- Optimize for a watchable video, not for perfect media recognition.
- Treat OCR and face detection as helper signals only.
- Do not turn OCR or face-detection accuracy into the main goal.
- Prefer scene-level flow, continuity, and pacing over local detection precision.

## Decision Rules

- When in doubt, choose the more natural scene flow over per-asset correctness.
- Treat representative assets and tags as editing summaries, not as ground truth.
- If an edge case adds too much complexity, prefer the simpler flow first.
- Before tuning recognition details, check whether the overall viewing flow improves.

## Working Style

- Follow the workflow described in `README.md`.
- Use `README.md` for steps and this file for priorities.
- For small tasks, avoid over-planning and move quickly to the smallest working change.
- Preserve scene-level grouping.
- Extract only the helper information that is needed.
- Use metadata and tags to improve the final editing JSON, not as an end in themselves.

## Implementation Notes

- Keep the code clean.
- Make the smallest correct change.
- Do not add complex branching only to improve OCR or face detection.
- Do not introduce duplication, dead code, or unnecessary abstractions.
- Judge features by whether they help editing decisions.
- Do not depend too heavily on a single detection result.
- Prefer designs that degrade gracefully when helper signals are incomplete.
- When modifying code, leave it cleaner than before.
- Remove anything unused or obsolete caused by the change.
- Do not weaken tests or checks.

## Before You Finalize Changes

- Check whether the output is easy to use for scene-level editing decisions.
- Check whether the generated artifacts are easy for humans to review.
- Check that the change is not a local optimization for OCR or face detection.
- Check that the final video flow still feels natural.
- Report `change made`, `cleanup done`, and `remaining risks`.

## Existing Assets

- Do not heavily change the existing CSV, JSON, or representative-selection logic without need.
- If you must change an output, review the impact from upstream steps forward.
- Add new metrics only when they clearly improve scene editing.

## In One Sentence

This project is not for identifying media correctly in isolation.
It is for building a travel video flow that feels good to watch.
