# AGENTS.md

Guidelines for agents working in this repository.

The goal is to turn travel photos and videos into a smooth, watchable video flow.

`README.md` explains how to run the project.
`AGENTS.md` explains how to make decisions while changing it.
`docs/ai-coding.md` maps the pipeline responsibilities and change boundaries.

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
- Use `docs/ai-coding.md` when you need the current pipeline map, step ownership, or safe change workflow.
- Use `README.md` for steps and this file for priorities.
- Preserve scene-level grouping.
- Extract only the helper information that is needed.
- Use metadata and tags to improve the final editing JSON, not as an end in themselves.

## Implementation Notes

- Do not add complex branching only to improve OCR or face detection.
- Judge features by whether they help editing decisions.
- Do not depend too heavily on a single detection result.
- Prefer designs that degrade gracefully when helper signals are incomplete.

## Before You Finalize Changes

- Check whether the output is easy to use for scene-level editing decisions.
- Check whether the generated artifacts are easy for humans to review.
- Check that the change is not a local optimization for OCR or face detection.
- Check that the final video flow still feels natural.

## Existing Assets

- Do not heavily change the existing CSV, JSON, or representative-selection logic without need.
- If you must change an output, review the impact from upstream steps forward.
- Add new metrics only when they clearly improve scene editing.

## In One Sentence

This project is not for identifying media correctly in isolation.
It is for building a travel video flow that feels good to watch.
