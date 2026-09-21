# Agent Review

[中文说明](README.md)

Agent Review is a lightweight, dependency-free audit toolkit for LLM graders and AI evaluation systems. It tests whether a scoring system rewards superficial changes, misses genuine content degradation, or reacts inconsistently to meaning-preserving paraphrases.

The project is designed for traceable pre-release screening rather than leaderboard construction or formal psychometric certification.

## What it tests

Each baseline item is paired with targeted variants:

- `gaming`: irrelevant fluency, verbosity, keyword stuffing, or other surface changes that should not improve the score;
- `degradation`: removal of evidence, reasoning, task requirements, or other substantive content that should reduce the score;
- `paraphrase`: meaning-preserving rewrites that should remain within a configured tolerance.

The toolkit also reports a screening metric:

```text
validity margin = mean degradation drop - largest positive gaming gain
```

A non-positive margin means that the system rewards a superficial change at least as strongly as it detects substantive degradation. This is a diagnostic signal, not a complete validity argument.

## Capabilities

- offline audits from scored CSV files;
- OpenAI-compatible `/chat/completions` scoring;
- model-drafted paraphrase candidates that only a human reviewer can admit to the case set;
- repeated sampling with sample standard deviations and conservative 95% t intervals;
- provisional risk labels when uncertainty is incomplete or crosses a decision threshold;
- resumable JSONL checkpoints with strict run-context validation;
- verified model or prompt-version comparisons using matching inputs, rubric hashes, thresholds, scale, temperature, and repeat count;
- Markdown, JSON, and secure self-contained HTML reports;
- no runtime dependencies beyond Python 3.10+.

## Quick start

```powershell
python -m agent_audit audit `
  --input examples/demo_scores.csv `
  --report outputs/demo_report.md `
  --json outputs/demo_result.json `
  --html outputs/demo_report.html `
  --score-min 0 `
  --score-max 10 `
  --data-provenance synthetic
```

Run the complete local review gate:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/review.ps1
```

The review script also enforces per-module line-coverage floors, measured with the standard library only:

```powershell
python scripts/coverage_report.py
```

Generate gaming and degradation variants from annotated baselines, so the most time-consuming part of a delivery is no longer hand-written:

```powershell
python -m agent_audit generate `
  --input examples/essay_baselines.csv `
  --output outputs/generated_cases.csv `
  --seed 0
```

The input marks which sentences carry the argument (`evidence_sentences`, 1-based). The tool never infers that: a missing, out-of-range, or all-covering annotation stops the run. Every variant must satisfy a machine-checkable postcondition before it is written — a gaming variant must still contain the whole baseline as a prefix, and a degradation variant must contain none of the annotated sentences while keeping the rest in order. The manifest records each intervention's size and the exact corpus sentences inserted, so a reviewer can confirm the padding really is irrelevant to that prompt.

English baselines use `--language english`; the language is never guessed. English periods are ambiguous (`Dr.`, `3.5`, `J. K.`) while annotations are given by sentence index, so one extra split silently points every later index at the wrong sentence. Besides an abbreviation-aware splitter, `--show-sentences` prints the numbered split without writing anything, and an optional `sentence_count` column turns a disagreement between the splitter and the reviewer into a refusal that names the splitter and shows what it found.

The case file holds your hand edits, so it is never overwritten: `generate` refuses when the output already exists. Use `--append` to merge new variants in, keeping every existing row. Merging also compares each kept row against what the generator would produce — one edited row or one hand-written variant marks the whole set `mixed`, and the audit then refuses to call it machine-generated. Appending therefore cannot launder hand-written content into a verifiable claim.

Auditing a generated set requires proof, not just a claim:

```powershell
python -m agent_audit audit `
  --input outputs/live_scores.csv `
  --report outputs/live_report.md `
  --variant-origin machine-generated `
  --generation-manifest outputs/generated_cases.manifest.json
```

The audit matches the generation manifest's `output_sha256` against the scoring manifest's `input_sha256`. If they differ, the cases were edited after generation, `machine-generated` would be false, and the command refuses and points at `mixed` instead. Editing generated rows is fine — it just has to be declared honestly.

Rule-based paraphrase generation is off by default: the conservative rewrite almost always scores the same, so including it would dilute the violation rate and make a grader look safer than it is.

Real rewrites are a different matter. Across four live runs in this repository, paraphrases violated 7 of 17 times while gaming variants violated 0 of 17 — and the paraphrases that found those failures were whole-sentence rewrites, not swapped connectives. Only a model or a person writes at that strength, so `paraphrase` is a separate subcommand that keeps `generate` offline and deterministic:

```powershell
python -m agent_audit paraphrase `
  --input examples/essay_baselines.csv `
  --output outputs/paraphrase_review.csv `
  --model YOUR_MODEL `
  --base-url https://YOUR_PROVIDER/v1
```

It writes a review file and never touches the case set. The prompt carries the baseline and "rewrite this without changing its meaning" — no rubric, because a rewrite optimised against the criteria under test is circular, and no statement of purpose, because telling a model it is probing a grader invites adversarial rather than faithful rewriting.

Checks are split in two. Empty, unchanged, baseline-echoing, and wildly out-of-band drafts are blocked. Missing numbers and changed negation counts are only listed as review notes, because as gates they rejected 2 of the 5 genuine hand-written paraphrases in this repository — a 40% false-rejection rate teaches reviewers to ignore the status column.

Approved rows merge in with `generate --append --paraphrase-review`. An unrecognised `status` or a `baseline_sha256` that no longer matches its baseline stops the run and names the case. Because a human, not a postcondition, judged the rewrites equivalent, the merged set is always `mixed` — the tool does not vouch for an equivalence claim it cannot verify. The generation manifest records the drafting model, and the audit refuses outright when that model is also the grader.

The committed [synthetic HTML example](docs/examples/example_audit_report.html) can be downloaded and opened locally without a server or network connection.

## Score with an OpenAI-compatible model

API keys are read from an environment variable and are never accepted as a plaintext command-line option.

```powershell
$env:OPENAI_API_KEY = "your-key"

python -m agent_audit score `
  --input examples/essay_cases.csv `
  --output outputs/live_scores.csv `
  --rubric-file examples/essay_rubric.md `
  --model YOUR_MODEL `
  --base-url https://YOUR_PROVIDER/v1 `
  --system-name "YOUR_MODEL + rubric-v1" `
  --score-min 0 --score-max 10 `
  --temperature 0.2 --repeats 3 `
  --checkpoint outputs/live_checkpoint.jsonl
```

Use the same command with `--resume` after an interruption. A resume is accepted only when the endpoint, model, scale, temperature, repeat count, input hash, rubric hash, and other run identifiers match exactly.

## Compare two verified runs

```powershell
python -m agent_audit compare `
  --reference outputs/reference_audit.json `
  --candidate outputs/candidate_audit.json `
  --report outputs/version_comparison.md `
  --json outputs/version_comparison.json `
  --html outputs/version_comparison.html
```

The comparison command rejects incompatible cases, variant types, thresholds, score ranges, provenance declarations, or verified scoring contexts. It separates cross-threshold changes from within-threshold magnitude changes and marks changes as provisional when either side has unresolved uncertainty.

## Public evidence

- [Gemma 3 4B reliability case study](docs/case_studies/gemma3_case_study.md): 5 baselines, 15 paired variants, and 60 local scoring calls.
- [Gemma 3 4B vs Gemma 4 E4B comparison](docs/case_studies/gemma_version_comparison.md): two runs with matching inputs, rubric, temperature, repeats, thresholds, and scale.
- [Review log](docs/governance/review_log.md): design risks, fixes, failure injection, browser review, and regression evidence.

Both model studies are explicitly exploratory. Five hand-authored baseline cases cannot establish population reliability, model superiority, statistical significance, or suitability for real educational decisions.

## Security and data handling

- Remote endpoints must use HTTPS; plain HTTP is allowed only for localhost.
- Variant labels are not sent to the scoring model.
- Raw replies and checkpoints are opt-in and must be treated as sensitive data.
- `--variant-origin` records whether the variants were written by hand (`human-authored`), produced by a generator (`machine-generated`), or both (`mixed`). A generated set is a floor test: failing it is conclusive, passing it proves little, and the report says so. A comparison is refused when the two sides disagree.
- Generated files under `outputs/` are ignored by Git by default.
- HTML reports escape untrusted labels and messages, use a restrictive Content Security Policy, and load no scripts or external resources.
- Private customer data must not be posted in public issues or sent to an unauthorized model provider.

## Service and contact

The repository includes a fixed-scope [7-day reliability audit](docs/service/service_one_pager.md), a [client intake form](docs/service/client_intake.md), and a [report template](docs/service/report_template.md).

For a non-sensitive initial inquiry, open an [audit request](https://github.com/Potato0705/Agent-Review/issues/new?template=audit-request.md). Do not attach private samples, API keys, personal data, or confidential business information to a public issue.

## License

MIT
