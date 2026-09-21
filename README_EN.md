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
- Generated files under `outputs/` are ignored by Git by default.
- HTML reports escape untrusted labels and messages, use a restrictive Content Security Policy, and load no scripts or external resources.
- Private customer data must not be posted in public issues or sent to an unauthorized model provider.

## Service and contact

The repository includes a fixed-scope [7-day reliability audit](docs/service/service_one_pager.md), a [client intake form](docs/service/client_intake.md), and a [report template](docs/service/report_template.md).

For a non-sensitive initial inquiry, open an [audit request](https://github.com/Potato0705/Agent-Review/issues/new?template=audit-request.md). Do not attach private samples, API keys, personal data, or confidential business information to a public issue.

## License

MIT
