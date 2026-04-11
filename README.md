# Task1 Final Repository

## Layout

- `src/upstream/`: upstream profile runners, hybrid routes, and v9 verifier
- `src/postprocess/`: template router and template tree runner
- `policies/`: routing policies used by the postprocess stages
- `tools/`: CSV path preparation and submission zip packaging
- `assets/frozen_inputs/mainline/`: frozen upstream inputs for the mainline replay path
- `evidence/run_secondary/`: archived logs and outputs for the secondary run
- `evidence/run_primary/`: archived logs and outputs for the primary run

## Requirements

Install packages from `requirements.txt`.

```bash
pip install -r requirements.txt
```

Input data can be provided in either of the following ways:

- Set `TASK1_DATA_ROOT` to the directory containing `test.csv`
- Set `INPUT_CSV` directly to the input csv path

The inference scripts read `VQA_API_KEY` from the environment. They also support a repository-local `.openai_api_key` file.

## Mainline Replay

Mainline replay uses the frozen upstream inputs in `assets/frozen_inputs/mainline/`.

```bash
bash scripts/run_main_snapshot.sh
```

Override input paths when needed:

```bash
TASK1_DATA_ROOT=/path/to/task1_data bash scripts/run_main_snapshot.sh
```

Outputs:

- `runs/mainline/output/prediction.txt`
- `runs/mainline/output/model.json`
- `runs/mainline/output/result.zip`

## Live Pipeline

Live pipeline rebuilds upstream outputs and then runs the final template router:

```bash
bash scripts/run_live_template_pipeline.sh
```

Override input paths when needed:

```bash
TASK1_DATA_ROOT=/path/to/task1_data bash scripts/run_live_template_pipeline.sh
```

Outputs:

- `runs/<tag>/upstream_clean_routes/`
- `runs/<tag>/upstream_v9/`
- `runs/<tag>/final_router/output/prediction.txt`
- `runs/<tag>/final_router/output/model.json`
- `runs/<tag>/final_router/output/result.zip`
