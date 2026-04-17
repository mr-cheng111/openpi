# Simulation Full Pipeline TODO Lists

## 0. Environment and Baseline

### 0.1 Environment setup
- [ ] Verify GPU and driver status (`nvidia-smi`)
- [ ] Install dependencies (`uv sync`)
- [ ] Install openpi package in editable mode (`uv pip install -e .`)
- [ ] Confirm submodules are initialized (`git submodule update --init --recursive`)
- [ ] Run a minimal import smoke test (`uv run python -c "import openpi; print('ok')"`)

Definition of done:
- [ ] Commands run successfully without import errors
- [ ] A clean Python environment is reproducible on this machine

### 0.2 Base model and task baseline
- [ ] Select target model family for simulation (recommended start: `pi05`)
- [ ] Select target simulation benchmark/task set (for example LIBERO task subset)
- [ ] Run one official inference example end-to-end to verify baseline works

Definition of done:
- [ ] Baseline inference produces valid action chunks
- [ ] Task and success metric are written down before data collection starts

## 1. Data Collection in Simulation

### 1.1 Simulator and task instrumentation
- [ ] Finalize observation keys to log (images, state, proprioception, language prompt)
- [ ] Finalize action representation and frequency
- [ ] Add episode metadata logging (task id, seed, success/failure, reset reason)
- [ ] Ensure synchronized timestamps across observation and action streams

Definition of done:
- [ ] One episode can be replayed and inspected step-by-step
- [ ] Logged fields match the expected policy input/output schema

### 1.2 Collection strategy
- [ ] Define train/val/test split policy by scenario and random seed
- [ ] Define coverage goals (object poses, distractors, lighting/domain randomization)
- [ ] Implement automatic quality filters (minimum episode length, non-zero action variance)
- [ ] Collect pilot set first (for example 100 to 500 episodes)
- [ ] Review pilot quality and fix collection bugs before large-scale rollout

Definition of done:
- [ ] Pilot data passes schema and quality checks
- [ ] Collection script can run unattended for large-scale generation

## 2. Data Packaging (to LeRobot format)

### 2.1 Conversion pipeline
- [ ] Create or adapt conversion script based on `examples/libero/convert_libero_data_to_lerobot.py`
- [ ] Map simulator observations/actions to openpi policy fields
- [ ] Export episodes to LeRobot-compatible structure
- [ ] Generate a dataset version tag and changelog

Definition of done:
- [ ] Converted dataset can be loaded without key/shape errors
- [ ] Dataset version is immutable and reproducible

### 2.2 Validation and normalization
- [ ] Run schema validation on a random subset of episodes
- [ ] Verify tensor shapes, dtypes, and action bounds
- [ ] Compute normalization stats (`uv run scripts/compute_norm_stats.py --config-name <your_config>`)
- [ ] Inspect generated norm stats for abnormal `std`, `q01`, `q99`

Definition of done:
- [ ] No schema mismatch in sampled episodes
- [ ] Norm stats are generated and sanity-checked

## 3. Training and Evaluation

### 3.1 Training config
- [ ] Add or adapt input/output mapping class for the simulator policy
- [ ] Add or adapt data config in training config
- [ ] Define experiment naming convention and checkpoint retention policy
- [ ] Set deterministic seed strategy for reproducibility

Definition of done:
- [ ] `config-name` is runnable and documented
- [ ] Training artifacts path structure is fixed

### 3.2 Training execution
- [ ] Launch first training run with a small budget (sanity run)
- [ ] Check loss curves and early overfitting signs
- [ ] Launch full run with target budget
- [ ] Save checkpoints at fixed intervals
- [ ] Log experiment metadata (git commit, config hash, dataset version)

Recommended command template:
- [ ] `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py <config_name> --exp-name=<exp_name> --overwrite`

Definition of done:
- [ ] Training completes without runtime errors
- [ ] At least one checkpoint reaches target validation metric

### 3.3 Offline and simulation evaluation
- [ ] Evaluate selected checkpoints on held-out scenarios
- [ ] Compute task success rate and episode efficiency metrics
- [ ] Record failure cases and cluster by failure mode
- [ ] Select candidate checkpoint(s) for deployment export

Definition of done:
- [ ] Best checkpoint is selected with quantitative evidence
- [ ] Evaluation report is stored with reproducible commands

## 4. Weight Export and Artifact Management

### 4.1 Export strategy
- [ ] Decide serving format (JAX native or converted PyTorch)
- [ ] If needed, convert checkpoint (`uv run examples/convert_jax_model_to_pytorch.py ...`)
- [ ] Validate exported weights by loading and running one inference pass

Definition of done:
- [ ] Exported artifact is loadable in target runtime
- [ ] Inference output from exported model is numerically reasonable

### 4.2 Versioning and release
- [ ] Add model card metadata (task scope, limitations, training data version)
- [ ] Tag artifact with semantic version and git commit
- [ ] Store artifact checksum and storage location
- [ ] Freeze dependency versions used for export

Definition of done:
- [ ] Another machine can fetch artifact and run the same inference test

## 5. Control Integration (Closed-loop)

### 5.1 Policy serving
- [ ] Start policy server with selected checkpoint
- [ ] Validate server health endpoint and latency budget
- [ ] Define retry and timeout behavior in control client

Recommended command template:
- [ ] `uv run scripts/serve_policy.py policy:checkpoint --policy.config=<config_name> --policy.dir=<checkpoint_dir>`

Definition of done:
- [ ] Control client can request actions continuously without protocol errors

### 5.2 Controller-side integration
- [ ] Implement observation pre-processing to exactly match training pipeline
- [ ] Implement action post-processing and safety clamps
- [ ] Add watchdog, emergency stop, and action-rate limiter
- [ ] Add rollout recorder for postmortem analysis

Definition of done:
- [ ] Closed-loop control runs stably for target episode duration
- [ ] Safety checks trigger correctly in simulated fault injection

### 5.3 End-to-end acceptance
- [ ] Run full scenario test suite with fixed seeds
- [ ] Measure success rate, latency, and failure recovery behavior
- [ ] Sign off deployment candidate and archive evidence

Definition of done:
- [ ] End-to-end pipeline is reproducible from raw simulation to closed-loop control

## 6. Suggested Milestones

- [ ] M1: Baseline ready (environment + official inference works)
- [ ] M2: Pilot dataset ready and validated
- [ ] M3: First convergent training run finished
- [ ] M4: Exported candidate model validated in server mode
- [ ] M5: Closed-loop simulation acceptance passed

## 7. Risk and Mitigation Checklist

- [ ] Data schema drift risk: enforce schema checks in CI before training
- [ ] Norm stats instability risk: inspect low-variance dimensions and clamp if needed
- [ ] Reproducibility risk: lock seed, config hash, dataset version, dependency versions
- [ ] Serving latency risk: benchmark with representative observation payload size
- [ ] Safety risk: keep action clipping and fail-safe behavior enabled by default
