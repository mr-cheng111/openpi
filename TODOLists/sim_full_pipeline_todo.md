# Raytron MuJoCo Full Pipeline TODO Lists

## 0. Project Scope Freeze

### 0.1 Robot spec (Raytron)
- [ ] Confirm robot name: `raytron`
- [ ] Confirm total DoF definition: `12 + 2 = 14` (dual-arm + two grippers)
- [ ] Freeze action order: `left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)`
- [ ] Freeze state/proprio order to exactly match the action order
- [ ] Freeze camera setup: 3 RGB cameras = `left_arm_rgb`, `right_arm_rgb`, `head_rgb`
- [ ] Freeze camera resolution, fps, and timestamp alignment policy across all 3 cameras
- [ ] Freeze control frequency and action horizon (ALOHA_SIM-like setup)

Definition of done:
- [ ] A one-page spec for `raytron` observation/action schema is written and versioned
- [ ] All downstream scripts use the same 14D ordering
- [ ] All downstream scripts use the same 3-camera RGB key names and image specs

### 0.2 Target objective in simulation
- [ ] Confirm MuJoCo task list for Raytron (train/val/test task names)
- [ ] Define success metric per task (binary success + optional efficiency metric)
- [ ] Define acceptance target (for example success rate threshold on held-out seeds)

Definition of done:
- [ ] Task list and acceptance threshold are fixed before large-scale collection

## 1. Environment and Baseline

### 1.1 Environment setup
- [ ] Verify GPU and driver status (`nvidia-smi`)
- [ ] Use target env (`/home/mr_cheng111/miniconda3/envs/pi0`) and verify Python path
- [ ] Install dependencies (`uv sync`)
- [ ] Install openpi package in editable mode (`uv pip install -e .`)
- [ ] Confirm submodules are initialized (`git submodule update --init --recursive`)
- [ ] Verify MuJoCo import and rendering (`python -c "import mujoco; print('ok')"`)
- [ ] Run minimal openpi smoke test (`python -c "import openpi; print('ok')"`)

Definition of done:
- [ ] No import/runtime errors for MuJoCo and openpi
- [ ] Environment can be recreated on this machine

### 1.2 Baseline inference sanity
- [ ] Run one official inference example end-to-end (already validated path)
- [ ] Save command history and baseline latency stats
- [ ] Record baseline server/client settings for reproducibility

Definition of done:
- [ ] Baseline inference runs successfully with valid action chunks
- [ ] Baseline logs are archived for comparison

## 2. MuJoCo Data Collection (Raytron)

### 2.1 Raytron MuJoCo environment instrumentation
- [ ] Implement/adapt Raytron MuJoCo env wrapper (ALOHA_SIM style runtime interface)
- [ ] Finalize observation keys (`observation/left_arm_rgb`, `observation/right_arm_rgb`, `observation/head_rgb`, `observation/state`, `prompt`)
- [ ] Finalize 14D action representation and clipping bounds
- [ ] Add episode metadata logging (task id, seed, success/failure, reset reason)
- [ ] Ensure timestamps are synchronized across observations/actions
- [ ] Add per-camera health checks (missing frame, repeated frame, wrong shape/dtype)
- [ ] Save one rollout video per episode (`.mp4`) for manual inspection

Definition of done:
- [ ] One episode can be replayed frame-by-frame
- [ ] Logged fields match expected policy input/output schema
- [ ] Video output is generated automatically per episode
- [ ] All 3 RGB camera streams are present and synchronized in sampled episodes

### 2.2 Collection strategy
- [ ] Define train/val/test split policy by task and random seed
- [ ] Define coverage goals (initial poses, object variation, distractors, lighting/randomization)
- [ ] Add quality filters (minimum episode length, non-zero action variance, valid termination)
- [ ] Add camera quality filters (exposure outliers, black/blank frames, heavy motion blur)
- [ ] Collect pilot set first (for example 100 to 500 episodes)
- [ ] Review pilot videos and metadata, then fix collection bugs
- [ ] Run large-scale unattended collection

Definition of done:
- [ ] Pilot data passes schema + quality checks
- [ ] Collection job can run stably and resume from interruption

## 3. Data Packaging (to LeRobot format)

### 3.1 Conversion pipeline
- [ ] Create/adapt conversion script for Raytron MuJoCo data
- [ ] Map Raytron observations/actions to openpi policy fields
- [ ] Ensure 3 RGB cameras map to fixed model inputs with deterministic channel order (HWC/RGB)
- [ ] Explicitly validate 14D state/action ordering in converted data
- [ ] Export episodes to LeRobot-compatible structure
- [ ] Add dataset version tag and changelog

Definition of done:
- [ ] Converted dataset loads without key/shape errors
- [ ] Dataset version is immutable and reproducible

### 3.2 Validation and normalization
- [ ] Run schema validation on random sampled episodes
- [ ] Verify tensor shapes, dtypes, and action bounds
- [ ] Verify all 3 camera tensors have expected resolution, dtype, and value range
- [ ] Compute normalization stats (`uv run scripts/compute_norm_stats.py --config-name <raytron_config>`)
- [ ] Inspect `std`, `q01`, `q99` for abnormal dimensions

Definition of done:
- [ ] No schema mismatch in sampled episodes
- [ ] Norm stats pass sanity checks

## 4. Training (Raytron)

### 4.1 Training config
- [ ] Add/adapt Raytron input/output mapping class (14D dual-arm schema + 3-camera RGB inputs)
- [ ] Add/adapt Raytron data config in training config
- [ ] Create a dedicated config name (for example `pi05_raytron_mujoco`)
- [ ] Define experiment naming and checkpoint retention policy
- [ ] Set deterministic seed strategy for reproducibility

### 4.3 Code integration checklist (pi0.5 for Raytron)
- [x] Add `src/openpi/policies/raytron_policy.py` with `RaytronInputs` and `RaytronOutputs`
- [x] Map 3 RGB cameras to model image slots: `head_rgb -> base_0_rgb`, `left_arm_rgb -> left_wrist_0_rgb`, `right_arm_rgb -> right_wrist_0_rgb`
- [x] Add `LeRobotRaytronDataConfig` in `src/openpi/training/config.py`
- [x] Add dataset repack mapping for Raytron LeRobot keys (`observation.images.*`, `observation.state`, `action`, `prompt`)
- [x] Register new train config `pi05_raytron_mujoco` in `_CONFIGS`
- [x] Set base weight loader to `gs://openpi-assets/checkpoints/pi05_base/params`
- [ ] Run `compute_norm_stats` with `pi05_raytron_mujoco` and fix any schema mismatches
- [ ] Run a short sanity train and verify action output is exactly 14D after output transform

Definition of done:
- [ ] `config-name` is runnable and documented
- [ ] Artifact path layout is fixed
- [ ] Ablation check completed: 3-camera input path works in a short sanity run

### 4.2 Training execution
- [ ] Run small-budget sanity training first
- [ ] Check loss curves and overfitting signals
- [ ] Run full-budget training
- [ ] Save checkpoints at fixed intervals
- [ ] Log metadata (git commit, config hash, dataset version)

Recommended command template:
- [ ] `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py <raytron_config> --exp-name=<exp_name> --overwrite`

Definition of done:
- [ ] Training completes without runtime errors
- [ ] At least one checkpoint reaches target validation metric

## 5. MuJoCo Evaluation and Video Review

### 5.1 Closed-loop MuJoCo evaluation
- [ ] Start policy server with selected Raytron checkpoint
- [ ] Run MuJoCo eval rollouts on held-out seeds/tasks
- [ ] Compute success rate and episode efficiency metrics
- [ ] Save rollout videos for both success and failure cases
- [ ] Verify inference uses all 3 camera inputs online (not silently dropping any stream)
- [ ] Cluster failure cases by mode (grasp miss, drift, collision, timeout, etc.)

Recommended command template:
- [ ] `uv run scripts/serve_policy.py policy:checkpoint --policy.config=<raytron_config> --policy.dir=<checkpoint_dir>`

Definition of done:
- [ ] Quantitative eval report is generated
- [ ] Videos can be replayed for qualitative inspection

### 5.2 Acceptance gate in MuJoCo sim
- [ ] Run full fixed-seed acceptance suite
- [ ] Verify latency budget and stability across long-horizon episodes
- [ ] Sign off deployment candidate and archive metrics + videos

Definition of done:
- [ ] End-to-end pipeline is reproducible from MuJoCo data collection to MuJoCo closed-loop evaluation

## 6. Weight Export and Artifact Management

### 6.1 Export strategy
- [ ] Decide serving format (JAX native or converted PyTorch)
- [ ] If needed, convert checkpoint (`uv run examples/convert_jax_model_to_pytorch.py ...`)
- [ ] Validate exported weights by loading and running one inference pass

Definition of done:
- [ ] Exported artifact is loadable in target runtime
- [ ] Inference outputs are numerically reasonable

### 6.2 Versioning and release
- [ ] Add model card metadata (task scope, limitations, training data version)
- [ ] Tag artifact with semantic version and git commit
- [ ] Store artifact checksum and storage location
- [ ] Freeze dependency versions used for export

Definition of done:
- [ ] Another machine can fetch artifact and reproduce inference test

## 7. Suggested Milestones

- [ ] M1: Raytron spec frozen + environment baseline ready
- [ ] M2: MuJoCo pilot dataset collected and validated
- [ ] M3: First convergent Raytron training run completed
- [ ] M4: Selected checkpoint passes MuJoCo held-out evaluation
- [ ] M5: MuJoCo acceptance suite passed with archived videos/reports

## 8. Risk and Mitigation Checklist

- [ ] Action/state order mismatch risk: add strict 14D schema assertions in collection + conversion
- [ ] Multi-camera mismatch risk: enforce strict key/shape/timestamp checks for 3 RGB streams
- [ ] Data schema drift risk: enforce schema checks before each training run
- [ ] Norm stats instability risk: inspect low-variance dims and clamp if needed
- [ ] Reproducibility risk: lock seed, config hash, dataset version, dependency versions
- [ ] Serving latency risk: benchmark with representative observation payloads
