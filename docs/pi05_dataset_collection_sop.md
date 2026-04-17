# pi0.5 稳定数据集采集 SOP（openpi）

这份 SOP 面向 `openpi` 的 `pi0.5`（`PI05`）训练流程，目标是让数据“可训练、可复现、可上线”。

---

## 1. 先冻结“训练契约”（采集前必须完成）

### 1.1 固定训练配置名
- 先确定你最终训练用的 config（例如 `pi05_droid_finetune` 或 `pi05_raytron_mujoco`）。
- 以后数据字段以这个 config 的 `repack + inputs transform` 为准，不再随意改字段名。

### 1.2 固定 observation/action schema（必须版本化）
- 为你的数据写一个 `schema_v1`（JSON/YAML/Markdown 都行），至少包含：
- observation keys（每个 key 的 shape/dtype/单位）
- action key（shape/dtype/单位，绝对 or 增量）
- prompt 来源（人工输入 / task 字段映射）
- 控制频率（Hz）
- 图像尺寸与编码格式

### 1.3 关键原则：只记录推理时可获得的信息
- observation 里只能放“当前时刻真实可观测”信号。
- 不要把未来信息/标注泄漏进 observation（例如未来轨迹点、未来目标）。
- 如果“目标值/控制目标”在推理时也实时可得，且你计划部署时同样提供，才可以作为 observation。

---

## 2. 采集字段最小闭环（PI05 推荐）

### 2.1 最小必需字段
- 图像（至少 1 个主视角 + 最好 1 个腕视角）
- 本体状态（state / joint_position + gripper_position）
- action（每帧对应执行动作）
- prompt/task（语言指令）
- episode 边界（开始/结束）

### 2.2 与 openpi 对齐的典型映射
- DROID 风格：
- `observation/exterior_image_1_left`
- `observation/wrist_image_left`
- `observation/joint_position`
- `observation/gripper_position`
- `actions`
- `prompt`（或 `task -> prompt`）

- LIBERO 风格：
- `observation/image`
- `observation/wrist_image`
- `observation/state`
- `actions`
- `prompt`

### 2.3 关于“目标位置控制值/目标值”
- 不是 openpi 的通用必填 observation 字段。
- 通常应体现在 `actions` 定义里（绝对位置目标 vs 速度/增量控制）。
- 若你采用绝对动作，训练时可用 `DeltaActions` 转换；若本身是速度/增量动作，则通常不需要再做额外 delta 转换。

---

## 3. 采集执行流程（建议按批次）

### 3.1 批次策略
- 每批次固定一个场景配置（相机位置、光照、物体集合、初始区域范围）。
- 每批次 30~100 条 episode，采完即做自动质检。
- 质检失败不进入主数据集。

### 3.2 单条 episode 采集流程
1. 记录元信息：`operator_id`、`scene_id`、`task_id`、`schema_version`、`control_hz`。
2. 下发 prompt（或绑定 task）。
3. 启动同步采集（相机/状态/动作同一时钟基准）。
4. 执行任务并结束 episode。
5. 立即写入本地 raw 包（不可覆盖，追加写）。

### 3.3 在线保护（防坏数据）
- 丢帧保护：图像帧缺失超过阈值（例如 >1%）直接标记失败。
- 维度保护：状态/action shape 不匹配立即中止本条。
- 值域保护：动作越界（例如 > 1.2x 预期范围）直接报警并标记。
- 时间保护：时间戳非单调直接丢弃该条。

---

## 4. 采后自动质检（入库前）

每批次必须跑一次自动质检并产出报告（`qc_report.json`）。

### 4.1 结构质检
- 所有样本 key 完整率 100%
- dtype 一致（图像 uint8，状态/动作 float32）
- shape 一致（允许最后一维按 schema 定义）

### 4.2 时间质检
- 图像/状态/动作时间戳单调递增
- 相邻步长接近 `1/control_hz`
- episode 长度在合理范围（去掉极短/极长异常）

### 4.3 行为质检
- 动作分布不过度塌缩（例如 90% 接近 0 需报警）
- 任务成功率/完成度有基本下限（按任务定义）
- prompt 非空，且与任务标签一致

### 4.4 视觉质检
- 黑帧/花屏检测
- 过曝/欠曝比例
- 相机错位检测（简单可用均值亮度+边缘统计）

---

## 5. 转换到 LeRobot 并做 schema 验证

### 5.1 转换
- 使用转换脚本将 raw 数据写为 LeRobot 格式。
- 转换后不做“隐式修复”；所有修复应回到 raw 阶段或显式记录。

### 5.2 验证（必须）
- 用 `openpi` 的目标训练 config 做一次“dataset dry-run”：
- 能成功读取 batch
- 无 key/shape/prompt 报错
- action chunk 生成正常

---

## 6. 训练前稳定性闸门（Gate）

进入正式训练前必须全部通过：
- `compute_norm_stats` 跑通
- 100~500 step 小跑无 NaN/无 shape error
- loss 下降趋势正常
- 随机抽样可视化（obs + action）无明显错配

---

## 7. 版本与可复现要求（强制）

每次数据发布都要记录：
- `dataset_version`（不可变）
- `schema_version`
- 采集代码 commit
- 转换代码 commit
- 训练 config 名与 hash
- 质检报告（含失败率）

建议目录：
- `raw/<date_batch>/...`
- `processed/<dataset_version>/...`
- `reports/<dataset_version>/qc_report.json`

---

## 8. 常见失败与修复策略

### 8.1 训练时报 “Prompt is required”
- 说明样本里没有 `prompt`，且没有打开 `prompt_from_task`。
- 修复：在转换时写入 `task`，并在 config 中开启 `prompt_from_task=True`，或直接写 `prompt` 字段。

### 8.2 动作维度不一致
- 常见于 gripper 维漏掉、左右臂拼接顺序错。
- 修复：先统一 action 定义（含维度顺序），再更新 `Outputs` 截断逻辑。

### 8.3 绝对/增量动作混用
- 会导致训练不稳定、推理漂移。
- 修复：只保留一种 action 语义；若原始是绝对动作，统一在 transform 中做 delta 转换。

### 8.4 线上可用信号与训练信号不一致
- 训练用了某字段，部署拿不到，必然掉性能。
- 修复：移除该字段或在部署链路补齐，保持同构输入。

---

## 9. 一套可直接执行的最小流程（建议你先跑这个）

1. 固定 config（例如 `pi05_raytron_mujoco`）并冻结 schema_v1。  
2. 先采 30 条 pilot 数据。  
3. 跑自动质检，失败条目全部剔除并定位原因。  
4. 转 LeRobot，跑 dry-run（能取 batch、无报错）。  
5. 跑 `compute_norm_stats`。  
6. 训练 500 step 冒烟。  
7. 通过后再扩到 300~1000 条主数据。  
8. 每新增一批都重复“质检 -> dry-run -> 小跑”。

---

## 10. 你现在最该优先定下来的 4 件事

1. action 语义：绝对位置还是速度/增量。  
2. prompt 来源：手工 prompt 还是 task 自动映射。  
3. 观测最小集：上线时 100% 能提供的字段。  
4. 采集频率与时间同步基准：统一 Hz + 同步策略。

这 4 件事不稳定，后面训练很容易反复返工。
