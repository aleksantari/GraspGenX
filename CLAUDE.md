# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

GraspGenX is a **cross-embodiment** 6-DOF grasp generator (NVlabs, CVPR 2026). Unlike GraspGen (one model per gripper), GraspGenX trains a *single* diffusion model conditioned on a gripper's **swept-volume** representation, so it generalizes zero-shot to novel grippers. Input is a partial point cloud or an object mesh + a `--gripper_name`; output is scored 6-DOF grasp poses (`(K,4,4)` homogeneous transforms + confidences).

## Setup & dependencies (read before running anything)

- **Install (inference):** `uv sync`. `pip install -e .` also works inside a conda/venv with Python 3.10/3.11. Use `uv run python ...` to run scripts (or activate the env). 3.12 needs torch>=2.2.
- **End-to-end extra:** `uv sync --extra end2end` pulls cuRobo + Newton + MuJoCo + USD. `nvidia-curobo` and `newton` resolve to **sibling editable checkouts** (`../curobo`, `../newton`) via `[tool.uv.sources]` — those repos must exist next to this one.
- **Two external asset trees auto-download on first `import graspgenx`** (see [graspgenx/_setup_dependencies.py](graspgenx/_setup_dependencies.py)), cloned into `ext/` (gitignored):
  - **Checkpoints** → `ext/graspgenx_checkpoints/` from HF `adithyamurali/GraspGenXModel`. Override with `GRASPGENX_CHECKPOINT_DIR`. Resolved version dir contains `gen/` and `dis/` subdirs (each `config.yaml` + `epoch_*.pth`).
  - **Gripper descriptions** (URDFs, meshes, `config.json` swept-volume metadata) → `ext/gripper_descriptions/` from `github.com/beininghan/gripper_descriptions`. Override with `GRASPGENX_GRIPPER_CFG_DIR`.
- Verify assets: `uv run python -c "from graspgenx import get_gripper_descriptions_root; print(get_gripper_descriptions_root())"`
- List grippers: `uv run python scripts/list_grippers.py` (add `--paths` / `--json`).
- This user's machine uses a **lane-based conda shell**: if you run inside a conda env rather than uv, wrap commands as `bash -ic 'use_conda <env> && <cmd>'` (the wrapper functions aren't on PATH in non-interactive shells). With `uv run` this is unnecessary.

## Gripper name mapping (important for this project — Unitree Dex3 / Dex1)

The user's goal is grasps for the Unitree **Dex3-1** (3-finger dexterous hand) and **Dex1** (parallel gripper) end effectors on the G1 humanoid.

- **Dex3-1 → `--gripper_name unitree_g1`.** The `unitree_g1` asset's URDF is the 7-DOF, 3-finger Dex3-1 hand (joints `right_zero_joint`…`right_six_joint`). Note: the upstream `gripper_descriptions` README *mislabels* it `revolute_2f` — that tag is wrong; it is the Dex3 hand.
- **Dex1 → no config exists.** There is no `dex1`/`unitree_dex1` in `gripper_descriptions`. To use it you must **onboard it via the wizard** (URDF → swept-volume `config.json`), see below. A generic parallel gripper (`franka_panda`, `robotiq_2f_85`) is only a rough geometric stand-in.

## Common commands

```bash
# Grasps for segmented object point clouds (viser GUI at http://localhost:8080)
uv run python scripts/demo_object_pc.py \
    --sample_data_dir assets/sample_data/real_world \
    --gripper_name unitree_g1 --plot_top_mesh

# Grasps for an object mesh (.obj/.stl/.ply/USD). Headless: --no-visualization --output_file /tmp/grasps.yml
uv run python scripts/demo_object_mesh.py \
    --mesh_file assets/sample_data/object_mesh/banana.obj --mesh_scale 1.0 \
    --gripper_name unitree_g1 --grasp_threshold -1.0 --return_topk --topk_num_grasps 100 --plot_top_mesh

# Grasps for every segmented object in a full scene point cloud (collisions filtered)
uv run python scripts/demo_scene_pc.py --sample_data_dir assets/sample_data/real_world --gripper_name unitree_g1
```

Planner flags (all demos): default is **GraspMoE** (diffusion ∪ OBB-swept, all discriminator-scored). `--planner diffusion` for diffusion-only. `--moe_obb_density sparse|dense` tunes the OBB sweep. `--no-filter_collisions` (scene demo) disables collision filtering.

```bash
# Onboard a NEW gripper (e.g. Dex1): interactive 6-panel wizard at http://localhost:8081
uv run python scripts/gripper_config_wizard.py --urdf /path/to/gripper.urdf --name <name> --port 8081
# Sanity-check a saved gripper config (animates open<->close with sweep volumes)
uv run python scripts/vis_gripper_desc.py --gripper <name> --port 8081

# Tests (end2end is slow + needs GPU/cuRobo/Newton — skip it normally)
uv run --no-sync pytest -m "not end2end"
uv run --no-sync pytest tests/test_graspmoe.py        # single file
uv run python scripts/train_xgrasp.py                 # training (hydra; tested only inside Docker)
bash docker/build.sh                                  # Docker image (recommended for training)
```

Lint/format (dev extra): `black .`, `isort .`, `flake8` (line-length 88, py310).

## Architecture (big picture)

**Inference flow:** point cloud / mesh → gripper swept-volume conditioning → diffusion sampling → discriminator scoring → ranked grasps.

- **[graspgenx/grasp_server.py](graspgenx/grasp_server.py) — `GraspGenXSampler`** is the main inference API. `__init__` calls `resolve_gripper_info()` then `GraspGen.from_config()` and loads the gen+dis checkpoints. Key methods: `sample()` (one PC → grasps+conf+contacts in world frame), `run_inference()` (retry/threshold/top-k wrapper), `run_inference_batch()` (N PCs, one forward pass).
- **Gripper conditioning — [graspgenx/x_grippers.py](graspgenx/x_grippers.py):** `resolve_gripper_info(name)` resolves `$GRASPGENX_GRIPPER_CFG_DIR` → `assets/x_grippers/` → `assets/proc_grippers/`, returning an `XGripperInfo` (swept-volume boxes from `config.json`, fingertip depth, `gripper_type` enum 0=parallel_2f/1=revolute_2f/2=revolute_3f, point clouds, TSDF, VAE repr). `load_gripper_input()` packs these into the model's tensor dict. **This swept volume is the only thing conditioning the model on the gripper** — that's the cross-embodiment mechanism.
- **Model — [graspgenx/models/grasp_gen.py](graspgenx/models/grasp_gen.py) `GraspGen`** composes `GraspGenGenerator` ([generator.py](graspgenx/models/generator.py), diffusion: object encoder PointNet++ or PTv3-vanilla + gripper MLP + DDPM noise-prediction head) and `GraspGenDiscriminator` ([discriminator.py](graspgenx/models/discriminator.py), scores each grasp ∈ [0,1]). `find_the_last_ckpt()` auto-picks the latest `epoch_*.pth`.
- **Planner — [graspgenx/samplers/planner.py](graspgenx/samplers/planner.py) `run_planner_on_object()`** dispatches on `planner=`: `"graspmoe"` → `run_graspmoe()` in [graspmoe.py](graspgenx/samplers/graspmoe.py) (diffusion branch **union** an OBB branch that computes an oriented bounding box and sweeps gripper poses across its faces, then scores *both* sets with the discriminator and takes a global top-k); `"diffusion"` → diffusion sampling only.
- **Serving — [graspgenx/serving/zmq_server.py](graspgenx/serving/zmq_server.py):** ZMQ REQ/REP, msgpack+msgpack-numpy wire format, lazy one-sampler-per-gripper caching. Actions: `health`, `metadata`, `infer`. CLI wrappers in [client-server/](client-server/) (`graspgenx_server.py` / `graspgenx_client.py`); needs the `serve` extra (`pyzmq msgpack msgpack-numpy`).
- **Training — [scripts/train_xgrasp.py](scripts/train_xgrasp.py)** + hydra config [scripts/config_xgrasp.yaml](scripts/config_xgrasp.yaml) (DDP multi-GPU; trains generator and discriminator). Dataset code under [graspgenx/dataset/](graspgenx/dataset/) (webdataset/h5 caches).
- **End-to-end pipeline — [end2end/](end2end/README.md):** GraspGenX grasps → cuRobo collision-free motion planning → Newton/MuJoCo physics replay → MP4 render + USD export.

## Gotchas

- Nothing works until the auto-download of checkpoints + gripper_descriptions into `ext/` succeeds (or the `GRASPGENX_*` env vars point at existing copies). A failed clone is swallowed at import — symptoms show up later as missing-gripper / missing-checkpoint errors.
- GraspGen checkpoints are **not** compatible with GraspGenX (different architecture).
- `scripts/inference_xgrasp.py` imports `scripts.curate_ord_eval_chunks`, which is not present in this checkout — that eval path won't run as-is; use the `demo_*.py` scripts for inference.
