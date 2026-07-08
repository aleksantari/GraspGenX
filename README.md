# GraspGenX grasp server — team setup (aleksantari fork)

> Upstream's original README (full GraspGenX docs: demos, training, gripper onboarding)
> is preserved at [README_UPSTREAM.md](README_UPSTREAM.md).

This fork of [NVlabs/GraspGenX](https://github.com/NVlabs/GraspGenX) runs as a standalone
**ZMQ grasp-inference server** for our Unitree G1 + Dex3 pipeline. A client (e.g. the
`G1_classical_manip` repo) sends a segmented object point cloud over TCP and gets back ranked
6-DoF grasp poses. The client needs **no GPU, no torch, no graspgenx install** — just
`pyzmq msgpack msgpack-numpy numpy`.

**What this fork changes vs upstream** (protocol v2):

- `graspgenx/serving/zmq_server.py` — `infer` now runs the **GraspMoE planner**
  (`run_planner_on_object`) instead of diffusion-only. New request fields
  `planner` (`diffusion` | `graspmoe` | `topdown`), `obb_density`, `skip_obb_rule`;
  new response fields `branch_tags` (per-grasp `"obb"`/`"diff"`) and `planner`.
  `planner: topdown` returns only OBB top-down/side grasps — used for grab-from-above.
- `pyproject.toml` — torch cap loosened to `<2.8` so cu128 wheels (torch 2.7.x) install on
  580.x drivers (RTX 5090 workstation).
- `requirements-server.txt` — frozen snapshot of the known-good conda env (backstop).

Full wire-contract reference (request/response schemas, planner semantics, frame
conventions): `docs/graspgenx_server_contract.md` **in the client repo**
(`G1_classical_manip`).

---

## 1. Setup (conda)

Known-good: Python **3.11**, torch **2.7.1+cu128**, numpy **1.26.4** (pinned by upstream —
do not share this env with numpy-2 projects).

```bash
git clone https://github.com/aleksantari/GraspGenX.git ~/repos/GraspGenX
cd ~/repos/GraspGenX

conda create -n graspgenx python=3.11 -y
conda activate graspgenx

# torch first, matching your CUDA driver (cu128 works on 570/580.x drivers):
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128

# the package + the serving extra (pyzmq / msgpack / msgpack-numpy):
pip install -e ".[serve]"
```

If dependency resolution drifts, fall back to the exact frozen versions:

```bash
pip install -r requirements-server.txt --extra-index-url https://download.pytorch.org/whl/cu128
pip install -e . --no-deps
```

### First-run asset download

The **first `import graspgenx` auto-downloads two asset trees** into `ext/` (gitignored):

- model checkpoints → `ext/graspgenx_checkpoints/` (from HF `adithyamurali/GraspGenXModel`, multi-GB)
- gripper URDFs/configs → `ext/gripper_descriptions/` (github.com/beininghan/gripper_descriptions)

Trigger + verify it before first launch:

```bash
python -c "from graspgenx import get_checkpoints_version_dir, get_gripper_descriptions_root; \
           print(get_checkpoints_version_dir()); print(get_gripper_descriptions_root())"
```

A failed clone is swallowed silently at import — if you later see missing-gripper /
missing-checkpoint errors, re-run the line above and check your network. Existing copies
elsewhere can be pointed at with `GRASPGENX_CHECKPOINT_DIR` / `GRASPGENX_GRIPPER_CFG_DIR`.

---

## 2. Launch the server

```bash
conda activate graspgenx
cd ~/repos/GraspGenX
python client-server/graspgenx_server.py \
    --config ext/graspgenx_checkpoints/release \
    --assets_dir assets \
    --default_gripper unitree_g1 \
    --port 5556
```

On the lab workstation we use the lane-based shell wrapper (one-liner):

```bash
bash -ic 'use_conda graspgenx && cd ~/repos/GraspGenX && python client-server/graspgenx_server.py --config ext/graspgenx_checkpoints/release --assets_dir assets --default_gripper unitree_g1 --port 5556'
```

(`use_conda` is our `~/.bashrc` wrapper that isolates conda from ROS 2; plain
`conda activate graspgenx` is equivalent if you don't have it.)

Flag notes:

- `--config` = checkpoint **root** containing `gen/` + `dis/` subdirs.
- `--default_gripper unitree_g1` pre-loads the **Dex3-1 hand** at startup (that's the model
  init cost — first launch takes a while) and is used when a client omits `gripper_name`.
- `--assets_dir assets` is the fallback root for procedural grippers; named grippers like
  `unitree_g1` resolve through the auto-cloned `ext/gripper_descriptions` package first, so
  this works as-is.
- Binds `tcp://0.0.0.0:5556` — reachable from other machines on the network; use
  `--host`/`--port` to change.

## 3. Smoke-test it

From any machine with `pyzmq msgpack msgpack-numpy numpy` (no GPU needed):

```python
import msgpack, msgpack_numpy, numpy as np, zmq
msgpack_numpy.patch()
s = zmq.Context().socket(zmq.REQ); s.connect("tcp://<server-ip>:5556")

s.send(msgpack.packb({"action": "health"}));   print(msgpack.unpackb(s.recv()))
s.send(msgpack.packb({"action": "metadata"})); print(msgpack.unpackb(s.recv()))  # protocol_version: 2

cloud = np.random.rand(2000, 3).astype(np.float32) * 0.08   # fake 8 cm object
s.send(msgpack.packb({"action": "infer", "point_cloud": cloud, "planner": "topdown"},
                     use_bin_type=True))
r = msgpack.unpackb(s.recv())
print(r["grasps"].shape, r["confidences"].shape, r["branch_tags"][:5], r["timing"])
```

Or use the shipped CLI client: `python client-server/graspgenx_client.py --mesh_file
assets/sample_data/object_mesh/banana.obj --gripper_name unitree_g1 --host <server-ip> --port 5556`.
(Note: the shipped CLI still speaks the v1 subset — fine for a smoke test; the full v2 client
lives in the `G1_classical_manip` repo.)

## 4. Quick contract summary

- Send an `(N,3) float32` **single-object** cloud in **your frame, meters** → grasps return
  `(K,4,4)` **in the same frame**, sorted by confidence (discriminator score ∈ [0,1]).
- `planner`: `topdown` = OBB-only grab-from-above (no fallback if none reachable);
  `graspmoe` = diffusion ∪ OBB (default); `diffusion` = original behavior.
- Grasp frame convention: **+Z = approach into the object, +X = closing axis**.
- One model load per gripper (lazy, cached) — first `infer` per gripper is slow, rest are hot.

## Syncing with upstream

`origin` = this fork, `upstream` = NVlabs. To pull upstream changes:
`git fetch upstream && git merge upstream/main` (our diff is small and isolated to
`graspgenx/serving/zmq_server.py` + `pyproject.toml`).
