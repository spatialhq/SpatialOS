# AI-assisted refinement (optional)

`spatial-os refine` adds a semantic/instance understanding layer on top of a
reconstructed point cloud, using a pretrained 2D semantic segmentation model
back-projected into 3D — plus optional classical mesh completion for
hole-filling.

Requires the `ai` extra (not installed by default — it pulls in `torch` +
`transformers`, which are large):

```bash
pip install -e ".[ai]"
```

```bash
spatial-os process ./session --output ./out --format ply
spatial-os refine ./session --cloud ./out/cloud.ply --output ./out/refined --max-frames 20 --mesh-completion
```

## How it works

1. **Semantic segmentation (AI)** — runs `nvidia/segformer-b0-finetuned-ade-512-512`
   (Hugging Face, ADE20K, 150 classes incl. wall/floor/ceiling/furniture) on
   up to `--max-frames` RGB keyframes.
2. **3D back-projection** — each labeled pixel is unprojected to a 3D point
   using the frame's known depth + ARKit pose (the same data the
   reconstruction stage uses); labels are then transferred onto the final
   reconstructed cloud via k-nearest-neighbor majority vote.
3. **Instance clustering (classical)** — non-structural classes (i.e. not
   wall/floor/ceiling) are split into individual object instances via
   per-class DBSCAN. There's no learned instance-segmentation model in the
   loop; this is deliberately simple and swappable.
4. **Mesh completion (classical, optional)** — `--mesh-completion` runs
   Poisson surface reconstruction with low-density trimming to fill small
   holes. This is *not* a learned completion model — see
   `src/spatial_os/refine/mesh_completion.py` for why, and swap it for one
   if a suitable pretrained checkpoint is identified later.

## Output (`--output` directory)

- `semantic_labels.npy`, `instance_ids.npy` — per-point arrays, aligned to
  the input `--cloud`'s point order.
- `labels.json` — id→class-name map, which class ids were treated as
  structural, and the instance count.
- `labeled_cloud.ply` — the input cloud recolored by semantic class, for
  quick visual QA.
- `mesh_completed.ply` — only if `--mesh-completion` was passed.

## Why 2D segmentation + back-projection instead of a native 3D model

Native 3D semantic/instance segmentation models (Mask3D, PTv3, ...) are
heavier to install, need task-specific pretrained checkpoints that aren't
reliably available, and often assume dense colored meshes rather than raw
fused point clouds. A well-supported 2D model applied per-keyframe, combined
with the depth+pose data the pipeline already has, gets most of the value
with far fewer moving parts. If a specific 3D model becomes a requirement,
swap the implementation behind the existing `Segmenter` protocol.
