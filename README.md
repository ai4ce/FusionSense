# FusionSense
Integrates the vision, touch, and common-sense information of foundational models, customized to the agent's perceptual needs.

## Dataset Format
```bash
tr-rabbit/
│
├── transforms.json
│
├── images/
│   ├── rgb_1.png
│   └── rgb_2.png
│
├── normals_from_pretrain/
│   ├── rgb_1.png
│   └── rgb_2.png
│
├── realsense_depth/
│   ├── depth_1.png
│   └── depth_2.png
│
├── object.ply
└── merged_pcd.ply
```

## Usage
```bash
ns-train dn-splatter --pipeline.model.use-depth-loss True       --pipeline.model.normal-lambda 0.4      --pipeline.model.sensor-depth-lambda 0.2    --pipeline.model.use-depth-smooth-loss True    --pipeline.model.use-normal-loss True  --pipeline.model.normal-supervision mono  --pipeline.model.random_init False normal-nerfstudio  --data tr-rabbit  --load-pcd-normals True --load-3D-points True  --normal-format opencv
```
