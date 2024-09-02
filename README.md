# FusionSense
Integrates the vision, touch, and common-sense information of foundational models, customized to the agent's perceptual needs.

## Usage

1. **Select frames**:  

    Run `delete.py` to select frames you want, or manually select, and you will get a folder of selected frames and transforms.json.  

    **Remember to set `transforms.json` in right format.**

2. **Generate Mask_imgs by [Grounded_SAM_2](https://github.com/IDEA-Research/Grounded-SAM-2)**:   

    set your scene path and prompt text with the end of '.'   
    `eg. 'transparent white statue.'`   

    ```python   
    python /home/ks8018/Grounded-SAM-2/grounded_sam2_hf_model_imgs_MaskExtract.py   
    ```   
    run the script to extract masks.   

    If the `num_no_detection` is not 0, you need to select the frame again. Then you will see mask_imgs in `path/masks`, and you can check `path/annotated` frames to see the results more directly.   
    
3. **Generate VisualHull by masks and transforms.json**:  

    run `VisualHull.py` to generate visual hull.  
    ```bash  
    python VisualHull.py --path your-path  
    ```
    
    You will get a point cloud file `foreground_pcd.ply`, and a screenshot `voxels.png` of checking whether the generated VisualHull is correct.    
    <img src="assets/voxels.png" width="300">

4. **RealSense depth & [Metric3Dv2](https://github.com/YvanYin/Metric3D) depth**:




```bash
ns-train dn-splatter --pipeline.model.use-depth-loss True       --pipeline.model.normal-lambda 0.4      --pipeline.model.sensor-depth-lambda 0.2    --pipeline.model.use-depth-smooth-loss True    --pipeline.model.use-normal-loss True  --pipeline.model.normal-supervision mono  --pipeline.model.random_init False normal-nerfstudio  --data tr-rabbit  --load-pcd-normals True --load-3D-points True  --normal-format opencv
```

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

