export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512'


# python generate_pcd.py --path your-path

# using dsine to generate normals using dsine model
# datasets/[DATA_SET]/image, rgb data
# datasets/
python dn_splatter/scripts/normals_from_pretrain.py --data-dir datasets/tr-rabbit3/  --model-type dsine


ns-train dn-splatter --steps-per-save 10000 --pipeline.model.use-depth-loss True --pipeline.model.normal-lambda 0.4  --pipeline.model.sensor-depth-lambda 0.2 --pipeline.model.use-depth-smooth-loss True  --pipeline.model.use-normal-loss True  --pipeline.model.normal-supervision mono  --pipeline.model.random_init False normal-nerfstudio --data datasets/tr-rabbit3 --load-pcd-normals False --load-3D-points False --load-touches False
