import os

def rendering_evaluation(input_dir, full_exp_dir, exp_name, past_n_trials=0):
    files = sorted(os.listdir(input_dir))
    os.makedirs(full_exp_dir, exist_ok=True)
    
    config_yml_path = os.path.join(input_dir, 'config.yml')
    exp_json = exp_name + '.json'
    full_exp_json = os.path.join(full_exp_dir, exp_json)
    full_cmd = f'ns-eval --load-config={config_yml_path} --output-path={full_exp_json}'
    print(full_cmd)
    os.system(full_cmd)
    
    full_cmd = f'ns-render dataset --load-config={config_yml_path} --output-path={full_exp_dir} --rendered_output_names rgb depth normal gt-sensor_depth'
    #    rendered_output_names = ['rgb', 'depth', 'normal', 'accumulation', 'background', 'normal_touch',  
    #   'raw-rgb', 'raw-depth', 'raw-normal', 'raw-accumulation', 'raw-background', 'raw-normal_touch', 'gt-image_idx',     
    #   'gt-sensor_depth', 'gt-normal', 'gt-rgb', 'raw-gt-image_idx', 'raw-gt-sensor_depth', 'raw-gt-normal', 'raw-gt-rgb']  
    print(full_cmd)
    os.system(full_cmd)