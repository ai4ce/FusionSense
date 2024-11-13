import os
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch


def open_image(image_path):
    image = Image.open(image_path)
    return image

def read_image(image_path):
    image = cv2.imread(image_path)[:, :, ::-1]
    return image

def compute_scale_and_offset(sparse_depth, dense_depth, weights=None):
    """
    Compute the scale factor and offset between sparse and dense depth maps.

    :param sparse_depth: Sparse depth map (2D numpy array).
    :param dense_depth: Dense depth map (2D numpy array, same size).
    :return: scale_factor, offset
    """
    # Mask to consider only the non-zero elements of the sparse depth map
    mask = sparse_depth > 0
    if weights is None:
        # Flattening the arrays and applying the mask
        sparse_depth_flat = sparse_depth[mask].flatten()
        dense_depth_flat = dense_depth[mask].flatten()

        # Performing linear regression
        A = np.vstack([dense_depth_flat, np.ones_like(dense_depth_flat)]).T
        scale_factor, offset = np.linalg.lstsq(A, sparse_depth_flat, rcond=None)[0]

        return scale_factor, offset
    else:
        sparse_depth_flat = sparse_depth[mask].flatten()
        dense_depth_flat = dense_depth[mask].flatten()
        weights_flat = weights[mask].flatten()

        # Incorporating weights into the linear regression
        W = np.diag(weights_flat)
        A = np.vstack([dense_depth_flat, np.ones_like(dense_depth_flat)]).T
        AW = np.dot(W, A)
        BW = np.dot(W, sparse_depth_flat)

        # Performing weighted least squares regression
        scale_factor, offset = np.linalg.lstsq(AW, BW, rcond=None)[0]
        
        return scale_factor, offset


class VisualPipeline:
    def __init__(self, root_img_dir, output_depth_path='output', output_normal_path='output_normal', scale_factor=1000, new_intrinsics = (641.299, 641.299, 636.707, 362.299), new_size=(1280, 720)):
        """Initializes the visual pipeline

        Args:
            root_img_dir (_type_): _description_
        """
        # self.dpt_model = DPT()
        # self.zoe_model = get_zoe_model()
        # self.depth_anything_model = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-base-hf")
        # # self.depth_anything_model = None
        # self.metric3d_model = get_metric3d_model()
        
        self.root_img_dir = root_img_dir
        
        self.img_paths = sorted(os.listdir(self.root_img_dir))
        
        self.images = []
        self.real_depth_images = []
        self.zoe_depth_images = []
        
        self.get_all_images()
        
        self.output_depth_path = output_depth_path
        self.output_normal_path = output_normal_path
        self.scale_factor = scale_factor

        self.intrinsics = new_intrinsics
        
        if not os.path.exists(self.output_depth_path):
            os.mkdir(self.output_depth_path)
        if not os.path.exists(self.output_normal_path):
            os.mkdir(self.output_normal_path)
            
            
    def get_all_images(self):
        for _, img_path in enumerate(self.img_paths):
            if img_path.endswith('.png'):
                full_path = os.path.join(self.root_img_dir, img_path)
                image = open_image(full_path)
                self.images.append(image)
                print('Loaded:', full_path)

            # full_depth_path = full_path.replace('imgs', 'realsense_depths').replace('c_', 'd_')
            # depth_image = open_image(full_depth_path)
            # self.real_depth_images.append(depth_image)

            # full_zoe_depth_path = full_path.replace('imgs', 'vision')
            # zoe_depth_image = open_image(full_zoe_depth_path)
            # self.zoe_depth_images.append(zoe_depth_image)
            
    def get_images(self):
        return self.image
    
    def predict(self, visualize=False):
        for i in range(len(self.images)):
            img_np = np.array(self.images[i])
            
            if len(img_np.shape) > 2 and img_np.shape[2] == 4:
                # convert the image from RGBA2RGB
                img_np = cv2.cvtColor(img_np, cv2.COLOR_BGRA2BGR)
            self.images[i] = Image.fromarray(img_np)

            #### adjust input size to fit pretrained model
            # keep ratio resize
            intrinsic = self.intrinsics
            rgb_origin = img_np
            input_size = (720, 1280) # for vit model
            # rgb_origin_np = np.array(rgb_origin)
            h, w = rgb_origin.shape[:2]
            scale = min(input_size[0] / h, input_size[1] / w)
            rgb = cv2.resize(rgb_origin, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
            # remember to scale intrinsic, hold depth
            intrinsic = [intrinsic[0] * scale, intrinsic[1] * scale, intrinsic[2] * scale, intrinsic[3] * scale]
            # padding to input_size
            padding = [123.675, 116.28, 103.53]
            h, w = rgb.shape[:2]
            pad_h = input_size[0] - h
            pad_w = input_size[1] - w
            pad_h_half = pad_h // 2
            pad_w_half = pad_w // 2
            rgb = cv2.copyMakeBorder(rgb, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=padding)
            pad_info = [pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]

            #### normalize
            mean = torch.tensor([123.675, 116.28, 103.53]).float()[:, None, None]
            std = torch.tensor([58.395, 57.12, 57.375]).float()[:, None, None]
            rgb = torch.from_numpy(rgb.transpose((2, 0, 1))).float()
            rgb = torch.div((rgb - mean), std)
            rgb = rgb[None, :, :, :].cuda()

            ###################### canonical camera space ######################
            # inference
            model = torch.hub.load('yvanyin/metric3d', 'metric3d_vit_giant2', pretrain=True)
            model.cuda().eval()
            with torch.no_grad():
                pred_depth, confidence, output_dict = model.inference({'input': rgb})

            # un pad
            pred_depth = pred_depth.squeeze()
            pred_depth = pred_depth[pad_info[0] : pred_depth.shape[0] - pad_info[1], pad_info[2] : pred_depth.shape[1] - pad_info[3]]
  
            # upsample to original size
            pred_depth = torch.nn.functional.interpolate(pred_depth[None, None, :, :], rgb_origin.shape[:2], mode='bilinear').squeeze()
            ###################### canonical camera space ######################

            #### de-canonical transform
            canonical_to_real_scale = intrinsic[0] / 1000.0 # 1000.0 is the focal length of canonical camera
            pred_depth = pred_depth * canonical_to_real_scale # now the depth is metric
            pred_depth = torch.clamp(pred_depth, 0, 300)


            # #### you can now do anything with the metric depth 
            # # such as evaluate predicted depth
            # # gt_depth = cv2.imread(depth_file, -1)
            # gt_depth_scale = 1000.0
            # gt_depth_np = np.array(self.real_depth_images[i])
            # gt_depth = gt_depth_np / gt_depth_scale
            # gt_depth = torch.from_numpy(gt_depth).float().cuda()
            # assert gt_depth.shape == pred_depth.shape
    
            # mask = (gt_depth > 1e-8)
            # abs_rel_err = (torch.abs(pred_depth[mask] - gt_depth[mask]) / gt_depth[mask]).mean()
            # print('abs_rel_err:', abs_rel_err.item())

            # zoe_depth_np = np.array(self.zoe_depth_images[i])
            # zoe_depth = zoe_depth_np / gt_depth_scale
            # zoe_depth = torch.from_numpy(zoe_depth).float().cuda()
            # assert zoe_depth.shape == pred_depth.shape
    
            # abs_zoe_rel_err = (torch.abs(zoe_depth[mask] - gt_depth[mask]) / gt_depth[mask]).mean()
            # print('abs_zoe_rel_err:', abs_zoe_rel_err.item())

            # abs_pred_zoe_err = (torch.abs(pred_depth[mask] - zoe_depth[mask]) / zoe_depth[mask]).mean()
            # print('abs_pred_zoe_err:', abs_pred_zoe_err.item())

            # predicted_depth = self.predict_depth_from_image(self.images[i], model_type='metric3d')
            
            final_depth_int = (self.scale_factor * pred_depth).cpu().numpy().astype(np.uint16)
            depth_paths = os.listdir(self.root_img_dir)
            
            depth_paths_sorted = sorted(depth_paths)
            
            depth_valid_selected = depth_paths_sorted[i]
            depth_valid_selected = depth_valid_selected.replace('c_', 'd_')
            
            if visualize:
                plt.imshow(final_depth_int, cmap='viridis')
                plt.show()
            
            depth_img_path = f'{self.output_depth_path}/{depth_valid_selected}'
            print(depth_img_path)
            cv2.imwrite(depth_img_path, final_depth_int)
            print(f'Saved depth image {depth_img_path}')

            #### normal are also available
            pred_normal = output_dict['prediction_normal'][:, :3, :, :]
            normal_confidence = output_dict['prediction_normal'][:, 3, :, :] # see https://arxiv.org/abs/2109.09881 for details
            # un pad and resize to some size if needed
            pred_normal = pred_normal.squeeze()
            pred_normal = pred_normal[:, pad_info[0] : pred_normal.shape[1] - pad_info[1], pad_info[2] : pred_normal.shape[2] - pad_info[3]]
            
            # you can now do anything with the normal
            # such as visualize pred_normal
            pred_normal_vis = pred_normal.cpu().numpy().transpose((1, 2, 0))
            pred_normal_vis = (pred_normal_vis + 1) / 2

            # final_depth_int = (self.scale_factor * pred_depth).astype(np.uint16)
            normal_paths = os.listdir(self.root_img_dir)
            
            normal_paths_sorted = sorted(normal_paths)
            
            normal_valid_selected = normal_paths_sorted[i]
            normal_valid_selected.replace('c_', 'n_')
            
            if visualize:
                plt.imshow(final_depth_int, cmap='viridis')
                plt.show()

            normal_img_path = f'{self.output_normal_path}/{normal_valid_selected}'
            cv2.imwrite(normal_img_path, (pred_normal_vis * 255).astype(np.uint8))
            print(f'Saved normal image {normal_img_path}')
        
                
    def visualize(self, colmap_depth, predicted_depth, refined_depth, labels=['Colmap Depth', 'Predicted Depth', 'Refined Depth']):
        # Apply a colormap for visualization
        # You can change 'plasma' to any other colormap (like 'viridis', 'magma', etc.)
        
        plt.figure(figsize=(12, 6))

        # Display the first depth image
        plt.subplot(1, 3, 1)  # (1 row, 2 columns, first subplot)
        plt.imshow(colmap_depth, cmap='viridis')
        plt.title(labels[0])
        plt.axis('off')  # Turn off axis numbers

        # Display the second depth image
        plt.subplot(1, 3, 2)  # (1 row, 2 columns, second subplot)
        plt.imshow(predicted_depth, cmap='viridis')
        plt.title(labels[1])
        plt.axis('off')  # Turn off axis numbers
        
        plt.subplot(1, 3, 3)  # (1 row, 2 columns, second subplot)
        plt.imshow(refined_depth, cmap='viridis')
        plt.title(labels[2])
        plt.axis('off')  # Turn off axis numbers

        # Show the plot
        plt.show()
        
        
    # def predict_depth_from_image(self, image, model_type='metric3d'):
    #     if model_type == 'metric3d':
    #         depth = self.metric3d_model(image)
    #     elif model_type == 'zoe':
    #         depth = self.zoe_model.infer_pil(image)
    #     elif model_type == 'depth_anything':
    #         depth = self.depth_anything_model(image)
    #         depth_tensor = depth['predicted_depth'].numpy().squeeze(axis=0) / 255
            
    #         depth = depth_tensor / np.max(depth_tensor)
            
    #         depth = 1 - depth_tensor
    #     else:
    #         depth = self.dpt_model(image)
            
    #     return depth
    
    # def predict_normal_from_image(self, image, model_type='metric3d'):
    #     if model_type == 'metric3d':
    #         depth = self.metric3d_model(image)
            
    #     return depth

def metric3d_depth_generation(root_dir, intrinsics, frame_size, output_depth_path='metric3d_depth_result', output_normal_path='metric3d_normal_result', img_dir='images', viz=False):
    
    imgs_path = os.path.join(root_dir, img_dir)
    # real_depth_path = os.path.join(args.root_dir, args.realsense_depth_image_dir)
    # zoe_depth_path = os.path.join(args.root_dir, args.zoe_depth_image_dir)
    
    output_depth_path = os.path.join(root_dir, output_depth_path)
    output_normal_path = os.path.join(root_dir, output_normal_path)
    
    visual_pipeline = VisualPipeline(root_img_dir=imgs_path, output_depth_path=output_depth_path, output_normal_path=output_normal_path, new_intrinsics=intrinsics, new_size=frame_size)
    
    visual_pipeline.predict(visualize=viz)