
import torch
from scene import Scene
import os, time
import numpy as np
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render, render_env_map
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.image_utils import psnr
from utils.loss_utils import ssim
from lpipsPyTorch import get_lpips_model
import imageio
import numpy as np
import shutil # 用于自动化文件拷贝

def export_transmission_cubemap_texture(cubemap_encoder, save_prefix, apply_sigmoid=True):
    """
    从 CubemapEncoder 中导出透射环境贴图的六个面纹理为 PNG 图片
    文件名使用方向命名：right, left, top, bottom, front, back
    """
    texture = cubemap_encoder.params['Cubemap_texture']  # shape: (6, C, H, W)
    num_faces, C, H, W = texture.shape
    assert C == 3, "Expected RGB channels"
    
    face_names = ["right", "left", "top", "bottom", "front", "back"]
    assert num_faces == len(face_names), "Unexpected number of faces"
    
    for i, name in enumerate(face_names):
        face = texture[i]                     # (3, H, W)
        face_img = face.permute(1, 2, 0).detach().cpu()  # (H, W, 3)
        
        if apply_sigmoid:
            face_img = torch.sigmoid(face_img)
        
        face_img = face_img.clamp(0, 1) * 255
        face_img = face_img.to(torch.uint8).numpy()
        
        save_path = f"{save_prefix}_{name}.png"
        imageio.imwrite(save_path, face_img)
        print(f"Saved transmission cubemap {name} -> {save_path}")

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, save_ims):
    if save_ims:
        render_path = os.path.join(model_path, name, "image_{}".format(iteration), "renders")
        # [修改] 建立独立文件夹，用来分别存放老师要求的图
        color_path = os.path.join(render_path, 'rgb')           # 最终图
        normal_path = os.path.join(render_path, 'normal')       # 法向图
        refl_path = os.path.join(render_path, 'reflection')     # 反射图
        trans_path = os.path.join(render_path, 'transmission')  # 透射图
        base_path = os.path.join(render_path, 'base_color')     # 底色图
        
        makedirs(color_path, exist_ok=True)
        makedirs(normal_path, exist_ok=True)
        makedirs(refl_path, exist_ok=True)
        makedirs(trans_path, exist_ok=True)
        makedirs(base_path, exist_ok=True)

    #LPIPS = get_lpips_model(net_type='vgg').cuda()
    ssims = []
    psnrs = []
    lpipss = []
    render_times = []

    if save_ims: # save env light
        ltres = render_env_map(gaussians)
        torchvision.utils.save_image(ltres['env_cood1'], os.path.join(model_path, 'light1.png'))
        torchvision.utils.save_image(ltres['env_cood2'], os.path.join(model_path, 'light2.png'))
        # 👇👇👇 透射 360 度全景图
        if 'env_trans_cood1' in ltres:
            torchvision.utils.save_image(ltres['env_trans_cood1'], os.path.join(model_path, 'light1_trans.png'))
            torchvision.utils.save_image(ltres['env_trans_cood2'], os.path.join(model_path, 'light2_trans.png'))
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        view.refl_mask = None # when evaluating, refl mask is banned
        t1 = time.time()
        rendering = render(view, gaussians, pipeline, background)
        render_time = time.time() - t1
        
        render_color = rendering["render"][None]
        gt = view.original_image[None, 0:3, :, :]

        ssims.append(ssim(render_color, gt).item())
        psnrs.append(psnr(render_color, gt).item())
        #lpipss.append(LPIPS(render_color, gt).item())
        lpip_v = 0.0
        render_times.append(render_time)

        if save_ims:
            # 1. 最终图
            torchvision.utils.save_image(render_color, os.path.join(color_path, '{0:05d}.png'.format(idx)))
            
            # 2. 法向图 (转换到 0~1 范围并保存)
            normal_map = rendering['normal_map'] * 0.5 + 0.5
            torchvision.utils.save_image(normal_map, os.path.join(normal_path, '{0:05d}.png'.format(idx)))
            
            # 3. 反射图
            refl_map = rendering['refl_color_map']
            torchvision.utils.save_image(refl_map, os.path.join(refl_path, '{0:05d}.png'.format(idx)))
            
            # 4. 透射图 (这就是你负责的核心部分！)
            trans_map = rendering['trans_color_map']
            torchvision.utils.save_image(trans_map, os.path.join(trans_path, '{0:05d}.png'.format(idx)))
            
            # 5. 基础底色图
            base_map = rendering['base_color_map']
            torchvision.utils.save_image(base_map, os.path.join(base_path, '{0:05d}.png'.format(idx)))
    
    ssim_v = np.array(ssims).mean()
    psnr_v = np.array(psnrs).mean()
    lpip_v = np.array(lpipss).mean()
    fps = 1.0/np.array(render_times).mean()
    print('psnr:{},ssim:{},lpips:{},fps:{}'.format(psnr_v, ssim_v, lpip_v, fps))
    dump_path = os.path.join(model_path, 'metric.txt')
    with open(dump_path, 'w') as f:
        f.write('psnr:{},ssim:{},lpips:{},fps:{}'.format(psnr_v, ssim_v, lpip_v, fps))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, save_ims : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
         # ========== 只导出透射 cubemap ==========
        if save_ims:
            if hasattr(gaussians, 'get_envmap_trans') and gaussians.get_envmap_trans is not None:
                export_transmission_cubemap_texture(
                    gaussians.get_envmap_trans,
                    os.path.join(dataset.model_path, "transmission_cubemap"),
                    apply_sigmoid=True
                )
        # =======================================
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, save_ims)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--save_images", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.save_images)

    import shutil
    import os
    import re
    import numpy as np
    from plyfile import PlyData, PlyElement

    def sync_to_web_viewer(model_path, iter_num, web_dir="web_viewer"):
        # 强制转换为绝对路径，防止相对路径坑人
        abs_model_path = os.path.abspath(model_path)
        abs_web_dir = os.path.abspath(web_dir)
        
        model_name = os.path.basename(abs_model_path)
        target_dir = os.path.join(abs_web_dir, "models", model_name)
        os.makedirs(target_dir, exist_ok=True)
        
        print(f"\n🚀 [系统流水线] 正在部署资产至: {target_dir}")

        # 1. 同步并清洗点云
        pc_dir = os.path.join(abs_model_path, "point_cloud")
        ply_src = None
        if iter_num == -1:
            if os.path.exists(pc_dir):
                iters = [int(d.split('_')[-1]) for d in os.listdir(pc_dir) if d.startswith("iteration_")]
                if iters:
                    ply_src = os.path.join(pc_dir, f"iteration_{max(iters)}", "point_cloud.ply")
        else:
            ply_src = os.path.join(pc_dir, f"iteration_{iter_num}", "point_cloud.ply")

        if ply_src and os.path.exists(ply_src):
            clean_ply_dest = os.path.join(target_dir, "point_cloud_clean.ply")
            print(f"  -> 🧹 正在清洗并同步点云: {os.path.basename(ply_src)}")
            # ... (此处省略之前的清洗 logic, 请确保保留在你的代码里) ...
            plydata = PlyData.read(ply_src)
            vertex_data = plydata.elements[0].data
            standard_props = [p.name for p in plydata.elements[0].properties if p.name != 'refl']
            new_dtype = [(name, vertex_data.dtype[name]) for name in standard_props]
            new_data = np.empty(len(vertex_data), dtype=new_dtype)
            for name in standard_props: new_data[name] = vertex_data[name]
            PlyData([PlyElement.describe(new_data, 'vertex')]).write(clean_ply_dest)
        else:
            print(f"  ⚠️ 警告: 未找到点云文件，请检查路径: {ply_src}")

        # 2. 【核心修复】同步 6 张环境贴图
        faces = ["right", "left", "top", "bottom", "front", "back"]
        found_images = 0
        for face in faces:
            # 尝试在模型根目录找图
            img_src = os.path.join(abs_model_path, f"transmission_cubemap_{face}.png")
            if os.path.exists(img_src):
                shutil.copy(img_src, os.path.join(target_dir, f"{face}.png"))
                print(f"  -> 🖼️ 已成功同步贴图: {face}.png")
                found_images += 1
            else:
                print(f"  ❌ 找不到贴图，预想路径为: {img_src}")
        
        if found_images == 6:
            print(f"✅ [系统流水线] 6 张环境贴图全部同步成功！")
        else:
            print(f"⚠️ [系统流水线] 贴图同步不完整 (成功 {found_images}/6)，请检查 output 文件夹！")

    # 执行同步
    sync_to_web_viewer(args.model_path, args.iteration)