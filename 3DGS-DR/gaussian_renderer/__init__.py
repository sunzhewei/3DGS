#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math, time
import torch.nn.functional as F
import diff_gaussian_rasterization_c3
import diff_gaussian_rasterization_c7 
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from utils.general_utils import sample_camera_rays, get_env_rayd1, get_env_rayd2
import numpy as np

# rayd: x,3, from camera to world points
# normal: x,3
# all normalized
def reflection(rayd, normal):
    refl = rayd - 2*normal*torch.sum(rayd*normal, dim=-1, keepdim=True)
    return refl

def sample_cubemap_color(rays_d, env_map):
    H,W = rays_d.shape[:2]
    outcolor = torch.sigmoid(env_map(rays_d.reshape(-1,3)))
    outcolor = outcolor.reshape(H,W,3).permute(2,0,1)
    return outcolor

def get_refl_color(envmap: torch.Tensor, HWK, R, T, normal_map): #RT W2C
    rays_d = sample_camera_rays(HWK, R, T)
    rays_d = reflection(rays_d, normal_map)
    #rays_d = rays_d.clamp(-1, 1) # avoid numerical error when arccos
    return sample_cubemap_color(rays_d, envmap)

def render_env_map(pc: GaussianModel):
    # 1. 先初始化 res 字典，把反射贴图装进去
    res = {
        'env_cood1': sample_cubemap_color(get_env_rayd1(512,1024), pc.get_envmap),
        'env_cood2': sample_cubemap_color(get_env_rayd2(512,1024), pc.get_envmap)
    }
    
    # 2. 判断如果有透射贴图，再往 res 里追加透射的渲染结果
    if hasattr(pc, 'get_envmap_trans') and pc.get_envmap_trans is not None:
        res['env_trans_cood1'] = sample_cubemap_color(get_env_rayd1(512,1024), pc.get_envmap_trans)
        res['env_trans_cood2'] = sample_cubemap_color(get_env_rayd2(512,1024), pc.get_envmap_trans)
        
    return res

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, initial_stage = False, more_debug_infos = False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    imH = int(viewpoint_camera.image_height)
    imW = int(viewpoint_camera.image_width)

    def get_setting(Setting):
        raster_settings = Setting(
            image_height=imH,
            image_width=imW,
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=pipe.debug
        )
        return raster_settings
    
    # init rasterizer with various channels
    Setting_c3 = diff_gaussian_rasterization_c3.GaussianRasterizationSettings
    Setting_c7 = diff_gaussian_rasterization_c7.GaussianRasterizationSettings
    rasterizer_c3 = diff_gaussian_rasterization_c3.GaussianRasterizer(get_setting(Setting_c3))
    rasterizer_c7 = diff_gaussian_rasterization_c7.GaussianRasterizer(get_setting(Setting_c7))

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacities = pc.get_opacity
    scales = pc.get_scaling
    rotations = pc.get_rotation
    shs = pc.get_features
    
    #bg_map_const = bg_color[:,None,None].cuda().expand(3, imH, imW)
    #bg_map_zero = torch.zeros_like(bg_map_const)
    # [新增] 动态透射背景计算
    # 如果模型里有透射贴图，就生成一张当前视角的透射背景图；否则退回纯色背景
    if hasattr(pc, 'get_envmap_trans') and pc.get_envmap_trans is not None:
        # 1. 计算当前屏幕所有像素发出的视线射线方向
        rays_d = sample_camera_rays(viewpoint_camera.HWK, viewpoint_camera.R, viewpoint_camera.T)
        # 2. 用这些射线去查询透射环境贴图，生成背景画面
        bg_map_trans = sample_cubemap_color(rays_d, pc.get_envmap_trans)
        # 3. 把查询到的透射图作为底层光栅化器的强行背景！
        bg_map_const = bg_map_trans
    else:
        bg_map_const = bg_color[:,None,None].cuda().expand(3, imH, imW)

    if initial_stage:
        base_color, _radii = rasterizer_c3(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = None,
            opacities = opacities,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = None,
            bg_map = bg_map_const)

        return {
            "render": base_color,
            "viewspace_points": screenspace_points,
            "visibility_filter" : _radii > 0,
            "radii": _radii}

    normals = pc.get_min_axis(viewpoint_camera.camera_center) # x,3
    refl_ratio = pc.get_refl

    input_ts = torch.cat([torch.zeros_like(normals), normals, refl_ratio], dim=-1)
    bg_map = torch.cat([bg_map_const, torch.zeros(4,imH,imW, device='cuda')], dim=0)
    out_ts, _radii = rasterizer_c7(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = input_ts,
        opacities = opacities,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = None,
        bg_map = bg_map)
    
    base_color = out_ts[:3,...] # 3,H,W
    refl_strength = out_ts[6:7,...] #
    normal_map = out_ts[3:6,...] 

    normal_map = normal_map.permute(1,2,0)
    normal_map = normal_map / (torch.norm(normal_map, dim=-1, keepdim=True)+1e-6)
    
    # 1. 计算反射图 (Reflection)
    refl_color = get_refl_color(pc.get_envmap, viewpoint_camera.HWK, viewpoint_camera.R, viewpoint_camera.T, normal_map)
    
    # 2. 提取透射图 (Transmission)
    # 利用你前面已经算出来的透射背景图 bg_map_trans
    if hasattr(pc, 'get_envmap_trans') and pc.get_envmap_trans is not None:
        trans_color = bg_map_trans 
    else:
        trans_color = torch.zeros_like(refl_color)
    
    # 3. 最终画面合成
    final_image = (1-refl_strength) * base_color + refl_strength * refl_color

    # 4. 把老师要求的所有图分离打包！
    results = {
        "render": final_image,                 # 最终 RGB 画面
        "refl_strength_map": refl_strength,    # 反射强度掩码
        "normal_map": normal_map.permute(2,0,1), # 法向图 (Normal)
        "refl_color_map": refl_color,          # 反射图 (Reflection)
        "trans_color_map": trans_color,        # 透射图 (Transmission) - [新增]
        "base_color_map": base_color,          # 基础底色图 (Base)
        "viewspace_points": screenspace_points,
        "visibility_filter" : _radii > 0,
        "radii": _radii
    }
        
    return results
