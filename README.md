

本项目实现了一种基于延迟渲染（Deferred Rendering）的三维高斯溅射（3DGS）渲染管线，专门用于解决传统 3DGS 在处理玻璃、水体等透明/半透明材质时出现的高光与背景混叠问题。

项目包含完整的离线训练代码（PyTorch/CUDA）以及一个支持动态 LOD 调度的 WebGL 前端查看器。

 主要特性

- **双环境光场解耦**：将反射高光与透射背景分离到独立的参数空间进行优化。
- **动态背景注入**：修改了 CUDA 光栅化内核，支持动态透射背景采样。
- **G-Buffer 特征导出**：支持导出基础底色、法向图、反射高光等多通道物理特征。
- **高性能 Web 查看器**：基于 Three.js 实现前端延迟着色，内置视距感知的动态 LOD 调度引擎，支持千万级点云流畅交互。

1. 后端训练环境 (Python & CUDA)
要求环境：Ubuntu 20.04/22.04 或 Windows 11, CUDA >= 11.8.
# 创建虚拟环境
conda create -n 3dgs_trans python=3.9 -y
conda activate 3dgs_trans

# 安装 PyTorch
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 
# 编译自定义 CUDA 光栅化器
cd rendering_backend/cuda_rasterizer
pip install .

# 安装其他依赖
cd ..
pip install -r requirements.txt

2. 前端 Web 运行环境
前端无需复杂配置，只需一个简单的静态服务器解决跨域问题即可。

训练模型

python rendering_backend/train.py -s ./dataset/your_scene -m ./output/result

导出前端资产
训练完成后，使用导出工具提取中间图层并生成前端适配的 .ply 资产：

python data_cleaning/export_ply.py --input ./output/result/chkpnt30000.ply --output ./web_
