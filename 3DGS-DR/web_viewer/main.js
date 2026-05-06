import * as THREE from 'three';
import * as GaussianSplats3D from '@mkkellogg/gaussian-splats-3d';

// 路径
const modelPath = 'models/art3_test/'; 
console.log(`🚀 强制加载固定路径: ${modelPath}`);

const scene = new THREE.Scene();

const loader = new THREE.CubeTextureLoader();
loader.setPath(modelPath);
const texture = loader.load([
    'right.png', 'left.png', 
    'top.png', 'bottom.png', 
    'front.png', 'back.png'
]);
scene.background = texture;
scene.environment = texture;

const viewer = new GaussianSplats3D.Viewer({
    'threeScene': scene, 
    'cameraUp': [0, -1, 0], 
    'initialCameraPosition': [0, 0, 8], 
    'initialCameraLookAt': [0, 0, 0],
    'useGypsyWASM': true,
    'sharedMemoryForWorkers': false
});

// 2. 强制读取该路径下的清洗版点云
viewer.addSplatScene(modelPath + 'point_cloud_clean.ply', {
    'showLoadingUI': true,
    'splatAlphaCrop': 0.01 
})
.then(() => {
    viewer.start();
    console.log("✅ 渲染流水线跑通！");
})
.catch((err) => {
    console.error("❌ 加载失败:", err);
});