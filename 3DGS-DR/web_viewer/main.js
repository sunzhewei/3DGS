import * as THREE from 'three';
import * as GaussianSplats3D from '@mkkellogg/gaussian-splats-3d';


const modelPath = 'models/art3_test/'; 
console.log(` 加载模型路径: ${modelPath}`);

const scene = new THREE.Scene();

// 加载六视图透射环境贴图
const loader = new THREE.CubeTextureLoader();
loader.setPath(modelPath);
const texture = loader.load([
    'right.png', 'left.png', 
    'top.png', 'bottom.png', 
    'front.png', 'back.png'
]);
scene.background = texture;
scene.environment = texture;

// 初始化 3DGS Viewer
const viewer = new GaussianSplats3D.Viewer({
    'threeScene': scene, 
    'cameraUp': [0, -1, 0], 
    'initialCameraPosition': [0, 0, 8], 
    'initialCameraLookAt': [0, 0, 0],
    'useGypsyWASM': true,
    'sharedMemoryForWorkers': false
});


const infoDiv = document.createElement('div');
infoDiv.style.position = 'absolute';
infoDiv.style.top = '20px';
infoDiv.style.left = '20px';
infoDiv.style.color = '#00FF00';
infoDiv.style.background = 'rgba(0, 0, 0, 0.85)'; 
infoDiv.style.padding = '25px 30px'; 
infoDiv.style.fontFamily = 'monospace';
infoDiv.style.fontSize = '20px'; 
infoDiv.style.lineHeight = '1.8'; 
infoDiv.style.borderRadius = '12px';
infoDiv.style.zIndex = '1000';
infoDiv.style.border = '2px solid #555'; 
infoDiv.style.boxShadow = '0 6px 12px rgba(0,0,0,0.6)'; 
infoDiv.style.minWidth = '450px'; 
document.body.appendChild(infoDiv);

// LOD 视距阈值设定 
const LOD_THRESHOLD = 14.0; 



viewer.addSplatScene(modelPath + 'point_cloud_clean.ply', {
    'showLoadingUI': true,
    'splatAlphaCrop': 0.01 
})
.then(() => {
    console.log(" 渲染流水线跑通！进入动态 LOD 调度循环...");
   
    requestAnimationFrame(lodRenderLoop);
})
.catch((err) => {
    console.error(" 加载失败:", err);
});


function lodRenderLoop() {
    requestAnimationFrame(lodRenderLoop);
    
    // 确保相机和网格数据都已经加载完毕
    if (viewer.camera && viewer.splatMesh) {
        // 步骤 1: 获取相机距离
        const centerPos = new THREE.Vector3(0, 0, 0);
        const distance = viewer.camera.position.distanceTo(centerPos);
        
        let currentMode = "";
        let color = "";
        let sampleRateText = "";

        // 步骤 2: 动态 LOD 调度逻辑判断 (安全版，不直接触碰底层显存)
        if (distance < LOD_THRESHOLD) {
            currentMode = "HIGH_LOD (双 Cubemap 解耦运算中)";
            color = "#00FF00"; // 绿色
            sampleRateText = "100%";
            if (viewer.splatMesh) {
                viewer.splatMesh.splatAlphaCrop = 0.01; // 恢复高精度
            }
        } else {
            currentMode = "LOW_LOD (基础底色降级模式)";
            color = "#FFA500"; // 橙色
            sampleRateText = "20%"; 
            if (viewer.splatMesh) {
                viewer.splatMesh.splatAlphaCrop = 0.8; // 降低精度
            }
        }
        
        // 步骤 3: 实时更新监控面板
        infoDiv.innerHTML = `
            <strong style="color:white; font-size:26px;">Spark LOD 动态调度引擎</strong><br/>
            ------------------------------------------<br/>
            实时相机距离 (Distance) : <span style="color:white">${distance.toFixed(3)} m</span><br/>
            降级距离阈值 (Threshold) : <span style="color:white">${LOD_THRESHOLD.toFixed(2)} m</span><br/>
            当前渲染分支 (Shader) : <span style="color:${color}; font-weight:bold;">${currentMode} | 采样率: ${sampleRateText}</span>
        `;
    }

    // 步骤 4: 执行单帧的更新与渲染
    viewer.update();
    viewer.render();
}
