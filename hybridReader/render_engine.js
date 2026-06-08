const fs = require('fs').promises
const path = require('path')
global.THREE = require('three')
global.Worker = require('worker_threads').Worker
const { createCanvas } = require('node-canvas-webgl/lib')
const Vec3 = require('vec3').Vec3
const { Viewer, WorldView, getBufferFromStream } = require('prismarine-viewer').viewer
const Block = require('prismarine-block')('1.16.4')

async function renderFromRaw(blockArray, width, height, length, outputFolder, numAngles=12, renderWidth=2048, renderHeight=renderWidth) {
  const version = '1.16.4'
  const viewDistance = 4
  const imgW = Number(renderWidth) || 2048
  const imgH = Number(renderHeight) || imgW
  
  const World = require('prismarine-world')(version)
  const Chunk = require('prismarine-chunk')(version)
  
  const canvas = createCanvas(imgW, imgH)
  const renderer = new THREE.WebGLRenderer({ canvas })
  const viewer = new Viewer(renderer)
  if (!viewer.setVersion(version)) {
    throw new Error('Unsupported version')
  }

  // Create empty world
  const world = new World(() => new Chunk())
  
  // Inject blocks from the 1D array
  const pastePos = new Vec3(0, 60, 0)
  
  let minX = width, minY = height, minZ = length;
  let maxX = 0, maxY = 0, maxZ = 0;
  
  const voxelNames = new Array(width * height * length);

  for (let x = 0; x < width; x++) {
    for (let y = 0; y < height; y++) {
      for (let z = 0; z < length; z++) {
        const idx = x * height * length + y * length + z;
        const blockStateId = blockArray[idx];
        
        const blockInfo = Block.fromStateId(blockStateId, 0);
        voxelNames[idx] = blockInfo ? blockInfo.name : 'air';
        
        if (blockStateId !== 0) {
          world.setBlockStateId(new Vec3(pastePos.x + x, pastePos.y + y, pastePos.z + z), blockStateId);
          if (x < minX) minX = x;
          if (y < minY) minY = y;
          if (z < minZ) minZ = z;
          if (x > maxX) maxX = x;
          if (y > maxY) maxY = y;
          if (z > maxZ) maxZ = z;
        }
      }
    }
  }
  
  await fs.mkdir(outputFolder, { recursive: true })
  await fs.writeFile(path.join(outputFolder, 'voxel_names.json'), JSON.stringify(voxelNames));

  if (maxX < minX) {
      // Empty schematic fallback
      minX = 0; minY = 0; minZ = 0;
      maxX = width; maxY = height; maxZ = length;
  }

  // Calculate center of the ACTUAL building
  const cx = pastePos.x + (minX + maxX) / 2
  const cy = pastePos.y + (minY + maxY) / 2
  const cz = pastePos.z + (minZ + maxZ) / 2
  const center = new Vec3(cx, cy, cz)
  const cameraCenter = new THREE.Vector3(cx, cy, cz)

  const worldView = new WorldView(world, viewDistance, center)
  viewer.listen(worldView)
  await worldView.init(center)
  await viewer.world.waitForChunksToRender()

  await fs.mkdir(outputFolder, { recursive: true })

  const elevation = Math.PI / 6
  const maxDim = Math.max(maxX - minX, maxY - minY, maxZ - minZ)
  const r = Math.max(maxDim * 1.5, 10) // ensure minimum radius

  for (let i = 0; i < numAngles; i++) {
    const angle = (i / numAngles) * Math.PI * 2
    
    const camX = cx + r * Math.cos(angle) * Math.cos(elevation)
    const camY = cy + r * Math.sin(elevation)
    const camZ = cz + r * Math.sin(angle) * Math.cos(elevation)
    
    viewer.camera.position.set(camX, camY, camZ)
    viewer.camera.lookAt(cameraCenter)
    
    renderer.render(viewer.scene, viewer.camera)
    
    const imageStream = canvas.createJPEGStream({ bufsize: 4096, quality: 100, progressive: false })
    const buf = await getBufferFromStream(imageStream)
    const viewName = `view_${i.toString().padStart(2, '0')}.jpg`
    await fs.writeFile(path.join(outputFolder, viewName), buf)
  }
}

module.exports = { renderFromRaw }
