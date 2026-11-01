import { describe, expect, it } from 'vitest';
import {
  cameraHeight,
  decomposeHomography,
  estimateFocalLength,
  fieldOfView,
  groundDistance,
  intrinsicsFromImage,
  invertHomography,
  pathLength,
  poseFromGroundHomography,
  projectToImage,
  backprojectToGround,
} from '../core/camera';
import { applyHomography, computeHomography, type Point2D } from '../core/homography';

/**
 * 用真实相机模型生成合成数据。
 *
 * 坐标约定：
 *  - 世界系：x 向右，y 向前，z 向上，地面为 z = 0 平面。
 *  - 相机系：x 向右，y 向下，z 沿光轴向前。
 *  - pitch 为向下俯视角（弧度，正值）。
 *
 * 相机在 (0, 0, 1.5) 米高处俯视地面，焦距 800 像素。
 */
function createCamera() {
  const focalLength = 800;
  const imageWidth = 1080;
  const imageHeight = 1920;
  const intrinsics = intrinsicsFromImage(focalLength, imageWidth, imageHeight);

  const pitch = 0.6;
  const cp = Math.cos(pitch);
  const sp = Math.sin(pitch);

  // 相机三个轴在世界系中的表示
  const right = [1, 0, 0];
  const down = [0, sp, cp];
  const forward = [0, cp, -sp];

  const center: [number, number, number] = [0, 0, 1.5];

  /** 世界点 → 像素 */
  const project = (world: [number, number, number]): Point2D => {
    const d: [number, number, number] = [
      world[0] - center[0],
      world[1] - center[1],
      world[2] - center[2],
    ];
    const xc = right[0] * d[0] + right[1] * d[1] + right[2] * d[2];
    const yc = down[0] * d[0] + down[1] * d[1] + down[2] * d[2];
    const zc = forward[0] * d[0] + forward[1] * d[1] + forward[2] * d[2];
    if (zc <= 1e-6) return { x: NaN, y: NaN };
    return {
      x: (focalLength * xc) / zc + intrinsics.principalX,
      y: (focalLength * yc) / zc + intrinsics.principalY,
    };
  };

  /** 地面点 → 像素 */
  const projectGround = (x: number, y: number): Point2D => project([x, y, 0]);

  /** 世界系 → 相机系 的旋转矩阵（行主序） */
  const rotation = [
    right[0], right[1], right[2],
    down[0], down[1], down[2],
    forward[0], forward[1], forward[2],
  ];

  return {
    focalLength,
    imageWidth,
    imageHeight,
    intrinsics,
    rotation,
    center,
    project,
    projectGround,
  };
}

describe('camera: 焦距解算', () => {
  it('从地面单应恢复真实焦距', () => {
    const cam = createCamera();
    // 在地面取 4 个点，构成矩形
    const groundPts: Point2D[] = [
      { x: -0.3, y: 0.6 },
      { x: 0.3, y: 0.6 },
      { x: 0.3, y: 1.6 },
      { x: -0.3, y: 1.6 },
    ];
    const imagePts = groundPts.map((g) => cam.projectGround(g.x, g.y));

    // H: 图像 → 地面
    const H = computeHomography(imagePts, groundPts)!;
    const groundToImage = invertHomography(H)!;

    const estimated = estimateFocalLength(
      groundToImage,
      cam.intrinsics.principalX,
      cam.intrinsics.principalY,
    );
    expect(estimated).not.toBeNull();
    // 焦距应接近真实值（允许 2% 误差）
    expect(estimated!).toBeCloseTo(cam.focalLength, -1);
    expect(Math.abs(estimated! - cam.focalLength) / cam.focalLength).toBeLessThan(0.02);
  });

  it('退化单应返回 null', () => {
    const H = [1, 0, 0, 0, 1, 0, 0, 0, 0];
    const result = estimateFocalLength(H, 100, 100);
    expect(result).toBeNull();
  });
});

describe('camera: 单应分解求位姿', () => {
  it('恢复的相机高度与真实值一致', () => {
    const cam = createCamera();
    const groundPts: Point2D[] = [
      { x: -0.3, y: 0.6 },
      { x: 0.3, y: 0.6 },
      { x: 0.3, y: 1.6 },
      { x: -0.3, y: 1.6 },
    ];
    const imagePts = groundPts.map((g) => cam.projectGround(g.x, g.y));

    const H = computeHomography(imagePts, groundPts)!;
    const groundToImage = invertHomography(H)!;

    const decomposed = decomposeHomography(groundToImage, cam.intrinsics);
    expect(decomposed).not.toBeNull();

    const height = cameraHeight(decomposed!.pose);
    expect(Math.abs(height - cam.center[2])).toBeLessThan(0.05);
  });

  it('位姿反解与直接求解一致', () => {
    const cam = createCamera();
    const groundPts: Point2D[] = [
      { x: -0.3, y: 0.6 },
      { x: 0.3, y: 0.6 },
      { x: 0.3, y: 1.6 },
      { x: -0.3, y: 1.6 },
    ];
    const imagePts = groundPts.map((g) => cam.projectGround(g.x, g.y));
    const H = computeHomography(imagePts, groundPts)!;

    const pose = poseFromGroundHomography(H, cam.intrinsics);
    expect(pose).not.toBeNull();
    expect(Math.abs(cameraHeight(pose!) - cam.center[2])).toBeLessThan(0.05);
  });

  it('奇异单应矩阵无法求逆', () => {
    const singular = [1, 2, 3, 2, 4, 6, 3, 6, 9];
    expect(invertHomography(singular)).toBeNull();
  });
});

describe('camera: 地面投影往返', () => {
  it('地面 → 图像 → 地面 往返无损', () => {
    const cam = createCamera();
    const groundPts: Point2D[] = [
      { x: -0.3, y: 0.6 },
      { x: 0.3, y: 0.6 },
      { x: 0.3, y: 1.6 },
      { x: -0.3, y: 1.6 },
    ];
    const imagePts = groundPts.map((g) => cam.projectGround(g.x, g.y));
    const H = computeHomography(imagePts, groundPts)!;

    // 任取一个地面点，投影到图像再反投影回来
    const probe = { x: 0.1, y: 1.0 };
    const pixel = applyHomography(invertHomography(H)!, probe)!;
    const back = backprojectToGround(H, pixel)!;

    expect(back.x).toBeCloseTo(probe.x, 6);
    expect(back.y).toBeCloseTo(probe.y, 6);
  });

  it('正投影与反投影互逆', () => {
    const cam = createCamera();
    const groundPts: Point2D[] = [
      { x: -0.3, y: 0.6 },
      { x: 0.3, y: 0.6 },
      { x: 0.3, y: 1.6 },
      { x: -0.3, y: 1.6 },
    ];
    const imagePts = groundPts.map((g) => cam.projectGround(g.x, g.y));
    const H = computeHomography(imagePts, groundPts)!;
    const groundToImage = invertHomography(H)!;

    const world = { x: 0.05, y: 1.05 };
    const pixel = projectToImage(groundToImage, world)!;
    const ground = backprojectToGround(H, pixel)!;

    expect(ground.x).toBeCloseTo(world.x, 6);
    expect(ground.y).toBeCloseTo(world.y, 6);
  });
});

describe('camera: 距离计算', () => {
  it('两点间距离计算正确', () => {
    expect(groundDistance({ x: 0, y: 0 }, { x: 3, y: 4 })).toBeCloseTo(5, 12);
    expect(groundDistance({ x: 1, y: 1 }, { x: 1, y: 1 })).toBeCloseTo(0, 12);
  });

  it('路径长度等于各段之和', () => {
    const path: Point2D[] = [
      { x: 0, y: 0 },
      { x: 3, y: 4 },
      { x: 3, y: 14 },
    ];
    expect(pathLength(path)).toBeCloseTo(15, 12);
  });

  it('单点路径长度为 0', () => {
    expect(pathLength([{ x: 5, y: 5 }])).toBeCloseTo(0, 12);
    expect(pathLength([])).toBeCloseTo(0, 12);
  });

  it('曲线距离大于直线距离（绕行情形）', () => {
    const path: Point2D[] = [
      { x: 0, y: 0 },
      { x: 1, y: 1 },
      { x: 2, y: 0 },
    ];
    const straight = groundDistance(path[0], path[2]);
    const curve = pathLength(path);
    expect(curve).toBeGreaterThan(straight);
    expect(curve).toBeCloseTo(2 * Math.SQRT2, 12);
  });
});

describe('camera: 视场角', () => {
  it('焦距越长视场角越窄', () => {
    const wide = intrinsicsFromImage(400, 1080, 1920);
    const tele = intrinsicsFromImage(1600, 1080, 1920);
    const fovWide = fieldOfView(wide, 1080, 1920);
    const fovTele = fieldOfView(tele, 1080, 1920);

    expect(fovWide.horizontal).toBeGreaterThan(fovTele.horizontal);
    expect(fovWide.vertical).toBeGreaterThan(fovTele.vertical);
  });

  it('水平视场角数值合理', () => {
    const intr = intrinsicsFromImage(800, 1080, 1920);
    const fov = fieldOfView(intr, 1080, 1920);
    // 2*atan(540/800) ≈ 68.1 度
    expect(fov.horizontal).toBeCloseTo(68.1, 0);
  });
});
