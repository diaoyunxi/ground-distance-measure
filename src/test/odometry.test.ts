import { describe, expect, it } from 'vitest';
import {
  accumulateFrameMotion,
  createOdometryState,
  detectCorners,
  estimateFrameHomography,
  homographyFromPose,
  toGray,
  trackOpticalFlow,
  type GrayImage,
} from '../core/odometry';
import { applyHomography, type Point2D } from '../core/homography';
import { intrinsicsFromImage } from '../core/camera';

/** 构造带随机纹理的合成图像（用于角点检测） */
function createTexturedImage(width: number, height: number, seed = 42): GrayImage {
  const data = new Uint8ClampedArray(width * height);
  let s = seed;
  const rand = () => {
    s = (s * 1103515245 + 12345) & 0x7fffffff;
    return s / 0x7fffffff;
  };

  // 随机方块纹理
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const blockSize = 6;
      const bx = Math.floor(x / blockSize);
      const by = Math.floor(y / blockSize);
      const h = Math.abs(Math.sin(bx * 12.9898 + by * 78.233 + seed) * 43758.5453) % 1;
      const base = h * 200 + rand() * 20;
      data[y * width + x] = base;
    }
  }
  return { data, width, height };
}

/**
 * 构造平移后的图像（模拟相机横移）。
 * 内容向 (dx, dy) 方向移动：新图像在 (x+dx, y+dy) 处取旧图像 (x, y) 的像素。
 */
function translateImage(image: GrayImage, dx: number, dy: number): GrayImage {
  const { data, width, height } = image;
  const out = new Uint8ClampedArray(width * height);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const sx = Math.round(x + dx);
      const sy = Math.round(y + dy);
      if (sx >= 0 && sx < width && sy >= 0 && sy < height) {
        out[y * width + x] = data[sy * width + sx];
      }
    }
  }
  return { data: out, width, height };
}

describe('odometry: 灰度转换', () => {
  it('RGB 转灰度符合加权公式', () => {
    const rgba = new Uint8ClampedArray([255, 0, 0, 255]); // 纯红
    const gray = toGray(rgba, 1, 1);
    // 0.299 * 255 ≈ 76
    expect(gray.data[0]).toBeCloseTo(76, 0);
  });

  it('灰度图尺寸正确', () => {
    const rgba = new Uint8ClampedArray(4 * 4 * 4).fill(128);
    const gray = toGray(rgba, 4, 4);
    expect(gray.width).toBe(4);
    expect(gray.height).toBe(4);
    expect(gray.data.length).toBe(16);
  });
});

describe('odometry: 角点检测', () => {
  it('在纹理图像上找到角点', () => {
    const image = createTexturedImage(160, 120);
    const corners = detectCorners(image, { maxCorners: 30 });
    expect(corners.length).toBeGreaterThan(0);
    expect(corners.length).toBeLessThanOrEqual(30);
  });

  it('角点坐标在图像范围内', () => {
    const image = createTexturedImage(120, 100);
    const corners = detectCorners(image, { maxCorners: 20 });
    for (const c of corners) {
      expect(c.x).toBeGreaterThan(0);
      expect(c.x).toBeLessThan(120);
      expect(c.y).toBeGreaterThan(0);
      expect(c.y).toBeLessThan(100);
    }
  });

  it('平坦图像不产生角点', () => {
    const width = 80;
    const height = 80;
    const data = new Uint8ClampedArray(width * height).fill(100);
    const corners = detectCorners({ data, width, height }, { maxCorners: 20 });
    expect(corners.length).toBe(0);
  });
});

describe('odometry: 光流跟踪', () => {
  it('能跟踪平移后的特征点', () => {
    const image = createTexturedImage(200, 150);
    const corners = detectCorners(image, { maxCorners: 25 });
    expect(corners.length).toBeGreaterThan(5);

    const dx = 4;
    const dy = -3;
    const shifted = translateImage(image, dx, dy);

    const tracked = trackOpticalFlow(image, shifted, corners, { windowSize: 15 });

    const okTracked = tracked.filter((t) => t.tracked);
    // 边缘点在金字塔粗层可能超出边界而失败，保留 40% 以上即可满足实际需求
    expect(okTracked.length).toBeGreaterThanOrEqual(corners.length * 0.4);

    let sumDx = 0;
    let sumDy = 0;
    let count = 0;
    tracked.forEach((t, i) => {
      if (!t.tracked) return;
      sumDx += t.point.x - corners[i].x;
      sumDy += t.point.y - corners[i].y;
      count++;
    });
    expect(count).toBeGreaterThan(0);
    // 图像内容向 (dx, dy) 移动，特征点位置也应随之移动
    expect(sumDx / count).toBeGreaterThan(1);
    expect(sumDy / count).toBeLessThan(-0.5);
  });

  it('越界点被标记为未跟踪', () => {
    const image = createTexturedImage(100, 100);
    const tracked = trackOpticalFlow(
      image,
      image,
      [{ x: 2, y: 2 }, { x: 99, y: 99 }],
      { windowSize: 15 },
    );
    expect(tracked[0].tracked).toBe(false);
    expect(tracked[1].tracked).toBe(false);
  });
});

describe('odometry: 帧间单应估计', () => {
  it('由对应点估计出帧间单应', () => {
    // 构造已知的地面→图像映射
    const groundPts: Point2D[] = [
      { x: -0.3, y: 0.6 },
      { x: 0.3, y: 0.6 },
      { x: 0.3, y: 1.6 },
      { x: -0.3, y: 1.6 },
      { x: 0, y: 1.0 },
      { x: -0.1, y: 0.9 },
    ];
    const imagePts: Point2D[] = groundPts.map((g) => ({
      x: 540 + g.x * 800,
      y: 960 - g.y * 300,
    }));

    const H = estimateFrameHomography(groundPts, imagePts);
    expect(H).not.toBeNull();
    for (let i = 0; i < groundPts.length; i++) {
      const mapped = applyHomography(H!, groundPts[i])!;
      expect(mapped.x).toBeCloseTo(imagePts[i].x, 3);
      expect(mapped.y).toBeCloseTo(imagePts[i].y, 3);
    }
  });

  it('点数不足时返回 null', () => {
    expect(estimateFrameHomography([{ x: 0, y: 0 }], [{ x: 1, y: 1 }])).toBeNull();
  });
});

describe('odometry: 曲线距离累计', () => {
  it('相机位移被正确累计为曲线距离', () => {
    const intrinsics = intrinsicsFromImage(800, 1080, 1920);

    // 构造两个已知位姿对应的地面→图像单应
    const makeH = (camX: number, camY: number): number[] => {
      // 简化的正交俯视模型
      const f = intrinsics.focalLength;
      const cx = intrinsics.principalX;
      const cy = intrinsics.principalY;
      const height = 1.5;
      return [
        f, 0, cx - (f * camX) / height,
        0, -f, cy - (f * camY) / height,
        0, 0, 1,
      ];
    };

    let state = createOdometryState();
    const points: Point2D[] = Array.from({ length: 8 }, (_, i) => ({ x: i * 10, y: i * 5 }));

    // 相机沿 X 方向每次移动 0.1 米，共 5 帧
    for (let frame = 0; frame <= 5; frame++) {
      const H = makeH(frame * 0.1, 0);
      state = accumulateFrameMotion(state, points, H, intrinsics);
    }

    // 曲线距离应接近 0.5 米
    expect(state.curveDistance).toBeGreaterThan(0.3);
    expect(state.curveDistance).toBeLessThan(0.8);
    expect(state.positions.length).toBeGreaterThan(1);
  });

  it('静止时曲线距离不累积', () => {
    const intrinsics = intrinsicsFromImage(800, 1080, 1920);
    const f = intrinsics.focalLength;
    const H = [
      f, 0, intrinsics.principalX,
      0, -f, intrinsics.principalY,
      0, 0, 1,
    ];
    const points: Point2D[] = Array.from({ length: 6 }, (_, i) => ({ x: i, y: i }));

    let state = createOdometryState();
    for (let i = 0; i < 5; i++) {
      state = accumulateFrameMotion(state, points, H, intrinsics);
    }
    expect(state.curveDistance).toBeCloseTo(0, 6);
  });

  it('初始状态为空', () => {
    const state = createOdometryState();
    expect(state.positions).toHaveLength(0);
    expect(state.curveDistance).toBe(0);
    expect(state.trackedPoints).toBe(0);
  });

  it('虚假的大幅位移被拒绝，不污染里程', () => {
    const intrinsics = intrinsicsFromImage(800, 1080, 1920);
    const f = intrinsics.focalLength;

    // 构造两个单应：第二个对应相机突然"瞬移"了 100 米
    const makeH = (camX: number): number[] => [
      f, 0, intrinsics.principalX - (f * camX) / 1.5,
      0, -f, intrinsics.principalY,
      0, 0, 1,
    ];

    const points: Point2D[] = Array.from({ length: 8 }, (_, i) => ({ x: i * 10, y: i * 5 }));

    let state = createOdometryState();
    state = accumulateFrameMotion(state, points, makeH(0), intrinsics);
    const afterFirst = state.curveDistance;

    // 瞬移 100 米 —— 应被上限拦截
    state = accumulateFrameMotion(state, points, makeH(100), intrinsics);
    expect(state.curveDistance).toBe(afterFirst);
    expect(state.positions).toHaveLength(1);
  });

  it('正常的小幅位移被正常累计', () => {
    const intrinsics = intrinsicsFromImage(800, 1080, 1920);
    const f = intrinsics.focalLength;
    const makeH = (camX: number): number[] => [
      f, 0, intrinsics.principalX - (f * camX) / 1.5,
      0, -f, intrinsics.principalY,
      0, 0, 1,
    ];
    const points: Point2D[] = Array.from({ length: 8 }, (_, i) => ({ x: i * 10, y: i * 5 }));

    let state = createOdometryState();
    // 相机每次移动 0.05 米，共 4 步
    for (let i = 0; i <= 4; i++) {
      state = accumulateFrameMotion(state, points, makeH(i * 0.05), intrinsics);
    }
    expect(state.curveDistance).toBeGreaterThan(0.1);
    expect(state.curveDistance).toBeLessThan(0.4);
  });
});

describe('odometry: 位姿与单应互转', () => {
  it('位姿可还原出对应的地面→图像单应', () => {
    const intrinsics = intrinsicsFromImage(800, 1080, 1920);
    // 构造一个简单的旋转与位置
    const pose = {
      rotation: [1, 0, 0, 0, 1, 0, 0, 0, 1],
      center: [0.2, -0.3, 1.5] as [number, number, number],
    };

    const H = homographyFromPose(pose, intrinsics);
    expect(H).not.toBeNull();

    // 世界原点 (0,0) 的投影：H 的第三列即 t = -R*C，这里 R = I
    const projected = applyHomography(H!, { x: 0, y: 0 })!;
    expect(projected.x).toBeCloseTo(intrinsics.principalX + (800 * 0.2) / 1.5, 4);
    expect(projected.y).toBeCloseTo(intrinsics.principalY - (800 * 0.3) / 1.5, 4);
  });
});
