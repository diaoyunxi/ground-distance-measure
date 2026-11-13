/**
 * 视觉里程计：从视频帧中跟踪地面特征点，估计相机的帧间运动，
 * 从而累计出摄像头在地面上的移动轨迹与曲线距离。
 *
 * 整体思路：
 *  1. 在首帧检测角点（Shi-Tomasi），作为地面特征点；
 *  2. 相邻帧之间用 Lucas-Kanade 光流跟踪这些点；
 *  3. 用跟踪到的对应关系估计两帧之间的地面单应增量；
 *  4. 由单应增量分解出相机的帧间平移，累计成轨迹。
 *
 * 所有算法均为纯函数实现，不依赖任何外部 CV 库，便于单元测试。
 */

import {
  computeHomography,
  estimateHomographyRANSAC,
  type Point2D,
} from './homography';
import {
  decomposeHomography,
  invertHomography,
  poseFromGroundHomography,
  type CameraPose,
  type Intrinsics,
} from './camera';

/** 灰度图（行主序，长度 = width * height） */
export interface GrayImage {
  data: Uint8ClampedArray;
  width: number;
  height: number;
}

/**
 * 由 RGB/RGBA 数据转为灰度图。
 * 采用 ITU-R BT.601 加权，视觉上最接近人眼亮度感知。
 */
export function toGray(
  rgba: Uint8ClampedArray | Uint8Array,
  width: number,
  height: number,
  channels = 4,
): GrayImage {
  const data = new Uint8ClampedArray(width * height);
  for (let i = 0; i < width * height; i++) {
    const base = i * channels;
    if (channels >= 3) {
      data[i] = 0.299 * rgba[base] + 0.587 * rgba[base + 1] + 0.114 * rgba[base + 2];
    } else {
      data[i] = rgba[base];
    }
  }
  return { data, width, height };
}

/**
 * Shi-Tomasi 角点检测。
 *
 * 先计算图像梯度与结构张量，再对每个像素求结构张量的最小特征值，
 * 该值在角点处取得局部极大。用 3x3 非极大值抑制挑选最终角点。
 */
export function detectCorners(
  image: GrayImage,
  options: {
    maxCorners?: number;
    qualityLevel?: number;
    minDistance?: number;
    blockSize?: number;
  } = {},
): Point2D[] {
  const maxCorners = options.maxCorners ?? 120;
  const qualityLevel = options.qualityLevel ?? 0.01;
  const minDistance = options.minDistance ?? 12;
  const blockSize = options.blockSize ?? 3;

  const { data, width, height } = image;
  if (width < 5 || height < 5) return [];

  // Sobel 梯度
  const gx = new Float32Array(width * height);
  const gy = new Float32Array(width * height);
  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      const i = y * width + x;
      const tl = data[i - width - 1];
      const tc = data[i - width];
      const tr = data[i - width + 1];
      const ml = data[i - 1];
      const mr = data[i + 1];
      const bl = data[i + width - 1];
      const bc = data[i + width];
      const br = data[i + width + 1];
      gx[i] = tr + 2 * mr + br - tl - 2 * ml - bl;
      gy[i] = bl + 2 * bc + br - tl - 2 * tc - tr;
    }
  }

  // 结构张量的三个分量（用积分图做 box filter，避免逐点遍历窗口）
  const half = blockSize >> 1;
  const sumXX = new Float32Array(width * height);
  const sumYY = new Float32Array(width * height);
  const sumXY = new Float32Array(width * height);

  // 逐像素的 gx*gy
  const gxyPix = new Float32Array(width * height);
  for (let i = 0; i < width * height; i++) {
    gxyPix[i] = gx[i] * gy[i];
  }

  // 积分图（(w+1) x (h+1)）
  const buildIntegral = (arr: Float32Array): Float64Array => {
    const integral = new Float64Array((width + 1) * (height + 1));
    for (let y = 0; y < height; y++) {
      let rowSum = 0;
      for (let x = 0; x < width; x++) {
        rowSum += arr[y * width + x];
        integral[(y + 1) * (width + 1) + (x + 1)] =
          integral[y * (width + 1) + (x + 1)] + rowSum;
      }
    }
    return integral;
  };

  const iXX = buildIntegral(gx.map((v) => v * v));
  const iYY = buildIntegral(gy.map((v) => v * v));
  const iXY = buildIntegral(gxyPix);

  /** 用积分图求以 (x,y) 为中心、边长 blockSize 的窗口和 */
  const windowSum = (integral: Float64Array, x: number, y: number): number => {
    const x0 = x - half;
    const y0 = y - half;
    const x1 = x + half + 1;
    const y1 = y + half + 1;
    const w1 = width + 1;
    return (
      integral[y1 * w1 + x1] - integral[y0 * w1 + x1] - integral[y1 * w1 + x0] + integral[y0 * w1 + x0]
    );
  };

  const candidates: { x: number; y: number; value: number }[] = [];

  for (let y = half + 1; y < height - half - 1; y++) {
    for (let x = half + 1; x < width - half - 1; x++) {
      const i = y * width + x;
      const sxx = windowSum(iXX, x, y);
      const syy = windowSum(iYY, x, y);
      const sxy = windowSum(iXY, x, y);

      sumXX[i] = sxx;
      sumYY[i] = syy;
      sumXY[i] = sxy;

      // 结构张量的最小特征值
      const trace = sxx + syy;
      const det = sxx * syy - sxy * sxy;
      const discriminant = Math.max(0, (trace * trace) / 4 - det);
      const lambdaMin = trace / 2 - Math.sqrt(discriminant);

      if (lambdaMin > 0) {
        candidates.push({ x, y, value: lambdaMin });
      }
    }
  }

  if (candidates.length === 0) return [];

  // 按响应值降序
  candidates.sort((a, b) => b.value - a.value);
  const threshold = candidates[0].value * qualityLevel;

  // 非极大值抑制（按最小间距）
  const selected: Point2D[] = [];
  const minDistSq = minDistance * minDistance;
  for (const c of candidates) {
    if (c.value < threshold) break;
    if (selected.length >= maxCorners) break;
    let ok = true;
    for (const s of selected) {
      const dx = s.x - c.x;
      const dy = s.y - c.y;
      if (dx * dx + dy * dy < minDistSq) {
        ok = false;
        break;
      }
    }
    if (ok) selected.push({ x: c.x, y: c.y });
  }

  return selected;
}

/**
 * Lucas-Kanade 稀疏光流（逆组合 + 图像金字塔）。
 *
 * 逆组合形式在收敛性与精度上都优于加性形式：
 *   - 梯度与结构张量始终在模板帧（第一帧）上计算，迭代过程中保持不变；
 *   - 位移增量通过"反向合成"求解，无需每帧重新计算梯度；
 *   - 配合图像金字塔，可处理较大位移（逐层由粗到细）。
 *
 * 求解的最小二乘方程：
 *   [Σ Ix²  Σ IxIy][dx]   [Σ Ix (I1(w) - I0(w))]
 *   [Σ IxIy Σ Iy²][dy] = [Σ Iy (I1(w) - I0(w))]
 */
export function trackOpticalFlow(
  prev: GrayImage,
  next: GrayImage,
  points: Point2D[],
  options: { windowSize?: number; maxIterations?: number; epsilon?: number; levels?: number } = {},
): { point: Point2D; tracked: boolean; residual: number }[] {
  const windowSize = options.windowSize ?? 21;
  const maxIterations = options.maxIterations ?? 15;
  const epsilon = options.epsilon ?? 0.02;
  const levels = options.levels ?? 3;

  // 构建图像金字塔（level 0 为原图）
  const prevPyramid = buildPyramid(prev, levels);
  const nextPyramid = buildPyramid(next, levels);

  return points.map((p) => {
    // 逐层由粗到细跟踪
    let current: Point2D = { x: p.x, y: p.y };

    for (let level = levels - 1; level >= 0; level--) {
      const scale = 1 / Math.pow(2, level);
      const guess: Point2D = { x: current.x * scale, y: current.y * scale };
      const result = refineAtLevel(
        prevPyramid[level],
        nextPyramid[level],
        guess,
        windowSize,
        maxIterations,
        epsilon,
      );
      if (!result.tracked) {
        // 粗层失败时不再往下细化，保留原位置并标记失败
        return { point: { x: p.x, y: p.y }, tracked: false, residual: result.residual };
      }
      current = { x: result.point.x / scale, y: result.point.y / scale };
    }

    const final = refineAtLevel(
      prevPyramid[0],
      nextPyramid[0],
      current,
      windowSize,
      maxIterations,
      epsilon,
    );
    return final;
  });
}

/** 构建图像金字塔（每层尺寸减半，用 2x2 平均池化） */
function buildPyramid(image: GrayImage, levels: number): GrayImage[] {
  const pyramid: GrayImage[] = [image];
  for (let level = 1; level < levels; level++) {
    const prev = pyramid[level - 1];
    const w = Math.max(1, prev.width >> 1);
    const h = Math.max(1, prev.height >> 1);
    const data = new Uint8ClampedArray(w * h);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const x0 = Math.min(prev.width - 1, x * 2);
        const y0 = Math.min(prev.height - 1, y * 2);
        const x1 = Math.min(prev.width - 1, x * 2 + 1);
        const y1 = Math.min(prev.height - 1, y * 2 + 1);
        data[y * w + x] =
          (prev.data[y0 * prev.width + x0] +
            prev.data[y0 * prev.width + x1] +
            prev.data[y1 * prev.width + x0] +
            prev.data[y1 * prev.width + x1]) /
          4;
      }
    }
    pyramid.push({ data, width: w, height: h });
  }
  return pyramid;
}

/**
 * 在单层图像上做逆组合 LK 精化。
 */
function refineAtLevel(
  prev: GrayImage,
  next: GrayImage,
  start: Point2D,
  windowSize: number,
  maxIterations: number,
  epsilon: number,
): { point: Point2D; tracked: boolean; residual: number } {
  const half = windowSize >> 1;
  const startX = Math.round(start.x);
  const startY = Math.round(start.y);

  if (
    startX - half < 1 ||
    startY - half < 1 ||
    startX + half >= prev.width - 1 ||
    startY + half >= prev.height - 1
  ) {
    return { point: start, tracked: false, residual: Infinity };
  }

  // 模板梯度与结构张量（固定不变）
  const gx = new Float32Array(windowSize * windowSize);
  const gy = new Float32Array(windowSize * windowSize);
  const i0 = new Float32Array(windowSize * windowSize);
  let gxx = 0;
  let gxy = 0;
  let gyy = 0;

  for (let dy = -half; dy <= half; dy++) {
    for (let dx = -half; dx <= half; dx++) {
      const idx = (dy + half) * windowSize + (dx + half);
      const px = startX + dx;
      const py = startY + dy;
      const idxImg = py * prev.width + px;

      // Sobel 梯度（3x3），比中心差分抗噪
      const gxVal =
        prev.data[idxImg - prev.width + 1] +
        2 * prev.data[idxImg + 1] +
        prev.data[idxImg + prev.width + 1] -
        prev.data[idxImg - prev.width - 1] -
        2 * prev.data[idxImg - 1] -
        prev.data[idxImg + prev.width - 1];
      const gyVal =
        prev.data[idxImg + prev.width - 1] +
        2 * prev.data[idxImg + prev.width] +
        prev.data[idxImg + prev.width + 1] -
        prev.data[idxImg - prev.width - 1] -
        2 * prev.data[idxImg - prev.width] -
        prev.data[idxImg - prev.width + 1];

      gx[idx] = gxVal;
      gy[idx] = gyVal;
      i0[idx] = prev.data[idxImg];
      gxx += gxVal * gxVal;
      gxy += gxVal * gyVal;
      gyy += gyVal * gyVal;
    }
  }

  const det = gxx * gyy - gxy * gxy;
  // 纹理过弱（角点附近梯度不足）时无法跟踪
  if (!isFinite(det) || Math.abs(det) < 1e-3) {
    return { point: start, tracked: false, residual: Infinity };
  }

  let vx = 0;
  let vy = 0;
  let residual = 0;

  for (let iter = 0; iter < maxIterations; iter++) {
    let bx = 0;
    let by = 0;
    let diffSum = 0;

    for (let dy = -half; dy <= half; dy++) {
      for (let dx = -half; dx <= half; dx++) {
        const idx = (dy + half) * windowSize + (dx + half);
        const sampleX = startX + dx + vx;
        const sampleY = startY + dy + vy;

        if (
          sampleX < 1 ||
          sampleY < 1 ||
          sampleX >= next.width - 1 ||
          sampleY >= next.height - 1
        ) {
          return { point: { x: startX + vx, y: startY + vy }, tracked: false, residual: Infinity };
        }

        const it = bilinearSample(next, sampleX, sampleY);
        const diff = it - i0[idx];
        bx += gx[idx] * diff;
        by += gy[idx] * diff;
        diffSum += diff * diff;
      }
    }

    // 逆组合更新：位移增量由模板梯度与残差求解
    const dxStep = (gyy * bx - gxy * by) / det;
    const dyStep = (gxx * by - gxy * bx) / det;

    vx += dxStep;
    vy += dyStep;
    residual = diffSum / (windowSize * windowSize);

    if (Math.hypot(dxStep, dyStep) < epsilon) break;
  }

  const finalX = startX + vx;
  const finalY = startY + vy;

  // 位移过大通常意味着跟踪失败
  if (Math.hypot(vx, vy) > windowSize * 1.5) {
    return { point: { x: finalX, y: finalY }, tracked: false, residual };
  }

  return { point: { x: finalX, y: finalY }, tracked: true, residual };
}

/** 双线性插值采样 */
function bilinearSample(image: GrayImage, x: number, y: number): number {
  const x0 = Math.floor(x);
  const y0 = Math.floor(y);
  const fx = x - x0;
  const fy = y - y0;

  const i00 = image.data[y0 * image.width + x0];
  const i10 = image.data[y0 * image.width + x0 + 1];
  const i01 = image.data[(y0 + 1) * image.width + x0];
  const i11 = image.data[(y0 + 1) * image.width + x0 + 1];

  const top = i00 * (1 - fx) + i10 * fx;
  const bottom = i01 * (1 - fx) + i11 * fx;
  return top * (1 - fy) + bottom * fy;
}

/**
 * 里程计状态：维护相机在地面系中的轨迹。
 */
export interface OdometryState {
  /** 每帧的相机地面位置（米） */
  positions: Point2D[];
  /** 累计曲线距离（米） */
  curveDistance: number;
  /** 跟踪成功的特征点数 */
  trackedPoints: number;
}

/** 初始里程计状态 */
export function createOdometryState(): OdometryState {
  return { positions: [], curveDistance: 0, trackedPoints: 0 };
}

/**
 * 处理一帧：估计相机相对上一帧的地面位移并累计。
 *
 * @param state 里程计状态
 * @param currPoints 当前帧跟踪到的特征点数（用于状态上报）
 * @param currH 当前帧的 地面→图像 单应
 * @param intrinsics 相机内参
 */
export function accumulateFrameMotion(
  state: OdometryState,
  currPoints: Point2D[],
  currH: number[],
  intrinsics: Intrinsics,
  options: { maxStepMeters?: number } = {},
): OdometryState {
  const positions = [...state.positions];
  let curveDistance = state.curveDistance;

  // 相机高度用于估算合理的单帧位移上限（防止虚假匹配导致的里程爆炸）
  const inverse = invertHomography(currH);
  if (!inverse) {
    return { ...state, trackedPoints: currPoints.length };
  }
  const pose = poseFromGroundHomography(inverse, intrinsics);
  if (!pose) {
    return { ...state, trackedPoints: currPoints.length };
  }

  const cameraGround: Point2D = { x: pose.center[0], y: pose.center[1] };
  const height = Math.abs(pose.center[2]);
  // 单帧位移上限：取相机高度的 30%，并设绝对上限 0.5 米。
  // 手机正常移动时每帧（10Hz）位移远小于此，可有效拦截虚假里程。
  const maxStep = options.maxStepMeters ?? Math.min(0.5, Math.max(0.05, height * 0.3));

  if (positions.length > 0) {
    const last = positions[positions.length - 1];
    const step = Math.hypot(cameraGround.x - last.x, cameraGround.y - last.y);
    // 抑制过小的抖动（阈值 1 毫米），避免静止时里程累积漂移
    if (step > 0.001 && step <= maxStep) {
      curveDistance += step;
      positions.push(cameraGround);
    }
  } else {
    positions.push(cameraGround);
  }

  return {
    positions,
    curveDistance,
    trackedPoints: currPoints.length,
  };
}

/**
 * 由两帧的特征对应关系估计地面→图像的单应。
 * 这是帧间运动估计的核心。
 */
export function estimateFrameHomography(
  prevPoints: Point2D[],
  currPoints: Point2D[],
): number[] | null {
  if (prevPoints.length < 4) return null;

  const result = estimateHomographyRANSAC(prevPoints, currPoints, {
    threshold: 2.5,
    iterations: 120,
  });

  if (result) return result.H;

  // RANSAC 失败时退回最小二乘
  return computeHomography(prevPoints, currPoints);
}

/**
 * 由标定的地面单应（图像→地面）求出对应的 地面→图像 单应。
 */
export function groundToImageHomography(
  H_imgToGround: number[],
  _intrinsics: Intrinsics,
): number[] | null {
  return invertHomography(H_imgToGround);
}

/**
 * 相机位姿 → 地面→图像 单应（用于实时反投影）。
 */
export function homographyFromPose(pose: CameraPose, intrinsics: Intrinsics): number[] | null {
  const f = intrinsics.focalLength;
  const { rotation, center } = pose;

  // r1/r2 为 R 的前两列（世界系基向量在相机系的表示）
  const r1 = [rotation[0], rotation[1], rotation[2]];
  const r2 = [rotation[3], rotation[4], rotation[5]];

  // t = -R * C（世界原点在相机系的坐标）
  const t = [
    -(rotation[0] * center[0] + rotation[1] * center[1] + rotation[2] * center[2]),
    -(rotation[3] * center[0] + rotation[4] * center[1] + rotation[5] * center[2]),
    -(rotation[6] * center[0] + rotation[7] * center[1] + rotation[8] * center[2]),
  ];

  const K = [f, 0, intrinsics.principalX, 0, f, intrinsics.principalY, 0, 0, 1];
  const H = [
    r1[0], r2[0], t[0],
    r1[1], r2[1], t[1],
    r1[2], r2[2], t[2],
  ];

  // H_groundToImg = K * H
  const out = new Array<number>(9);
  for (let r = 0; r < 3; r++) {
    for (let c = 0; c < 3; c++) {
      out[r * 3 + c] =
        K[r * 3] * H[c] + K[r * 3 + 1] * H[3 + c] + K[r * 3 + 2] * H[6 + c];
    }
  }
  return out;
}

export { decomposeHomography };
