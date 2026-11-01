/**
 * 单应矩阵（Homography）相关：归一化 DLT 求解、RANSAC 鲁棒估计。
 *
 * 用于把图像像素坐标映射到地面世界坐标（米），是"判定摄像头内对应的现实地面长度"的基础。
 */

import {
  mat3Apply,
  solveLinearSystem,
  symmetricEigenDecomposition,
  type Mat3,
  type Vec3,
} from './linalg';

export interface Point2D {
  x: number;
  y: number;
}

/** 单应矩阵把 (x,y,1) 映射到 (u,v,1) */
export type Homography = Mat3;

/**
 * Hartley 归一化：把点集平移到质心、缩放到平均距离 sqrt(2)。
 * 显著提升 DLT 的数值稳定性。
 */
export function normalizePoints(points: Point2D[]): {
  normalized: Point2D[];
  transform: Mat3;
  inverse: Mat3;
} {
  const n = points.length;
  let cx = 0;
  let cy = 0;
  for (const p of points) {
    cx += p.x;
    cy += p.y;
  }
  cx /= n;
  cy /= n;

  let meanDist = 0;
  for (const p of points) {
    meanDist += Math.hypot(p.x - cx, p.y - cy);
  }
  meanDist /= n;

  const scale = meanDist > 1e-12 ? Math.sqrt(2) / meanDist : 1;

  // T = [s, 0, -s*cx; 0, s, -s*cy; 0, 0, 1]
  const transform: Mat3 = [scale, 0, -scale * cx, 0, scale, -scale * cy, 0, 0, 1];
  // T^-1 = [1/s, 0, cx; 0, 1/s, cy; 0, 0, 1]
  const inverse: Mat3 = [1 / scale, 0, cx, 0, 1 / scale, cy, 0, 0, 1];

  const normalized = points.map((p) => ({
    x: scale * (p.x - cx),
    y: scale * (p.y - cy),
  }));
  return { normalized, transform, inverse };
}

/** 3x3 矩阵乘法 */
export function multiply3x3(a: Mat3, b: Mat3): Mat3 {
  const out = new Array<number>(9);
  for (let r = 0; r < 3; r++) {
    for (let c = 0; c < 3; c++) {
      out[r * 3 + c] = a[r * 3] * b[c] + a[r * 3 + 1] * b[3 + c] + a[r * 3 + 2] * b[6 + c];
    }
  }
  return out;
}

/**
 * 检测点集是否共线（退化）。
 * 共线点无法确定单应矩阵，必须显式拦截，否则会返回数值不稳定的解。
 */
export function isCollinear(points: Point2D[], epsilon = 1e-9): boolean {
  if (points.length < 3) return true;
  const [a, b, c] = points;
  const cross =
    (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
  const scale = Math.max(
    1,
    Math.hypot(b.x - a.x, b.y - a.y) * Math.hypot(c.x - a.x, c.y - a.y),
  );
  return Math.abs(cross) / scale < epsilon;
}

/**
 * 用 4 组及以上对应点求解单应矩阵（归一化 DLT + 最小二乘）。
 * 返回 null 表示退化（点不足、源点共线或目标点共线）。
 */
export function computeHomography(src: Point2D[], dst: Point2D[]): Homography | null {
  const count = Math.min(src.length, dst.length);
  if (count < 4) return null;

  // 共线点无法确定单应，直接判为退化
  if (isCollinear(src.slice(0, count)) || isCollinear(dst.slice(0, count))) {
    return null;
  }

  const normSrc = normalizePoints(src.slice(0, count));
  const normDst = normalizePoints(dst.slice(0, count));

  // 每个对应点产生 2 个方程，A 为 2N x 9
  const rows: number[][] = [];
  for (let i = 0; i < count; i++) {
    const { x, y } = normSrc.normalized[i];
    const { x: u, y: v } = normDst.normalized[i];
    rows.push([-x, -y, -1, 0, 0, 0, u * x, u * y, u]);
    rows.push([0, 0, 0, -x, -y, -1, v * x, v * y, v]);
  }

  // 最小奇异向量解：A^T A h = 0
  const ata: number[][] = [];
  for (let i = 0; i < 9; i++) {
    ata.push(new Array<number>(9).fill(0));
  }
  for (const row of rows) {
    for (let i = 0; i < 9; i++) {
      for (let j = 0; j < 9; j++) {
        ata[i][j] += row[i] * row[j];
      }
    }
  }

  const { vectors } = symmetricEigenDecomposition(ata);
  const h = vectors[vectors.length - 1]; // 最小特征值对应的向量
  if (!h || h.every((v) => Math.abs(v) < 1e-12)) return null;

  const Hn: Mat3 = [h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], h[8]];

  // 反归一化：H = T_dst^-1 * Hn * T_src
  const H = multiply3x3(multiply3x3(normDst.inverse, Hn), normSrc.transform);

  // 归一化，使 H[8] = 1（若可行），保证数值尺度稳定
  if (Math.abs(H[8]) > 1e-12) {
    for (let i = 0; i < 9; i++) H[i] /= H[8];
  }
  return H;
}

/** 用单应矩阵变换一个点，返回归一化后的 (x, y)；w 为 0 时返回 null */
export function applyHomography(H: Homography, p: Point2D): Point2D | null {
  const res: Vec3 = mat3Apply(H, [p.x, p.y, 1]);
  if (Math.abs(res[2]) < 1e-12) return null;
  return { x: res[0] / res[2], y: res[1] / res[2] };
}

/**
 * 计算点到映射点的欧氏误差。
 * 注意误差单位与 dst 的坐标单位一致。
 */
export function reprojectionError(H: Homography, src: Point2D, dst: Point2D): number {
  const mapped = applyHomography(H, src);
  if (!mapped) return Number.POSITIVE_INFINITY;
  return Math.hypot(mapped.x - dst.x, mapped.y - dst.y);
}

/**
 * 在源空间（通常是图像像素）度量重投影误差。
 *
 * 单应矩阵是 src → dst 的映射，但当 src 是像素、dst 是地面米制坐标时，
 * dst 的单位是米，用 dst 空间的距离做阈值判定会因量纲差异而失效。
 * 因此这里用逆映射把 dst 拉回 src 空间比较，误差单位即为像素。
 */
export function reprojectionErrorInSource(
  H: Homography,
  src: Point2D,
  dst: Point2D,
): number {
  const inverse = invertHomography3x3(H);
  if (!inverse) return Number.POSITIVE_INFINITY;
  const mapped = applyHomography(inverse, dst);
  if (!mapped) return Number.POSITIVE_INFINITY;
  return Math.hypot(mapped.x - src.x, mapped.y - src.y);
}

/** 3x3 矩阵求逆（内部工具） */
function invertHomography3x3(H: Homography): Homography | null {
  const det =
    H[0] * (H[4] * H[8] - H[5] * H[7]) -
    H[1] * (H[3] * H[8] - H[5] * H[6]) +
    H[2] * (H[3] * H[7] - H[4] * H[6]);
  if (Math.abs(det) < 1e-16) return null;
  const invDet = 1 / det;
  return [
    (H[4] * H[8] - H[5] * H[7]) * invDet,
    (H[2] * H[7] - H[1] * H[8]) * invDet,
    (H[1] * H[5] - H[2] * H[4]) * invDet,
    (H[5] * H[6] - H[3] * H[8]) * invDet,
    (H[0] * H[8] - H[2] * H[6]) * invDet,
    (H[2] * H[3] - H[0] * H[5]) * invDet,
    (H[3] * H[7] - H[4] * H[6]) * invDet,
    (H[1] * H[6] - H[0] * H[7]) * invDet,
    (H[0] * H[4] - H[1] * H[3]) * invDet,
  ];
}

/**
 * RANSAC 鲁棒单应估计：对每个外点（误匹配）具有抵抗力。
 *
 * 关键设计（应对"巧合解"问题）：
 *  4 点最小样本存在本质歧义——任取 4 个点都能精确拟合出一个零误差单应，
 *  当外点恰好也构成合法 4 点时，错误解与正确解的重投影误差完全打平，
 *  无法用误差大小区分。因此本实现采用：
 *  1. 最小样本数取 min(6, 可用点数)，用更多约束打破平局；
 *  2. 若可用点数仅 4~5，则退化为"全组合枚举 + 留出验证"，
 *     用未参与拟合的点来判别候选解的真伪；
 *  3. 局部优化（LO）用内点重拟合，反复迭代直至内点集合稳定；
 *  4. 内点判定一律在源空间（像素）进行，保证阈值量纲与像素一致。
 */
export function estimateHomographyRANSAC(
  src: Point2D[],
  dst: Point2D[],
  options: {
    threshold?: number;
    iterations?: number;
    minSamples?: number;
    rng?: () => number;
    refineRounds?: number;
  } = {},
): { H: Homography; inliers: boolean[]; inlierCount: number } | null {
  const threshold = options.threshold ?? 3;
  const iterations = options.iterations ?? 200;
  const count = Math.min(src.length, dst.length);
  const refineRounds = options.refineRounds ?? 3;
  if (count < 4) return null;

  // 最小样本数：优先 6 点以打破 4 点巧合解的平局；点数不足时自动下调
  const preferredSamples = options.minSamples ?? 6;
  const minSamples = Math.min(count, Math.max(4, Math.min(preferredSamples, count)));

  const rng = options.rng ?? Math.random;
  const indices = Array.from({ length: count }, (_, i) => i);

  /** 评估一个模型，返回内点标记、内点数与内点误差中位数 */
  const evaluate = (H: Homography) => {
    const inliers: boolean[] = [];
    const errors: number[] = [];
    for (let i = 0; i < count; i++) {
      const err = reprojectionErrorInSource(H, src[i], dst[i]);
      const isInlier = err < threshold;
      inliers.push(isInlier);
      if (isInlier) errors.push(err);
    }
    errors.sort((a, b) => a - b);
    const median = errors.length > 0 ? errors[Math.floor(errors.length / 2)] : Number.POSITIVE_INFINITY;
    return { inliers, inlierCount: errors.length, meanError: median };
  };

  /** 留出验证：错误解往往无法同时解释"未参与拟合"的点 */
  const holdoutScore = (H: Homography, sampleSet: Set<number>) => {
    let consistent = 0;
    let total = 0;
    for (let i = 0; i < count; i++) {
      if (sampleSet.has(i)) continue;
      total++;
      const err = reprojectionErrorInSource(H, src[i], dst[i]);
      if (err < threshold) consistent++;
    }
    return total > 0 ? consistent / total : 0;
  };

  /** 候选解 */
  interface Candidate {
    H: Homography;
    evaluation: ReturnType<typeof evaluate>;
    holdout: number;
  }

  const candidates: Candidate[] = [];

  // 阶段一：随机采样生成候选
  for (let iter = 0; iter < iterations; iter++) {
    for (let i = count - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      [indices[i], indices[j]] = [indices[j], indices[i]];
    }
    const sample = indices.slice(0, minSamples);
    const H = computeHomography(
      sample.map((i) => src[i]),
      sample.map((i) => dst[i]),
    );
    if (!H) continue;

    const evaluation = evaluate(H);
    if (evaluation.inlierCount < 4) continue;

    candidates.push({
      H,
      evaluation,
      holdout: holdoutScore(H, new Set(sample)),
    });
  }

  if (candidates.length === 0) return null;

  // 阶段二：对候选做局部优化，用内点重拟合直到稳定
  let best: Candidate | null = null;

  for (const candidate of candidates) {
    let currentH = candidate.H;
    let currentEval = candidate.evaluation;

    for (let round = 0; round < refineRounds; round++) {
      const inlierSrc = src.filter((_, i) => currentEval.inliers[i]);
      const inlierDst = dst.filter((_, i) => currentEval.inliers[i]);
      if (inlierSrc.length < 4) break;

      const refined = computeHomography(inlierSrc, inlierDst);
      if (!refined) break;

      const refinedEval = evaluate(refined);
      if (refinedEval.inlierCount < currentEval.inlierCount) break;

      const improved =
        refinedEval.inlierCount > currentEval.inlierCount ||
        refinedEval.meanError < currentEval.meanError;
      currentH = refined;
      currentEval = refinedEval;
      if (!improved) break;
    }

    const refinedCandidate: Candidate = {
      H: currentH,
      evaluation: currentEval,
      holdout: candidate.holdout,
    };

    // 判优顺序：内点数 → 留出验证一致率 → 内点误差中位数
    const isBetter =
      best === null ||
      currentEval.inlierCount > best.evaluation.inlierCount ||
      (currentEval.inlierCount === best.evaluation.inlierCount &&
        (refinedCandidate.holdout > best.holdout ||
          (refinedCandidate.holdout === best.holdout &&
            currentEval.meanError < best.evaluation.meanError)));

    if (isBetter) {
      best = refinedCandidate;
    }
  }

  if (!best) return null;
  return {
    H: best.H,
    inliers: best.evaluation.inliers,
    inlierCount: best.evaluation.inlierCount,
  };
}

/**
 * 计算单位距离（米/像素）：在图像原点处 100 像素对应的地面距离。
 */
export function computeUnitScale(H: Homography): { metersPerPixel: number } | null {
  const origin = applyHomography(H, { x: 0, y: 0 });
  const probe = applyHomography(H, { x: 100, y: 0 });
  if (!origin || !probe) return null;
  const dWorld = Math.hypot(probe.x - origin.x, probe.y - origin.y);
  if (dWorld < 1e-9) return null;
  return { metersPerPixel: dWorld / 100 };
}

/**
 * 计算图像中任意点处的单位距离（米/像素）。
 * 由于透视投影，尺度随位置变化，因此按点采样更准确。
 */
export function measureScaleAt(H: Homography, p: Point2D): { metersPerPixel: number } | null {
  const origin = applyHomography(H, p);
  const probe = applyHomography(H, { x: p.x + 100, y: p.y });
  if (!origin || !probe) return null;
  const dWorld = Math.hypot(probe.x - origin.x, probe.y - origin.y);
  if (dWorld < 1e-9) return null;
  return { metersPerPixel: dWorld / 100 };
}

export { solveLinearSystem };
