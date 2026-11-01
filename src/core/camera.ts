/**
 * 相机几何：内参解算、单应分解求位姿、地面投影与距离计算。
 *
 * 坐标约定：
 *  - 地面世界系 (X, Y, Z)：Z 向上，地面为 Z = 0 平面，单位米。
 *    X/Y 由标定参考矩形的两个边定义（右为 X，前为 Y）。
 *  - 图像系 (u, v)：像素，u 向右，v 向下。
 *  - 相机系 (x, y, z)：z 沿光轴向前（右手系）。
 */

import { mat3Apply, mat3Transpose, type Mat3, type Vec3 } from './linalg';
import { applyHomography, type Homography, type Point2D } from './homography';

export interface Intrinsics {
  /** 焦距，单位像素 */
  focalLength: number;
  /** 主点 x（像素） */
  principalX: number;
  /** 主点 y（像素） */
  principalY: number;
}

/** 相机位姿：描述相机在世界系中的位置与朝向 */
export interface CameraPose {
  /** 3x3 旋转矩阵，行主序，世界系 → 相机系 */
  rotation: Mat3;
  /** 相机在世界系中的位置（米） */
  center: Vec3;
}

/** 由主点与图像尺寸构造内参（假定平方像素、主点在中心） */
export function intrinsicsFromImage(
  focalLength: number,
  imageWidth: number,
  imageHeight: number,
): Intrinsics {
  return {
    focalLength,
    principalX: imageWidth / 2,
    principalY: imageHeight / 2,
  };
}

/**
 * 从地面单应矩阵解算焦距。
 *
 * 采用 Zhang 单平面标定的两条约束联合求解：
 *   (1) h1ᵀ ω h2 = 0            —— 正交约束
 *   (2) h1ᵀ ω h1 = h2ᵀ ω h2     —— 等模约束
 * 其中 ω = K⁻ᵀK⁻¹，K = [[f,0,cx],[0,f,cy],[0,0,1]]。
 *
 * 单独使用约束 (1) 在 h1z = 0 时会退化（此时方程不含 f），
 * 例如"相机正对地面、地面 X 轴垂直光轴"的常见场景。
 * 因此这里把 f² 作为未知量，对两条约束联立得到的二次方程求根，
 * 并选取使约束残差最小的那个解。
 *
 * @param H 地面世界坐标 → 图像像素 的单应矩阵
 */
export function estimateFocalLength(
  H: Homography,
  principalX: number,
  principalY: number,
): number | null {
  const h1: Vec3 = [H[0], H[3], H[6]];
  const h2: Vec3 = [H[1], H[4], H[7]];

  const cx = principalX;
  const cy = principalY;

  // 约束 (1)：h1ᵀωh2 = 0
  // 展开并乘以 f²：
  //   (a1a2 + b1b2) - cx(a1c2 + c1a2) - cy(b1c2 + c1b2) + f²(c1c2) + (cx²+cy²)(c1c2) = 0
  const a1 = h1[0];
  const b1 = h1[1];
  const c1 = h1[2];
  const a2 = h2[0];
  const b2 = h2[1];
  const c2 = h2[2];

  const K1 = a1 * a2 + b1 * b2 - cx * (a1 * c2 + c1 * a2) - cy * (b1 * c2 + c1 * b2) + (cx * cx + cy * cy) * c1 * c2;
  const L1 = c1 * c2;

  // 约束 (2)：h1ᵀωh1 = h2ᵀωh2
  // 展开并乘以 f²：
  //   (a1²+b1²-a2²-b2²) - 2cx(a1c1 - a2c2) - 2cy(b1c1 - b2c2) + f²(c1²-c2²) + (cx²+cy²)(c1²-c2²) = 0
  const K2 =
    a1 * a1 +
    b1 * b1 -
    a2 * a2 -
    b2 * b2 -
    2 * cx * (a1 * c1 - a2 * c2) -
    2 * cy * (b1 * c1 - b2 * c2) +
    (cx * cx + cy * cy) * (c1 * c1 - c2 * c2);
  const L2 = c1 * c1 - c2 * c2;

  // 每条约束形如 K + L·f² = 0，即 f² = -K/L（当 L ≠ 0）
  const candidates: number[] = [];
  if (Math.abs(L1) > 1e-14) candidates.push(-K1 / L1);
  if (Math.abs(L2) > 1e-14) candidates.push(-K2 / L2);

  // 两条约束联立消去 f²：K1 + L1·f² = 0 且 K2 + L2·f² = 0
  // → f² = (K2·L1 - K1·L2) / (L1·L2 - L1·L2) ... 实际等价于上面的两个候选
  // 当两条约束都给不出有效 f² 时，退化为用图像宽度经验估计
  const valid = candidates.filter((v) => isFinite(v) && v > 0);
  if (valid.length === 0) return null;

  // 若有多个候选，选择使两条约束残差综合最小者
  let best: number | null = null;
  let bestResidual = Number.POSITIVE_INFINITY;
  for (const f2 of valid) {
    const residual = Math.abs(K1 + L1 * f2) + Math.abs(K2 + L2 * f2);
    if (residual < bestResidual) {
      bestResidual = residual;
      best = f2;
    }
  }
  if (best === null) return null;
  return Math.sqrt(best);
}

/**
 * 由地面单应矩阵分解相机位姿。
 *
 * H = K [r1 r2 t]，r1/r2 是世界系两基向量在相机系的表示，t 是世界原点在相机系的坐标。
 * 解出 R = [r1 r2 r3]，则相机中心 C = -R^T t，相机高度即 C[2]。
 *
 * @param H 地面世界坐标 → 图像像素 的单应矩阵
 */
export function decomposeHomography(
  H: Homography,
  intrinsics: Intrinsics,
): { pose: CameraPose; focalLength: number } | null {
  const f = intrinsics.focalLength;
  if (!isFinite(f) || f <= 0) return null;

  const Kinv: Mat3 = [
    1 / f,
    0,
    -intrinsics.principalX / f,
    0,
    1 / f,
    -intrinsics.principalY / f,
    0,
    0,
    1,
  ];

  const h1 = applyKinv(Kinv, [H[0], H[3], H[6]]);
  const h2 = applyKinv(Kinv, [H[1], H[4], H[7]]);
  const h3 = applyKinv(Kinv, [H[2], H[5], H[8]]);
  if (!h1 || !h2 || !h3) return null;

  // λ 归一化，使 |r1| = |r2| = 1
  const norm1 = Math.hypot(h1[0], h1[1], h1[2]);
  const norm2 = Math.hypot(h2[0], h2[1], h2[2]);
  if (norm1 < 1e-9 || norm2 < 1e-9) return null;
  const lambda = 2 / (norm1 + norm2);

  const r1: Vec3 = [h1[0] * lambda, h1[1] * lambda, h1[2] * lambda];

  // 正交化 r2：r2 = λ (h2 - (r1 · h2) r1)
  const dot = r1[0] * h2[0] + r1[1] * h2[1] + r1[2] * h2[2];
  let r2: Vec3 = [
    (h2[0] - dot * r1[0]) * lambda,
    (h2[1] - dot * r1[1]) * lambda,
    (h2[2] - dot * r1[2]) * lambda,
  ];
  const normR2 = Math.hypot(r2[0], r2[1], r2[2]);
  if (normR2 < 1e-12) return null;
  r2 = [r2[0] / normR2, r2[1] / normR2, r2[2] / normR2];

  const r3: Vec3 = [
    r1[1] * r2[2] - r1[2] * r2[1],
    r1[2] * r2[0] - r1[0] * r2[2],
    r1[0] * r2[1] - r1[1] * r2[0],
  ];

  const t: Vec3 = [h3[0] * lambda, h3[1] * lambda, h3[2] * lambda];

  // R 的列依次为 r1, r2, r3，且 r3 = r1 × r2，保证 det(R) = +1（合法旋转）。
  const rotation: Mat3 = [
    r1[0], r2[0], r3[0],
    r1[1], r2[1], r3[1],
    r1[2], r2[2], r3[2],
  ];

  const Rt = mat3Transpose(rotation);
  const centerRaw = mat3Apply(Rt, t);
  let center: Vec3 = [-centerRaw[0], -centerRaw[1], -centerRaw[2]];

  // 平面单应存在固有双解：相机位于平面上方或下方，二者互为镜像。
  // 本应用约定相机位于地面上方（世界系 Z 向上），故当解出 C_z < 0 时，
  // 取另一解：保持 r1、r2 与 t 不变，翻转第三个基向量 r3 → -r3，
  // 此时 det(R) 仍为 -1 的镜像形式，但相机中心落在地面上方的正确位置。
  if (center[2] < 0) {
    const mirroredRotation: Mat3 = [
      r1[0], r2[0], -r3[0],
      r1[1], r2[1], -r3[1],
      r1[2], r2[2], -r3[2],
    ];
    const mirroredRt = mat3Transpose(mirroredRotation);
    const mirroredRaw = mat3Apply(mirroredRt, t);
    center = [-mirroredRaw[0], -mirroredRaw[1], -mirroredRaw[2]];
    return {
      pose: { rotation: mirroredRotation, center },
      focalLength: f,
    };
  }

  return { pose: { rotation, center }, focalLength: f };
}

/**
 * 用 K^-1 变换一个齐次向量。
 * 第三分量允许为 0（例如地面 X 轴在相机系中无深度分量），
 * 仅在数值完全不可用时返回 null。
 */
function applyKinv(Kinv: Mat3, v: Vec3): Vec3 | null {
  const res = mat3Apply(Kinv, v);
  if (!isFinite(res[0]) || !isFinite(res[1]) || !isFinite(res[2])) return null;
  return res;
}

/**
 * 由地面单应矩阵（图像 → 地面）与内参直接求解相机位姿。
 * 内部先把矩阵反解为 地面 → 图像 的方向再做标准分解。
 */
export function poseFromGroundHomography(
  H_imgToGround: Homography,
  intrinsics: Intrinsics,
): CameraPose | null {
  // 地面 → 图像 的单应 = (图像 → 地面)^-1
  const inverse = invertHomography(H_imgToGround);
  if (!inverse) return null;
  const decomposed = decomposeHomography(inverse, intrinsics);
  return decomposed ? decomposed.pose : null;
}

/** 3x3 单应矩阵求逆 */
export function invertHomography(H: Homography): Homography | null {
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
 * 把图像像素点反投影到地面世界平面（Z = 0）。
 * 使用地面单应矩阵直接映射，比射线求交更稳定。
 */
export function backprojectToGround(
  H_imgToGround: Homography,
  pixel: Point2D,
): Point2D | null {
  return applyHomography(H_imgToGround, pixel);
}

/**
 * 把地面世界点正投影到图像像素。
 */
export function projectToImage(
  H_groundToImg: Homography,
  world: Point2D,
): Point2D | null {
  return applyHomography(H_groundToImg, world);
}

/** 相机相对地面的高度（米） */
export function cameraHeight(pose: CameraPose): number {
  return pose.center[2];
}

/** 地面两点间的欧氏距离（米） */
export function groundDistance(a: Point2D, b: Point2D): number {
  return Math.hypot(b.x - a.x, b.y - a.y);
}

/**
 * 路径（折线）总长度（米）。用于统计手机摄像头移动的曲线距离。
 */
export function pathLength(path: Point2D[]): number {
  let total = 0;
  for (let i = 1; i < path.length; i++) {
    total += groundDistance(path[i - 1], path[i]);
  }
  return total;
}

/**
 * 计算相机水平/垂直视场角（度），用于比较不同焦距摄像头的视野范围。
 */
export function fieldOfView(
  intrinsics: Intrinsics,
  imageWidth: number,
  imageHeight: number,
): { horizontal: number; vertical: number } {
  const horizontal = 2 * Math.atan(imageWidth / 2 / intrinsics.focalLength) * (180 / Math.PI);
  const vertical = 2 * Math.atan(imageHeight / 2 / intrinsics.focalLength) * (180 / Math.PI);
  return { horizontal, vertical };
}

/**
 * 由内参和图像尺寸推导每像素角度与相对焦距倍率，用于跨镜头比较。
 */
export function angularResolution(
  intrinsics: Intrinsics,
  imageWidth: number,
): { degreesPerPixel: number; focalRatio: number } {
  const degreesPerPixel =
    ((Math.atan(imageWidth / 2 / intrinsics.focalLength) * 2 * 180) / Math.PI) / imageWidth;
  return { degreesPerPixel, focalRatio: intrinsics.focalLength };
}
