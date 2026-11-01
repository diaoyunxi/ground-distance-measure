/**
 * 标定与测量会话。
 *
 * 流程：
 *  1. 用户在地面放置一个已知尺寸的参考矩形（如 A4 纸 210×297mm）。
 *  2. 在画面上点选矩形的 4 个角 → DLT 解单应 → 得到"像素 ↔ 地面米"的映射，
 *     即该镜头下的单位距离（米/像素）。
 *  3. 用单应分解求出焦距、相机位姿与相机高度。
 *  4. 用户选择起点与终点 → 反投影到地面 → 得到直线距离。
 *  5. 移动过程中按帧跟踪地面特征 → 位姿估计 → 累计出摄像头移动的曲线距离，
 *     并实时计算距起点的距离。
 */

import {
  computeHomography,
  estimateHomographyRANSAC,
  measureScaleAt,
  type Point2D,
} from './homography';
import {
  backprojectToGround,
  decomposeHomography,
  estimateFocalLength,
  fieldOfView,
  groundDistance,
  intrinsicsFromImage,
  invertHomography,
  pathLength,
  projectToImage,
  type CameraPose,
  type Intrinsics,
} from './camera';

/** 标定参考矩形的真实尺寸（米） */
export interface ReferenceRect {
  /** 世界系 X 方向长度（米） */
  widthMeters: number;
  /** 世界系 Y 方向长度（米） */
  heightMeters: number;
  /** 名称，用于界面提示 */
  name: string;
}

/** 常用参考物 */
export const REFERENCE_RECTS: ReferenceRect[] = [
  { name: 'A4 纸 (210×297mm)', widthMeters: 0.21, heightMeters: 0.297 },
  { name: 'A5 纸 (148×210mm)', widthMeters: 0.148, heightMeters: 0.21 },
  { name: '身份证 (85.6×54mm)', widthMeters: 0.0856, heightMeters: 0.054 },
  { name: '1元纸币 (130×63mm)', widthMeters: 0.13, heightMeters: 0.063 },
  { name: '方格纸 10×10cm', widthMeters: 0.1, heightMeters: 0.1 },
];

/** 一次标定的结果 */
export interface Calibration {
  /** 图像像素 → 地面世界坐标 的单应 */
  H_imgToGround: number[];
  /** 地面世界坐标 → 图像像素 的单应 */
  H_groundToImg: number[];
  /** 解算出的内参（焦距可能为 null，表示该视角无法唯一确定焦距） */
  intrinsics: Intrinsics | null;
  /** 标定时刻的相机位姿（焦距不可解时为 null） */
  pose: CameraPose | null;
  /** 相机高度（米），不可解时为 null */
  cameraHeightMeters: number | null;
  /** 使用的参考矩形 */
  reference: ReferenceRect;
  /** 四个角点的重投影误差（像素 RMS），用于质量提示 */
  reprojectionError: number;
}

/**
 * 求解标定：4 个角点的图像坐标 → 地面世界坐标。
 * 世界系原点放在矩形角点0，X 沿 widthMeters，Y 沿 heightMeters。
 *
 * 重要：焦距并非总能解出。当参考矩形在画面中呈轴对齐矩形时
 * （即正对拍摄、透视消失），平面单应的焦距约束退化，
 * 此时只能完成"像素↔地面"映射（反投影与直线距离依然精确），
 * 但无法解出相机位姿（依赖焦距）。
 */
export function solveCalibration(
  imageCorners: Point2D[],
  reference: ReferenceRect,
  imageWidth: number,
  imageHeight: number,
): { calibration: Calibration; error: number } | null {
  if (imageCorners.length < 4) return null;

  const worldCorners: Point2D[] = [
    { x: 0, y: 0 },
    { x: reference.widthMeters, y: 0 },
    { x: reference.widthMeters, y: reference.heightMeters },
    { x: 0, y: reference.heightMeters },
  ];

  // 用全部 4 点做 DLT；RANSAC 只在点更多时有意义，这里点少直接用最小二乘
  const H = computeHomography(imageCorners, worldCorners);
  if (!H) return null;

  const H_groundToImg = invertHomography(H);
  if (!H_groundToImg) return null;

  // 计算重投影误差（像素 RMS）
  let sumSq = 0;
  for (let i = 0; i < 4; i++) {
    const projected = projectToImage(H_groundToImg, worldCorners[i]);
    if (!projected) return null;
    const dx = projected.x - imageCorners[i].x;
    const dy = projected.y - imageCorners[i].y;
    sumSq += dx * dx + dy * dy;
  }
  const rmsError = Math.sqrt(sumSq / 4);

  // 估计焦距（Zhang 闭式解，主点取图像中心）
  const principalX = imageWidth / 2;
  const principalY = imageHeight / 2;
  const focalLength = estimateFocalLength(H_groundToImg, principalX, principalY);

  let intrinsics: Intrinsics | null = null;
  let pose: CameraPose | null = null;
  let cameraHeightMeters: number | null = null;

  if (focalLength) {
    intrinsics = intrinsicsFromImage(focalLength, imageWidth, imageHeight);
    const decomposed = decomposeHomography(H_groundToImg, intrinsics);
    if (decomposed) {
      pose = decomposed.pose;
      cameraHeightMeters = decomposed.pose.center[2];
    } else {
      intrinsics = null;
    }
  }

  return {
    calibration: {
      H_imgToGround: H,
      H_groundToImg,
      intrinsics,
      pose,
      cameraHeightMeters,
      reference,
      reprojectionError: rmsError,
    },
    error: rmsError,
  };
}

/** 标定质量的评估等级 */
export function assessCalibrationQuality(rmsError: number): {
  level: 'excellent' | 'good' | 'fair' | 'poor';
  message: string;
} {
  if (rmsError < 2) {
    return { level: 'excellent', message: '标定质量优秀，测量精度高' };
  }
  if (rmsError < 5) {
    return { level: 'good', message: '标定质量良好，可正常测量' };
  }
  if (rmsError < 12) {
    return { level: 'fair', message: '标定质量一般，建议重新标定以提升精度' };
  }
  return { level: 'poor', message: '标定误差过大，请确保四角点贴合参考物边缘后重试' };
}

/**
 * 单位距离：给定图像位置，返回该处的地面尺度（米/像素）。
 * 这是"判定摄像头内对应的现实地面长度"的直接依据。
 */
export function unitDistanceAt(
  calibration: Calibration,
  pixel: Point2D,
): { metersPerPixel: number; pixelsPerMeter: number } | null {
  const scale = measureScaleAt(calibration.H_imgToGround, pixel);
  if (!scale) return null;
  return {
    metersPerPixel: scale.metersPerPixel,
    pixelsPerMeter: 1 / scale.metersPerPixel,
  };
}

/** 起止点的直线距离（米） */
export function straightLineDistance(
  calibration: Calibration,
  startPixel: Point2D,
  endPixel: Point2D,
): { meters: number; startGround: Point2D; endGround: Point2D } | null {
  const startGround = backprojectToGround(calibration.H_imgToGround, startPixel);
  const endGround = backprojectToGround(calibration.H_imgToGround, endPixel);
  if (!startGround || !endGround) return null;
  return {
    meters: groundDistance(startGround, endGround),
    startGround,
    endGround,
  };
}

/**
 * 位姿轨迹：由连续帧的地面单应推算相机在地面系中的位置。
 * 用于计算摄像头移动的曲线距离。
 */
export interface PoseTrack {
  /** 每帧的相机地面位置（米） */
  positions: Point2D[];
  /** 累计曲线距离（米） */
  curveDistance: number;
  /** 相机高度序列（米） */
  heights: number[];
}

/**
 * 依据两帧之间的地面单应增量，推算相机平移量。
 *
 * 地面平面场景下，两帧之间的地面映射关系为
 *   H_delta = H_new * H_old^-1  （地面 → 图像）
 * 若已知内参，可由 H_delta 分解出两帧之间的相对旋转与相对平移，
 * 从而累计得到相机在世界系中的轨迹。
 *
 * 这里采用简化的做法：直接用两帧各自的地面→图像单应分别分解位姿，
 * 取相机中心的差作为位移量。这样避免处理平面单应分解的双解歧义。
 */
/**
 * 依据连续帧的相机位姿，累加地面轨迹。
 *
 * @param previousPose 上一帧的相机位姿（用于计算帧间位移，校验轨迹连续性）
 * @param currentPose 当前帧的相机位姿
 * @param accumulator 已有轨迹
 */
export function accumulatePose(
  previousPose: CameraPose,
  currentPose: CameraPose,
  accumulator: PoseTrack,
): PoseTrack {
  const prev2D: Point2D = { x: previousPose.center[0], y: previousPose.center[1] };
  const curr2D: Point2D = { x: currentPose.center[0], y: currentPose.center[1] };

  // 帧间位移过小视为抖动，不计入轨迹，避免静止时里程漂移
  const step = groundDistance(prev2D, curr2D);
  if (step < 0.001) {
    return accumulator;
  }

  const positions = [...accumulator.positions, curr2D];
  const heights = [...accumulator.heights, currentPose.center[2]];

  return {
    positions,
    heights,
    curveDistance: pathLength(positions),
  };
}

/**
 * 从地面单应增量估计两帧之间的相对位移（米）。
 * 用于没有绝对位姿时的增量式里程计。
 */
export function estimateTranslation(
  previousH: number[],
  currentH: number[],
  intrinsics: Intrinsics,
): { dx: number; dy: number; meters: number } | null {
  const prev = decomposeHomography(previousH, intrinsics);
  const curr = decomposeHomography(currentH, intrinsics);
  if (!prev || !curr) return null;

  const dx = curr.pose.center[0] - prev.pose.center[0];
  const dy = curr.pose.center[1] - prev.pose.center[1];
  return { dx, dy, meters: Math.hypot(dx, dy) };
}

/**
 * 实时距起点的距离（米）。
 * 起点是地面上的一个点；当前位置由相机所在位置（或画面中心对应地面点）给出。
 */
export function liveDistanceFromStart(
  startGround: Point2D,
  currentGround: Point2D,
): number {
  return groundDistance(startGround, currentGround);
}

/**
 * 综合测量结果。
 */
export interface MeasurementResult {
  /** 起止点直线距离（米） */
  straightDistanceMeters: number;
  /** 摄像头移动曲线距离（米） */
  curveDistanceMeters: number;
  /** 曲直比：曲线距离 / 直线距离 */
  detourRatio: number;
  /** 起点地面坐标 */
  startGround: Point2D;
  /** 终点地面坐标 */
  endGround: Point2D;
  /** 相机轨迹 */
  track: Point2D[];
  /** 测量时刻 */
  timestamp: number;
}

/**
 * 汇总最终结果。
 */
export function buildMeasurementResult(
  calibration: Calibration,
  startPixel: Point2D,
  endPixel: Point2D,
  track: Point2D[],
): MeasurementResult | null {
  const straight = straightLineDistance(calibration, startPixel, endPixel);
  if (!straight) return null;

  const curve = pathLength(track);

  return {
    straightDistanceMeters: straight.meters,
    curveDistanceMeters: curve,
    detourRatio: straight.meters > 1e-6 ? curve / straight.meters : 0,
    startGround: straight.startGround,
    endGround: straight.endGround,
    track,
    timestamp: Date.now(),
  };
}

/**
 * 格式化距离显示：自动选择米 / 厘米 / 毫米。
 */
export function formatDistance(meters: number): string {
  if (!isFinite(meters)) return '--';
  if (meters >= 1) return `${meters.toFixed(3)} m`;
  if (meters >= 0.01) return `${(meters * 100).toFixed(1)} cm`;
  return `${(meters * 1000).toFixed(1)} mm`;
}

/**
 * 生成报告文本（便于复制分享）。
 */
export function formatReport(result: MeasurementResult, label = '地面测距报告'): string {
  const lines = [
    `【${label}】`,
    `起止点直线距离：${formatDistance(result.straightDistanceMeters)}`,
    `摄像头移动曲线距离：${formatDistance(result.curveDistanceMeters)}`,
    `绕行系数：${result.detourRatio.toFixed(3)}`,
    `起点坐标：(${result.startGround.x.toFixed(3)}, ${result.startGround.y.toFixed(3)}) m`,
    `终点坐标：(${result.endGround.x.toFixed(3)}, ${result.endGround.y.toFixed(3)}) m`,
    `轨迹点数：${result.track.length}`,
    `时间：${new Date(result.timestamp).toLocaleString('zh-CN')}`,
  ];
  return lines.join('\n');
}

export { estimateHomographyRANSAC, fieldOfView };
