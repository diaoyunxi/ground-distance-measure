import { describe, expect, it } from 'vitest';
import {
  REFERENCE_RECTS,
  assessCalibrationQuality,
  buildMeasurementResult,
  formatDistance,
  formatReport,
  liveDistanceFromStart,
  solveCalibration,
  straightLineDistance,
  unitDistanceAt,
  type ReferenceRect,
} from '../core/calibration';
import { applyHomography, type Point2D } from '../core/homography';

/**
 * 构造一个合成相机，用于生成标定数据。
 * 世界系：x 右，y 前，z 上；相机在 (0,0,1.5) 俯视地面。
 */
function createSyntheticCamera() {
  const focalLength = 800;
  const imageWidth = 1080;
  const imageHeight = 1920;
  const pitch = 0.6;
  const cp = Math.cos(pitch);
  const sp = Math.sin(pitch);
  const right = [1, 0, 0];
  const down = [0, sp, cp];
  const forward = [0, cp, -sp];
  const center: [number, number, number] = [0, 0, 1.5];

  const projectGround = (x: number, y: number): Point2D => {
    const d: [number, number, number] = [x, y, -center[2]];
    const xc = right[0] * d[0] + right[1] * d[1] + right[2] * d[2];
    const yc = down[0] * d[0] + down[1] * d[1] + down[2] * d[2];
    const zc = forward[0] * d[0] + forward[1] * d[1] + forward[2] * d[2];
    return {
      x: (focalLength * xc) / zc + imageWidth / 2,
      y: (focalLength * yc) / zc + imageHeight / 2,
    };
  };

  return { focalLength, imageWidth, imageHeight, center, projectGround };
}

/** 生成参考矩形的四角图像坐标 */
function rectangleCorners(
  cam: ReturnType<typeof createSyntheticCamera>,
  reference: ReferenceRect,
  origin: { x: number; y: number } = { x: -0.15, y: 0.7 },
): Point2D[] {
  return [
    cam.projectGround(origin.x, origin.y),
    cam.projectGround(origin.x + reference.widthMeters, origin.y),
    cam.projectGround(origin.x + reference.widthMeters, origin.y + reference.heightMeters),
    cam.projectGround(origin.x, origin.y + reference.heightMeters),
  ];
}

describe('calibration: 标定求解', () => {
  it('从 4 个角点解出标定结果', () => {
    const cam = createSyntheticCamera();
    const reference = REFERENCE_RECTS[0]; // A4
    const corners = rectangleCorners(cam, reference);

    const result = solveCalibration(corners, reference, cam.imageWidth, cam.imageHeight);
    expect(result).not.toBeNull();
    expect(result!.calibration.intrinsics!.focalLength).toBeGreaterThan(0);
    // 重投影误差应极小
    expect(result!.error).toBeLessThan(0.5);
  });

  it('相机高度解算准确', () => {
    const cam = createSyntheticCamera();
    const reference = REFERENCE_RECTS[0];
    const corners = rectangleCorners(cam, reference);

    const result = solveCalibration(corners, reference, cam.imageWidth, cam.imageHeight)!;
    expect(result.calibration.cameraHeightMeters).not.toBeNull();
    expect(Math.abs(result.calibration.cameraHeightMeters! - cam.center[2])).toBeLessThan(0.1);
  });

  it('焦距解算接近真实值', () => {
    const cam = createSyntheticCamera();
    const reference = REFERENCE_RECTS[0];
    const corners = rectangleCorners(cam, reference);

    const result = solveCalibration(corners, reference, cam.imageWidth, cam.imageHeight)!;
    expect(result.calibration.intrinsics).not.toBeNull();
    const relError =
      Math.abs(result.calibration.intrinsics!.focalLength - cam.focalLength) / cam.focalLength;
    expect(relError).toBeLessThan(0.05);
  });

  it('轴对齐视角下焦距不可解，但尺度映射仍可用', () => {
    // 参考矩形在画面中呈轴对齐矩形（正对拍摄、透视消失），
    // 此时焦距约束退化，但"像素↔地面"映射依然精确，
    // 因此直线距离测量必须仍然可用。
    const reference = REFERENCE_RECTS[0];
    const corners: Point2D[] = [
      { x: 575, y: 306 },
      { x: 1340, y: 306 },
      { x: 1340, y: 652 },
      { x: 575, y: 652 },
    ];

    const result = solveCalibration(corners, reference, 1920, 1080);
    expect(result).not.toBeNull();
    expect(result!.calibration.intrinsics).toBeNull();
    expect(result!.calibration.pose).toBeNull();

    // 直线距离依然精确
    const H = result!.calibration.H_imgToGround;
    const p1 = applyHomography(H, { x: 575, y: 306 })!;
    const p2 = applyHomography(H, { x: 1340, y: 306 })!;
    expect(p1.x).toBeCloseTo(0, 6);
    expect(p1.y).toBeCloseTo(0, 6);
    expect(p2.x).toBeCloseTo(reference.widthMeters, 6);
    expect(Math.hypot(p2.x - p1.x, p2.y - p1.y)).toBeCloseTo(reference.widthMeters, 6);
  });

  it('角点数量不足时返回 null', () => {
    const cam = createSyntheticCamera();
    const result = solveCalibration(
      [cam.projectGround(0, 1)],
      REFERENCE_RECTS[0],
      cam.imageWidth,
      cam.imageHeight,
    );
    expect(result).toBeNull();
  });

  it('角点共线时判定为退化', () => {
    const cam = createSyntheticCamera();
    const collinear: Point2D[] = [
      cam.projectGround(-0.2, 0.7),
      cam.projectGround(0, 0.7),
      cam.projectGround(0.2, 0.7),
      cam.projectGround(0.4, 0.7),
    ];
    const result = solveCalibration(collinear, REFERENCE_RECTS[0], cam.imageWidth, cam.imageHeight);
    expect(result).toBeNull();
  });
});

describe('calibration: 单位距离', () => {
  it('单位距离反映真实比例', () => {
    const cam = createSyntheticCamera();
    const reference = REFERENCE_RECTS[0];
    const corners = rectangleCorners(cam, reference);
    const calibration = solveCalibration(corners, reference, cam.imageWidth, cam.imageHeight)!
      .calibration;

    // 参考矩形中心附近的单位距离
    const centerPixel = corners[0];
    const scale = unitDistanceAt(calibration, centerPixel);
    expect(scale).not.toBeNull();

    // 校验：100 像素对应的地面距离，应与米/像素 × 100 一致
    const p1 = applyHomography(calibration.H_imgToGround, centerPixel)!;
    const p2 = applyHomography(calibration.H_imgToGround, {
      x: centerPixel.x + 100,
      y: centerPixel.y,
    })!;
    const groundDelta = Math.hypot(p2.x - p1.x, p2.y - p1.y);
    expect(groundDelta).toBeCloseTo(scale!.metersPerPixel * 100, 5);
  });

  it('远处单位距离更大（透视缩小效应）', () => {
    const cam = createSyntheticCamera();
    const reference = REFERENCE_RECTS[0];
    const corners = rectangleCorners(cam, reference);
    const calibration = solveCalibration(corners, reference, cam.imageWidth, cam.imageHeight)!
      .calibration;

    // 该俯视构型下，像素 y 越大对应地面越远：
    // y=1100 处地面约 2.6m，y=1300 处约 6.8m。
    // 远处同样的像素数覆盖更大的地面范围，故米/像素更大。
    const mid = unitDistanceAt(calibration, { x: 540, y: 1100 })!;
    const far = unitDistanceAt(calibration, { x: 540, y: 1300 })!;
    expect(far.metersPerPixel).toBeGreaterThan(mid.metersPerPixel);
  });
});

describe('calibration: 直线距离测量', () => {
  it('测得的距离接近真实距离', () => {
    const cam = createSyntheticCamera();
    const reference = REFERENCE_RECTS[0];
    const corners = rectangleCorners(cam, reference);
    const calibration = solveCalibration(corners, reference, cam.imageWidth, cam.imageHeight)!
      .calibration;

    // 地面上相隔 1 米的两个点
    const a = cam.projectGround(0, 1.0);
    const b = cam.projectGround(0, 2.0);
    const result = straightLineDistance(calibration, a, b);

    expect(result).not.toBeNull();
    expect(result!.meters).toBeCloseTo(1.0, 2);
  });

  it('同一点距离为 0', () => {
    const cam = createSyntheticCamera();
    const reference = REFERENCE_RECTS[0];
    const corners = rectangleCorners(cam, reference);
    const calibration = solveCalibration(corners, reference, cam.imageWidth, cam.imageHeight)!
      .calibration;

    const p = cam.projectGround(0, 1.0);
    const result = straightLineDistance(calibration, p, p)!;
    expect(result.meters).toBeCloseTo(0, 6);
  });
});

describe('calibration: 实时距起点距离', () => {
  it('随移动单调变化', () => {
    const start = { x: 0, y: 0 };
    const d1 = liveDistanceFromStart(start, { x: 0, y: 0.5 });
    const d2 = liveDistanceFromStart(start, { x: 0, y: 1.0 });
    expect(d1).toBeCloseTo(0.5, 12);
    expect(d2).toBeCloseTo(1.0, 12);
    expect(d2).toBeGreaterThan(d1);
  });
});

describe('calibration: 结果汇总', () => {
  it('直线距离与曲线距离同时输出', () => {
    const cam = createSyntheticCamera();
    const reference = REFERENCE_RECTS[0];
    const corners = rectangleCorners(cam, reference);
    const calibration = solveCalibration(corners, reference, cam.imageWidth, cam.imageHeight)!
      .calibration;

    const startPixel = cam.projectGround(0, 1.0);
    const endPixel = cam.projectGround(0, 2.0);

    // 模拟一条弯曲的相机轨迹
    const track: Point2D[] = [
      { x: 0, y: 0 },
      { x: 0.3, y: 0.4 },
      { x: 0.2, y: 0.8 },
      { x: 0, y: 1.0 },
    ];

    const result = buildMeasurementResult(calibration, startPixel, endPixel, track);
    expect(result).not.toBeNull();
    expect(result!.straightDistanceMeters).toBeCloseTo(1.0, 2);
    expect(result!.curveDistanceMeters).toBeGreaterThan(result!.straightDistanceMeters);
    expect(result!.detourRatio).toBeGreaterThan(1);
    expect(result!.track).toHaveLength(4);
  });

  it('报告文本包含关键字段', () => {
    const cam = createSyntheticCamera();
    const reference = REFERENCE_RECTS[0];
    const corners = rectangleCorners(cam, reference);
    const calibration = solveCalibration(corners, reference, cam.imageWidth, cam.imageHeight)!
      .calibration;

    const result = buildMeasurementResult(
      calibration,
      cam.projectGround(0, 1.0),
      cam.projectGround(0, 2.0),
      [{ x: 0, y: 0 }, { x: 0, y: 1 }],
    )!;

    const report = formatReport(result);
    expect(report).toContain('直线距离');
    expect(report).toContain('曲线距离');
    expect(report).toContain('绕行系数');
  });
});

describe('calibration: 格式化与质量评估', () => {
  it('距离格式化按量级切换单位', () => {
    expect(formatDistance(2.5)).toContain('m');
    expect(formatDistance(0.25)).toContain('cm');
    expect(formatDistance(0.005)).toContain('mm');
    expect(formatDistance(Number.NaN)).toBe('--');
  });

  it('标定质量分级正确', () => {
    expect(assessCalibrationQuality(1).level).toBe('excellent');
    expect(assessCalibrationQuality(3).level).toBe('good');
    expect(assessCalibrationQuality(8).level).toBe('fair');
    expect(assessCalibrationQuality(50).level).toBe('poor');
  });

  it('预置参考物尺寸合理', () => {
    expect(REFERENCE_RECTS.length).toBeGreaterThan(0);
    for (const rect of REFERENCE_RECTS) {
      expect(rect.widthMeters).toBeGreaterThan(0);
      expect(rect.heightMeters).toBeGreaterThan(0);
      expect(rect.name.length).toBeGreaterThan(0);
    }
  });
});

describe('calibration: 单应方向一致性', () => {
  it('H_imgToGround 与 H_groundToImg 互为逆', () => {
    const cam = createSyntheticCamera();
    const reference = REFERENCE_RECTS[0];
    const corners = rectangleCorners(cam, reference);
    const calibration = solveCalibration(corners, reference, cam.imageWidth, cam.imageHeight)!
      .calibration;

    const point = { x: 540, y: 800 };
    const ground = applyHomography(calibration.H_imgToGround, point)!;
    const backToImage = applyHomography(calibration.H_groundToImg, ground)!;

    expect(backToImage.x).toBeCloseTo(point.x, 5);
    expect(backToImage.y).toBeCloseTo(point.y, 5);
  });
});
