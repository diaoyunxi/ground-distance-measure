import { describe, expect, it } from 'vitest';
import {
  applyHomography,
  computeHomography,
  computeUnitScale,
  estimateHomographyRANSAC,
  isCollinear,
  measureScaleAt,
  normalizePoints,
  type Point2D,
} from '../core/homography';
import { invertHomography } from '../core/camera';

/**
 * 构造一个已知的透视映射：地面矩形 → 图像四点。
 * 用于验证 DLT 是否能恢复真实单应。
 */
function makePerspectiveCase() {
  // 图像四角（模拟俯视地面的梯形透视）
  const imagePts: Point2D[] = [
    { x: 420, y: 180 },
    { x: 780, y: 210 },
    { x: 700, y: 560 },
    { x: 300, y: 520 },
  ];
  // 对应地面矩形（米）
  const groundPts: Point2D[] = [
    { x: 0, y: 0 },
    { x: 0.297, y: 0 },
    { x: 0.297, y: 0.21 },
    { x: 0, y: 0.21 },
  ];
  return { imagePts, groundPts };
}

describe('homography: 归一化', () => {
  it('归一化后质心接近原点', () => {
    const pts: Point2D[] = [
      { x: 100, y: 200 },
      { x: 400, y: 250 },
      { x: 300, y: 600 },
      { x: 50, y: 480 },
    ];
    const { normalized } = normalizePoints(pts);
    const cx = normalized.reduce((s, p) => s + p.x, 0) / normalized.length;
    const cy = normalized.reduce((s, p) => s + p.y, 0) / normalized.length;
    expect(cx).toBeCloseTo(0, 8);
    expect(cy).toBeCloseTo(0, 8);
  });

  it('变换与逆变换互为反函数', () => {
    const pts: Point2D[] = [
      { x: 100, y: 200 },
      { x: 400, y: 250 },
      { x: 300, y: 600 },
    ];
    const { transform, inverse } = normalizePoints(pts);
    const p = { x: 123, y: 456 };
    const fwd = applyHomography(transform, p)!;
    const back = applyHomography(inverse, fwd)!;
    expect(back.x).toBeCloseTo(p.x, 8);
    expect(back.y).toBeCloseTo(p.y, 8);
  });
});

describe('homography: DLT 求解', () => {
  it('点数不足时返回 null', () => {
    expect(computeHomography([{ x: 0, y: 0 }], [{ x: 0, y: 0 }])).toBeNull();
  });

  it('从 4 组对应点恢复透视映射', () => {
    const { imagePts, groundPts } = makePerspectiveCase();
    const H = computeHomography(imagePts, groundPts);
    expect(H).not.toBeNull();

    for (let i = 0; i < 4; i++) {
      const mapped = applyHomography(H!, imagePts[i])!;
      expect(mapped.x).toBeCloseTo(groundPts[i].x, 8);
      expect(mapped.y).toBeCloseTo(groundPts[i].y, 8);
    }
  });

  it('用多余点时仍能精确恢复', () => {
    const { imagePts, groundPts } = makePerspectiveCase();
    // 先由 4 点求出真实单应，再用它生成内部点（保证测试数据严格满足射影关系）
    const H = computeHomography(imagePts, groundPts)!;

    const extraImg: Point2D[] = [];
    const extraGnd: Point2D[] = [];
    for (const u of [0.25, 0.5, 0.75]) {
      for (const v of [0.25, 0.5, 0.75]) {
        const g = { x: u * 0.297, y: v * 0.21 };
        // 该地面点的真实像素位置 = H^-1 * g（H 的方向是 图像 → 地面）
        const inv = invertHomography(H)!;
        const p = applyHomography(inv, g)!;
        extraImg.push(p);
        extraGnd.push(g);
      }
    }
    const H2 = computeHomography([...imagePts, ...extraImg], [...groundPts, ...extraGnd]);
    expect(H2).not.toBeNull();
    for (let i = 0; i < 4; i++) {
      const mapped = applyHomography(H2!, imagePts[i])!;
      expect(mapped.x).toBeCloseTo(groundPts[i].x, 6);
      expect(mapped.y).toBeCloseTo(groundPts[i].y, 6);
    }
  });

  it('共线点判为退化并返回 null', () => {
    const src: Point2D[] = [
      { x: 0, y: 0 },
      { x: 10, y: 0 },
      { x: 20, y: 0 },
      { x: 30, y: 0 },
    ];
    const dst: Point2D[] = [
      { x: 0, y: 0 },
      { x: 10, y: 5 },
      { x: 20, y: 10 },
      { x: 30, y: 15 },
    ];
    expect(computeHomography(src, dst)).toBeNull();
    expect(isCollinear(src)).toBe(true);
  });

  it('非共线点集不被误判为退化', () => {
    expect(
      isCollinear([
        { x: 0, y: 0 },
        { x: 10, y: 0 },
        { x: 5, y: 10 },
      ]),
    ).toBe(false);
  });
});

describe('homography: RANSAC 鲁棒估计', () => {
  it('能剔除大部分错误匹配点', () => {
    const { imagePts, groundPts } = makePerspectiveCase();
    // H 的方向是 图像 → 地面，生成内点时需要反向映射（地面 → 图像）
    const H = computeHomography(imagePts, groundPts)!;
    const groundToImage = invertHomography(H)!;

    // 构造真实场景：16 个正确内点 + 6 个外点
    const src = [...imagePts];
    const dst = [...groundPts];
    for (let i = 0; i < 16; i++) {
      const g = { x: (i % 4) * 0.099, y: Math.floor(i / 4) * 0.0525 };
      src.push(applyHomography(groundToImage, g)!);
      dst.push(g);
    }
    const realCount = src.length;

    const outliers: Point2D[] = [
      { x: 120, y: 640 },
      { x: 640, y: 120 },
      { x: 900, y: 900 },
      { x: 50, y: 50 },
      { x: 1000, y: 300 },
      { x: 300, y: 1000 },
    ];
    for (const o of outliers) {
      src.push(o);
      dst.push({ x: 50, y: 50 });
    }

    let seed = 999;
    const rng = () => {
      seed = (seed * 1103515245 + 12345) & 0x7fffffff;
      return seed / 0x7fffffff;
    };

    const result = estimateHomographyRANSAC(src, dst, { threshold: 2, iterations: 500, rng });
    expect(result).not.toBeNull();

    // 绝大多数真实内点应被识别
    let detected = 0;
    for (let i = 0; i < realCount; i++) {
      if (result!.inliers[i]) detected++;
    }
    expect(detected).toBeGreaterThanOrEqual(realCount - 2);

    // 外点大多数应被剔除
    let falsePositives = 0;
    for (let i = realCount; i < src.length; i++) {
      if (result!.inliers[i]) falsePositives++;
    }
    expect(falsePositives).toBeLessThanOrEqual(1);
  });

  it('全匹配时所有点均为内点', () => {
    const { imagePts, groundPts } = makePerspectiveCase();
    const H = computeHomography(imagePts, groundPts)!;
    const groundToImage = invertHomography(H)!;

    const src = [...imagePts];
    const dst = [...groundPts];
    for (let i = 0; i < 12; i++) {
      const g = { x: (i % 4) * 0.099, y: Math.floor(i / 4) * 0.0525 };
      src.push(applyHomography(groundToImage, g)!);
      dst.push(g);
    }

    let seed = 4242;
    const rng = () => {
      seed = (seed * 1103515245 + 12345) & 0x7fffffff;
      return seed / 0x7fffffff;
    };

    const result = estimateHomographyRANSAC(src, dst, { threshold: 2, iterations: 200, rng });
    expect(result).not.toBeNull();
    expect(result!.inlierCount).toBe(src.length);
  });

  it('匹配点不足 4 个时返回 null', () => {
    const result = estimateHomographyRANSAC(
      [{ x: 0, y: 0 }, { x: 1, y: 1 }],
      [{ x: 0, y: 0 }, { x: 1, y: 1 }],
    );
    expect(result).toBeNull();
  });
});

describe('homography: 单位距离', () => {
  it('单位距离反映真实尺度比例', () => {
    // 构造：图像 100 像素对应地面 0.297 米（即宽度方向）
    const imagePts: Point2D[] = [
      { x: 0, y: 0 },
      { x: 100, y: 0 },
      { x: 100, y: 70 },
      { x: 0, y: 70 },
    ];
    const groundPts: Point2D[] = [
      { x: 0, y: 0 },
      { x: 0.297, y: 0 },
      { x: 0.297, y: 0.21 },
      { x: 0, y: 0.21 },
    ];
    const H = computeHomography(imagePts, groundPts)!;
    const scale = computeUnitScale(H);
    expect(scale).not.toBeNull();
    expect(scale!.metersPerPixel).toBeCloseTo(0.00297, 6);
  });

  it('可在任意位置测量局部尺度', () => {
    const { imagePts, groundPts } = makePerspectiveCase();
    const H = computeHomography(imagePts, groundPts)!;
    const scale = measureScaleAt(H, { x: 500, y: 300 });
    expect(scale).not.toBeNull();
    expect(scale!.metersPerPixel).toBeGreaterThan(0);
  });
});
