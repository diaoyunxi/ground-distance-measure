import { describe, expect, it } from 'vitest';
import {
  mat3Apply,
  mat3Det,
  mat3Inverse,
  mat3Mul,
  mat3Transpose,
  solveLinearSystem,
  symmetricEigenDecomposition,
  vec3Cross,
  vec3Dot,
  vec3Normalize,
  type Vec3,
} from '../core/linalg';

describe('linalg: 3x3 矩阵运算', () => {
  it('单位矩阵乘任意矩阵等于自身', () => {
    const identity = [1, 0, 0, 0, 1, 0, 0, 0, 1];
    const m = [1, 2, 3, 4, 5, 6, 7, 8, 10];
    expect(mat3Mul(identity, m)).toEqual(m);
    expect(mat3Mul(m, identity)).toEqual(m);
  });

  it('逆矩阵与原矩阵相乘为单位矩阵', () => {
    const m = [4, 7, 2, 3, 6, 1, 9, 8, 5];
    const inv = mat3Inverse(m);
    expect(inv).not.toBeNull();
    const product = mat3Mul(m, inv!);
    const expected = [1, 0, 0, 0, 1, 0, 0, 0, 1];
    for (let i = 0; i < 9; i++) {
      expect(product[i]).toBeCloseTo(expected[i], 10);
    }
  });

  it('奇异矩阵求逆返回 null', () => {
    const singular = [1, 2, 3, 2, 4, 6, 3, 6, 9];
    expect(mat3Inverse(singular)).toBeNull();
  });

  it('行列式计算正确', () => {
    expect(mat3Det([1, 0, 0, 0, 1, 0, 0, 0, 1])).toBeCloseTo(1, 12);
    expect(mat3Det([2, 0, 0, 0, 3, 0, 0, 0, 4])).toBeCloseTo(24, 12);
    expect(mat3Det([1, 2, 3, 4, 5, 6, 7, 8, 10])).toBeCloseTo(-3, 12);
  });

  it('转置是逆操作', () => {
    const m = [1, 2, 3, 4, 5, 6, 7, 8, 10];
    const t = mat3Transpose(m);
    expect(mat3Transpose(t)).toEqual(m);
    expect(t[1]).toBe(m[3]);
  });

  it('矩阵作用于齐次坐标', () => {
    const m = [2, 0, 1, 0, 3, 2, 0, 0, 1];
    const v: Vec3 = [1, 1, 1];
    const r = mat3Apply(m, v);
    expect(r[0]).toBeCloseTo(3, 12);
    expect(r[1]).toBeCloseTo(5, 12);
    expect(r[2]).toBeCloseTo(1, 12);
  });
});

describe('linalg: 向量运算', () => {
  it('叉积方向符合右手系', () => {
    const x: Vec3 = [1, 0, 0];
    const y: Vec3 = [0, 1, 0];
    const z = vec3Cross(x, y);
    expect(z[0]).toBeCloseTo(0, 12);
    expect(z[1]).toBeCloseTo(0, 12);
    expect(z[2]).toBeCloseTo(1, 12);
  });

  it('点积计算正确', () => {
    expect(vec3Dot([1, 2, 3], [4, 5, 6])).toBeCloseTo(32, 12);
  });

  it('归一化后模长为 1', () => {
    const n = vec3Normalize([3, 4, 0]);
    expect(Math.hypot(n[0], n[1], n[2])).toBeCloseTo(1, 12);
  });

  it('零向量归一化不产生 NaN', () => {
    const n = vec3Normalize([0, 0, 0]);
    expect(n).toEqual([0, 0, 0]);
  });
});

describe('linalg: 线性方程组求解', () => {
  it('求解唯一解', () => {
    const A = [
      [2, 1, -1],
      [-3, -1, 2],
      [-2, 1, 2],
    ];
    const b = [8, -11, -3];
    const x = solveLinearSystem(A, b);
    expect(x).not.toBeNull();
    expect(x![0]).toBeCloseTo(2, 10);
    expect(x![1]).toBeCloseTo(3, 10);
    expect(x![2]).toBeCloseTo(-1, 10);
  });

  it('奇异矩阵返回 null', () => {
    const A = [
      [1, 2, 3],
      [2, 4, 6],
      [1, 1, 1],
    ];
    expect(solveLinearSystem(A, [1, 2, 3])).toBeNull();
  });
});

describe('linalg: 对称矩阵特征分解', () => {
  it('分解后可用特征值重构原矩阵', () => {
    const A = [
      [4, 1, 2],
      [1, 3, 0],
      [2, 0, 5],
    ];
    const { values, vectors } = symmetricEigenDecomposition(A);
    expect(values).toHaveLength(3);
    // 按降序排列
    expect(values[0]).toBeGreaterThanOrEqual(values[1]);
    expect(values[1]).toBeGreaterThanOrEqual(values[2]);

    // 重建 A = V diag(values) V^T
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        let sum = 0;
        for (let k = 0; k < 3; k++) {
          sum += vectors[k][i] * values[k] * vectors[k][j];
        }
        expect(sum).toBeCloseTo(A[i][j], 8);
      }
    }
  });

  it('对角矩阵的特征值就是对角元', () => {
    const A = [
      [5, 0, 0],
      [0, 2, 0],
      [0, 0, 9],
    ];
    const { values } = symmetricEigenDecomposition(A);
    expect([...values].sort((a, b) => a - b)).toEqual([2, 5, 9]);
  });
});
