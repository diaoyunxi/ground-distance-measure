/**
 * 基础线性代数工具：3x3 矩阵、向量、线性方程组求解、对称矩阵 Jacobi 特征分解。
 * 全部为纯函数，无外部依赖，便于单元测试。
 */

export type Vec2 = [number, number];
export type Vec3 = [number, number, number];

/** 行主序 3x3 矩阵： [m00,m01,m02, m10,m11,m12, m20,m21,m22] */
export type Mat3 = number[];

export function mat3Identity(): Mat3 {
  return [1, 0, 0, 0, 1, 0, 0, 0, 1];
}

export function mat3Mul(a: Mat3, b: Mat3): Mat3 {
  const out = new Array<number>(9);
  for (let r = 0; r < 3; r++) {
    for (let c = 0; c < 3; c++) {
      out[r * 3 + c] = a[r * 3] * b[c] + a[r * 3 + 1] * b[3 + c] + a[r * 3 + 2] * b[6 + c];
    }
  }
  return out;
}

export function mat3Transpose(a: Mat3): Mat3 {
  return [a[0], a[3], a[6], a[1], a[4], a[7], a[2], a[5], a[8]];
}

/** 3x3 行列式 */
export function mat3Det(a: Mat3): number {
  return (
    a[0] * (a[4] * a[8] - a[5] * a[7]) -
    a[1] * (a[3] * a[8] - a[5] * a[6]) +
    a[2] * (a[3] * a[7] - a[4] * a[6])
  );
}

/** 3x3 逆矩阵；奇异时返回 null */
export function mat3Inverse(a: Mat3): Mat3 | null {
  const det = mat3Det(a);
  if (Math.abs(det) < 1e-12) return null;
  const invDet = 1 / det;
  return [
    (a[4] * a[8] - a[5] * a[7]) * invDet,
    (a[2] * a[7] - a[1] * a[8]) * invDet,
    (a[1] * a[5] - a[2] * a[4]) * invDet,
    (a[5] * a[6] - a[3] * a[8]) * invDet,
    (a[0] * a[8] - a[2] * a[6]) * invDet,
    (a[2] * a[3] - a[0] * a[5]) * invDet,
    (a[3] * a[7] - a[4] * a[6]) * invDet,
    (a[1] * a[6] - a[0] * a[7]) * invDet,
    (a[0] * a[4] - a[1] * a[3]) * invDet,
  ];
}

/** 3x3 矩阵乘以 3 维齐次坐标，返回齐次结果（未归一化） */
export function mat3Apply(m: Mat3, v: Vec3): Vec3 {
  return [
    m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
    m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
    m[6] * v[0] + m[7] * v[1] + m[8] * v[2],
  ];
}

export function vec3Sub(a: Vec3, b: Vec3): Vec3 {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

export function vec3Cross(a: Vec3, b: Vec3): Vec3 {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

export function vec3Dot(a: Vec3, b: Vec3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

export function vec3Norm(a: Vec3): number {
  return Math.sqrt(vec3Dot(a, a));
}

export function vec3Normalize(a: Vec3): Vec3 {
  const n = vec3Norm(a);
  if (n < 1e-12) return [0, 0, 0];
  return [a[0] / n, a[1] / n, a[2] / n];
}

/**
 * 高斯消元求解 A x = b（A 为 n x n 行数组）。奇异时返回 null。
 */
export function solveLinearSystem(A: number[][], b: number[]): number[] | null {
  const n = b.length;
  const m = A.map((row, i) => [...row, b[i]]);
  for (let col = 0; col < n; col++) {
    let pivot = col;
    for (let r = col + 1; r < n; r++) {
      if (Math.abs(m[r][col]) > Math.abs(m[pivot][col])) pivot = r;
    }
    if (Math.abs(m[pivot][col]) < 1e-12) return null;
    if (pivot !== col) {
      const tmp = m[pivot];
      m[pivot] = m[col];
      m[col] = tmp;
    }
    const p = m[col][col];
    for (let r = col + 1; r < n; r++) {
      const factor = m[r][col] / p;
      if (factor === 0) continue;
      for (let c = col; c <= n; c++) {
        m[r][c] -= factor * m[col][c];
      }
    }
  }
  const x = new Array<number>(n).fill(0);
  for (let r = n - 1; r >= 0; r--) {
    let sum = m[r][n];
    for (let c = r + 1; c < n; c++) sum -= m[r][c] * x[c];
    x[r] = sum / m[r][r];
  }
  return x;
}

/**
 * 实对称矩阵的 Jacobi 特征分解（支持任意 n）。
 * 返回按特征值降序排列的特征值与对应特征向量（每个特征向量为长度 n 的数组）。
 */
export function symmetricEigenDecomposition(input: number[][]): {
  values: number[];
  vectors: number[][];
} {
  const n = input.length;
  const a = input.map((r) => [...r]);
  const v: number[][] = [];
  for (let i = 0; i < n; i++) {
    v.push(new Array<number>(n).fill(0));
    v[i][i] = 1;
  }

  for (let sweep = 0; sweep < 100; sweep++) {
    let offDiag = 0;
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) offDiag += a[i][j] * a[i][j];
    }
    if (offDiag < 1e-22) break;

    for (let p = 0; p < n - 1; p++) {
      for (let q = p + 1; q < n; q++) {
        if (Math.abs(a[p][q]) < 1e-20) continue;
        const theta = (a[q][q] - a[p][p]) / (2 * a[p][q]);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1);
        const s = t * c;

        for (let k = 0; k < n; k++) {
          const akp = a[k][p];
          const akq = a[k][q];
          a[k][p] = c * akp - s * akq;
          a[k][q] = s * akp + c * akq;
        }
        for (let k = 0; k < n; k++) {
          const apk = a[p][k];
          const aqk = a[q][k];
          a[p][k] = c * apk - s * aqk;
          a[q][k] = s * apk + c * aqk;
        }
        a[p][q] = 0;
        a[q][p] = 0;

        for (let k = 0; k < n; k++) {
          const vkp = v[k][p];
          const vkq = v[k][q];
          v[k][p] = c * vkp - s * vkq;
          v[k][q] = s * vkp + c * vkq;
        }
      }
    }
  }

  const idx = Array.from({ length: n }, (_, i) => i);
  idx.sort((x, y) => a[y][y] - a[x][x]);

  const values = idx.map((i) => a[i][i]);
  const vectors = idx.map((i) => v.map((row) => row[i]));

  return { values, vectors };
}
