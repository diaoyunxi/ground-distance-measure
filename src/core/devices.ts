/**
 * 摄像头设备枚举与能力探测。
 *
 * 目标：同一台手机上枚举出多个不同焦距的摄像头（主摄 / 超广角 / 长焦 / 前置），
 * 每个设备独立标定内参，从而在任意镜头下都能判定"摄像头内对应的现实地面长度"。
 */

export interface CameraDevice {
  deviceId: string;
  /** 设备标签，例如 "后置主摄" / "超广角" */
  label: string;
  /** 分组 id，同组通常是同一个物理摄像头的不同配置 */
  groupId: string;
  /** 是否为前置摄像头 */
  facing: 'front' | 'back' | 'unknown';
  /** 是否为虚拟/外接设备 */
  virtual: boolean;
}

/** 摄像头能力：焦距信息、变焦范围等 */
export interface CameraCapabilities {
  deviceId: string;
  /** 硬件报告的等效焦距（毫米），部分浏览器可提供 */
  focalLengthMM: number | null;
  /** 支持的最大变焦倍率 */
  maxZoom: number | null;
  /** 支持的分辨率宽 */
  width: number | null;
  /** 支持的分辨率高 */
  height: number | null;
}

/**
 * 依据设备标签推断镜头类型与焦距倾向。
 * 不同厂商的标签差异很大，这里用关键词做稳健匹配，未命中时返回 unknown。
 */
export function classifyCameraLabel(label: string): {
  facing: 'front' | 'back' | 'unknown';
  focalHint: 'ultrawide' | 'wide' | 'tele' | 'front' | 'unknown';
} {
  const lower = label.toLowerCase();

  const isFront =
    lower.includes('front') ||
    lower.includes('user') ||
    lower.includes('face') ||
    lower.includes('前置');

  const isBack =
    lower.includes('back') ||
    lower.includes('rear') ||
    lower.includes('后置');

  let focalHint: 'ultrawide' | 'wide' | 'tele' | 'front' | 'unknown' = 'unknown';
  if (lower.includes('ultra') || lower.includes('0.5') || lower.includes('超广角')) {
    focalHint = 'ultrawide';
  } else if (
    lower.includes('tele') ||
    lower.includes('3x') ||
    lower.includes('5x') ||
    lower.includes('10x') ||
    lower.includes('长焦')
  ) {
    focalHint = 'tele';
  } else if (lower.includes('wide') || lower.includes('主摄') || lower.includes('广角')) {
    focalHint = 'wide';
  }

  if (isFront && focalHint === 'unknown') focalHint = 'front';

  return {
    facing: isFront ? 'front' : isBack ? 'back' : 'unknown',
    focalHint,
  };
}

/**
 * 枚举所有视频输入设备。
 * 注意：必须先获得媒体权限，label 才会被填充。
 */
export async function enumerateCameraDevices(): Promise<CameraDevice[]> {
  if (!navigator.mediaDevices?.enumerateDevices) {
    throw new Error('当前浏览器不支持 mediaDevices.enumerateDevices');
  }

  const devices = await navigator.mediaDevices.enumerateDevices();
  const videoInputs = devices.filter((d) => d.kind === 'videoinput');

  return videoInputs.map((device, index) => {
    const label = device.label || `摄像头 ${index + 1}`;
    const { facing } = classifyCameraLabel(label);
    return {
      deviceId: device.deviceId,
      label,
      groupId: device.groupId,
      facing,
      virtual: /virtual|obs|virtualcam|screen|mirror/i.test(label),
    };
  });
}

/**
 * 打开指定摄像头，返回媒体流与实际能力信息。
 */
export async function openCamera(
  deviceId: string,
  constraints: MediaTrackConstraints = {},
): Promise<{ stream: MediaStream; capabilities: CameraCapabilities }> {
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: false,
    video: {
      deviceId: { exact: deviceId },
      ...constraints,
    },
  });

  const track = stream.getVideoTracks()[0];
  const capabilities = await probeCameraCapabilities(track, deviceId);
  return { stream, capabilities };
}

/**
 * 读取一条视频轨的能力信息（焦距、变焦、分辨率）。
 */
export async function probeCameraCapabilities(
  track: MediaStreamTrack,
  deviceId: string,
): Promise<CameraCapabilities> {
  const result: CameraCapabilities = {
    deviceId,
    focalLengthMM: null,
    maxZoom: null,
    width: null,
    height: null,
  };

  try {
    const caps = track.getCapabilities() as MediaTrackCapabilities & {
      zoom?: { min: number; max: number };
      focusDistance?: { min: number; max: number };
    };
    const settings = track.getSettings() as MediaTrackSettings & {
      focalLength?: number;
    };

    if (typeof caps.zoom?.max === 'number') {
      result.maxZoom = caps.zoom.max;
    }
    if (typeof settings.focalLength === 'number') {
      result.focalLengthMM = settings.focalLength;
    }
    if (typeof settings.width === 'number') {
      result.width = settings.width;
    }
    if (typeof settings.height === 'number') {
      result.height = settings.height;
    }
  } catch {
    // 部分浏览器不支持 getCapabilities，忽略即可
  }

  return result;
}

/**
 * 判断两个设备是否属于不同的物理镜头。
 * 用于提示用户"当前切换到了另一个焦距的镜头，需要重新标定"。
 */
export function isDistinctLens(a: CameraDevice, b: CameraDevice): boolean {
  if (a.deviceId === b.deviceId) return false;
  const aInfo = classifyCameraLabel(a.label);
  const bInfo = classifyCameraLabel(b.label);
  if (aInfo.focalHint !== 'unknown' && bInfo.focalHint !== 'unknown') {
    return aInfo.focalHint !== bInfo.focalHint;
  }
  return a.groupId !== b.groupId || a.label !== b.label;
}

/**
 * 估算设备的相对焦距倍率（相对于主摄），用于展示镜头排序。
 * 依据标签线索给出粗略先验，实际倍率仍以标定解算结果为准。
 */
export function estimateRelativeFocal(capabilities: CameraCapabilities, label: string): number {
  const { focalHint } = classifyCameraLabel(label);
  switch (focalHint) {
    case 'ultrawide':
      return 0.5;
    case 'tele':
      return 3;
    case 'front':
      return 0.8;
    case 'wide':
      return 1;
    default:
      return capabilities.focalLengthMM ?? 1;
  }
}
