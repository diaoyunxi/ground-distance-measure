/**
 * 应用主界面。
 *
 * 交互流程：
 *  1. 选择摄像头（枚举手机上的多个不同焦距镜头）
 *  2. 放置参考物 → 点选 4 个角 → 完成标定，得到该镜头下的单位距离
 *  3. 点选起点 → 移动手机（实时显示距起点距离 + 曲线里程）→ 点选终点
 *  4. 输出起止点直线距离 与 摄像头移动曲线距离
 */

import {
  REFERENCE_RECTS,
  assessCalibrationQuality,
  buildMeasurementResult,
  formatDistance,
  formatReport,
  liveDistanceFromStart,
  solveCalibration,
  unitDistanceAt,
  type Calibration,
  type MeasurementResult,
  type ReferenceRect,
} from '../core/calibration';
import {
  classifyCameraLabel,
  enumerateCameraDevices,
  estimateRelativeFocal,
  openCamera,
  type CameraDevice,
} from '../core/devices';
import { backprojectToGround, fieldOfView } from '../core/camera';
import { applyHomography, type Point2D } from '../core/homography';
import {
  accumulateFrameMotion,
  createOdometryState,
  detectCorners,
  estimateFrameHomography,
  toGray,
  trackOpticalFlow,
  type GrayImage,
  type OdometryState,
} from '../core/odometry';

/** 交互模式 */
type Mode = 'idle' | 'calibrating' | 'calibrated' | 'measuring' | 'finished';

const CORNER_LABELS = ['左上角', '右上角', '右下角', '左下角'];

/** 视觉里程计的跟踪分辨率宽度（像素）。降采样以保证交互流畅。 */
const TRACKING_WIDTH = 640;

class GroundDistanceApp {
  private video: HTMLVideoElement;
  private canvas: HTMLCanvasElement;
  private overlay: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;

  private devices: CameraDevice[] = [];
  private currentDevice: CameraDevice | null = null;
  private stream: MediaStream | null = null;

  private calibration: Calibration | null = null;
  private reference: ReferenceRect = REFERENCE_RECTS[0];
  private cornerPoints: Point2D[] = [];
  private startPixel: Point2D | null = null;
  private endPixel: Point2D | null = null;
  private track: Point2D[] = [];
  private result: MeasurementResult | null = null;

  private mode: Mode = 'idle';
  private frameHandle = 0;

  // 视觉里程计状态
  private odometry: OdometryState = createOdometryState();
  private prevGray: GrayImage | null = null;
  private trackedPoints: Point2D[] = [];
  /** 最近一帧估计出的 地面→图像 单应 */
  private prevGroundToImage: number[] | null = null;

  // DOM 引用
  private els: Record<string, HTMLElement> = {};

  constructor() {
    this.video = document.getElementById('video') as HTMLVideoElement;
    this.canvas = document.getElementById('canvas') as HTMLCanvasElement;
    this.overlay = document.getElementById('overlay') as HTMLCanvasElement;
    this.ctx = this.canvas.getContext('2d')!;

    this.cacheElements();
    this.bindEvents();
    void this.init();
  }

  private cacheElements(): void {
    const ids = [
      'device-select',
      'device-info',
      'calibrate-btn',
      'reset-btn',
      'reference-select',
      'hint',
      'start-btn',
      'end-btn',
      'live-distance',
      'live-curve',
      'live-scale',
      'calib-panel',
      'calib-detail',
      'result-panel',
      'result-straight',
      'result-curve',
      'result-ratio',
      'result-detail',
      'copy-btn',
      'status',
      'zoom-slider',
      'zoom-value',
    ];
    for (const id of ids) {
      const el = document.getElementById(id);
      if (el) this.els[id] = el;
    }
  }

  private bindEvents(): void {
    this.els['device-select']?.addEventListener('change', (e) => {
      const id = (e.target as HTMLSelectElement).value;
      void this.switchDevice(id);
    });

    this.els['calibrate-btn']?.addEventListener('click', () => this.startCalibration());
    this.els['reset-btn']?.addEventListener('click', () => this.reset());
    this.els['start-btn']?.addEventListener('click', () => this.startMeasuring());
    this.els['end-btn']?.addEventListener('click', () => this.finishMeasuring());
    this.els['copy-btn']?.addEventListener('click', () => void this.copyReport());

    const referenceSelect = this.els['reference-select'] as HTMLSelectElement | undefined;
    if (referenceSelect) {
      referenceSelect.innerHTML = REFERENCE_RECTS.map(
        (r, i) => `<option value="${i}">${r.name}</option>`,
      ).join('');
      referenceSelect.addEventListener('change', (e) => {
        const idx = Number((e.target as HTMLSelectElement).value);
        this.reference = REFERENCE_RECTS[idx] ?? REFERENCE_RECTS[0];
      });
    }

    // 画面交互：用 pointerdown 而非 click。
    // click 需等待按下与抬起的完整序列，在图像处理占用主线程时容易丢失；
    // pointerdown 立即响应，交互更可靠。
    this.overlay.addEventListener('pointerdown', (e) => this.handleCanvasClick(e));

    // 数字变焦（部分设备支持）
    const zoom = this.els['zoom-slider'] as HTMLInputElement | undefined;
    zoom?.addEventListener('input', (e) => {
      const value = (e.target as HTMLInputElement).value;
      if (this.els['zoom-value']) this.els['zoom-value'].textContent = `${value}×`;
      void this.applyZoom(Number(value));
    });
  }

  private async init(): Promise<void> {
    this.setStatus('正在枚举摄像头…');
    try {
      // 先请求一次权限，才能拿到设备 label
      await navigator.mediaDevices.getUserMedia({ video: true, audio: false }).then((s) => {
        s.getTracks().forEach((t) => t.stop());
      });
      this.devices = await enumerateCameraDevices();
      this.renderDeviceList();
      if (this.devices.length > 0) {
        await this.switchDevice(this.devices[0].deviceId);
      } else {
        this.setStatus('未检测到可用摄像头');
      }
    } catch (err) {
      this.setStatus(`无法访问摄像头：${(err as Error).message}`);
    }
    this.loop();
  }

  private renderDeviceList(): void {
    const select = this.els['device-select'] as HTMLSelectElement | undefined;
    if (!select) return;
    select.innerHTML = this.devices
      .map(
        (d, i) =>
          `<option value="${d.deviceId}">${escapeHtml(d.label)}（${d.facing === 'front' ? '前置' : '后置'} #${
            i + 1
          }）</option>`,
      )
      .join('');
  }

  private async switchDevice(deviceId: string): Promise<void> {
    const device = this.devices.find((d) => d.deviceId === deviceId);
    if (!device) return;

    this.stopStream();
    this.setStatus(`正在打开 ${device.label}…`);

    try {
      const { stream, capabilities } = await openCamera(deviceId, {
        width: { ideal: 1920 },
        height: { ideal: 1080 },
      });
      this.stream = stream;
      this.currentDevice = device;
      this.video.srcObject = stream;
      await this.video.play();

      const track = stream.getVideoTracks()[0];
      const settings = track.getSettings();
      const width = settings.width ?? 1280;
      const height = settings.height ?? 720;
      this.canvas.width = width;
      this.canvas.height = height;
      this.overlay.width = width;
      this.overlay.height = height;
      this.updateViewportAspect(width, height);

      // 切换镜头后原标定失效（焦距不同）
      this.calibration = null;
      this.resetMeasurementState();
      this.mode = 'idle';

      this.renderDeviceInfo(device, capabilities.focalLengthMM, capabilities.maxZoom);
      this.setStatus(`已就绪：${device.label}`);
    } catch (err) {
      this.setStatus(`打开摄像头失败：${(err as Error).message}`);
    }
  }

  /**
   * 依据视频真实宽高比与可用空间调整视口高度。
   *
   * 视口高度受两个约束：
   *  1. 不能超过父容器可用高度（否则整页会滚动，画面被顶出屏幕）；
   *  2. 应贴合视频宽高比，避免画布内部出现信箱黑边。
   */
  private updateViewportAspect(width: number, height: number): void {
    const viewport = this.overlay.parentElement;
    if (!viewport || !width || !height) return;

    const stage = viewport.parentElement;
    const available = stage ? stage.clientHeight : window.innerHeight;
    const widthBased = Math.round((viewport.clientWidth * height) / width);
    const targetHeight = Math.max(160, Math.min(available, widthBased));
    viewport.style.height = `${targetHeight}px`;
  }

  private renderDeviceInfo(
    device: CameraDevice,
    focalLengthMM: number | null,
    maxZoom: number | null,
  ): void {
    const el = this.els['device-info'];
    if (!el) return;

    const { focalHint } = classifyCameraLabel(device.label);
    const hintText: Record<string, string> = {
      ultrawide: '超广角',
      wide: '主摄/广角',
      tele: '长焦',
      front: '前置',
      unknown: '未知焦距',
    };

    const parts: string[] = [];
    parts.push(`镜头：${hintText[focalHint] ?? focalHint}`);
    if (focalLengthMM) parts.push(`等效焦距 ${focalLengthMM.toFixed(1)}mm`);
    if (maxZoom && maxZoom > 1) parts.push(`最大变焦 ${maxZoom.toFixed(1)}×`);
    parts.push(`相对倍率 ≈ ${estimateRelativeFocal({ deviceId: device.deviceId, focalLengthMM, maxZoom, width: null, height: null }, device.label).toFixed(2)}×`);
    parts.push(`分辨率 ${this.canvas.width}×${this.canvas.height}`);

    el.innerHTML = parts.map((p) => `<span class="chip">${escapeHtml(p)}</span>`).join('');

    // 变焦滑块
    const zoom = this.els['zoom-slider'] as HTMLInputElement | undefined;
    if (zoom) {
      const max = maxZoom && maxZoom > 1 ? maxZoom : 1;
      zoom.max = String(max);
      zoom.value = '1';
      zoom.disabled = max <= 1;
      if (this.els['zoom-value']) this.els['zoom-value'].textContent = '1×';
    }

    // 窗口尺寸变化时重新计算视口高度
    window.removeEventListener('resize', this.handleResize);
    window.addEventListener('resize', this.handleResize);
  }

  private handleResize = (): void => {
    this.updateViewportAspect(this.canvas.width, this.canvas.height);
  };

  private async applyZoom(factor: number): Promise<void> {
    const track = this.stream?.getVideoTracks()[0];
    if (!track) return;
    try {
      const caps = track.getCapabilities() as MediaTrackCapabilities & {
        zoom?: { min: number; max: number };
      };
      if (!caps.zoom) return;
      const clamped = Math.max(caps.zoom.min, Math.min(caps.zoom.max, factor));
      // zoom 属于厂商扩展约束，需通过 advanced 传入
      await track.applyConstraints({
        advanced: [{ zoom: clamped } as MediaTrackConstraintSet],
      });
      // 变焦会改变有效焦距，原标定失效
      if (this.calibration && clamped !== 1) {
        this.calibration = null;
        this.resetMeasurementState();
        this.mode = 'idle';
        this.setStatus('变焦已改变有效焦距，请重新标定');
      }
    } catch {
      // 部分浏览器不支持动态变焦，忽略
    }
  }

  /** 主循环：绘制画面与叠加层 */
  private loop = (): void => {
    if (this.video.readyState >= 2 && this.video.videoWidth > 0) {
      this.ctx.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);
      this.drawOverlay();
    }
    this.frameHandle = requestAnimationFrame(this.loop);
  };

  /**
   * 跟踪循环（独立于渲染循环）。
   *
   * 视觉里程计的计算量较大（光流 + 单应估计），
   * 若与渲染共用同一帧回调，会阻塞 UI 事件响应、导致点击无响应。
   * 因此这里用独立定时器并限制处理频率，让交互始终保持流畅。
   */
  private trackingTimer = 0;

  private startTrackingLoop(): void {
    if (this.trackingTimer) return;
    const TICK_MS = 120; // 约 8Hz，足够跟踪缓慢移动且不占用过多主线程
    this.trackingTimer = window.setInterval(() => {
      if (this.mode !== 'measuring' || !this.calibration) return;
      this.processTrackingFrame();
    }, TICK_MS);
  }

  private stopTrackingLoop(): void {
    if (this.trackingTimer) {
      window.clearInterval(this.trackingTimer);
      this.trackingTimer = 0;
    }
  }

  private drawOverlay(): void {
    const ctx = this.overlay.getContext('2d')!;
    ctx.clearRect(0, 0, this.overlay.width, this.overlay.height);
    const { width, height } = this.overlay;

    // 标定角点
    if (this.mode === 'calibrating') {
      this.cornerPoints.forEach((p, i) => {
        ctx.beginPath();
        ctx.arc(p.x, p.y, 12, 0, Math.PI * 2);
        ctx.strokeStyle = '#ffb020';
        ctx.lineWidth = 3;
        ctx.stroke();
        ctx.fillStyle = '#ffb020';
        ctx.font = 'bold 18px sans-serif';
        ctx.fillText(String(i + 1), p.x + 16, p.y - 12);
      });
      // 连线展示矩形
      if (this.cornerPoints.length >= 2) {
        ctx.beginPath();
        this.cornerPoints.forEach((p, i) => {
          if (i === 0) ctx.moveTo(p.x, p.y);
          else ctx.lineTo(p.x, p.y);
        });
        if (this.cornerPoints.length === 4) ctx.closePath();
        ctx.strokeStyle = 'rgba(255,176,32,0.7)';
        ctx.lineWidth = 2;
        ctx.stroke();
      }
    }

    // 地面轨迹（把世界坐标投影回当前帧）
    if (this.track.length >= 2 && this.prevGroundToImage) {
      ctx.beginPath();
      let started = false;
      for (const p of this.track) {
        const projected = applyHomography(this.prevGroundToImage, p);
        if (!projected) continue;
        if (!started) {
          ctx.moveTo(projected.x, projected.y);
          started = true;
        } else {
          ctx.lineTo(projected.x, projected.y);
        }
      }
      ctx.strokeStyle = 'rgba(34,197,94,0.85)';
      ctx.lineWidth = 3;
      ctx.stroke();
    }

    // 起点 / 终点
    if (this.startPixel) this.drawMarker(ctx, this.startPixel, '#22c55e', '起');
    if (this.endPixel) this.drawMarker(ctx, this.endPixel, '#ef4444', '终');

    // 地面比例尺：显示画面中心处 1 米对应的像素长度
    if (this.calibration) {
      const center = { x: width / 2, y: height / 2 };
      const scale = unitDistanceAt(this.calibration, center);
      if (scale && scale.pixelsPerMeter > 0) {
        // 水平方向 1 米的像素长度
        const p1 = backprojectToGround(this.calibration.H_imgToGround, center);
        const p2 = backprojectToGround(this.calibration.H_imgToGround, {
          x: center.x + 100,
          y: center.y,
        });
        if (p1 && p2) {
          const metersPerPixel = Math.hypot(p2.x - p1.x, p2.y - p1.y) / 100;
          const pxPerMeter = 1 / metersPerPixel;
          const barPx = Math.min(pxPerMeter, width * 0.5);
          const y0 = height - 40;
          ctx.strokeStyle = '#22d3ee';
          ctx.lineWidth = 4;
          ctx.beginPath();
          ctx.moveTo(width / 2 - barPx / 2, y0);
          ctx.lineTo(width / 2 + barPx / 2, y0);
          ctx.stroke();
          ctx.fillStyle = '#22d3ee';
          ctx.font = 'bold 20px sans-serif';
          ctx.textAlign = 'center';
          ctx.fillText('1 米', width / 2, y0 - 12);
          ctx.textAlign = 'left';
        }
      }
    }
  }

  private drawMarker(ctx: CanvasRenderingContext2D, p: Point2D, color: string, label: string): void {
    ctx.beginPath();
    ctx.arc(p.x, p.y, 16, 0, Math.PI * 2);
    ctx.strokeStyle = color;
    ctx.lineWidth = 4;
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(p.x - 24, p.y);
    ctx.lineTo(p.x - 8, p.y);
    ctx.moveTo(p.x + 8, p.y);
    ctx.lineTo(p.x + 24, p.y);
    ctx.moveTo(p.x, p.y - 24);
    ctx.lineTo(p.x, p.y - 8);
    ctx.moveTo(p.x, p.y + 8);
    ctx.lineTo(p.x, p.y + 24);
    ctx.stroke();
    ctx.fillStyle = color;
    ctx.font = 'bold 20px sans-serif';
    ctx.fillText(label, p.x + 20, p.y - 18);
  }

  /**
   * 把显示层的点击坐标转换为画布（图像）坐标。
   *
   * 画布使用 `object-fit: contain` 显示，宽高比与容器不一致时会出现信箱（黑边），
   * 因此必须先算出内容区域再换算，否则点击会整体偏移。
   */
  private eventToCanvas(e: MouseEvent): Point2D {
    const rect = this.overlay.getBoundingClientRect();
    const canvasWidth = this.overlay.width;
    const canvasHeight = this.overlay.height;

    // contain 模式下的实际内容区域
    const scale = Math.min(rect.width / canvasWidth, rect.height / canvasHeight);
    const contentWidth = canvasWidth * scale;
    const contentHeight = canvasHeight * scale;
    const offsetX = (rect.width - contentWidth) / 2;
    const offsetY = (rect.height - contentHeight) / 2;

    const x = (e.clientX - rect.left - offsetX) / scale;
    const y = (e.clientY - rect.top - offsetY) / scale;
    return { x, y };
  }

  private handleCanvasClick(e: MouseEvent): void {
    // 注意：不能用 video.videoWidth 判断可用性——
    // video 元素被 CSS 隐藏时其 videoWidth 可能为 0，导致点击被静默忽略。
    // 这里以画布尺寸作为判据。
    if (!this.canvas.width || !this.canvas.height) return;
    const { x, y } = this.eventToCanvas(e);

    if (this.mode === 'calibrating') {
      if (this.cornerPoints.length >= 4) return;
      this.cornerPoints.push({ x, y });
      this.updateHint();
      if (this.cornerPoints.length === 4) this.completeCalibration();
    } else if (this.mode === 'calibrated' && !this.startPixel) {
      this.startPixel = { x, y };
      this.mode = 'measuring';
      this.updateHint();
      this.updateLiveReadout();
    } else if (this.mode === 'measuring' && this.startPixel && !this.endPixel) {
      this.endPixel = { x, y };
      this.finishMeasuring();
    }
  }

  private startCalibration(): void {
    if (!this.currentDevice) return;
    this.mode = 'calibrating';
    this.cornerPoints = [];
    this.resetMeasurementState();
    this.showPanel('calib-panel');
    this.hidePanel('result-panel');
    this.updateHint();
  }

  private completeCalibration(): void {
    const result = solveCalibration(
      this.cornerPoints,
      this.reference,
      this.canvas.width,
      this.canvas.height,
    );

    if (!result) {
      this.setStatus('标定失败：角点共线或数据退化，请重新点选');
      this.mode = 'calibrating';
      this.cornerPoints = [];
      this.updateHint();
      return;
    }

    this.calibration = result.calibration;
    this.mode = 'calibrated';
    this.startPixel = null;
    this.endPixel = null;

    const quality = assessCalibrationQuality(result.error);
    const hasIntrinsics = result.calibration.intrinsics !== null;
    const fov = hasIntrinsics
      ? fieldOfView(
          result.calibration.intrinsics!,
          this.canvas.width,
          this.canvas.height,
        )
      : null;

    const detail = this.els['calib-detail'];
    if (detail) {
      const rows: string[] = [];
      if (hasIntrinsics) {
        rows.push(
          `<div class="metric"><span>解算焦距</span><b>${result.calibration.intrinsics!.focalLength.toFixed(
            1,
          )} px</b></div>`,
          `<div class="metric"><span>水平视场角</span><b>${fov!.horizontal.toFixed(1)}°</b></div>`,
          `<div class="metric"><span>垂直视场角</span><b>${fov!.vertical.toFixed(1)}°</b></div>`,
          `<div class="metric"><span>相机高度</span><b>${result.calibration.cameraHeightMeters!.toFixed(
            3,
          )} m</b></div>`,
        );
      } else {
        rows.push(
          `<div class="metric"><span>解算焦距</span><b class="q-fair">该视角无法确定</b></div>`,
          `<div class="metric"><span>说明</span><b class="q-fair">请斜拍重标定</b></div>`,
        );
      }
      rows.push(
        `<div class="metric"><span>重投影误差</span><b>${result.error.toFixed(2)} px</b></div>`,
        `<div class="metric"><span>标定质量</span><b class="q-${quality.level}">${quality.message}</b></div>`,
      );
      detail.innerHTML = rows.join('');
    }

    if (hasIntrinsics) {
      this.setStatus('标定成功，现在点击画面选择起点');
    } else {
      this.setStatus(
        '标定完成（尺度可用）。当前视角正对导致焦距不可解，斜拍重标定可启用曲线里程',
      );
    }
    this.updateHint();
  }

  private startMeasuring(): void {
    if (!this.calibration) {
      this.setStatus('请先完成标定');
      return;
    }
    if (!this.startPixel) {
      this.setStatus('请先在画面中点击起点');
      return;
    }
    this.mode = 'measuring';
    this.track = [];
    this.odometry = createOdometryState();
    this.prevGray = null;
    this.trackedPoints = [];
    this.prevGroundToImage = null;
    this.startTrackingLoop();
    this.updateHint();
  }

  /**
   * 实时追踪：通过帧间光流跟踪地面特征，估计相机的地面位移，
   * 累计得到摄像头移动的曲线距离。
   *
   * 流程：
   *  1. 首帧检测角点作为地面特征点；
   *  2. 逐帧用 LK 光流跟踪这些点；
   *  3. 由跟踪结果估计帧间地面→图像单应；
   *  4. 分解单应得到相机在地面上的位置，累计曲线里程。
   */
  private processTrackingFrame(): void {
    if (!this.calibration) return;

    // 从视频中抓取当前帧的灰度图
    const frame = this.captureGrayFrame();
    if (!frame) return;

    if (!this.prevGray) {
      // 首帧：检测角点并初始化
      this.prevGray = frame;
      this.trackedPoints = detectCorners(frame, { maxCorners: 120 });
      this.prevGroundToImage = this.calibration.H_groundToImg;
      this.odometry = createOdometryState();
      this.updateLiveReadout();
      return;
    }

    if (this.trackedPoints.length >= 4) {
      // 光流跟踪（在降采样的跟踪分辨率上进行）
      const tracked = trackOpticalFlow(this.prevGray, frame, this.trackedPoints, {
        windowSize: 21,
        levels: 3,
      });

      // 跟踪坐标需换算回原图分辨率，才能与标定单应处于同一坐标系
      const ratio = this.canvas.width / this.prevGray.width;

      const prevPoints: Point2D[] = [];
      const currPoints: Point2D[] = [];
      tracked.forEach((t, i) => {
        if (!t.tracked) return;
        prevPoints.push({
          x: this.trackedPoints[i].x * ratio,
          y: this.trackedPoints[i].y * ratio,
        });
        currPoints.push({ x: t.point.x * ratio, y: t.point.y * ratio });
      });

      if (prevPoints.length >= 4 && this.calibration.intrinsics) {
        // 估计帧间地面→图像单应
        const H = estimateFrameHomography(prevPoints, currPoints);
        if (H) {
          this.odometry = accumulateFrameMotion(
            this.odometry,
            currPoints,
            H,
            this.calibration.intrinsics,
          );
          this.prevGroundToImage = H;
          this.track = [...this.odometry.positions];
        }
      }

      // 更新被跟踪的点位（保持跟踪分辨率坐标）
      this.trackedPoints = tracked
        .map((t) => (t.tracked ? t.point : null))
        .filter((p): p is Point2D => p !== null);

      if (this.trackedPoints.length < 30) {
        const fresh = detectCorners(frame, { maxCorners: 60 });
        for (const p of fresh) {
          const tooClose = this.trackedPoints.some(
            (t) => Math.hypot(t.x - p.x, t.y - p.y) < 15,
          );
          if (!tooClose) this.trackedPoints.push(p);
        }
        this.trackedPoints = this.trackedPoints.slice(0, 160);
      }
    } else {
      this.trackedPoints = detectCorners(frame, { maxCorners: 120 });
    }

    this.prevGray = frame;
    this.updateLiveReadout();
  }

  /**
   * 从视频画面抓取灰度帧。
   *
   * 光流计算量随像素数平方增长，在 1080p 上逐帧处理会明显拖慢交互，
   * 因此这里降采样到较小尺寸再处理——跟踪精度对尺度不敏感，
   * 只影响亚像素精度，不影响整体里程精度。
   */
  private captureGrayFrame(): GrayImage | null {
    const targetWidth = TRACKING_WIDTH;
    if (!this.canvas.width || !this.canvas.height) return null;
    try {
      // 使用离屏 canvas 抓帧，避免 getImageData 在主 canvas 上强制同步刷新，
      // 减少与 UI 交互的相互干扰。
      if (!this.trackCanvas) {
        this.trackCanvas = document.createElement('canvas');
      }
      const trackCanvas = this.trackCanvas;
      if (trackCanvas.width !== this.canvas.width || trackCanvas.height !== this.canvas.height) {
        trackCanvas.width = this.canvas.width;
        trackCanvas.height = this.canvas.height;
      }
      const tctx = trackCanvas.getContext('2d', { willReadFrequently: true })!;
      tctx.drawImage(this.canvas, 0, 0);
      const imageData = tctx.getImageData(0, 0, this.canvas.width, this.canvas.height);
      const full = toGray(imageData.data, this.canvas.width, this.canvas.height);

      // 降采样到跟踪分辨率
      const scale = targetWidth / this.canvas.width;
      const w = Math.max(1, Math.round(this.canvas.width * scale));
      const h = Math.max(1, Math.round(this.canvas.height * scale));
      const data = new Uint8ClampedArray(w * h);
      for (let y = 0; y < h; y++) {
        const sy = Math.min(this.canvas.height - 1, Math.round(y / scale));
        for (let x = 0; x < w; x++) {
          const sx = Math.min(this.canvas.width - 1, Math.round(x / scale));
          data[y * w + x] = full.data[sy * this.canvas.width + sx];
        }
      }
      return { data, width: w, height: h };
    } catch {
      return null;
    }
  }

  /** 用于抓帧的离屏画布 */
  private trackCanvas: HTMLCanvasElement | null = null;

  private updateLiveReadout(): void {
    if (!this.calibration || !this.startPixel) return;

    // 当前画面中心对应的地面点，作为"当前位置"
    const center = { x: this.canvas.width / 2, y: this.canvas.height / 2 };
    const currentGround = backprojectToGround(this.calibration.H_imgToGround, center);
    const startGround = backprojectToGround(this.calibration.H_imgToGround, this.startPixel);

    if (currentGround && startGround) {
      const distance = liveDistanceFromStart(startGround, currentGround);
      const el = this.els['live-distance'];
      if (el) el.textContent = formatDistance(distance);
    }

    // 单位距离（画面中心处）
    const scale = unitDistanceAt(this.calibration, center);
    const scaleEl = this.els['live-scale'];
    if (scaleEl && scale) {
      scaleEl.textContent = `${(scale.metersPerPixel * 100).toFixed(2)} cm/px`;
    }

    // 曲线距离（摄像头实际移动轨迹累计）
    const curveEl = this.els['live-curve'];
    if (curveEl) {
      curveEl.textContent = formatDistance(this.odometry.curveDistance);
    }

    // 跟踪点数提示
    const trackEl = this.els['live-track'];
    if (trackEl) {
      trackEl.textContent = `${this.odometry.trackedPoints} 个特征点`;
    }
  }

  private finishMeasuring(): void {
    if (!this.calibration || !this.startPixel || !this.endPixel) return;

    const result = buildMeasurementResult(
      this.calibration,
      this.startPixel,
      this.endPixel,
      this.track,
    );
    if (!result) {
      this.setStatus('测量结果计算失败');
      return;
    }

    this.result = result;
    this.mode = 'finished';
    this.stopTrackingLoop();

    const straightEl = this.els['result-straight'];
    const curveEl = this.els['result-curve'];
    const ratioEl = this.els['result-ratio'];
    if (straightEl) straightEl.textContent = formatDistance(result.straightDistanceMeters);
    if (curveEl) curveEl.textContent = formatDistance(result.curveDistanceMeters);
    if (ratioEl) ratioEl.textContent = result.detourRatio.toFixed(3);

    const detail = this.els['result-detail'];
    if (detail) {
      detail.innerHTML = `
        <div class="metric"><span>起点坐标</span><b>(${result.startGround.x.toFixed(3)}, ${result.startGround.y.toFixed(3)}) m</b></div>
        <div class="metric"><span>终点坐标</span><b>(${result.endGround.x.toFixed(3)}, ${result.endGround.y.toFixed(3)}) m</b></div>
        <div class="metric"><span>绕行系数</span><b>${result.detourRatio.toFixed(3)}</b></div>
        <div class="metric"><span>测量时间</span><b>${new Date(result.timestamp).toLocaleString('zh-CN')}</b></div>
      `;
    }

    this.showPanel('result-panel');
    this.setStatus('测量完成');
  }

  private async copyReport(): Promise<void> {
    if (!this.result) return;
    const text = formatReport(this.result);
    try {
      await navigator.clipboard.writeText(text);
      this.setStatus('报告已复制到剪贴板');
    } catch {
      this.setStatus('复制失败，请手动选择文本');
    }
  }

  private resetMeasurementState(): void {
    this.stopTrackingLoop();
    this.startPixel = null;
    this.endPixel = null;
    this.track = [];
    this.result = null;
    this.odometry = createOdometryState();
    this.prevGray = null;
    this.trackedPoints = [];
    this.prevGroundToImage = null;
    this.hidePanel('result-panel');
  }

  private reset(): void {
    this.calibration = null;
    this.cornerPoints = [];
    this.resetMeasurementState();
    this.mode = 'idle';
    this.hidePanel('calib-panel');
    this.updateHint();
    this.setStatus('已重置');
  }

  private updateHint(): void {
    const hint = this.els['hint'];
    if (!hint) return;

    switch (this.mode) {
      case 'calibrating':
        hint.innerHTML = `标定中：请依次点击参考矩形的 <b>${
          CORNER_LABELS[this.cornerPoints.length] ?? ''
        }</b>（已选 ${this.cornerPoints.length}/4）`;
        break;
      case 'calibrated':
        hint.innerHTML = '标定完成：<b>点击画面选择起点</b>';
        break;
      case 'measuring':
        hint.innerHTML = '移动手机进行测量：实时显示距起点距离，点击画面选择终点';
        break;
      case 'finished':
        hint.innerHTML = '测量完成，可重新选择起点再次测量';
        break;
      default:
        hint.innerHTML = '请先放置参考物并点击「开始标定」';
    }
  }

  private setStatus(text: string): void {
    const el = this.els['status'];
    if (el) el.textContent = text;
  }

  private showPanel(id: string): void {
    const el = this.els[id];
    if (el) el.classList.remove('hidden');
  }

  private hidePanel(id: string): void {
    const el = this.els[id];
    if (el) el.classList.add('hidden');
  }

  private stopStream(): void {
    if (this.stream) {
      this.stream.getTracks().forEach((t) => t.stop());
      this.stream = null;
    }
  }

  /** 页面卸载时释放资源 */
  dispose(): void {
    this.stopTrackingLoop();
    if (this.frameHandle) {
      cancelAnimationFrame(this.frameHandle);
      this.frameHandle = 0;
    }
    this.stopStream();
  }
}

function escapeHtml(text: string): string {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

// 启动应用
const app = new GroundDistanceApp();
window.addEventListener('beforeunload', () => app.dispose());

// 调试入口：便于在浏览器控制台检查运行状态（如 __app.mode / __app.calibration）
(window as unknown as { __app: GroundDistanceApp }).__app = app;
