# -*- coding: utf-8 -*-
"""Camera config and worker module."""

import json
import os
import sys
import time
import logging
import threading
import ctypes
from ctypes import POINTER, byref, c_long, cast, create_string_buffer, memset, sizeof
from queue import Empty, Full, Queue
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

from detector import create_detector, AsyncImageSaver
try:
    from HCNetSDK import (
        NET_DVR_DEVICEINFO_V30,
        NET_DVR_PREVIEWINFO,
        NET_DVR_STREAMDATA,
        NET_DVR_SYSHEAD,
        REALDATACALLBACK,
        netsdkdllpath,
    )
    from PlayCtrl import DECCBFUNWIN, playM4dllpath
    HIK_SDK_AVAILABLE = True
except Exception as _hik_import_error:
    HIK_SDK_AVAILABLE = False
    HIK_SDK_IMPORT_ERROR = str(_hik_import_error)

try:
    from MvImport.MvCameraControl_class import MvCamera
    from MvImport.CameraParams_header import (
        MV_CC_DEVICE_INFO,
        MV_CC_DEVICE_INFO_LIST,
        MV_FRAME_OUT,
        MV_GIGE_DEVICE,
        MV_TRIGGER_MODE_OFF,
        PixelType_Gvsp_BayerBG10,
        PixelType_Gvsp_BayerBG10_Packed,
        PixelType_Gvsp_BayerBG12,
        PixelType_Gvsp_BayerBG12_Packed,
        PixelType_Gvsp_BayerBG8,
        PixelType_Gvsp_BayerGB10,
        PixelType_Gvsp_BayerGB10_Packed,
        PixelType_Gvsp_BayerGB12,
        PixelType_Gvsp_BayerGB12_Packed,
        PixelType_Gvsp_BayerGB8,
        PixelType_Gvsp_BayerGR10,
        PixelType_Gvsp_BayerGR10_Packed,
        PixelType_Gvsp_BayerGR12,
        PixelType_Gvsp_BayerGR12_Packed,
        PixelType_Gvsp_BayerGR8,
        PixelType_Gvsp_BayerRG10,
        PixelType_Gvsp_BayerRG10_Packed,
        PixelType_Gvsp_BayerRG12,
        PixelType_Gvsp_BayerRG12_Packed,
        PixelType_Gvsp_BayerRG8,
        PixelType_Gvsp_Mono10,
        PixelType_Gvsp_Mono12,
        PixelType_Gvsp_Mono8,
        PixelType_Gvsp_YUV422_Packed,
        PixelType_Gvsp_YUV422_YUYV_Packed,
    )
    INDUSTRIAL_CAMERA_AVAILABLE = True
except Exception as _mv_import_error:
    INDUSTRIAL_CAMERA_AVAILABLE = False
    MV_IMPORT_ERROR = str(_mv_import_error)

logger = logging.getLogger(__name__)


class CameraError(Exception):
    pass


def get_resource_path(relative_path: str) -> str:
    try:
        base_path = sys._MEIPASS  # type: ignore[attr-defined]
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


def get_config_path(config_file: str) -> str:
    if getattr(sys, "frozen", False):
        current_dir = os.path.dirname(sys.executable)
    else:
        current_dir = os.path.dirname(os.path.abspath(__file__))

    external_config_path = os.path.join(current_dir, config_file)
    if os.path.exists(external_config_path):
        return external_config_path

    internal_config_path = get_resource_path(config_file)
    if os.path.exists(internal_config_path):
        return internal_config_path

    return external_config_path


class CameraConfig:
    def __init__(self, config_file: str = "config.json"):
        self.config_file = get_config_path(config_file)
        self.cameras: List[Dict[str, Any]] = []
        self.load_config()

    def load_config(self):
        try:
            with open(self.config_file, "r", encoding="utf-8") as f:
                config = json.load(f)

            cameras_data = config.get("cameras", [])
            if cameras_data and "cameras_config" in cameras_data[0]:
                merged = []
                for camera_data in cameras_data:
                    item: Dict[str, Any] = {}
                    item.update(camera_data.get("cameras_config", {}))
                    item.update(camera_data.get("detection_config", {}))

                    plc_conf = camera_data.get("plc_config", {})
                    for key, value in plc_conf.items():
                        if key in ["ip", "port", "enabled"]:
                            item[f"plc_{key}"] = value
                        else:
                            item[key] = value

                    trigger_conf = camera_data.get("hardware_trigger", {})
                    for key, value in trigger_conf.items():
                        item[f"hardware_trigger_{key}"] = value

                    merged.append(item)
                self.cameras = merged
            else:
                self.cameras = cameras_data

            logger.info(f"Loaded {len(self.cameras)} camera configs")
        except FileNotFoundError:
            logger.error(f"Config not found: {self.config_file}")
            self.cameras = []
        except json.JSONDecodeError:
            logger.error(f"Config parse failed: {self.config_file}")
            self.cameras = []

    def get_enabled_cameras(self, max_count: Optional[int] = None) -> List[Dict[str, Any]]:
        enabled = [cam for cam in self.cameras if cam.get("enabled", True)]
        if max_count is None:
            return enabled
        return enabled[:max_count]


def _is_bayer_data(pixel_type: int) -> bool:
    return pixel_type in {
        PixelType_Gvsp_BayerGR8, PixelType_Gvsp_BayerRG8, PixelType_Gvsp_BayerGB8, PixelType_Gvsp_BayerBG8,
        PixelType_Gvsp_BayerGR10, PixelType_Gvsp_BayerRG10, PixelType_Gvsp_BayerGB10, PixelType_Gvsp_BayerBG10,
        PixelType_Gvsp_BayerGR12, PixelType_Gvsp_BayerRG12, PixelType_Gvsp_BayerGB12, PixelType_Gvsp_BayerBG12,
        PixelType_Gvsp_BayerGR10_Packed, PixelType_Gvsp_BayerRG10_Packed,
        PixelType_Gvsp_BayerGB10_Packed, PixelType_Gvsp_BayerBG10_Packed,
        PixelType_Gvsp_BayerGR12_Packed, PixelType_Gvsp_BayerRG12_Packed,
        PixelType_Gvsp_BayerGB12_Packed, PixelType_Gvsp_BayerBG12_Packed,
    }


def _is_color_data(pixel_type: int) -> bool:
    return _is_bayer_data(pixel_type) or pixel_type in {
        PixelType_Gvsp_YUV422_Packed,
        PixelType_Gvsp_YUV422_YUYV_Packed,
    }


def _color_numpy(data: np.ndarray, width: int, height: int, pixel_type: int) -> np.ndarray:
    if pixel_type == PixelType_Gvsp_YUV422_Packed:
        yuv_data = data.reshape((height, width * 2))
        return cv2.cvtColor(yuv_data, cv2.COLOR_YUV2BGR_UYVY)
    if pixel_type == PixelType_Gvsp_YUV422_YUYV_Packed:
        yuv_data = data.reshape((height, width * 2))
        return cv2.cvtColor(yuv_data, cv2.COLOR_YUV2BGR_YUYV)
    if _is_bayer_data(pixel_type):
        gray = data.reshape((height, width))
        if pixel_type in {PixelType_Gvsp_BayerGR8, PixelType_Gvsp_BayerGR10, PixelType_Gvsp_BayerGR12}:
            return cv2.cvtColor(gray, cv2.COLOR_BAYER_BG2BGR)
        if pixel_type in {PixelType_Gvsp_BayerRG8, PixelType_Gvsp_BayerRG10, PixelType_Gvsp_BayerRG12}:
            return cv2.cvtColor(gray, cv2.COLOR_BAYER_GB2BGR)
        if pixel_type in {PixelType_Gvsp_BayerGB8, PixelType_Gvsp_BayerGB10, PixelType_Gvsp_BayerGB12}:
            return cv2.cvtColor(gray, cv2.COLOR_BAYER_GB2RGB)
        return cv2.cvtColor(gray, cv2.COLOR_BAYER_GR2BGR)
    gray = data[: width * height].reshape((height, width))
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _mono_numpy(data: np.ndarray, width: int, height: int, pixel_type: int) -> np.ndarray:
    if pixel_type in {PixelType_Gvsp_Mono8, PixelType_Gvsp_Mono10, PixelType_Gvsp_Mono12}:
        gray = data[: width * height].reshape((height, width))
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    gray = data[: width * height].reshape((height, width))
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


class IndustrialCameraStream:
    """Hikrobot industrial camera stream via MVS SDK."""

    def __init__(self, camera_config: Dict[str, Any]):
        if not INDUSTRIAL_CAMERA_AVAILABLE:
            raise CameraError(f"Industrial camera SDK unavailable: {globals().get('MV_IMPORT_ERROR', 'unknown')}")

        self.camera_config = camera_config
        self.CAMERA_NAME = camera_config.get("name", "Unknown")
        self.CAMERA_ID = str(camera_config.get("id", self.CAMERA_NAME))
        self.IP = str(camera_config.get("camera_ip", camera_config.get("ip", "")))
        self.obj_cam = MvCamera()
        self.is_running = True

        self.enable_detection = camera_config.get("enable_detection", False)
        self.trigger_detection_enabled = camera_config.get("trigger_detection_enabled", False)
        self.has_plc = camera_config.get("plc_enabled", False)
        self.detection_allowed = True
        self.detection_frame = None
        self._frozen_result_emitted = False
        self.last_captured_frame = None
        self.triggered_frame = None
        self.triggered_detection_result = None

        self.save_detected_images = camera_config.get("save_detected_images", False)
        self.save_all = camera_config.get("save_all", False)
        self.image_saver = None
        if self.save_detected_images or self.save_all:
            save_dir = camera_config.get("save_dir", "detected_images")
            max_queue_size = camera_config.get("save_queue_size", 20)
            self.image_saver = AsyncImageSaver(save_dir=save_dir, max_queue_size=max_queue_size)

        self.detector = create_detector(camera_config) if self.enable_detection else None

        self._connect_by_ip()
        self._start_grabbing()

    def _connect_by_ip(self):
        st_device_list = MV_CC_DEVICE_INFO_LIST()
        ret = MvCamera.MV_CC_EnumDevices(MV_GIGE_DEVICE, st_device_list)
        if ret != 0:
            raise CameraError(f"Enum devices failed: {ret}")

        found = None
        for i in range(st_device_list.nDeviceNum):
            dev = cast(st_device_list.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
            if dev.nTLayerType != MV_GIGE_DEVICE:
                continue
            nip1 = ((dev.SpecialInfo.stGigEInfo.nCurrentIp & 0xFF000000) >> 24)
            nip2 = ((dev.SpecialInfo.stGigEInfo.nCurrentIp & 0x00FF0000) >> 16)
            nip3 = ((dev.SpecialInfo.stGigEInfo.nCurrentIp & 0x0000FF00) >> 8)
            nip4 = (dev.SpecialInfo.stGigEInfo.nCurrentIp & 0x000000FF)
            current_ip = f"{nip1}.{nip2}.{nip3}.{nip4}"
            if current_ip == self.IP:
                found = dev
                break

        if found is None:
            raise CameraError(f"Industrial camera not found by ip: {self.IP}")

        ret = self.obj_cam.MV_CC_CreateHandle(found)
        if ret != 0:
            raise CameraError(f"Create industrial camera handle failed: {ret}")

        ret = self.obj_cam.MV_CC_OpenDevice()
        if ret != 0:
            raise CameraError(f"Open industrial camera failed: {ret}")

        config_file_path = self.camera_config.get("config_file_path")
        if config_file_path:
            profile_path = get_resource_path(config_file_path)
            if os.path.exists(profile_path):
                try:
                    self.obj_cam.MV_CC_FeatureLoad(profile_path)
                except Exception:
                    pass

        self.obj_cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
        logger.info(f"Industrial camera connected: {self.CAMERA_NAME} ({self.IP})")

    def _start_grabbing(self):
        ret = self.obj_cam.MV_CC_StartGrabbing()
        if ret != 0:
            raise CameraError(f"Industrial camera start grabbing failed: {ret}")

    def set_detection_allowed(self, allowed: bool):
        self.detection_allowed = allowed
        if allowed:
            self.detection_frame = None
            self._frozen_result_emitted = False

    def _detect_with_result(self, image, camera_name="Unknown"):
        if not self.detector:
            return image, []
        try:
            if hasattr(self.detector, "detect_with_objects"):
                return self.detector.detect_with_objects(image, camera_name, self.image_saver)
            return self.detector.detect(image, camera_name), []
        except Exception as exc:
            logger.error(f"Detection failed: {exc}")
            return image, []

    def get_frame(self):
        if not self.is_running:
            return None

        st_out = MV_FRAME_OUT()
        memset(byref(st_out), 0, sizeof(st_out))
        ret = self.obj_cam.MV_CC_GetImageBuffer(st_out, 1000)
        if ret != 0:
            return None

        try:
            width = st_out.stFrameInfo.nWidth
            height = st_out.stFrameInfo.nHeight
            frame_len = st_out.stFrameInfo.nFrameLen
            pixel_type = st_out.stFrameInfo.enPixelType
            data_buf = cast(st_out.pBufAddr, POINTER(ctypes.c_ubyte * frame_len)).contents
            data = np.frombuffer(data_buf, dtype=np.uint8)

            if _is_color_data(pixel_type):
                frame = _color_numpy(data, width, height, pixel_type)
            else:
                frame = _mono_numpy(data, width, height, pixel_type)
        finally:
            self.obj_cam.MV_CC_FreeImageBuffer(st_out)

        self.last_captured_frame = frame.copy()

        if self.trigger_detection_enabled and self.triggered_frame is not None:
            return self.triggered_frame

        if not self.detection_allowed and self.detection_frame is not None:
            if (
                not self._frozen_result_emitted
                and self.has_plc
                and hasattr(self, "worker")
                and hasattr(self.worker, "detection_result")
            ):
                self.worker.detection_result.emit(self.CAMERA_ID, True)
                self._frozen_result_emitted = True
            return self.detection_frame

        detection_result = None
        if self.enable_detection and self.detection_allowed and not self.trigger_detection_enabled:
            frame, detection_result = self._detect_with_result(frame, self.CAMERA_NAME)
            if detection_result and len(detection_result) > 0:
                self.detection_frame = frame.copy()
                self._frozen_result_emitted = False
                if hasattr(self, "worker") and hasattr(self.worker, "detection_control"):
                    self.worker.detection_control.emit(self.CAMERA_ID, False)

        if self.has_plc and detection_result is not None:
            detected = len(detection_result) > 0
            if hasattr(self, "worker") and hasattr(self.worker, "detection_result"):
                self.worker.detection_result.emit(self.CAMERA_ID, detected)
            if detected:
                self.detection_frame = frame.copy()

        return frame

    def trigger_detection(self):
        if not (self.enable_detection and self.last_captured_frame is not None):
            return False
        try:
            result_image, detection_result = self._detect_with_result(
                self.last_captured_frame.copy(), self.CAMERA_NAME
            )
            self.triggered_frame = result_image
            self.triggered_detection_result = detection_result
            if self.has_plc and hasattr(self, "worker") and hasattr(self.worker, "detection_result"):
                self.worker.detection_result.emit(self.CAMERA_ID, len(detection_result) > 0)
            return True
        except Exception as exc:
            logger.error(f"Trigger detection failed: {exc}")
            return False

    def release(self):
        self.is_running = False
        try:
            self.obj_cam.MV_CC_StopGrabbing()
        except Exception:
            pass
        try:
            self.obj_cam.MV_CC_CloseDevice()
        except Exception:
            pass
        try:
            self.obj_cam.MV_CC_DestroyHandle()
        except Exception:
            pass


class HKCameraStream:
    """Hikvision SDK stream implementation (old_version style)."""

    def __init__(self, camera_config: Dict[str, Any]):
        if not HIK_SDK_AVAILABLE:
            raise CameraError(f"Hik SDK unavailable: {globals().get('HIK_SDK_IMPORT_ERROR', 'unknown')}")

        self.camera_config = camera_config
        self.CAMERA_NAME = camera_config.get("name", "Unknown")
        self.CAMERA_ID = str(camera_config.get("id", self.CAMERA_NAME))
        self.DEV_IP = create_string_buffer(
            str(camera_config.get("camera_ip", camera_config.get("ip", ""))).encode()
        )
        sdk_port = camera_config.get("sdk_port", camera_config.get("hik_sdk_port"))
        if sdk_port is None:
            base_port = camera_config.get("camera_port", camera_config.get("port", 8000))
            # If config port is RTSP(554), use Hik SDK default service port(8000).
            sdk_port = 8000 if int(base_port) == 554 else base_port
        self.DEV_PORT = int(sdk_port)
        self.DEV_USER_NAME = create_string_buffer(str(camera_config.get("username", "")).encode())
        self.DEV_PASSWORD = create_string_buffer(str(camera_config.get("password", "")).encode())
        self.CHANNEL = int(camera_config.get("channel", 1))
        self.Stream_Type = int(camera_config.get("stream_type", 0))

        self.enable_detection = camera_config.get("enable_detection", False)
        self.acquisition_mode = camera_config.get("acquisition_mode", "realtime")
        self.hardware_trigger_enabled = bool(camera_config.get("hardware_trigger_enabled", False))
        self.hardware_trigger_mode = str(camera_config.get("hardware_trigger_trigger_mode", "Off")).lower()
        if self.hardware_trigger_enabled and self.hardware_trigger_mode == "on":
            self.acquisition_mode = "photoelectric_trigger"

        if "plc_config" in camera_config:
            plc_conf = camera_config.get("plc_config", {})
            self.trigger_detection_enabled = plc_conf.get("trigger_detection_enabled", False)
            self.has_plc = plc_conf.get("enabled", False)
        else:
            self.trigger_detection_enabled = camera_config.get("trigger_detection_enabled", False)
            self.has_plc = camera_config.get("plc_enabled", False)

        self.detection_allowed = True
        self.detection_frame = None
        self._frozen_result_emitted = False
        self.last_captured_frame = None
        self.triggered_frame = None
        self.triggered_detection_result = None

        self.save_detected_images = camera_config.get("save_detected_images", False)
        self.save_all = camera_config.get("save_all", False)
        self.image_saver = None
        if self.save_detected_images or self.save_all:
            save_dir = camera_config.get("save_dir", "detected_images")
            max_queue_size = camera_config.get("save_queue_size", 20)
            self.image_saver = AsyncImageSaver(save_dir=save_dir, max_queue_size=max_queue_size)
        self.detector = create_detector(camera_config) if self.enable_detection else None

        self.frame_queue = Queue(maxsize=max(1, int(camera_config.get("hk_frame_queue_size", 5))))
        self.display_buf = max(1, int(camera_config.get("hk_display_buf", 1)))
        self.error_count = 0
        self.max_error_count = 5

        self.is_running = True
        self.lUserId = -1
        self.lRealPlayHandle = -1
        self.PlayCtrl_Port = c_long(-1)
        self.Objdll = None
        self.Playctrldll = None
        self.device_info = None
        self._sdk_initialized = False
        self.funcRealDataCallBack_V30 = None
        self.FuncDecCB = None

        self._load_sdk()
        self._login_device()
        self._start_play()
        time.sleep(0.2)

    def _load_sdk(self):
        self.Objdll = ctypes.CDLL(netsdkdllpath)
        self.Playctrldll = ctypes.CDLL(playM4dllpath)
        if not self.Objdll.NET_DVR_Init():
            raise CameraError(f"NET_DVR_Init failed for {self.CAMERA_NAME}")
        self._sdk_initialized = True
        try:
            self.Objdll.NET_DVR_SetLogToFile(3, bytes("./SdkLog_Python/", encoding="utf-8"), False)
        except Exception:
            pass

    def _login_device(self):
        self.device_info = NET_DVR_DEVICEINFO_V30()
        self.lUserId = self.Objdll.NET_DVR_Login_V30(
            self.DEV_IP,
            self.DEV_PORT,
            self.DEV_USER_NAME,
            self.DEV_PASSWORD,
            byref(self.device_info),
        )
        if self.lUserId < 0:
            err_code = self.Objdll.NET_DVR_GetLastError()
            raise CameraError(f"Login failed [{self.CAMERA_NAME}] error={err_code}")
        logger.info(f"Hik SDK camera login success: {self.CAMERA_NAME}")

    def _start_play(self):
        if not self.Playctrldll.PlayM4_GetPort(byref(self.PlayCtrl_Port)):
            raise CameraError(f"PlayM4_GetPort failed [{self.CAMERA_NAME}]")

        self.funcRealDataCallBack_V30 = REALDATACALLBACK(self.RealDataCallBack_V30)
        preview_info = NET_DVR_PREVIEWINFO()
        preview_info.hPlayWnd = 0
        preview_info.lChannel = self.CHANNEL
        preview_info.dwStreamType = self.Stream_Type
        preview_info.dwLinkMode = 0
        preview_info.bBlocked = 1

        self.lRealPlayHandle = self.Objdll.NET_DVR_RealPlay_V40(
            self.lUserId, byref(preview_info), self.funcRealDataCallBack_V30, None
        )
        if self.lRealPlayHandle < 0:
            err_code = self.Objdll.NET_DVR_GetLastError()
            raise CameraError(f"RealPlay failed [{self.CAMERA_NAME}] error={err_code}")

        self.Playctrldll.PlayM4_SetDisplayBuf(self.PlayCtrl_Port, self.display_buf)
        logger.info(f"Hik SDK camera streaming: {self.CAMERA_NAME}")

    def set_detection_allowed(self, allowed: bool):
        self.detection_allowed = allowed
        if allowed:
            self.detection_frame = None
            self._frozen_result_emitted = False

    def _detect_with_result(self, image, camera_name="Unknown"):
        if not self.detector:
            return image, []
        try:
            if hasattr(self.detector, "detect_with_objects"):
                return self.detector.detect_with_objects(image, camera_name, self.image_saver)
            return self.detector.detect(image, camera_name), []
        except Exception as exc:
            logger.error(f"Detection failed: {exc}")
            return image, []

    def RealDataCallBack_V30(self, lPlayHandle, dwDataType, pBuffer, dwBufSize, pUser):
        if dwDataType == NET_DVR_SYSHEAD:
            if self.Playctrldll.PlayM4_OpenStream(self.PlayCtrl_Port, pBuffer, dwBufSize, 1024 * 1024):
                self.FuncDecCB = DECCBFUNWIN(self.DecCBFun)
                self.Playctrldll.PlayM4_SetDecCallBackExMend(
                    self.PlayCtrl_Port, self.FuncDecCB, None, 0, None
                )
                self.Playctrldll.PlayM4_Play(self.PlayCtrl_Port, None)
        elif dwDataType == NET_DVR_STREAMDATA:
            self.Playctrldll.PlayM4_InputData(self.PlayCtrl_Port, pBuffer, dwBufSize)

    def DecCBFun(self, nPort, pBuf, nSize, pFrameInfo, nUser, nReserved2):
        if not pFrameInfo:
            return
        # T_YV12 frame type
        if pFrameInfo.contents.nType != 3:
            return
        try:
            if self.frame_queue.full():
                self.frame_queue.get_nowait()
            frame_data = ctypes.string_at(pBuf, nSize)
            self.frame_queue.put_nowait((frame_data, pFrameInfo.contents.nWidth, pFrameInfo.contents.nHeight))
        except Empty:
            pass
        except Full:
            try:
                self.frame_queue.get_nowait()
                self.frame_queue.put_nowait((frame_data, pFrameInfo.contents.nWidth, pFrameInfo.contents.nHeight))
            except Exception:
                pass
        except Exception:
            pass

    def get_frame(self):
        if not self.is_running:
            return None

        try:
            yuv_data, width, height = self.frame_queue.get(timeout=0.1)
            yuv = np.frombuffer(yuv_data, dtype=np.uint8).reshape((height * 3 // 2, width))
            frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_YV12)
        except Empty:
            return None
        except Exception:
            self.error_count += 1
            if self.error_count >= self.max_error_count:
                logger.warning(f"Hik frame decode unstable [{self.CAMERA_NAME}]")
                self.error_count = 0
            return None

        self.last_captured_frame = frame.copy()

        if self.acquisition_mode == "photoelectric_trigger" and self.triggered_frame is not None:
            return self.triggered_frame

        if self.trigger_detection_enabled and self.triggered_frame is not None:
            return self.triggered_frame

        if not self.detection_allowed and self.detection_frame is not None:
            if (
                not self._frozen_result_emitted
                and self.has_plc
                and hasattr(self, "worker")
                and hasattr(self.worker, "detection_result")
            ):
                self.worker.detection_result.emit(self.CAMERA_ID, True)
                self._frozen_result_emitted = True
            return self.detection_frame

        detection_result = None
        if self.enable_detection and self.detection_allowed and not self.trigger_detection_enabled:
            frame, detection_result = self._detect_with_result(frame, self.CAMERA_NAME)
            if detection_result and len(detection_result) > 0:
                self.detection_frame = frame.copy()
                self._frozen_result_emitted = False
                if hasattr(self, "worker") and hasattr(self.worker, "detection_control"):
                    self.worker.detection_control.emit(self.CAMERA_ID, False)

        if self.has_plc and detection_result is not None:
            detected = len(detection_result) > 0
            if hasattr(self, "worker") and hasattr(self.worker, "detection_result"):
                self.worker.detection_result.emit(self.CAMERA_ID, detected)
            if detected:
                self.detection_frame = frame.copy()

        return frame

    def trigger_detection(self):
        if not (self.enable_detection and self.last_captured_frame is not None):
            return False

        try:
            result_image, detection_result = self._detect_with_result(
                self.last_captured_frame.copy(), self.CAMERA_NAME
            )
            self.triggered_frame = result_image
            self.triggered_detection_result = detection_result

            if self.has_plc and hasattr(self, "worker") and hasattr(self.worker, "detection_result"):
                detected = len(detection_result) > 0
                self.worker.detection_result.emit(self.CAMERA_ID, detected)
            return True
        except Exception as exc:
            logger.error(f"Trigger detection failed: {exc}")
            return False

    def release(self):
        self.is_running = False
        try:
            if self.lRealPlayHandle >= 0 and self.Objdll is not None:
                self.Objdll.NET_DVR_StopRealPlay(self.lRealPlayHandle)
                self.lRealPlayHandle = -1
        except Exception:
            pass
        try:
            if self.PlayCtrl_Port.value > -1 and self.Playctrldll is not None:
                self.Playctrldll.PlayM4_Stop(self.PlayCtrl_Port)
                self.Playctrldll.PlayM4_CloseStream(self.PlayCtrl_Port)
                self.Playctrldll.PlayM4_FreePort(self.PlayCtrl_Port)
                self.PlayCtrl_Port = c_long(-1)
        except Exception:
            pass
        try:
            if self.lUserId >= 0 and self.Objdll is not None:
                self.Objdll.NET_DVR_Logout(self.lUserId)
                self.lUserId = -1
        except Exception:
            pass
        try:
            if self._sdk_initialized and self.Objdll is not None:
                self.Objdll.NET_DVR_Cleanup()
        except Exception:
            pass


class OpenCVCameraStream:
    """Unified camera stream implementation.

    Notes:
    - Supports realtime and photoelectric_trigger modes via config.
    - In photoelectric_trigger mode, camera should naturally output frames only on trigger.
    """

    def __init__(self, camera_config: Dict[str, Any]):
        self.camera_config = camera_config
        self.CAMERA_NAME = camera_config.get("name", "Unknown")
        self.CAMERA_ID = str(camera_config.get("id", self.CAMERA_NAME))
        self.camera_type = str(camera_config.get("type", "network")).lower()
        self.config_file_path = camera_config.get("config_file_path")
        self.stream_type = camera_config.get("stream_type", 0)

        self.enable_detection = camera_config.get("enable_detection", False)
        self.acquisition_mode = camera_config.get("acquisition_mode", "realtime")
        # Realtime mode prefers freshness over continuity: drop stale buffered frames.
        self.low_latency = bool(camera_config.get("low_latency", True))
        self.max_drain_grabs = max(0, int(camera_config.get("max_drain_grabs", 3)))
        self.hardware_trigger_enabled = bool(camera_config.get("hardware_trigger_enabled", False))
        self.hardware_trigger_mode = str(camera_config.get("hardware_trigger_trigger_mode", "Off")).lower()
        if self.hardware_trigger_enabled and self.hardware_trigger_mode == "on":
            self.acquisition_mode = "photoelectric_trigger"

        if "plc_config" in camera_config:
            plc_conf = camera_config.get("plc_config", {})
            self.trigger_detection_enabled = plc_conf.get("trigger_detection_enabled", False)
            self.has_plc = plc_conf.get("enabled", False)
        else:
            self.trigger_detection_enabled = camera_config.get("trigger_detection_enabled", False)
            self.has_plc = camera_config.get("plc_enabled", False)

        self.detection_allowed = True
        self.detection_frame = None
        self._frozen_result_emitted = False

        self.last_captured_frame = None
        self.triggered_frame = None
        self.triggered_detection_result = None

        self.save_detected_images = camera_config.get("save_detected_images", False)
        self.save_all = camera_config.get("save_all", False)
        self.image_saver = None
        if self.save_detected_images or self.save_all:
            save_dir = camera_config.get("save_dir", "detected_images")
            max_queue_size = camera_config.get("save_queue_size", 20)
            self.image_saver = AsyncImageSaver(save_dir=save_dir, max_queue_size=max_queue_size)

        self.detector = create_detector(camera_config) if self.enable_detection else None

        self.capture = None
        self.is_running = True
        self._open_capture()

    def _build_source_candidates(self):
        source = self.camera_config.get("source")
        if source is not None:
            direct_source = int(source) if isinstance(source, str) and source.isdigit() else source
            return [direct_source]

        url = self.camera_config.get("url")
        if url:
            return [url]

        ip = self.camera_config.get("camera_ip", self.camera_config.get("ip"))
        if not ip:
            index = self.camera_config.get("index", 0)
            return [int(index)]

        username = self.camera_config.get("username", "")
        password = self.camera_config.get("password", "")
        configured_port = self.camera_config.get("camera_port", self.camera_config.get("port", 554))
        channel = self.camera_config.get("channel", 1)
        stream_type = self.stream_type

        channel_candidates = []
        try:
            channel_int = int(channel)
            stream_type_int = int(stream_type)
            if channel_int < 100:
                channel_candidates.append(channel_int * 100 + stream_type_int + 1)
            else:
                channel_candidates.append(channel_int)
        except (TypeError, ValueError):
            channel_candidates.append(channel)

        if channel not in channel_candidates:
            channel_candidates.append(channel)

        ports = [configured_port]
        if str(configured_port) != "554":
            ports.append(554)

        if username:
            auth_variants = [
                f"{username}:{password}@",
                f"{quote(str(username), safe='')}:{quote(str(password), safe='')}@",
            ]
        else:
            auth_variants = [""]

        sources = []
        for auth in auth_variants:
            for port in ports:
                for ch in channel_candidates:
                    sources.append(f"rtsp://{auth}{ip}:{port}/Streaming/Channels/{ch}")
                    sources.append(f"rtsp://{auth}{ip}:{port}/Streaming/channels/{ch}")

        # Keep order while removing duplicates.
        dedup_sources = []
        seen = set()
        for item in sources:
            key = str(item)
            if key in seen:
                continue
            seen.add(key)
            dedup_sources.append(item)
        return dedup_sources

    def _open_capture(self):
        if self.config_file_path and self.camera_type == "industrial":
            profile_path = get_resource_path(self.config_file_path)
            if os.path.exists(profile_path):
                logger.info(f"Using industrial camera profile: {profile_path}")
            else:
                logger.warning(f"Industrial camera profile not found: {profile_path}")

        sources = self._build_source_candidates()
        last_source = None
        for source in sources:
            last_source = source
            capture = cv2.VideoCapture(source)
            if capture.isOpened():
                # Try to keep decoder queue short to reduce end-to-end latency.
                try:
                    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception:
                    pass
                self.capture = capture
                logger.info(f"Camera connected: {self.CAMERA_NAME} source={source}")
                return
            capture.release()

        raise CameraError(
            f"Open camera failed: {self.CAMERA_NAME} tried={len(sources)} last_source={last_source}"
        )

    def set_detection_allowed(self, allowed: bool):
        self.detection_allowed = allowed
        if allowed:
            self.detection_frame = None
            self._frozen_result_emitted = False

    def _detect_with_result(self, image, camera_name="Unknown"):
        if not self.detector:
            return image, []
        try:
            if hasattr(self.detector, "detect_with_objects"):
                return self.detector.detect_with_objects(image, camera_name, self.image_saver)
            return self.detector.detect(image, camera_name), []
        except Exception as exc:
            logger.error(f"Detection failed: {exc}")
            return image, []

    def get_frame(self):
        if not self.capture or not self.is_running:
            return None

        if self.low_latency and self.acquisition_mode == "realtime":
            # Drain stale frames and decode the newest available frame.
            drained = False
            for _ in range(self.max_drain_grabs):
                if not self.capture.grab():
                    break
                drained = True

            if drained:
                ok, frame = self.capture.retrieve()
                if not ok or frame is None:
                    ok, frame = self.capture.read()
            else:
                ok, frame = self.capture.read()
        else:
            ok, frame = self.capture.read()

        if not ok or frame is None:
            return None

        self.last_captured_frame = frame.copy()

        if self.acquisition_mode == "photoelectric_trigger" and self.triggered_frame is not None:
            return self.triggered_frame

        if self.trigger_detection_enabled and self.triggered_frame is not None:
            return self.triggered_frame

        if not self.detection_allowed and self.detection_frame is not None:
            if (
                not self._frozen_result_emitted
                and self.has_plc
                and hasattr(self, "worker")
                and hasattr(self.worker, "detection_result")
            ):
                self.worker.detection_result.emit(self.CAMERA_ID, True)
                self._frozen_result_emitted = True
            return self.detection_frame

        detection_result = None
        if self.enable_detection and self.detection_allowed and not self.trigger_detection_enabled:
            frame, detection_result = self._detect_with_result(frame, self.CAMERA_NAME)
            if detection_result and len(detection_result) > 0:
                self.detection_frame = frame.copy()
                self._frozen_result_emitted = False
                if hasattr(self, "worker") and hasattr(self.worker, "detection_control"):
                    self.worker.detection_control.emit(self.CAMERA_ID, False)

        if self.has_plc and detection_result is not None:
            detected = len(detection_result) > 0
            if hasattr(self, "worker") and hasattr(self.worker, "detection_result"):
                self.worker.detection_result.emit(self.CAMERA_ID, detected)
            if detected:
                self.detection_frame = frame.copy()

        return frame

    def trigger_detection(self):
        if not (self.enable_detection and self.last_captured_frame is not None):
            return False

        try:
            result_image, detection_result = self._detect_with_result(
                self.last_captured_frame.copy(), self.CAMERA_NAME
            )
            self.triggered_frame = result_image
            self.triggered_detection_result = detection_result

            if self.has_plc and hasattr(self, "worker") and hasattr(self.worker, "detection_result"):
                detected = len(detection_result) > 0
                self.worker.detection_result.emit(self.CAMERA_ID, detected)
            return True
        except Exception as exc:
            logger.error(f"Trigger detection failed: {exc}")
            return False

    def release(self):
        self.is_running = False
        if self.capture:
            self.capture.release()
            self.capture = None


class CameraWorker(QThread):
    frame_ready = pyqtSignal(object, str)          # frame, camera_id
    detection_result = pyqtSignal(str, bool)       # camera_id, detected
    detection_control = pyqtSignal(str, bool)      # camera_id, detection_allowed

    def __init__(self, camera_config: Dict[str, Any]):
        super().__init__()
        self.camera_config = camera_config
        self.camera_id = str(camera_config.get("id", camera_config.get("name", "unknown")))
        self.camera = None
        self.is_running = True
        self.resources_released = False

        self.fps = max(1, int(camera_config.get("detection_fps", 15)))
        self.interval = 1.0 / self.fps
        self.hardware_trigger_enabled = bool(camera_config.get("hardware_trigger_enabled", False))
        self.hardware_trigger_mode = str(camera_config.get("hardware_trigger_trigger_mode", "Off")).lower()
        self.hardware_trigger_source = str(camera_config.get("hardware_trigger_trigger_source", "Line0"))
        self.hardware_trigger_activation = str(
            camera_config.get("hardware_trigger_trigger_activation", "RisingEdge")
        ).lower()
        delay_us = camera_config.get("hardware_trigger_trigger_delay", 0)
        try:
            self.hardware_trigger_delay_sec = max(0, int(delay_us)) / 1_000_000.0
        except (TypeError, ValueError):
            self.hardware_trigger_delay_sec = 0.0
        self.last_trigger_state = False

        self.detection_control.connect(self.set_detection_allowed)

    def _hardware_trigger_active(self) -> bool:
        return self.hardware_trigger_enabled and self.hardware_trigger_mode == "on"

    def _should_fire_trigger(self, status: bool) -> bool:
        previous = self.last_trigger_state
        fire = False
        activation = self.hardware_trigger_activation
        if activation == "risingedge":
            fire = (not previous) and status
        elif activation == "fallingedge":
            fire = previous and (not status)
        elif activation == "levelhigh":
            fire = status
        elif activation == "levellow":
            fire = not status
        else:
            fire = (not previous) and status
        self.last_trigger_state = status
        return fire

    def _execute_hardware_trigger(self):
        if self.hardware_trigger_delay_sec > 0:
            time.sleep(self.hardware_trigger_delay_sec)
        self.trigger_detection()

    def handle_plc_trigger(self, status: bool) -> bool:
        status = bool(status)
        if not self._hardware_trigger_active():
            self.last_trigger_state = status
            return False

        if self.hardware_trigger_source.lower() != "line0":
            self.last_trigger_state = status
            return False

        if not self._should_fire_trigger(status):
            return False

        threading.Thread(target=self._execute_hardware_trigger, daemon=True).start()
        return True

    def set_detection_allowed(self, camera_id: str, allowed: bool):
        if self.camera and self.camera.CAMERA_ID == str(camera_id):
            self.camera.set_detection_allowed(allowed)

    def run(self):
        try:
            camera_type = str(self.camera_config.get("type", "network")).lower()
            stream_backend = str(
                self.camera_config.get(
                    "stream_backend",
                    "hik_sdk" if camera_type == "network" else "opencv",
                )
            ).lower()

            if camera_type == "industrial":
                if INDUSTRIAL_CAMERA_AVAILABLE:
                    try:
                        self.camera = IndustrialCameraStream(self.camera_config)
                    except CameraError as exc:
                        logger.error(
                            f"Industrial SDK stream failed [{self.camera_id}], fallback OpenCV: {exc}"
                        )
                        self.camera = OpenCVCameraStream(self.camera_config)
                else:
                    logger.warning(
                        f"Industrial SDK unavailable for [{self.camera_id}], fallback OpenCV"
                    )
                    self.camera = OpenCVCameraStream(self.camera_config)
            elif camera_type == "network" and stream_backend in {"hik_sdk", "hcnetsdk", "old_version"}:
                if HIK_SDK_AVAILABLE:
                    try:
                        self.camera = HKCameraStream(self.camera_config)
                    except CameraError as exc:
                        logger.error(
                            f"Hik SDK stream failed [{self.camera_id}], fallback OpenCV RTSP: {exc}"
                        )
                        self.camera = OpenCVCameraStream(self.camera_config)
                else:
                    logger.warning(
                        f"Hik SDK unavailable for [{self.camera_id}], fallback OpenCV RTSP"
                    )
                    self.camera = OpenCVCameraStream(self.camera_config)
            else:
                self.camera = OpenCVCameraStream(self.camera_config)
            self.camera.worker = self

            last_emit_time = 0.0
            while self.is_running and self.camera.is_running:
                frame = self.camera.get_frame()
                if frame is None:
                    time.sleep(0.01)
                    continue

                now = time.time()
                if now - last_emit_time >= self.interval:
                    self.frame_ready.emit(frame, self.camera.CAMERA_ID)
                    last_emit_time = now
                else:
                    time.sleep(0.002)
        except CameraError as exc:
            logger.error(f"Camera connect error [{self.camera_id}]: {exc}")
            error_frame = np.zeros((240, 320, 3), dtype=np.uint8)
            self.frame_ready.emit(error_frame, self.camera_id)
        except Exception as exc:
            logger.error(f"Camera worker error [{self.camera_id}]: {exc}")
        finally:
            if self.camera and not self.resources_released:
                self.camera.release()
                self.resources_released = True

    def trigger_detection(self):
        try:
            if self.camera and (self.camera.trigger_detection_enabled or self._hardware_trigger_active()):
                return self.camera.trigger_detection()
        except Exception as exc:
            logger.error(f"Trigger detection error [{self.camera_id}]: {exc}")
        return False

    def stop(self):
        self.is_running = False
        if self.camera and not self.resources_released:
            if hasattr(self.camera, "image_saver") and self.camera.image_saver:
                self.camera.image_saver.stop()
            self.camera.release()
            self.resources_released = True
        self.wait(5000)
