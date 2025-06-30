"""
camera_model.py
카메라의 내부 및 외부 매개변수를 관리하는 모듈
"""

import cv2
import numpy as np
import json
import math

class CameraModel:
    """카메라의 내부 및 외부 매개변수를 관리하는 클래스"""
    
    def __init__(self, calibration_file=None):
        self.camera_matrix = None
        self.dist_coeffs = None
        self.rvecs = None
        self.tvecs = None
        self.ret = None
        
        if calibration_file:
            self.load_calibration(calibration_file)
        else:
            self._set_default_parameters()
    
    def load_calibration(self, calibration_file):
        """캘리브레이션 파일에서 카메라 매개변수 로드"""
        try:
            with open(calibration_file, 'r') as f:
                calib_data = json.load(f)
            
            # 카메라 매트릭스 로드
            self.camera_matrix = np.array(calib_data['CameraMatrix'], dtype=np.float32)
            
            # 왜곡 계수 로드
            self.dist_coeffs = np.array(calib_data['DistCoeffs'], dtype=np.float32)
            
            # 회전 벡터들 로드 (첫 번째 것을 기본값으로 사용)
            if 'RVecs' in calib_data and len(calib_data['RVecs']) > 0:
                rvec_data = calib_data['RVecs'][0]
                self.rvecs = np.array([rvec_data['Item0'], rvec_data['Item1'], rvec_data['Item2']], dtype=np.float32)
            
            # 변환 벡터들 로드 (첫 번째 것을 기본값으로 사용)
            if 'TVecs' in calib_data and len(calib_data['TVecs']) > 0:
                tvec_data = calib_data['TVecs'][0]
                self.tvecs = np.array([tvec_data['Item0'], tvec_data['Item1'], tvec_data['Item2']], dtype=np.float32)
            
            self.ret = calib_data.get('Ret', 1.0)
            
            print(f"✓ 캘리브레이션 데이터 로드 완료")
            print(f"  - 재투영 오차: {self.ret:.3f}")
            print(f"  - 초점거리: fx={self.camera_matrix[0,0]:.1f}, fy={self.camera_matrix[1,1]:.1f}")
            print(f"  - 주점: cx={self.camera_matrix[0,2]:.1f}, cy={self.camera_matrix[1,2]:.1f}")
            
        except Exception as e:
            print(f"⚠ 캘리브레이션 파일 로드 오류: {e}")
            print("기본 매개변수를 사용합니다.")
            self._set_default_parameters()
    
    def _set_default_parameters(self):
        """기본 카메라 매개변수 설정 (640x480 웹캠 기준)"""
        self.camera_matrix = np.array([
            [600.0, 0.0, 320.0],  # fx, 0, cx
            [0.0, 600.0, 240.0],  # 0, fy, cy
            [0.0, 0.0, 1.0]       # 0, 0, 1
        ], dtype=np.float32)
        
        # 왜곡 없음으로 설정
        self.dist_coeffs = np.zeros(5, dtype=np.float32)
        
        print("📷 기본 카메라 매개변수를 사용합니다.")
    
    def set_camera_pose(self, height, pitch_deg=0.0, yaw_deg=0.0, roll_deg=0.0):
        """
        카메라의 높이와 자세 설정
        
        Args:
            height: 지면으로부터의 높이 (미터)
            pitch_deg: 피치 각도 (아래쪽이 양수)
            yaw_deg: 요 각도 (오른쪽이 양수)
            roll_deg: 롤 각도 (시계방향이 양수)
        """
        # 각도를 라디안으로 변환
        pitch_rad = math.radians(pitch_deg)
        yaw_rad = math.radians(yaw_deg)
        roll_rad = math.radians(roll_deg)
        
        # 회전 벡터 (Rodrigues 형식)
        self.rvecs = np.array([pitch_rad, yaw_rad, roll_rad], dtype=np.float32)
        
        # 변환 벡터 (카메라 위치: 월드 원점에서 높이만큼 위)
        self.tvecs = np.array([0.0, 0.0, -height], dtype=np.float32)
        
        print(f"📐 카메라 자세 설정:")
        print(f"  - 높이: {height}m")
        print(f"  - 피치: {pitch_deg}° (아래쪽: {pitch_deg}°)")
        print(f"  - 요: {yaw_deg}°")
        print(f"  - 롤: {roll_deg}°")
    
    def get_info(self):
        """카메라 정보 반환"""
        if self.camera_matrix is None:
            return "카메라가 초기화되지 않았습니다."
        
        fx, fy = self.camera_matrix[0,0], self.camera_matrix[1,1]
        cx, cy = self.camera_matrix[0,2], self.camera_matrix[1,2]
        
        info = f"""
🎥 카메라 정보:
  - 초점거리: fx={fx:.1f}, fy={fy:.1f}
  - 주점: cx={cx:.1f}, cy={cy:.1f}
  - 왜곡계수: {len(self.dist_coeffs)}개
"""
        if self.rvecs is not None and self.tvecs is not None:
            height = -self.tvecs[2]
            pitch = math.degrees(self.rvecs[0])
            info += f"  - 높이: {height:.1f}m, 피치: {pitch:.1f}°"
        
        return info