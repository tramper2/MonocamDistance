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
        pitch_rad = math.radians(pitch_deg) # X축 회전
        yaw_rad = math.radians(yaw_deg)   # Y축 회전
        roll_rad = math.radians(roll_deg)  # Z축 회전
        
        # 회전 벡터 (Rodrigues 형식) - 월드 좌표계를 카메라 좌표계로 변환하는 회전
        # OpenCV 표준: rvec은 월드에서 카메라로의 회전을 나타냄
        self.rvecs = np.array([pitch_rad, yaw_rad, roll_rad], dtype=np.float32)
        
        # 변환 벡터 (tvec) - 월드 원점을 카메라 좌표계로 가져오는 변환 벡터
        # P_camera = R * P_world + tvec
        # 카메라가 월드 좌표 (0, 0, height)에 위치하고, 월드 X,Y,Z 축 방향으로 향하다가 pitch 만큼 기울어졌다고 가정.
        # 초기 카메라 위치 C_w = [0, 0, height]^T
        # 초기 카메라 회전 R_init = Identity (카메라 Z축이 월드 Z축, Y축이 월드 Y축 등)
        # 실제 카메라 회전 R_actual_cam_to_world 는 rvecs에 해당. (월드->카메라 회전의 역)
        # R_world_to_cam, _ = cv2.Rodrigues(self.rvecs)
        
        # tvec은 카메라 좌표계에서 월드 원점의 위치를 나타냄.
        # 또는, 월드 좌표계에서 카메라의 위치 C_w = (tx, ty, tz) 일때,
        # tvec = -R_world_to_cam * C_w

        # 카메라의 월드 좌표계에서의 위치를 (0, -height, 0)으로 가정하고 (Y축이 위, Z축이 앞)
        # 피치각은 X축 중심 회전 (아래를 보면 양수)
        # 월드좌표계: X 오른쪽, Y 위쪽, Z 카메라 방향 (전방)
        # 카메라 위치 C_w = [0, -height, 0] (지면으로부터 Y축으로 height만큼 위에 있음)
        # rvecs는 x축으로 pitch_rad 회전.

        R_world_to_camera, _ = cv2.Rodrigues(self.rvecs)

        # 카메라의 월드 좌표 (C_w): X=0, Y=height (지면이 Y=0일 때), Z=0 (카메라가 YZ 평면상에 위치)
        # 이 프로젝트의 3D 점들은 Z=0인 XY평면(지면)에 생성됨 (x,y,0)
        # 카메라 높이는 Z축 방향으로 해석하는 것이 일반적.
        # 월드: X-오른쪽, Y-앞, Z-위. 카메라 위치: (0,0,height)
        # 카메라 초기 방향: 월드 -Y축을 바라봄 (지면을 향함). 또는 월드 +Y축을 바라보다가 pitch.

        # 현재 distance_visualizer.generate_circle_points는 X,Y 평면에 점을 만듦 (Z=0)
        # 카메라가 Z=height에 있고, 처음에는 -Z 방향을 바라보다가 (지면을 봄)
        # X축 기준으로 pitch_deg 만큼 회전 (양수면 더 아래로, 음수면 위로)

        # OpenCV extrinsic: R, tvec طوریکه X_cam = R * X_world + tvec
        # 카메라가 월드 (0,0,height)에 있고, x축으로 pitch_rad 회전했다고 가정.
        # C_w = [0, 0, height]^T.
        # R은 월드에서 카메라로의 회전.
        # tvec = -R * C_w

        # 회전 행렬 (월드 -> 카메라), X축 중심 회전 (pitch)
        # R_x(pitch) = [[1, 0, 0], [0, cos(p), -sin(p)], [0, sin(p), cos(p)]]
        # self.rvecs = [pitch_rad, yaw_rad, roll_rad]
        # 여기서는 yaw, roll은 0으로 가정하고 pitch만 고려하여 tvec 계산

        R_x_pitch, _ = cv2.Rodrigues(np.array([pitch_rad, 0.0, 0.0], dtype=np.float32))

        # 카메라의 월드 기준 초기 위치 C_w = [0, 0, height]^T (Z축이 위)
        camera_position_world_initial = np.array([0.0, 0.0, height], dtype=np.float32).reshape(3,1)

        # tvec = -R_x(pitch) * C_w
        # (카메라가 먼저 이동 후, 월드 원점 기준으로 회전하는게 아니라,
        # 카메라 자체를 회전시키고 그 자세에서의 tvec을 구하는 것)
        # OpenCV의 tvec은 카메라 좌표계에서 월드 원점의 위치임.
        # P_cam = R * P_world + tvec
        # 만약 카메라가 월드 C_w에 있고 회전이 R(월드->캠)이라면, tvec = -R * C_w

        # 월드: X 오른쪽, Y 앞, Z 위
        # 카메라 초기 자세: Z축이 월드 -Z축을 향하도록 (지면을 바라봄)
        # 이는 월드 Y축 기준 180도 회전으로 달성 가능: r_initial = [0, pi, 0]
        # 또는 월드 X축 기준 180도 회전: r_initial = [pi, 0, 0] (카메라 Y가 -Y월드, Z가 -Z월드) - 이쪽이 더 직관적일 수 있음
        # (카메라 X는 그대로, Y는 아래, Z는 뒤쪽 -> 여기서 Z는 카메라 뒤쪽이므로, 앞을 보려면 추가 회전 필요)

        # 표준적인 카메라 자세: 카메라가 (0,0,H)에 있고, 수평(월드 Y축)에서 X축 기준으로 pitch만큼 아래로 (-pitch)
        # rvecs = [pitch_rad, 0, 0] 이고, tvecs = -R*C_w 가 맞음.
        # 문제는 이 설정에서 가까운 점들이 카메라 뒤로 투영된다는 것.

        # 시도: 카메라가 지면을 수직으로 내려다보는 것을 기본 자세로 설정 (pitch=0일 때)
        # 즉, 카메라 Z축이 월드 -Z축과 평행. 이는 월드 X축 기준 +90도 회전.
        # 이 상태에서 사용자가 입력하는 pitch_deg는 이 기본 자세에서의 추가적인 X축 회전으로 해석.
        # 표준적인 카메라 자세 정의로 복귀
        # 월드: X 오른쪽, Y 앞, Z 위
        # 카메라 위치 C_w = (0,0,height)
        # 카메라 회전 rvecs = [pitch_rad, yaw_rad, roll_rad] (X, Y, Z축 순차 회전이 아님, 축-각도 표현)
        # tvecs = -R_w2c * C_w

        self.rvecs = np.array([pitch_rad, yaw_rad, roll_rad], dtype=np.float32)

        R_w2c, _ = cv2.Rodrigues(self.rvecs)

        camera_position_world = np.array([0.0, 0.0, height], dtype=np.float32).reshape(3,1)

        self.tvecs = (-R_w2c @ camera_position_world).flatten()

        print(f"📐 카메라 자세 설정 (표준 방식):")
        print(f"  - 높이: {height}m (월드 Z축)")
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