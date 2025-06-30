"""
ground_projector.py
3D 지면 점을 2D 이미지 점으로 투영하는 모듈
"""

import cv2
import numpy as np

class GroundProjector:
    """3D 지면 점을 2D 이미지 점으로 투영하는 클래스"""
    
    def __init__(self, camera_model, image_width, image_height):
        """
        Args:
            camera_model: CameraModel 인스턴스
            image_width: 이미지 너비
            image_height: 이미지 높이
        """
        self.camera_model = camera_model
        self.image_width = image_width
        self.image_height = image_height
        
        print(f"🖼 투영기 초기화: {image_width}x{image_height}")
    
    def project_3d_to_2d(self, points_3d):
        """
        3D 점들을 2D 이미지 좌표로 투영
        
        Args:
            points_3d: 3D 점들의 numpy 배열 (N, 3)
            
        Returns:
            tuple: (투영된 2D 점들, 가시성 마스크)
        """
        if self.camera_model.camera_matrix is None or self.camera_model.rvecs is None:
            raise ValueError("카메라 매개변수가 설정되지 않았습니다.")
        
        # cv2.projectPoints를 사용하여 투영
        points_2d, _ = cv2.projectPoints(
            points_3d,
            self.camera_model.rvecs,
            self.camera_model.tvecs,
            self.camera_model.camera_matrix,
            self.camera_model.dist_coeffs
        )
        
        # 결과 형태 변환 (N, 1, 2) -> (N, 2)
        points_2d = points_2d.reshape(-1, 2)
        
        # 가시성 필터링
        valid_mask = self._filter_visible_points(points_3d, points_2d)
        
        return points_2d, valid_mask
    
    def _filter_visible_points(self, points_3d, points_2d):
        """
        가시적인 점들만 필터링
        
        Args:
            points_3d: 원본 3D 점들
            points_2d: 투영된 2D 점들
            
        Returns:
            numpy.ndarray: 가시성 마스크 (True: 가시적, False: 비가시적)
        """
        valid_mask = np.ones(len(points_3d), dtype=bool)
        
        # 1. 카메라 좌표계로 변환하여 Z값 확인 (카메라 전방 확인)
        #    월드 좌표계의 점들(points_3d)을 카메라 좌표계로 변환합니다.
        #    카메라 좌표계에서 Z > 0 이어야 카메라 앞에 있는 점입니다.
        
        # Rodrigues 변환을 사용하여 회전 벡터(rvecs)를 회전 행렬(R)로 변환
        R, _ = cv2.Rodrigues(self.camera_model.rvecs)
        t = self.camera_model.tvecs.reshape(3, 1) # tvecs를 컬럼 벡터로

        # 월드 좌표를 카메라 좌표로 변환: P_camera = R * P_world + t
        # 그러나 OpenCV의 projectPoints 내부 계산 방식과 일치시키려면,
        # P_camera = R * (P_world - C) 여기서 C는 월드 원점에서의 카메라 위치.
        # tvecs는 카메라에서 월드 원점으로의 변환이므로, C = -R_transpose * tvecs.
        # 또는 더 간단히, projectPoints는 월드좌표계 점을 입력받으므로,
        # 해당 함수가 내부적으로 좌표 변환을 수행합니다.
        # 여기서의 Z값 확인은 projectPoints 투영 후, 투영된 점들 중 화면 뒤로 간 것을 걸러내는 목적.
        # projectPoints는 카메라 앞(Z>0)에 있는 점들만 투영하는 것이 기본.
        # 따라서, points_3d가 이미 카메라 앞을 향하도록 generate_circle_points에서 수정했으므로,
        # 여기서 behind_camera 필터는 대부분 통과해야 정상입니다.

        # points_camera는 각 점의 카메라 좌표계에서의 위치 (X, Y, Z)
        # points_3d는 (N,3). R은 (3,3). tvecs는 (3,) 또는 (3,1)
        # 올바른 변환: points_camera = (R @ points_3d.T).T + tvecs_broadcastable
        # tvecs를 (1,3) 형태로 만들어 브로드캐스팅 가능하게 함
        tvecs_broadcastable = self.camera_model.tvecs.reshape(1, 3)
        points_camera = (R @ points_3d.T).T + tvecs_broadcastable

        # Z > 0 (카메라 앞) 인 점들 선택 -> 이 부분을 cv2.projectPoints에 맡김.
        # cv2.projectPoints는 카메라 뒤의 점들을 처리 (예: 매우 큰 좌표값으로 투영).
        # 따라서, 수동 Z 필터링은 제거하거나 매우 관대하게 설정.
        # in_front_of_camera_mask = points_camera[:, 2] > -float('inf') # 사실상 모든 점 통과
        # 또는, generate_circle_points가 이미 카메라 전방을 가정하므로, 이 필터는 단순화.
        # 가장 중요한 것은 화면 경계 필터링임.

        # 2. 이미지 경계 필터링 (투영된 점들이 이미지 내부에 있는지 확인)
        #    points_2d는 projectPoints로부터 나온 이미지 평면상의 좌표.
        on_screen_mask = (points_2d[:, 0] >= 0) & (points_2d[:, 0] < self.image_width) & \
                         (points_2d[:, 1] >= 0) & (points_2d[:, 1] < self.image_height)
        
        valid_mask = on_screen_mask

        if not hasattr(self, '_debug_filter_count'):
            self._debug_filter_count = 0

        if self._debug_filter_count < 7: # 처음 7번 (모든 거리에 대해 한번씩)
            print(f"--- Projector Filter (Call {self._debug_filter_count + 1}) ---")
            print(f"  Input 3D points: {len(points_3d)}")
            if len(points_3d) > 0:
                print(f"  World P3D[0]: {points_3d[0].round(2)}")
                # R_w2c, _ = cv2.Rodrigues(self.camera_model.rvecs)
                # cam_P3D_0 = (R_w2c @ points_3d[0].reshape(3,1) + self.camera_model.tvecs.reshape(3,1)).flatten()
                # print(f"  Cam P3D[0] (manual): {cam_P3D_0.round(2)}")
                print(f"  Projected P2D[0]: {points_2d[0].round(2)}")

            print(f"  On-screen points (strict): {np.sum(on_screen_mask)}")

            # 추가: 만약 모든 점이 화면 밖에 있다면, 어떤 값으로 투영되었는지 일부를 보여줌
            if np.sum(on_screen_mask) == 0 and len(points_2d) > 0:
                print(f"  All points off-screen. First 3 projected points: {points_2d[:3].round(2)}")
            elif np.sum(on_screen_mask) < 5 and len(points_2d) > 0 :
                 print(f"  Few points on-screen. On-screen points: {points_2d[on_screen_mask].round(2)}")


            self._debug_filter_count += 1

        return valid_mask
    
    def project_single_point(self, x, y, z=0.0):
        """
        단일 3D 점을 2D로 투영
        
        Args:
            x, y, z: 3D 좌표 (z=0은 지면)
            
        Returns:
            tuple: (투영된 픽셀 좌표, 가시성 여부)
        """
        point_3d = np.array([[x, y, z]], dtype=np.float32)
        points_2d, valid_mask = self.project_3d_to_2d(point_3d)
        
        if valid_mask[0]:
            return (int(points_2d[0, 0]), int(points_2d[0, 1])), True
        else:
            return (0, 0), False