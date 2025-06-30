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
        
        # 1. 카메라 좌표계로 변환하여 Z값 확인
        rvec_matrix, _ = cv2.Rodrigues(self.camera_model.rvecs)
        points_camera = (rvec_matrix @ points_3d.T).T + self.camera_model.tvecs
        
        # 카메라 뒤편 필터링 (Z > 0이어야 카메라 앞쪽)
        behind_camera = points_camera[:, 2] <= 0
        valid_mask[behind_camera] = False
        
        # 2. 이미지 경계 필터링
        out_of_bounds = (
            (points_2d[:, 0] < 0) | (points_2d[:, 0] >= self.image_width) |
            (points_2d[:, 1] < 0) | (points_2d[:, 1] >= self.image_height)
        )
        valid_mask[out_of_bounds] = False
        
        # 필터링 결과 요약
        total_points = len(points_3d)
        behind_count = np.sum(behind_camera)
        out_of_bounds_count = np.sum(out_of_bounds)
        visible_count = np.sum(valid_mask)
        
        # 디버그 정보 (처음 몇 번만 출력)
        if not hasattr(self, '_debug_count'):
            self._debug_count = 0
        
        if self._debug_count < 3:  # 처음 3번만 출력
            print(f"🔍 가시성 필터링:")
            print(f"  - 전체 점: {total_points}개")
            print(f"  - 카메라 뒤편: {behind_count}개")
            print(f"  - 경계 밖: {out_of_bounds_count}개")
            print(f"  - 가시적: {visible_count}개")
            self._debug_count += 1
        
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