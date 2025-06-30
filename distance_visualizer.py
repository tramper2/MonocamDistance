"""
distance_visualizer.py
거리 호와 레이블을 이미지에 그리는 모듈
"""

import cv2
import numpy as np
import math

class DistanceVisualizer:
    """거리 호와 레이블을 이미지에 그리는 클래스"""
    
    def __init__(self, projector):
        """
        Args:
            projector: GroundProjector 인스턴스
        """
        self.projector = projector
        
        # 거리별 색상 정의 (BGR 형식)
        self.colors = [
            (0, 255, 0),    # 녹색
            (255, 0, 0),    # 파란색
            (0, 0, 255),    # 빨간색
            (255, 255, 0),  # 청록색
            (255, 0, 255),  # 자홍색
            (0, 255, 255),  # 노란색
            (128, 255, 0),  # 연두색
            (255, 128, 0),  # 주황색
            (128, 0, 255),  # 보라색
            (0, 128, 255),  # 오렌지색
        ]
        
        print(f"🎨 시각화 도구 초기화: {len(self.colors)}가지 색상")
    
    def generate_circle_points(self, radius, num_points=180):
        """
        지면상의 원을 구성하는 3D 점들 생성 (카메라 시야각 고려)
        
        Args:
            radius: 원의 반지름 (미터)
            num_points: 원을 구성할 점의 개수
            
        Returns:
            numpy.ndarray: 3D 점들 (N, 3)
        """
        # 카메라의 시야각을 고려하여 점들을 생성
        # 카메라 중심에서 방사형으로 퍼지는 형태
        
        # 카메라의 대략적인 수평 시야각 (라디안)
        if hasattr(self.projector.camera_model, 'camera_matrix') and self.projector.camera_model.camera_matrix is not None:
            fx = self.projector.camera_model.camera_matrix[0, 0]
            fov_horizontal = 2 * np.arctan(self.projector.image_width / (2 * fx))
        else:
            fov_horizontal = np.radians(60)  # 기본값 60도
        
        # 카메라 중심에서 좌우로 퍼지는 각도 범위 설정
        # 시야각보다 약간 넓게 설정하여 화면 경계 밖 부분도 포함
        max_angle = min(fov_horizontal * 0.8, np.radians(90))  # 최대 90도
        
        # 각도 범위: -max_angle ~ +max_angle
        angles = np.linspace(-max_angle, max_angle, num_points)
        
        # 극좌표에서 직교좌표로 변환
        # x: 카메라 기준 좌우 방향 (음수: 왼쪽, 양수: 오른쪽)
        # y: 카메라 기준 전후 방향 (양수: 앞쪽)
        x = radius * np.sin(angles)  # 좌우 방향
        y = radius * np.cos(angles)  # 전후 방향 (모두 양수 = 카메라 앞쪽)
        z = np.zeros_like(x)  # 지면 (Z=0)
        
        return np.column_stack([x, y, z]).astype(np.float32)
    
    def fit_ellipse_to_points(self, points_2d):
        """
        2D 점들에 타원 피팅
        
        Args:
            points_2d: 2D 점들의 배열
            
        Returns:
            타원 매개변수 또는 None (피팅 실패시)
        """
        if len(points_2d) < 5:  # 타원 피팅을 위한 최소 점 개수
            return None
        
        try:
            # cv2.fitEllipse는 정수형 점들을 요구
            points_int = points_2d.astype(np.int32)
            ellipse = cv2.fitEllipse(points_int)
            return ellipse
        except:
            return None
    
    def calculate_arc_angles(self, points_2d, ellipse_params):
        """
        타원 호의 시작/끝 각도 계산 (카메라 중심에서 방사형으로 퍼지도록)
        
        Args:
            points_2d: 투영된 2D 점들
            ellipse_params: 타원 매개변수
            
        Returns:
            tuple: (시작 각도, 끝 각도) - 도 단위
        """
        center, axes, angle = ellipse_params
        cx, cy = center
        
        # 카메라 중심점 찾기 (이미지 하단 중앙 근처)
        camera_center_2d, is_visible = self.projector.project_single_point(0.0, 0.0, 0.0)
        
        if not is_visible:
            # 카메라 중심이 보이지 않으면 이미지 하단 중앙을 카메라 중심으로 가정
            camera_center_2d = (self.projector.image_width // 2, self.projector.image_height - 50)
        
        # 카메라 중심에서 각 점까지의 벡터 계산
        camera_x, camera_y = camera_center_2d
        
        # 각 투영된 점이 카메라 중심에서 보이는 방향인지 확인
        # 카메라 중심에서 위쪽으로 향하는 점들만 선택
        valid_points = []
        for point in points_2d:
            px, py = point
            # 카메라 중심보다 위쪽에 있고, 합리적인 거리 내에 있는 점들만 선택
            if py < camera_y and abs(px - camera_x) < self.projector.image_width * 0.8:
                valid_points.append(point)
        
        if len(valid_points) < 2:
            # 유효한 점이 부족하면 전체 타원 사용
            return 0, 360
        
        valid_points = np.array(valid_points)
        
        # 타원 중심을 원점으로 이동
        centered_points = valid_points - np.array([cx, cy])
        
        # 타원 회전 각도만큼 역회전
        angle_rad = math.radians(-angle)
        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        
        rotated_points = (rotation_matrix @ centered_points.T).T
        
        # 각도 계산
        angles = np.arctan2(rotated_points[:, 1], rotated_points[:, 0])
        angles_deg = np.degrees(angles) % 360
        
        # 카메라 중심 방향으로 정렬하기 위해 각도 조정
        # 하단에서 시작해서 좌우로 퍼지는 형태가 되도록
        start_angle = np.min(angles_deg)
        end_angle = np.max(angles_deg)
        
        # 각도 범위가 180도를 넘으면 조정
        if end_angle - start_angle > 180:
            angles_adjusted = angles_deg.copy()
            angles_adjusted[angles_deg > 180] -= 360
            start_angle = np.min(angles_adjusted)
            end_angle = np.max(angles_adjusted)
            
            # 음수 각도 처리
            if start_angle < 0:
                start_angle += 360
                end_angle += 360
        
        # 최소 호 범위 보장 (너무 작으면 보이지 않음)
        if end_angle - start_angle < 10:
            mid_angle = (start_angle + end_angle) / 2
            start_angle = mid_angle - 5
            end_angle = mid_angle + 5
        
        return start_angle, end_angle
    
    def draw_distance_arc(self, image, distance, color_index):
        """
        단일 거리 호를 그리기
        
        Args:
            image: 그릴 이미지
            distance: 거리 (미터)
            color_index: 색상 인덱스
            
        Returns:
            bool: 성공 여부
        """
        color = self.colors[color_index % len(self.colors)]
        
        # 3D 원 점들 생성
        circle_points_3d = self.generate_circle_points(distance)
        
        # 2D로 투영
        points_2d, valid_mask = self.projector.project_3d_to_2d(circle_points_3d)
        
        if not np.any(valid_mask):
            return False  # 가시적인 점이 없음
        
        # 가시적인 점들만 선택
        visible_points = points_2d[valid_mask]
        
        if len(visible_points) < 5:
            return False  # 타원 피팅을 위한 최소 점 개수 부족
        
        # 타원 피팅
        ellipse_params = self.fit_ellipse_to_points(visible_points)
        if ellipse_params is None:
            return False
        
        # 호 각도 계산
        start_angle, end_angle = self.calculate_arc_angles(visible_points, ellipse_params)
        
        # 타원 호 그리기
        center, axes, angle = ellipse_params
        center = (int(center[0]), int(center[1]))
        axes = (int(axes[0]/2), int(axes[1]/2))  # 반경으로 변환
        
        # 호 그리기
        cv2.ellipse(image, center, axes, angle, start_angle, end_angle, color, 2)
        
        # 거리 레이블 추가
        self._draw_distance_label(image, distance, center, axes, color)
        
        return True
    
    def _draw_distance_label(self, image, distance, center, axes, color):
        """거리 레이블을 호의 중앙 부근에 그리기"""
        label_text = f"{distance:.1f}m"
        
        # 호의 중앙 지점 계산 (타원의 상단 중앙)
        label_x = center[0]
        label_y = center[1] - axes[1] - 15  # 타원 위쪽
        
        # 이미지 경계 확인 및 조정
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        text_size = cv2.getTextSize(label_text, font, font_scale, thickness)[0]
        
        # X 좌표 조정 (텍스트가 중앙에 오도록)
        label_x = label_x - text_size[0] // 2
        
        # 경계 확인
        if label_x < 10:
            label_x = 10
        elif label_x + text_size[0] > self.projector.image_width - 10:
            label_x = self.projector.image_width - text_size[0] - 10
            
        if label_y < 20:
            label_y = center[1] + axes[1] + 25  # 타원 아래쪽으로 이동
        
        label_pos = (max(10, label_x), max(20, label_y))
        
        # 텍스트 배경 그리기 (더 작고 깔끔하게)
        bg_x1 = label_pos[0] - 3
        bg_y1 = label_pos[1] - text_size[1] - 3
        bg_x2 = label_pos[0] + text_size[0] + 3
        bg_y2 = label_pos[1] + 3
        
        # 반투명 배경
        overlay = image.copy()
        cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
        
        # 테두리
        cv2.rectangle(image, (bg_x1, bg_y1), (bg_x2, bg_y2), color, 1)
        
        # 텍스트 그리기
        cv2.putText(image, label_text, label_pos, font, font_scale, color, thickness)
    
    def draw_all_distance_arcs(self, image, distances):
        """
        모든 거리 호들을 그리기
        
        Args:
            image: 그릴 이미지
            distances: 거리 목록 (미터)
            
        Returns:
            int: 성공적으로 그려진 호의 개수
        """
        success_count = 0
        
        for i, distance in enumerate(distances):
            try:
                if self.draw_distance_arc(image, distance, i):
                    success_count += 1
            except Exception as e:
                print(f"⚠ 거리 {distance}m 호 그리기 실패: {e}")
        
        return success_count
    
    def draw_center_point(self, image):
        """카메라 바로 아래 지점에 중심점 표시"""
        center_pixel, is_visible = self.projector.project_single_point(0.0, 0.0, 0.0)
        
        if is_visible:
            # 중심점 그리기
            cv2.circle(image, center_pixel, 8, (255, 255, 255), -1)
            cv2.circle(image, center_pixel, 8, (0, 0, 0), 2)
            
            # 레이블 추가
            label_pos = (center_pixel[0] + 15, center_pixel[1] - 10)
            cv2.putText(image, "Camera Center", label_pos,
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            return True
        return False