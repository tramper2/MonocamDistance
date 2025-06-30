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
        # 카메라의 시야각을 고려하여 점들을 생성 (카메라 앞쪽 부채꼴 모양)

        # 카메라의 수평 시야각 (라디안) 가져오기
        # camera_model에 fov가 정의되어 있지 않으므로, projector에서 이미지 너비와 fx를 이용해 추정
        # 또는 적절한 기본값 사용
        fov_horizontal_rad = np.radians(90) # 기본값: 90도 (카메라 전방 반원)
        if self.projector and hasattr(self.projector, 'camera_model') and \
           self.projector.camera_model.camera_matrix is not None and \
           self.projector.camera_model.camera_matrix[0, 0] > 0:
            fx = self.projector.camera_model.camera_matrix[0, 0]
            image_width = self.projector.image_width
            if image_width > 0 and fx > 0:
                 # 일반적인 웹캠의 경우 수평 FoV는 60~90도 사이가 많음
                 # 여기서는 좀 더 넓게 잡아 120도 정도로 설정하여 충분한 영역을 커버
                calculated_fov = 2 * np.arctan(image_width / (2 * fx))
                # 너무 넓거나 좁은 FoV 방지, 60도 ~ 150도 사이로 제한
                fov_horizontal_rad = np.clip(calculated_fov, np.radians(60), np.radians(150))


        # 부채꼴의 각도 범위 설정 (카메라 전방을 중심으로)
        # -fov_horizontal_rad / 2  에서 +fov_horizontal_rad / 2 까지
        # 결과적으로 카메라 정면을 중심으로 좌우로 fov_horizontal_rad 만큼의 각도를 커버
        start_angle_rad = -fov_horizontal_rad / 2
        end_angle_rad = fov_horizontal_rad / 2

        angles = np.linspace(start_angle_rad, end_angle_rad, num_points)

        # 3D 지면 좌표계 (X: 좌우, Y: 전후, Z: 상하)
        # 카메라 좌표계와 일치시키기 위해 X는 좌우, Y는 전후(카메라 앞쪽이 +)
        # Z는 지면이므로 0
        x_coords = radius * np.sin(angles)  # X = r * sin(theta)
        y_coords = radius * np.cos(angles)  # Y = r * cos(theta) (카메라 앞쪽)
        z_coords = np.zeros_like(x_coords) # 지면 Z=0

        # 생성된 점들이 카메라 앞쪽에만 있도록 y_coords > 0 필터링 (cos(angles)이므로 이미 양수)
        # points_3d = np.column_stack([x_coords, y_coords, z_coords]).astype(np.float32)
        # return points_3d[y_coords > 0] # 이미 angles 범위로 인해 y_coords는 양수

        # (N, 3) 형태의 배열로 반환
        return np.column_stack([x_coords, y_coords, z_coords]).astype(np.float32)
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
        center, axes, ellipse_angle_deg = ellipse_params
        cx_ellipse, cy_ellipse = center

        # 투영된 2D 점들 중 유효한 (이미지 내에 있는) 점들만 사용
        # generate_circle_points 에서 이미 카메라 전방 점들만 생성하므로,
        # 여기서는 이미지 경계 내에 있는 점들만 고려

        # 이미지의 대략적인 카메라 투영 중심점 (소실점 근처 또는 이미지 하단 중앙)
        # 이 점을 기준으로 방사형으로 퍼져나가는 호를 그리기 위함.
        # vanishing_point_y = self.projector.image_height * 0.5 # 예시: 이미지 높이의 중간
        # 실제로는 카메라 피치각에 따라 달라짐. 간단하게 이미지 하단 중앙으로 가정.
        camera_origin_on_image_x = self.projector.image_width / 2
        camera_origin_on_image_y = self.projector.image_height

        # 각 점들을 타원 중심 기준으로 변환하고, 타원의 회전을 고려하여 각도 계산
        # 이는 cv2.ellipse가 사용하는 각도 체계에 맞추기 위함

        # 점들을 타원 중심 기준으로 이동
        points_relative_to_ellipse_center = points_2d - np.array([cx_ellipse, cy_ellipse])

        # 타원의 회전각(ellipse_angle_deg)만큼 점들을 반대로 회전시켜서
        # 타원의 주축/단축이 좌표축과 나란하도록 만듦
        ellipse_rotation_rad = np.radians(-ellipse_angle_deg) # 반대 방향 회전
        cos_rot = np.cos(ellipse_rotation_rad)
        sin_rot = np.sin(ellipse_rotation_rad)

        # 2D 회전 행렬
        # [[cos, -sin],
        #  [sin,  cos]]
        rotation_matrix = np.array([[cos_rot, -sin_rot],
                                    [sin_rot,  cos_rot]])

        # 점들 회전
        # (2, N) 형태로 만들기 위해 전치 후 행렬곱, 다시 전치
        rotated_points = (rotation_matrix @ points_relative_to_ellipse_center.T).T

        # 회전된 점들로부터 각도 계산 (arctan2 사용, 결과는 -pi ~ pi)
        # Y축이 위로 갈수록 작아지는 이미지 좌표계를 고려하여 Y 부호 반전 후 각도 계산
        # 또는 arctan2(y,x) 대신 arctan2(-y,x) 등을 사용
        angles_rad = np.arctan2(rotated_points[:, 1], rotated_points[:, 0])
        angles_deg = np.degrees(angles_rad) # -180 ~ 180 범위

        # cv2.ellipse는 0~360도 범위, x축 양의 방향에서 반시계 방향으로 각도 사용
        # angles_deg를 0~360 범위로 변환
        angles_deg = (angles_deg + 360) % 360

        # 방사형으로 퍼지는 느낌을 위해, 이미지 하단 중앙에서 가장 멀리 떨어진
        # 두 점을 호의 시작과 끝으로 결정하는 대신,
        # 생성된 점들의 각도 분포에서 최소/최대 각도를 사용.
        # generate_circle_points가 이미 부채꼴 점들을 생성하므로,
        # 이 점들이 만드는 각도 범위를 그대로 사용하는 것이 자연스러움.

        # points_2d는 이미 generate_circle_points로부터 온 "보이는" 점들의 투영임
        # 이 점들이 타원 상에서 어떤 각도 범위를 가지는지 확인

        # 시작 각도와 끝 각도 찾기
        # 여기서 중요한 것은, 각도들이 연속적인 범위를 이루도록 하는 것
        # 예를 들어, 350도와 10도 사이의 호는 (350, 10)이 아니라, (350, 370) 또는 (-10, 10) 등으로 표현되어야 함

        min_angle = np.min(angles_deg)
        max_angle = np.max(angles_deg)

        # 각도 범위가 180도를 크게 넘어서는 경우 (예: 10도와 350도),
        # 작은 각도에 360을 더해서 범위를 조정 (예: 350도와 370도)
        if (max_angle - min_angle) > 270: # 360에 가까운 큰 차이 (거의 한바퀴)
            # 180도보다 작은 각도들에 360을 더해줌
            angles_deg[angles_deg < 180] += 360
            min_angle = np.min(angles_deg)
            max_angle = np.max(angles_deg)

        # 호가 너무 길게 그려지는 것을 방지 (예: 화면 상단을 넘어 뒤로 감기는 경우)
        # 보통 카메라 전방의 호는 180도 미만으로 그려짐
        # 만약 min_angle이 90도보다 크고 max_angle이 270도보다 작다면, 이는 주로 화면 상단에 해당.
        # 그리고 각도 차이가 너무 크다면, 이는 점들이 화면을 거의 한 바퀴 도는 경우일 수 있음.
        # 이 부분은 generate_circle_points 에서 FoV를 적절히 설정하면 자연스럽게 해결될 것으로 기대.
        # 여기서는 계산된 min_angle, max_angle을 그대로 사용.

        # cv2.ellipse의 startAngle, endAngle은 x축의 양의 방향(오른쪽)에서 시작하여 반시계 방향으로 증가.
        # "하단에서 시작해서 좌우로 퍼지는" 모양은 일반적으로 타원의 아래쪽 절반에 해당.
        # 예를 들어, 타원이 똑바로 서 있다면 180도에서 360도(또는 0도)까지의 범위.
        # 하지만 타원이 회전되어 있으므로, 이 각도 범위는 달라짐.
        # 위에서 계산된 min_angle, max_angle이 이 역할을 함.

        # 최종적으로 시작각도와 끝각도 설정
        # generate_circle_points에서 생성된 점들의 투영이 만드는 각도 범위를 사용
        start_angle_final = min_angle
        end_angle_final = max_angle

        # 호의 최소 길이 보장 (너무 짧으면 안보임)
        if abs(end_angle_final - start_angle_final) < 5: # 5도 미만이면
             # 중심각을 기준으로 약간 넓힘
             mid = (start_angle_final + end_angle_final) / 2
             start_angle_final = mid - 5
             end_angle_final = mid + 5

        return start_angle_final, end_angle_final
    
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
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5 # 약간 작게 조정
        thickness = 1
        
        text_size, _ = cv2.getTextSize(label_text, font, font_scale, thickness)
        text_width, text_height = text_size

        # 레이블 위치: 호의 중앙 상단
        # 타원의 중심(center)과 y축 반경(axes[1])을 사용하여 타원의 가장 높은 점을 찾음.
        # 호가 타원의 일부이므로, 이 점이 호 위에 있을 가능성이 높음.
        # 타원의 회전각(ellipse_angle_deg)을 고려해야 함.
        # ellipse_params: (타원 중심점 (x,y), 타원 축 길이 (장축, 단축), 타원 회전 각도)
        # center: 타원의 중심 (x,y) 튜플
        # axes: (타원의 장축 길이 / 2, 타원의 단축 길이 / 2) - 이미 반경으로 계산됨
        ellipse_center_x, ellipse_center_y = center

        # 레이블 위치 결정: 호의 대략적인 중앙 상단.
        # 타원의 중심 x좌표를 사용하고, y좌표는 타원의 가장 높은 지점(근사)보다 약간 위로 설정.
        # 타원의 회전을 고려한 정확한 호 상단점 계산은 복잡하므로 근사치 사용.

        # 텍스트의 x 좌표: 타원 중심 x에서 텍스트 너비의 절반을 빼서 중앙 정렬.
        label_x_candidate = int(ellipse_center_x - text_width / 2)

        # 텍스트의 y 좌표 (텍스트의 baseline 기준):
        # 타원의 이론적 상단 (ellipse_center_y - axes[1]) 보다
        # 텍스트 높이(text_height)와 추가적인 수직 간격(vertical_offset)만큼 위로.
        # cv2.putText는 y좌표를 텍스트의 좌하단 기준으로 삼으므로, 이를 고려하여 계산.
        # 레이블이 타원 호의 약간 "위"에 위치하도록 함.
        vertical_offset_from_ellipse_top = 10  # 타원 상단 경계에서 레이블 하단까지의 여유 공간
        label_y_candidate = int(ellipse_center_y - axes[1] - vertical_offset_from_ellipse_top)

        # 이미지 경계 클리핑: 레이블이 이미지 밖으로 나가지 않도록 좌표 조정.
        # X 좌표 클리핑: 레이블 전체가 화면 좌우 경계 내에 있도록.
        label_x = np.clip(label_x_candidate, 5, self.projector.image_width - text_width - 5)

        # Y 좌표 클리핑: 레이블 전체가 화면 상하 경계 내에 있도록.
        # label_y는 텍스트의 baseline이므로, 텍스트 상단은 label_y - text_height.
        # 이미지 상단 경계: label_y - text_height > 5  => label_y > text_height + 5
        # 이미지 하단 경계: label_y < self.projector.image_height - 5
        min_y = text_height + 10 # 최소 y값 (상단 여백 고려, 기존 5에서 좀 더 늘림)
        max_y = self.projector.image_height - 5 # 최대 y값 (하단 여백 고려)
        label_y = np.clip(label_y_candidate, min_y, max_y)

        label_pos = (int(label_x), int(label_y)) # cv2.putText를 위한 최종 위치 (텍스트 좌하단)

        # 반투명 배경 추가: 텍스트 가독성 향상
        bg_x1 = label_pos[0] - 3
        bg_y1 = label_pos[1] - text_height - 1  # text_height 기준으로 y1 조정
        bg_x2 = label_pos[0] + text_width + 3
        bg_y2 = label_pos[1] + 3               # text_height 기준으로 y2 조정

        try:
            # ROI (Region of Interest) 추출
            roi = image[bg_y1:bg_y2, bg_x1:bg_x2]

            # 검은색 배경 사각형 생성 (텍스트 배경용)
            black_rect = np.zeros(roi.shape, dtype=image.dtype)

            # 반투명 효과 적용: alpha * foreground + (1-alpha) * background
            alpha = 0.6 # 투명도 (0.0 완전 투명 ~ 1.0 완전 불투명)
            blended_roi = cv2.addWeighted(black_rect, alpha, roi, 1 - alpha, 0)

            image[bg_y1:bg_y2, bg_x1:bg_x2] = blended_roi

            # 테두리 (선택 사항, 더 얇게)
            # cv2.rectangle(image, (bg_x1, bg_y1), (bg_x2, bg_y2), color, 1)

        except Exception as e:
            # ROI가 이미지 경계를 벗어나는 등 예외 발생 시 배경 없이 텍스트만 그림
            # print(f"Label background drawing error: {e}")
            pass

        # 텍스트 그리기 (흰색 또는 밝은 색으로 가독성 확보)
        text_color = (220, 220, 220) # 밝은 회색 계열
        cv2.putText(image, label_text, (label_pos[0], label_pos[1]), font, font_scale, text_color, thickness, cv2.LINE_AA)
    
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