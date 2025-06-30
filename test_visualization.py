"""
test_visualization.py
테스트 및 디버깅을 위한 도구

실행 방법:
python test_visualization.py
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from camera_model import CameraModel
from ground_projector import GroundProjector
from distance_visualizer import DistanceVisualizer

def test_camera_calibration():
    """카메라 캘리브레이션 데이터 테스트"""
    print("🧪 카메라 캘리브레이션 테스트")
    print("-" * 40)
    
    # 캘리브레이션 파일 있을 때와 없을 때 비교
    print("1. 캘리브레이션 파일 사용:")
    try:
        camera_with_calib = CameraModel("GasCameraCalibrationLogitec.json")
        print(camera_with_calib.get_info())
    except:
        print("  캘리브레이션 파일 로드 실패")
    
    print("\n2. 기본 매개변수 사용:")
    camera_default = CameraModel()
    print(camera_default.get_info())

def test_projection():
    """3D-2D 투영 테스트"""
    print("\n🎯 3D-2D 투영 테스트")
    print("-" * 40)
    
    # 카메라 설정
    camera = CameraModel()
    camera.set_camera_pose(height=2.0, pitch_deg=15.0)
    
    # 투영기 설정
    projector = GroundProjector(camera, 640, 480)
    
    # 테스트 점들 (지면상의 몇 개 점)
    test_points_3d = np.array([
        [0.0, 0.0, 0.0],    # 중심점
        [1.0, 0.0, 0.0],    # 1m 앞
        [0.0, 1.0, 0.0],    # 1m 오른쪽
        [5.0, 0.0, 0.0],    # 5m 앞
        [0.0, 5.0, 0.0],    # 5m 오른쪽
        [-1.0, -1.0, 0.0],  # 뒤쪽 왼쪽
    ], dtype=np.float32)
    
    # 투영 실행
    points_2d, valid_mask = projector.project_3d_to_2d(test_points_3d)
    
    print("투영 결과:")
    for i, (point_3d, point_2d, is_valid) in enumerate(zip(test_points_3d, points_2d, valid_mask)):
        status = "✓" if is_valid else "✗"
        print(f"  {status} 3D{point_3d} -> 2D({point_2d[0]:.1f}, {point_2d[1]:.1f})")

def test_circle_generation():
    """원 생성 테스트"""
    print("\n⭕ 원 생성 테스트")
    print("-" * 40)
    
    visualizer = DistanceVisualizer(None)  # projector는 나중에 설정
    
    # 다양한 반지름으로 원 생성
    radii = [1.0, 5.0, 10.0]
    
    for radius in radii:
        circle_points = visualizer.generate_circle_points(radius, num_points=8)
        print(f"반지름 {radius}m 원 (8점):")
        for i, point in enumerate(circle_points):
            print(f"  점 {i}: ({point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f})")
        
        # 실제 거리 확인
        distances = np.sqrt(circle_points[:, 0]**2 + circle_points[:, 1]**2)
        print(f"  실제 거리: 평균={np.mean(distances):.3f}, 표준편차={np.std(distances):.6f}")

def create_comparison_image():
    """서로 다른 카메라 설정으로 비교 이미지 생성"""
    print("\n📊 비교 이미지 생성")
    print("-" * 40)
    
    # 이미지 설정
    img_width, img_height = 640, 480
    distances = [2.0, 5.0, 10.0]
    
    # 서로 다른 설정들
    settings = [
        {"height": 1.0, "pitch": 0.0, "title": "Height 1m, Pitch 0deg"},
        {"height": 2.0, "pitch": 15.0, "title": "Height 2m, Pitch 15deg"},
        {"height": 3.0, "pitch": 30.0, "title": "Height 3m, Pitch 30deg"},
        {"height": 2.0, "pitch": 0.0, "title": "Height 2m, Pitch 0deg"},
    ]
    
    # 2x2 그리드로 이미지 생성
    grid_image = np.zeros((img_height * 2, img_width * 2, 3), dtype=np.uint8)
    grid_image.fill(40)  # 어두운 배경
    
    for i, setting in enumerate(settings):
        # 위치 계산
        row = i // 2
        col = i % 2
        y_start = row * img_height
        y_end = y_start + img_height
        x_start = col * img_width
        x_end = x_start + img_width
        
        # 카메라 설정
        camera = CameraModel()
        camera.set_camera_pose(
            height=setting["height"], 
            pitch_deg=setting["pitch"]
        )
        
        # 투영 및 시각화
        projector = GroundProjector(camera, img_width, img_height)
        visualizer = DistanceVisualizer(projector)
        
        # 개별 이미지 생성
        sub_image = np.zeros((img_height, img_width, 3), dtype=np.uint8)
        sub_image.fill(40)
        
        # 거리 호 그리기
        success_count = visualizer.draw_all_distance_arcs(sub_image, distances)
        visualizer.draw_center_point(sub_image)
        
        # 제목 추가
        title = setting["title"]
        cv2.putText(sub_image, title, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(sub_image, f"Arcs: {success_count}/{len(distances)}", (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # 그리드에 복사
        grid_image[y_start:y_end, x_start:x_end] = sub_image
        
        print(f"  설정 {i+1}: {title} -> {success_count}/{len(distances)} 호 성공")
    
    # 결과 저장 및 표시
    cv2.imwrite("comparison_visualization.jpg", grid_image)
    cv2.imshow("Comparison - Different Camera Settings", grid_image)
    print("✅ 비교 이미지 저장: comparison_visualization.jpg")
    print("아무 키나 눌러 계속...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def test_ellipse_fitting():
    """타원 피팅 테스트"""
    print("\n🔄 타원 피팅 테스트")
    print("-" * 40)
    
    # 가상의 타원 점들 생성
    center_x, center_y = 320, 240
    a, b = 100, 50  # 장축, 단축
    angle = 30  # 회전 각도
    
    # 타원 위의 점들 생성
    t = np.linspace(0, 2*np.pi, 20)
    x = a * np.cos(t)
    y = b * np.sin(t)
    
    # 회전 적용
    angle_rad = np.radians(angle)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    x_rot = x * cos_a - y * sin_a + center_x
    y_rot = x * sin_a + y * cos_a + center_y
    
    points = np.column_stack([x_rot, y_rot]).astype(np.float32)
    
    # 타원 피팅
    try:
        ellipse = cv2.fitEllipse(points.astype(np.int32))
        fitted_center, fitted_axes, fitted_angle = ellipse
        
        print(f"원본 타원: 중심=({center_x}, {center_y}), 축=({a}, {b}), 각도={angle}°")
        print(f"피팅 결과: 중심=({fitted_center[0]:.1f}, {fitted_center[1]:.1f}), "
              f"축=({fitted_axes[0]/2:.1f}, {fitted_axes[1]/2:.1f}), 각도={fitted_angle:.1f}°")
        
        # 시각화
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # 원본 점들
        for point in points:
            cv2.circle(img, tuple(point.astype(int)), 3, (0, 255, 0), -1)
        
        # 피팅된 타원
        cv2.ellipse(img, ellipse, (255, 0, 0), 2)
        
        cv2.imshow("Ellipse Fitting Test", img)
        cv2.imwrite("ellipse_fitting_test.jpg", img)
        print("✅ 타원 피팅 테스트 이미지 저장: ellipse_fitting_test.jpg")
        print("아무 키나 눌러 계속...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"❌ 타원 피팅 실패: {e}")

def analyze_camera_fov():
    """카메라 시야각 분석"""
    print("\n👁 카메라 시야각 분석")
    print("-" * 40)
    
    camera = CameraModel()
    
    if camera.camera_matrix is not None:
        fx = camera.camera_matrix[0, 0]
        fy = camera.camera_matrix[1, 1]
        cx = camera.camera_matrix[0, 2]
        cy = camera.camera_matrix[1, 2]
        
        # 이미지 크기 (일반적인 웹캠 해상도)
        img_width, img_height = 640, 480
        
        # 시야각 계산
        fov_x = 2 * np.arctan(img_width / (2 * fx)) * 180 / np.pi
        fov_y = 2 * np.arctan(img_height / (2 * fy)) * 180 / np.pi
        
        print(f"이미지 크기: {img_width} x {img_height}")
        print(f"초점거리: fx={fx:.1f}, fy={fy:.1f}")
        print(f"주점: cx={cx:.1f}, cy={cy:.1f}")
        print(f"수평 시야각: {fov_x:.1f}°")
        print(f"수직 시야각: {fov_y:.1f}°")
        print(f"대각선 시야각: {np.sqrt(fov_x**2 + fov_y**2):.1f}°")

def main():
    """테스트 메인 함수"""
    print("🧪 거리 시각화 테스트 도구")
    print("=" * 50)
    
    try:
        # 각종 테스트 실행
        test_camera_calibration()
        test_projection()
        test_circle_generation()
        analyze_camera_fov()
        test_ellipse_fitting()
        create_comparison_image()
        
        print("\n✅ 모든 테스트 완료!")
        
    except Exception as e:
        print(f"\n❌ 테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()