"""
main.py
메인 실행 프로그램 - 웹캠 문제 해결 및 실시간 거리 시각화

실행 방법:
python main.py

키 조작:
- q: 종료
- s: 스크린샷 저장
- 1,2,3: 카메라 높이 조정 (1m, 2m, 3m)
- +,-: 피치 각도 조정
"""

import cv2
import numpy as np
import sys
import os

# 다른 모듈들 임포트
from camera_model import CameraModel
from ground_projector import GroundProjector
from distance_visualizer import DistanceVisualizer

class DistanceVisualizationApp:
    """거리 시각화 애플리케이션 메인 클래스"""
    
    def __init__(self):
        print("🚀 거리 시각화 애플리케이션 시작")
        
        # 카메라 설정
        self.camera_height = 2.0  # 기본 높이 2미터
        self.camera_pitch = 15.0  # 기본 피치 15도
        
        # 시각화할 거리들 (미터)
        self.distances = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0]
        
        # 카메라 모델 초기화
        self.camera_model = self._initialize_camera()
        
        # 웹캠 초기화
        self.cap = self._initialize_webcam()
        
        if self.cap is None:
            print("❌ 웹캠을 초기화할 수 없습니다.")
            sys.exit(1)
        
        # 투영기와 시각화 도구 초기화
        frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"📹 웹캠 해상도: {frame_width}x{frame_height}")
        
        self.projector = GroundProjector(self.camera_model, frame_width, frame_height)
        self.visualizer = DistanceVisualizer(self.projector)
        
        self.screenshot_count = 0
    
    def _initialize_camera(self):
        """카메라 모델 초기화"""
        print("📷 카메라 모델 초기화 중...")
        
        # 캘리브레이션 파일 확인
        calibration_file = "GasCameraCalibrationLogitec.json"
        if os.path.exists(calibration_file):
            print(f"✓ 캘리브레이션 파일 발견: {calibration_file}")
            camera = CameraModel(calibration_file)
        else:
            print(f"⚠ 캘리브레이션 파일 없음: {calibration_file}")
            print("기본 카메라 매개변수를 사용합니다.")
            camera = CameraModel()
        
        # 카메라 자세 설정
        camera.set_camera_pose(
            height=self.camera_height, 
            pitch_deg=self.camera_pitch
        )
        
        print(camera.get_info())
        return camera
    
    def _initialize_webcam(self):
        """웹캠 초기화 (다양한 방법 시도)"""
        print("🎥 웹캠 초기화 중...")
        
        # 다양한 캡처 방법 시도
        capture_methods = [
            (0, cv2.CAP_DSHOW),    # DirectShow (Windows)
            (0, cv2.CAP_V4L2),     # Video4Linux (Linux)
            (0, cv2.CAP_AVFOUNDATION),  # AVFoundation (macOS)
            (0, cv2.CAP_ANY),      # 자동 선택
            (0, None),             # 기본값
        ]
        
        for camera_id, backend in capture_methods:
            try:
                if backend is not None:
                    cap = cv2.VideoCapture(camera_id, backend)
                else:
                    cap = cv2.VideoCapture(camera_id)
                
                if cap.isOpened():
                    # 해상도 설정 시도
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    cap.set(cv2.CAP_PROP_FPS, 30)
                    
                    # 테스트 프레임 읽기
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        print(f"✓ 웹캠 초기화 성공: 방법={backend}")
                        print(f"  실제 해상도: {frame.shape[1]}x{frame.shape[0]}")
                        return cap
                    else:
                        cap.release()
                        
            except Exception as e:
                print(f"  웹캠 초기화 실패 (방법={backend}): {e}")
        
        print("❌ 모든 웹캠 초기화 방법이 실패했습니다.")
        print("다음을 확인해주세요:")
        print("  1. 웹캠이 연결되어 있는지")
        print("  2. 다른 프로그램에서 웹캠을 사용하고 있지 않은지")
        print("  3. 웹캠 드라이버가 설치되어 있는지")
        return None
    
    def update_camera_settings(self):
        """카메라 설정 업데이트"""
        self.camera_model.set_camera_pose(
            height=self.camera_height,
            pitch_deg=self.camera_pitch
        )
    
    def process_frame(self, frame):
        """프레임 처리 및 거리 호 그리기"""
        # 원본 프레임 복사
        display_frame = frame.copy()
        
        # 중심점 그리기
        self.visualizer.draw_center_point(display_frame)
        
        # 거리 호들 그리기
        success_count = self.visualizer.draw_all_distance_arcs(display_frame, self.distances)
        
        # 정보 텍스트 추가
        self._draw_info_text(display_frame, success_count)
        
        return display_frame
    
    def _draw_info_text(self, frame, success_count):
        """정보 텍스트를 프레임에 그리기"""
        info_lines = [
            f"Height: {self.camera_height:.1f}m",
            f"Pitch: {self.camera_pitch:.1f}deg",
            f"Arcs: {success_count}/{len(self.distances)}",
            "",
            "Controls:",
            "q: Quit",
            "s: Screenshot", 
            "1,2,3: Height 1m,2m,3m",
            "+,-: Pitch +/-5deg"
        ]
        
        # 텍스트 배경
        bg_height = len(info_lines) * 25 + 20
        cv2.rectangle(frame, (10, 10), (250, bg_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (250, bg_height), (255, 255, 255), 1)
        
        # 텍스트 그리기
        for i, line in enumerate(info_lines):
            y_pos = 30 + i * 25
            color = (0, 255, 0) if line.startswith(("Height", "Pitch", "Arcs")) else (255, 255, 255)
            cv2.putText(frame, line, (20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    def handle_keypress(self, key):
        """키 입력 처리"""
        if key == ord('q'):
            return False  # 종료
        
        elif key == ord('s'):
            # 스크린샷 저장
            self.screenshot_count += 1
            filename = f"distance_viz_screenshot_{self.screenshot_count:03d}.jpg"
            ret, frame = self.cap.read()
            if ret:
                processed_frame = self.process_frame(frame)
                cv2.imwrite(filename, processed_frame)
                print(f"📸 스크린샷 저장: {filename}")
        
        elif key == ord('1'):
            self.camera_height = 1.0
            self.update_camera_settings()
            print(f"📏 카메라 높이: {self.camera_height}m")
        
        elif key == ord('2'):
            self.camera_height = 2.0
            self.update_camera_settings()
            print(f"📏 카메라 높이: {self.camera_height}m")
        
        elif key == ord('3'):
            self.camera_height = 3.0
            self.update_camera_settings()
            print(f"📏 카메라 높이: {self.camera_height}m")
        
        elif key == ord('+') or key == ord('='):
            self.camera_pitch += 5.0
            self.camera_pitch = min(self.camera_pitch, 90.0)  # 최대 90도
            self.update_camera_settings()
            print(f"📐 피치 각도: {self.camera_pitch}°")
        
        elif key == ord('-'):
            self.camera_pitch -= 5.0
            self.camera_pitch = max(self.camera_pitch, -30.0)  # 최소 -30도
            self.update_camera_settings()
            print(f"📐 피치 각도: {self.camera_pitch}°")
        
        return True  # 계속 실행
    
    def run(self):
        """메인 실행 루프"""
        print("\n🎬 실시간 거리 시각화 시작!")
        print("조작법:")
        print("  q: 종료")
        print("  s: 스크린샷 저장")
        print("  1,2,3: 카메라 높이 1m, 2m, 3m")
        print("  +,-: 피치 각도 ±5도 조정")
        print("-" * 50)
        
        frame_count = 0
        
        while True:
            # 프레임 읽기
            ret, frame = self.cap.read()
            
            if not ret:
                print("❌ 프레임을 읽을 수 없습니다.")
                break
            
            frame_count += 1
            
            # 프레임 처리
            try:
                processed_frame = self.process_frame(frame)
                cv2.imshow("Distance Visualization", processed_frame)
                
                # 첫 번째 프레임에서 성공 메시지
                if frame_count == 1:
                    print("✅ 웹캠 영상 표시 성공!")
                
            except Exception as e:
                print(f"⚠ 프레임 처리 오류: {e}")
                cv2.imshow("Distance Visualization", frame)  # 원본 프레임 표시
            
            # 키 입력 처리
            key = cv2.waitKey(1) & 0xFF
            if key != 255:  # 키가 눌렸을 때
                if not self.handle_keypress(key):
                    break
        
        # 정리
        self.cleanup()
    
    def cleanup(self):
        """리소스 정리"""
        print("🧹 리소스 정리 중...")
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("✅ 프로그램 종료")

def test_camera_detection():
    """사용 가능한 카메라 장치 검색"""
    print("🔍 사용 가능한 카메라 장치 검색 중...")
    
    available_cameras = []
    
    for i in range(5):  # 0-4번 장치 확인
        try:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"✓ 카메라 {i}: 사용 가능 ({frame.shape[1]}x{frame.shape[0]})")
                    available_cameras.append(i)
                else:
                    print(f"✗ 카메라 {i}: 연결되었으나 영상 없음")
                cap.release()
            else:
                print(f"✗ 카메라 {i}: 사용 불가")
        except:
            print(f"✗ 카메라 {i}: 오류")
    
    return available_cameras

def create_test_image():
    """테스트용 정적 이미지 생성"""
    print("🖼 테스트 이미지 모드로 실행")
    
    # 카메라 모델 초기화
    camera = CameraModel("GasCameraCalibrationLogitec.json" if os.path.exists("GasCameraCalibrationLogitec.json") else None)
    camera.set_camera_pose(height=2.0, pitch_deg=15.0)
    
    # 테스트 이미지 생성
    image_width, image_height = 640, 480
    image = np.zeros((image_height, image_width, 3), dtype=np.uint8)
    image.fill(50)  # 어두운 회색 배경
    
    # 격자 그리기
    for i in range(0, image_width, 50):
        cv2.line(image, (i, 0), (i, image_height), (80, 80, 80), 1)
    for i in range(0, image_height, 50):
        cv2.line(image, (0, i), (image_width, i), (80, 80, 80), 1)
    
    # 투영기와 시각화 도구 초기화
    projector = GroundProjector(camera, image_width, image_height)
    visualizer = DistanceVisualizer(projector)
    
    # 거리 호 그리기
    distances = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0]
    success_count = visualizer.draw_all_distance_arcs(image, distances)
    visualizer.draw_center_point(image)
    
    # 정보 텍스트
    cv2.putText(image, f"Test Image - {success_count}/{len(distances)} arcs", 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # 결과 표시
    cv2.imshow("Distance Visualization - Test Image", image)
    cv2.imwrite("test_distance_visualization.jpg", image)
    print("✅ 테스트 이미지 저장: test_distance_visualization.jpg")
    print("아무 키나 눌러 종료...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    """메인 함수"""
    print("=" * 60)
    print("🎯 단일 카메라 거리 시각화 프로그램")
    print("=" * 60)
    
    # 카메라 장치 검색
    available_cameras = test_camera_detection()
    
    if not available_cameras:
        print("\n❌ 사용 가능한 카메라가 없습니다.")
        print("테스트 이미지 모드로 실행합니다.")
        create_test_image()
        return
    
    print(f"\n✅ {len(available_cameras)}개의 카메라를 발견했습니다.")
    
    try:
        # 애플리케이션 실행
        app = DistanceVisualizationApp()
        app.run()
        
    except KeyboardInterrupt:
        print("\n⏹ 사용자가 프로그램을 중단했습니다.")
    except Exception as e:
        print(f"\n❌ 예상치 못한 오류: {e}")
        print("테스트 이미지 모드로 실행합니다.")
        create_test_image()

if __name__ == "__main__":
    main()