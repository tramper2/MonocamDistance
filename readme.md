# 단일 카메라 거리 시각화 프로그램

카메라 이미지에 지면상의 거리를 동심원 호로 시각화하는 Python 프로그램입니다.

## 📋 Requirements (requirements.txt)

```txt
opencv-python>=4.5.0
numpy>=1.19.0
matplotlib>=3.3.0
```

## 🚀 설치 및 실행

### 1. 필요한 라이브러리 설치
```bash
pip install opencv-python numpy matplotlib
```

또는

```bash
pip install -r requirements.txt
```

### 2. 파일 구성
```
📁 project/
├── 📄 main.py                           # 메인 실행 프로그램
├── 📄 camera_model.py                   # 카메라 모델 관리
├── 📄 ground_projector.py               # 3D-2D 투영 처리
├── 📄 distance_visualizer.py            # 거리 시각화
├── 📄 test_visualization.py             # 테스트 도구
├── 📄 GasCameraCalibrationLogitec.json  # 카메라 캘리브레이션 파일
├── 📄 requirements.txt                  # 필요 라이브러리
└── 📄 README.md                         # 이 파일
```

### 3. 실행 방법

#### 🎥 실시간 웹캠 모드
```bash
python main.py
```

#### 🧪 테스트 모드
```bash
python test_visualization.py
```

## 🎮 조작법

### 실시간 모드 (main.py)
- **q**: 프로그램 종료
- **s**: 현재 화면 스크린샷 저장
- **1, 2, 3**: 카메라 높이를 1m, 2m, 3m로 설정
- **+, -**: 카메라 피치 각도를 ±5도씩 조정

## 📐 주요 매개변수

### 카메라 설정
- **높이**: 지면으로부터의 카메라 높이 (기본값: 2.0m)
- **피치**: 아래쪽을 바라보는 각도 (기본값: 15°)
- **거리**: 시각화할 거리들 (기본값: 1, 2, 3, 5, 7, 10, 15m)

### 캘리브레이션 파일
프로그램은 `GasCameraCalibrationLogitec.json` 파일을 자동으로 찾아 사용합니다.
파일이 없으면 기본 매개변수를 사용합니다.

## 🔧 웹캠 문제 해결

### 웹캠이 인식되지 않는 경우:

1. **다른 프로그램 확인**: 다른 화상회의 프로그램이나 카메라 앱이 실행 중인지 확인
2. **드라이버 확인**: 웹캠 드라이버가 최신인지 확인
3. **권한 확인**: 카메라 접근 권한이 허용되어 있는지 확인

### Windows에서:
```bash
# 장치 관리자에서 카메라 장치 확인
# 설정 > 개인정보 > 카메라에서 앱 권한 확인
```

### Linux에서:
```bash
# 카메라 장치 확인
ls /dev/video*

# 권한 확인
sudo usermod -a -G video $USER
```

### macOS에서:
```bash
# 시스템 환경설정 > 보안 및 개인 정보 보호 > 카메라에서 권한 확인
```

## 📊 테스트 도구 사용법

`test_visualization.py`는 다음 기능을 제공합니다:

1. **카메라 캘리브레이션 테스트**: 캘리브레이션 파일 로드 및 매개변수 확인
2. **3D-2D 투영 테스트**: 지면 좌표의 이미지 투영 결과 확인
3. **원 생성 테스트**: 동심원 점 생성 및 정확도 검증
4. **타원 피팅 테스트**: OpenCV 타원 피팅 알고리즘 검증
5. **비교 이미지 생성**: 다양한 카메라 설정의 시각화 결과 비교
6. **시야각 분석**: 카메라의 시야각 계산 및 분석

## 🏗️ 프로그램 구조

### 모듈별 역할

#### 📷 `camera_model.py`
- 카메라 내부/외부 매개변수 관리
- 캘리브레이션 파일 로드
- 카메라 높이 및 자세 설정

```python
# 사용 예시
camera = CameraModel("calibration.json")
camera.set_camera_pose(height=2.0, pitch_deg=15.0)
```

#### 🎯 `ground_projector.py`
- 3D 지면 좌표를 2D 이미지 좌표로 투영
- 가시성 필터링 (카메라 뒤편, 이미지 경계 밖 제거)
- OpenCV의 `cv2.projectPoints` 활용

```python
# 사용 예시
projector = GroundProjector(camera, 640, 480)
points_2d, valid_mask = projector.project_3d_to_2d(points_3d)
```

#### 🎨 `distance_visualizer.py`
- 거리별 동심원 생성
- 타원 피팅 및 호 그리기
- 거리 레이블 표시

```python
# 사용 예시
visualizer = DistanceVisualizer(projector)
distances = [1.0, 3.0, 5.0, 10.0]
visualizer.draw_all_distance_arcs(image, distances)
```

## ⚙️ 설정 사용자 정의

### 거리 변경
`main.py`에서 `distances` 리스트를 수정:
```python
self.distances = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]  # 원하는 거리로 변경
```

### 색상 변경
`distance_visualizer.py`에서 `colors` 리스트를 수정:
```python
self.colors = [
    (0, 255, 0),    # 녹색
    (255, 0, 0),    # 파란색
    # 원하는 색상 추가 (BGR 형식)
]
```

### 카메라 매개변수 수동 설정
캘리브레이션 파일 없이 사용하려면:
```python
camera = CameraModel()  # 기본 매개변수 사용
# 또는 수동으로 설정
camera.camera_matrix = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0, 0, 1]
], dtype=np.float32)
```

## 🐛 문제 해결

### 일반적인 문제들

#### 1. 웹캠 화면이 검은색으로 나올 때
```bash
# 해결 방법:
# 1. 웹캠 LED가 켜져 있는지 확인
# 2. 다른 카메라 앱을 모두 종료
# 3. 컴퓨터 재부팅 후 재시도
```

#### 2. "카메라를 열 수 없습니다" 오류
```python
# main.py에서 카메라 인덱스 변경
# 0 대신 1, 2, 3 등을 시도
cap = cv2.VideoCapture(1)  # 다른 인덱스 시도
```

#### 3. 거리 호가 표시되지 않을 때
- 카메라 높이를 낮춰보세요 (1m로 설정)
- 피치 각도를 크게 해보세요 (30도 이상)
- 테스트 모드로 확인: `python test_visualization.py`

#### 4. 왜곡된 호가 나타날 때
- 캘리브레이션 파일이 올바른지 확인
- 카메라와 실제 설정이 일치하는지 확인

### 성능 최적화

#### 실시간 성능 향상
```python
# main.py에서 점 개수 줄이기
circle_points_3d = self.generate_circle_points(distance, num_points=90)  # 기본값: 180

# 거리 개수 줄이기
self.distances = [1.0, 3.0, 5.0, 10.0]  # 적은 개수로 시작
```

## 📈 고급 사용법

### 1. 다른 평면 사용
지면이 아닌 다른 평면을 사용하려면:
```python
# Z=1 평면 (지면에서 1미터 위)
circle_points_3d[:, 2] = 1.0
```

### 2. 비원형 거리 표시
정사각형이나 다른 모양으로 거리 표시:
```python
def generate_square_points(self, size, num_points=40):
    # 정사각형 경계의 점들 생성
    points = []
    side_points = num_points // 4
    for i in range(side_points):
        t = i / side_points
        # 각 변의 점들 생성...
```

### 3. 실시간 카메라 자세 추정
IMU 센서나 다른 방법으로 카메라 자세를 실시간으로 업데이트:
```python
def update_camera_pose_from_imu(self, roll, pitch, yaw):
    self.camera_model.set_camera_pose(
        height=self.camera_height,
        pitch_deg=pitch,
        yaw_deg=yaw,
        roll_deg=roll
    )
```

## 📚 참고 자료

### OpenCV 함수들
- `cv2.projectPoints()`: 3D 점을 2D로 투영
- `cv2.fitEllipse()`: 점들에 타원 피팅
- `cv2.ellipse()`: 타원/호 그리기
- `cv2.calibrateCamera()`: 카메라 캘리브레이션

### 관련 논문 및 자료
- Computer Vision: Algorithms and Applications (Richard Szeliski)
- Multiple View Geometry (Hartley & Zisserman)
- OpenCV Documentation: Camera Calibration and 3D Reconstruction

## 🤝 기여 및 라이선스

이 프로젝트는 교육 및 연구 목적으로 개발되었습니다.
개선 사항이나 버그 리포트는 언제든 환영합니다!

### 개발자 정보
- 기반 연구: 단일 카메라 및 알려진 지면 평면을 활용한 단안 거리 추정
- 구현: OpenCV와 Python을 활용한 실시간 시각화

---

## 🎯 빠른 시작 가이드

1. **라이브러리 설치**: `pip install opencv-python numpy`
2. **파일 다운로드**: 모든 .py 파일을 같은 폴더에 저장
3. **웹캠 연결**: USB 웹캠 연결 및 인식 확인
4. **실행**: `python main.py`
5. **조작**: 키보드로 높이와 각도 조정하며 테스트

문제가 있으면 먼저 `python test_visualization.py`로 기본 기능을 확인해보세요!