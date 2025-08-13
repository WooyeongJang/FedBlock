# FedBlock Android Client

FedBlock Android 클라이언트는 모바일 환경에서 연합학습과 블록체인 감사 로깅을 지원하는 네이티브 Android 애플리케이션입니다.

## 🚀 주요 기능

### 연합학습
- **실시간 훈련**: 모바일 디바이스에서 직접 머신러닝 모델 훈련
- **다양한 훈련 타입**: MNIST, Deterministic MNIST, Gossip MNIST, Chest X-Ray 지원
- **훈련 진행 상황**: 실시간 에포크, 손실, 정확도 모니터링
- **히스토리 관리**: 과거 훈련 결과 저장 및 조회

### 블록체인 감사
- **감사 로그**: 모든 훈련 활동을 블록체인에 불변 기록
- **실시간 조회**: 블록체인에서 감사 로그 실시간 조회
- **무결성 검증**: 모델 파라미터의 암호화 해시 검증

### 사용자 인터페이스
- **Material Design 3**: 현대적이고 직관적인 UI/UX
- **반응형 디자인**: 다양한 화면 크기 지원
- **실시간 업데이트**: 훈련 상태와 진행 상황 실시간 표시

## 🛠️ 기술 스택

- **Kotlin**: 주 개발 언어
- **Jetpack Compose**: 선언형 UI 프레임워크
- **Hilt**: 의존성 주입
- **Retrofit**: HTTP API 통신
- **Web3j**: 블록체인 상호작용
- **TensorFlow Lite**: 온디바이스 머신러닝
- **Coroutines**: 비동기 프로그래밍

## 📱 화면 구성

### 1. 훈련 화면 (Training)
- 클라이언트 상태 모니터링
- 훈련 설정 (학습률, 에포크, 배치 크기)
- 훈련 시작/중지
- 실시간 진행 상황
- 훈련 히스토리

### 2. 감사 로그 화면 (Audit)
- 블록체인 감사 로그 조회
- 이벤트 타입별 필터링
- 감사 통계 대시보드
- 트랜잭션 해시 확인

### 3. 블록체인 화면 (Blockchain)
- 네트워크 연결 상태
- 블록체인 정보 (체인 ID, 블록 높이, 가스 가격)
- 스마트 컨트랙트 정보
- 익스플로러 연결

### 4. 설정 화면 (Settings)
- 서버 URL 설정
- 클라이언트 ID 설정
- 블록체인 RPC 설정
- 앱 정보 및 라이센스

## 🏗️ 아키텍처

```
ui/
├── features/
│   ├── training/     # 훈련 화면
│   ├── audit/        # 감사 로그 화면
│   ├── blockchain/   # 블록체인 화면
│   └── settings/     # 설정 화면
├── theme/            # Material Design 테마
└── MainActivity.kt   # 메인 액티비티

domain/
├── model/            # 도메인 모델
└── repository/       # 저장소 인터페이스

data/
├── api/              # REST API 서비스
├── blockchain/       # 블록체인 서비스
├── local/            # 로컬 데이터베이스
├── ml/               # 머신러닝 트레이너
└── repository/       # 저장소 구현체

di/                   # 의존성 주입 모듈
```

## 🔧 빌드 및 실행

### 요구사항
- Android Studio Hedgehog (2023.1.1) 이상
- Android SDK API 24 이상
- Kotlin 1.9.22 이상

### 빌드 방법
```bash
# 프로젝트 클론 후
cd client-android

# Gradle 래퍼 권한 설정 (Linux/macOS)
chmod +x gradlew

# 디버그 빌드
./gradlew assembleDebug

# 릴리즈 빌드
./gradlew assembleRelease
```

### 설치 방법
```bash
# ADB를 통한 설치
adb install app/build/outputs/apk/debug/app-debug.apk

# 또는 Android Studio에서 Run 버튼 클릭
```

## ⚙️ 설정

### 서버 연결
1. 설정 화면에서 "Server URL" 설정
2. FedBlock 서버가 실행 중인지 확인
3. "Test Connection" 버튼으로 연결 테스트

### 블록체인 연결
1. 설정 화면에서 "RPC URL" 설정
2. 스마트 컨트랙트 주소 확인
3. 블록체인 화면에서 연결 상태 확인

## 🔐 보안 고려사항

- **프라이빗 키**: 개발용 키만 포함 (프로덕션에서는 안전한 키 관리 필요)
- **HTTPS**: 가능한 모든 통신에 HTTPS 사용 권장
- **권한**: 최소 필요 권한만 요청
- **데이터 보호**: 로컬 모델 데이터 암호화 고려

## 🤝 기여 방법

1. 이슈 생성 또는 기존 이슈 확인
2. 피처 브랜치 생성
3. 변경 사항 구현 및 테스트
4. Pull Request 생성

## 📄 라이센스

MIT License - 자세한 내용은 [LICENSE](../LICENSE) 파일 참조
