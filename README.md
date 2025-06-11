# 🎤 Whisper 음성 인식 프로젝트

이 프로젝트는 [OpenAI Whisper](https://github.com/openai/whisper)와 **빠른 음성 구간 분리**([webrtcvad](https://github.com/wiseman/py-webrtcvad))를 활용하여  
MP3 파일을 손쉽게 텍스트로 변환하는 자동화 도구입니다.

---

## ✅ 주요 기능

- MP3 → WAV(16kHz, mono) 변환 (ffmpeg 사용)
- **최적화된 VAD** 처리로 빠른 음성 구간 분리
- Whisper 모델 기반 음성→텍스트(STT) 변환
- 중복 텍스트 자동 제거
- **선택적 GPU 가속** (NVIDIA CUDA)
- **성능 최적화** (벡터화된 연산, 배치 처리)

---

## ⚡ 설치 & 실행 가이드

### 1️⃣ 최초 설치 (처음 한 번만)

```bash
############################################################
# Whisper 음성 인식 프로젝트 - 최초 설치
############################################################

# 1. PowerShell(윈도우)에서 실행 정책 에러가 나면 ↓ 먼저 입력
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# 2. 프로젝트 폴더로 이동 (본인 경로로 변경)
cd project-path

# 3. requirements.txt 만들기
#  아래 내용을 복사해서 requirements.txt 파일로 저장
: '
# PyTorch with CUDA 12.1
--extra-index-url https://download.pytorch.org/whl/cu121
torch==2.5.1+cu121
torchaudio==2.5.1+cu121

# Audio processing
openai-whisper>=20231117
soundfile>=0.13.1
numpy>=1.24.0
ffmpeg-python>=0.2.0
pydub>=0.25.1
webrtcvad>=2.0.10
librosa>=0.10.0

# Performance optimization
numba>=0.58.0  # librosa 성능 향상
'

# 4. 가상환경 생성 및 활성화
python -m venv venv
venv\Scripts\activate     # (리눅스/Mac: source venv/bin/activate)

# 5. 필수 패키지 설치
pip install -r requirements.txt

# 6. FFmpeg 설치
#    - 윈도우: https://ffmpeg.org/download.html 참고, bin 폴더 PATH 추가
#    - 리눅스: sudo apt-get install ffmpeg
#    - Mac:    brew install ffmpeg

# 7. webrtcvad 설치 오류시 (윈도우)
#    - https://visualstudio.microsoft.com/visual-cpp-build-tools/ 에서 C++ 빌드툴 설치
#    - 또는 https://www.lfd.uci.edu/~gohlke/pythonlibs/#webrtcvad 에서 whl 다운받아 아래처럼 설치
#    pip install webrtcvad‑2.0.10‑cp310‑cp310‑win_amd64.whl
```

### 2️⃣ 이후 실행 (매번)

```bash
############################################################
# Whisper 음성 인식 프로젝트 - 이후 실행
############################################################

# 1. 프로젝트 폴더로 이동
cd project-path

# 2. 가상환경 활성화
venv\Scripts\activate     # (리눅스/Mac: source venv/bin/activate)

# 3. 실행 예시
#    기본 실행 (CPU 모드):
python test_whisper.py --input "C:/audio/lecture.mp3" --output "output"

#    GPU 사용 (NVIDIA 그래픽카드 필요):
python test_whisper.py --use-gpu --input "C:/audio/lecture.mp3" --output "output"

#    모델 크기 선택:
python test_whisper.py --model tiny    # 가장 빠름, 정확도 낮음
python test_whisper.py --model base    # 빠름, 정확도 보통
python test_whisper.py --model small   # 보통, 정확도 좋음
python test_whisper.py --model medium  # 느림, 정확도 매우 좋음 (기본값)
python test_whisper.py --model large   # 매우 느림, 정확도 최상

#    여러 파일 한번에 처리:
python test_whisper.py --input "C:/audio/lecture*.mp3" --output "output"

#    전체 옵션 조합 예시:
python test_whisper.py --use-gpu --model medium --input "C:/audio/*.mp3" --output "output"

############################################################
# 참고:
#  - Whisper 공식문서: https://github.com/openai/whisper
#  - ffmpeg 설치: https://ffmpeg.org/download.html
#  - webrtcvad:   https://github.com/wiseman/py-webrtcvad
############################################################
```

### 3️⃣ 주의사항

1. **최초 설치 후에는 가상환경을 지우지 마세요!**
   - 가상환경은 한 번만 생성하면 됩니다
   - 매번 실행할 때는 `venv\Scripts\activate`만 하면 됩니다

2. **가상환경을 지워야 하는 경우**
   - 패키지 버전 충돌이 발생할 때
   - requirements.txt의 패키지 버전을 변경했을 때
   - 가상환경이 손상되었을 때

3. **실행 전 확인사항**
   - 가상환경이 활성화되어 있는지 확인 (프롬프트 앞에 `(venv)` 표시)
   - FFmpeg가 설치되어 있고 PATH에 추가되어 있는지 확인
   - GPU 사용 시 NVIDIA 드라이버와 CUDA가 설치되어 있는지 확인

## 🎯 성능 최적화 가이드

### 1. 오디오 처리 최적화
- librosa를 사용한 빠른 오디오 로딩
- 벡터화된 연산으로 처리 속도 향상
- 배치 처리로 메모리 효율성 개선

### 2. VAD 처리 최적화
- 최소 음성 구간: 2초
- 최대 허용 간격: 0.6초
- 배치 단위 처리 (100개씩)

### 3. Whisper 처리 최적화
- `condition_on_previous_text=False`로 처리 속도 향상
- 오디오 정규화 개선
- 메모리 사용량 최적화

### 4. GPU 사용 시 주의사항
- 단일 스레드 모드 권장
- GPU 메모리 관리 자동화
- 대형 모델 사용 시 메모리 모니터링

## 🎯 GPU 설정 가이드 (선택사항)

### NVIDIA GPU 사용을 위한 필수 요구사항

1. **NVIDIA 드라이버 설치**
   - [NVIDIA 드라이버 다운로드](https://www.nvidia.com/Download/index.aspx)
   - 최신 버전 설치 권장

2. **CUDA Toolkit 설치**
   - [CUDA Toolkit 다운로드](https://developer.nvidia.com/cuda-downloads)
   - CUDA 12.1 버전 권장 (PyTorch 2.5.1과 호환)

3. **GPU 사용 확인**
   ```python
   import torch
   print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
   print(f"GPU 이름: {torch.cuda.get_device_name(0)}")
   print(f"CUDA 버전: {torch.version.cuda}")
   ```

### 입출력 경로 설정

1. **입력 파일 지정**
   - 단일 파일: `--input "path/to/file.mp3"`
   - 여러 파일: `--input "path/to/*.mp3"` (와일드카드 사용)
   - 필수 옵션입니다

2. **출력 폴더 설정**
   - 기본값: `./output`
   - 지정: `--output "path/to/output"`
   - 자동으로 생성됩니다

3. **출력 파일 이름**
   - 기본값: `transcript_001.txt`, `transcript_002.txt`, ...
   - 접두사 변경: `--prefix "lecture"` → `lecture_001.txt`, `lecture_002.txt`, ...

### 멀티스레딩 설정 가이드

1. **CPU 모드 스레드 설정**
   - CPU 코어 수에 맞게 설정 (예: 4코어 → `--threads 4`)
   - 일반적으로 CPU 코어 수의 50-100% 권장

2. **GPU 모드 스레드 설정 (중요)**
   - ⚠️ GPU 메모리 관리를 위해 1개 스레드 사용 권장 (`--threads 1`)
   - 여러 스레드 사용 시 GPU 메모리 부족 오류 발생 가능
   - 대형 모델(`large`) 사용 시 반드시 1개 스레드 사용

3. **주의사항**
   - GPU 메모리가 부족하면 처리 실패 가능
   - CPU 모드에서는 코어 수를 초과하지 않도록 주의
   - 각 스레드는 독립적인 GPU 메모리를 사용

### Whisper 모델별 특징

| 모델 크기 | 속도 | 정확도 | 메모리 사용량 | 용도 |
|----------|------|--------|--------------|------|
| tiny     | ⚡⚡⚡ | ⭐     | ~1GB        | 빠른 테스트 |
| base     | ⚡⚡  | ⭐⭐    | ~1GB        | 일반 사용 |
| small    | ⚡    | ⭐⭐⭐   | ~2GB        | 정확도 중요 |
| medium   | 🐢   | ⭐⭐⭐⭐  | ~5GB        | 고품질 변환 |
| large    | 🐢🐢 | ⭐⭐⭐⭐⭐ | ~10GB       | 최고 품질 |

### 주의사항

- GPU 메모리 부족 시 `torch.cuda.empty_cache()`로 메모리 정리
- CPU 모드에서는 `tiny` 또는 `base` 모델 권장
- GPU 모드에서는 `medium` 또는 `large` 모델 권장
- ⚠️ GPU 사용 시 반드시 `--threads 1` 사용 권장

## 🔧 문제 해결

### GPU 관련 문제
1. GPU가 인식되지 않는 경우:
   - NVIDIA 드라이버 재설치
   - CUDA Toolkit 재설치
   - 시스템 재부팅

2. CUDA 오류 발생 시:
   - PyTorch 재설치: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`
   - 가상환경 재생성

### 기타 문제
- FFmpeg 오류: PATH 환경변수 확인
- webrtcvad 설치 실패: Visual C++ 빌드 도구 설치
- 메모리 부족: 더 작은 Whisper 모델 사용 (`tiny`, `base`, `small`)
- librosa 설치 실패: numba 패키지 재설치
