import whisper
import torch
import os
import time
import torchaudio
import numpy as np
import subprocess
import tempfile
import webrtcvad
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
import threading
import librosa
import soundfile as sf
import shutil

# ----------- (1) MP3를 WAV로 변환 (ffmpeg) -----------
def convert_audio_to_wav(audio_path):
    """
    다양한 오디오 파일을 WAV로 변환
    지원 형식: mp3, m4a, flac, ogg, wma, aac 등
    """
    temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    temp_wav.close()
    
    # 파일 확장자 확인
    ext = os.path.splitext(audio_path)[1].lower()
    if ext == '.wav':
        # WAV 파일은 그대로 복사
        shutil.copy2(audio_path, temp_wav.name)
        return temp_wav.name
        
    # ffmpeg로 변환
    command = [
        'ffmpeg', '-y',
        '-i', audio_path,
        '-acodec', 'pcm_s16le',
        '-ar', '16000',
        '-ac', '1',
        temp_wav.name
    ]
    completed = subprocess.run(command, capture_output=True)
    if completed.returncode != 0:
        print(f"ffmpeg 변환 실패: {completed.stderr.decode()}")
        raise RuntimeError(f"ffmpeg 변환 실패: {audio_path}")
    return temp_wav.name

def load_audio_fast(audio_path):
    """빠른 오디오 로딩 - librosa 사용"""
    try:
        # librosa로 직접 로딩 (더 빠름)
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        return audio, sr
    except Exception as e:
        print(f"librosa 로딩 실패, soundfile 시도: {e}")
        # fallback to soundfile
        data, samplerate = sf.read(audio_path)
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)
        if samplerate != 16000:
            data = librosa.resample(data, orig_sr=samplerate, target_sr=16000)
        return data, 16000

# ----------- (2) 개선된 VAD 처리 -----------
def split_audio_by_vad_optimized(audio_path, min_voice_duration=3.0, aggressiveness=1):
    """
    최적화된 VAD 처리
    - 벡터화된 연산 사용
    - 메모리 효율적인 처리
    - 빠른 오디오 로딩
    """
    print(f"오디오 로딩 중: {audio_path}")
    start_time = time.time()
    
    # 오디오 파일을 WAV로 변환
    wav_path = convert_audio_to_wav(audio_path)
    
    try:
        # 빠른 오디오 로딩
        audio_data, sample_rate = load_audio_fast(wav_path)
        total_duration = len(audio_data) / sample_rate
        print(f"오디오 로딩 완료 (길이: {total_duration:.1f}초, 소요: {time.time()-start_time:.2f}초)")
        
        # int16으로 변환 (VAD 요구사항)
        audio_int16 = (audio_data * 32767).astype(np.int16)
        pcm_audio = audio_int16.tobytes()
        
        print("VAD 처리 중...")
        vad_start = time.time()
        
        vad = webrtcvad.Vad(aggressiveness)
        frame_duration_ms = 30  # 30ms 프레임
        frame_size = int(sample_rate * frame_duration_ms / 1000)  # 샘플 수
        frame_size_bytes = frame_size * 2  # 16bit = 2 bytes
        
        # 벡터화된 프레임 분할
        total_samples = len(audio_int16)
        n_frames = total_samples // frame_size
        
        # 프레임별로 VAD 처리 (배치 방식)
        voice_flags = []
        frames_data = []
        
        print(f"총 {n_frames}개 프레임 처리 중...")
        
        for i in range(0, n_frames, 100):  # 100개씩 배치 처리
            batch_end = min(i + 100, n_frames)
            batch_flags = []
            
            for j in range(i, batch_end):
                start_idx = j * frame_size
                end_idx = start_idx + frame_size
                
                if end_idx > total_samples:
                    break
                    
                frame_bytes = audio_int16[start_idx:end_idx].tobytes()
                if len(frame_bytes) == frame_size_bytes:
                    try:
                        is_speech = vad.is_speech(frame_bytes, sample_rate)
                        batch_flags.append(is_speech)
                        frames_data.append((start_idx, end_idx))
                    except:
                        batch_flags.append(False)
                        frames_data.append((start_idx, end_idx))
                else:
                    break
            
            voice_flags.extend(batch_flags)
            
            # 진행률 표시
            progress = (batch_end * 100) // n_frames
            if i % 1000 == 0:
                print(f"VAD 진행률: {progress}%")
        
        print(f"VAD 처리 완료 (소요: {time.time()-vad_start:.2f}초)")
        
        # 음성 구간 병합 (벡터화)
        print("음성 구간 병합 중...")
        merge_start = time.time()
        
        voice_flags = np.array(voice_flags, dtype=bool)
        min_frames = int(min_voice_duration * 1000 / frame_duration_ms)  # 최소 프레임 수
        max_gap_frames = 20  # 최대 허용 gap (0.6초)
        
        segments = []
        
        # 연속된 음성 구간 찾기
        in_segment = False
        segment_start = 0
        gap_count = 0
        
        for i, is_voice in enumerate(voice_flags):
            if is_voice:
                if not in_segment:
                    segment_start = i
                    in_segment = True
                gap_count = 0
            else:
                if in_segment:
                    gap_count += 1
                    if gap_count > max_gap_frames:
                        # 세그먼트 종료
                        segment_end = i - gap_count
                        if segment_end - segment_start >= min_frames:
                            # 충분히 긴 세그먼트만 추가
                            start_sample = frames_data[segment_start][0]
                            end_sample = frames_data[min(segment_end, len(frames_data)-1)][1]
                            segments.append((start_sample, end_sample))
                        in_segment = False
                        gap_count = 0
        
        # 마지막 세그먼트 처리
        if in_segment and len(voice_flags) - segment_start >= min_frames:
            start_sample = frames_data[segment_start][0]
            end_sample = frames_data[-1][1]
            segments.append((start_sample, end_sample))
        
        print(f"구간 병합 완료 (소요: {time.time()-merge_start:.2f}초)")
        print(f"총 {len(segments)}개의 음성 구간 추출")
        
        # torch tensor로 변환
        print("텐서 변환 중...")
        tensor_start = time.time()
        
        audio_segments = []
        for i, (start_sample, end_sample) in enumerate(segments):
            segment_audio = audio_data[start_sample:end_sample]
            
            # 최소 길이 체크
            if len(segment_audio) >= sample_rate * min_voice_duration:
                tensor = torch.from_numpy(segment_audio).float().unsqueeze(0)
                audio_segments.append(tensor)
            
            if (i + 1) % 10 == 0:
                print(f"텐서 변환: {i+1}/{len(segments)}")
        
        print(f"텐서 변환 완료 (소요: {time.time()-tensor_start:.2f}초)")
        print(f"최종 {len(audio_segments)}개 세그먼트 생성")
        
        return audio_segments, sample_rate
    finally:
        # 임시 WAV 파일 삭제
        if os.path.exists(wav_path):
            os.unlink(wav_path)

# ----------- (3) 중복 텍스트 제거 -----------
def remove_duplicates(texts):
    unique_texts = []
    seen = set()
    for text in texts:
        text_clean = text.strip()
        if text_clean and text_clean not in seen:
            unique_texts.append(text_clean)
            seen.add(text_clean)
    return unique_texts

def process_file(audio_path, output_dir, model, device, file_num, total_files):
    """단일 파일 처리 함수 - 개선된 VAD 사용"""
    print(f"\n[{file_num}/{total_files}] 파일 처리 시작: {os.path.basename(audio_path)}")
    file_start_time = time.time()

    # GPU 메모리 정리
    if device == "cuda":
        torch.cuda.empty_cache()

    # 개선된 VAD로 음성 구간 나누기
    try:
        segments, sample_rate = split_audio_by_vad_optimized(audio_path, min_voice_duration=2.0)
    except Exception as e:
        print(f"VAD 분할 실패: {e}")
        return

    if not segments:
        print("경고: 추출된 음성 구간이 없습니다!")
        return

    # Whisper 처리
    print(f"\nWhisper 처리 시작 ({len(segments)}개 구간)")
    whisper_start = time.time()
    
    all_text = []
    for j, segment in enumerate(segments):
        print(f"구간 {j+1}/{len(segments)} 처리 중... (길이: {segment.size(1)/sample_rate:.1f}초)")
        
        # 임시 오디오 파일 생성
        temp_path = f"temp_segment_{file_num}_{j}.wav"
        try:
            # 오디오 정규화
            max_val = torch.max(torch.abs(segment))
            if max_val > 0:
                segment_normalized = segment / max_val * 0.9
            else:
                segment_normalized = segment
            
            # 저장
            torchaudio.save(
                temp_path,
                segment_normalized,
                sample_rate,
                encoding="PCM_S",
                bits_per_sample=16
            )
            
            # Whisper 처리
            st = time.time()
            result = model.transcribe(
                temp_path,
                language='ko',
                verbose=False,  # 로그 줄이기
                fp16=False if device == "cpu" else True,
                condition_on_previous_text=False,  # 처리 속도 향상
                initial_prompt="강의 내용:"
            )
            
            if result and "text" in result and result["text"].strip():
                all_text.append(result["text"].strip())
                print(f"  완료 (소요: {time.time()-st:.1f}초)")
            else:
                print(f"  결과 없음")
                
        except Exception as e:
            print(f"  오류: {e}")
        finally:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass

    print(f"Whisper 처리 완료 (소요: {time.time()-whisper_start:.1f}초)")

    if not all_text:
        print("경고: 변환된 텍스트가 없습니다!")
        return

    # 중복 제거 및 저장
    unique_texts = remove_duplicates(all_text)
    
    # 입력 파일 이름을 기반으로 출력 파일 이름 생성
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    txt_path = os.path.join(output_dir, f"{base_name}.txt")
    
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(unique_texts))

    total_time = time.time() - file_start_time
    print(f"\n파일 {file_num} 처리 완료!")
    print(f"  저장 위치: {txt_path}")
    print(f"  중복 제거: {len(all_text)} -> {len(unique_texts)} 구간")
    print(f"  총 소요시간: {total_time:.1f}초")

# ----------- (4) Main Routine -----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='개선된 Whisper 음성 인식 도구')
    parser.add_argument('--use-gpu', action='store_true', help='GPU 사용')
    parser.add_argument('--model', type=str, default='medium', 
                      choices=['tiny', 'base', 'small', 'medium', 'large'],
                      help='Whisper 모델 크기')
    parser.add_argument('--input', type=str, required=True,
                      help='입력 오디오 파일 경로 (지원 형식: mp3, wav, m4a, flac, ogg, wma, aac 등)')
    parser.add_argument('--output', type=str, default='output',
                      help='출력 폴더 경로')
    args = parser.parse_args()

    # GPU 설정
    if args.use_gpu and torch.cuda.is_available():
        device = "cuda"
        print(f"GPU 사용: {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache()
    else:
        device = "cpu"
        print("CPU 모드로 실행")

    print(f"사용 장치: {device}")

    # 모델 로딩
    print("Whisper 모델 로딩 중...")
    start_time = time.time()
    model = whisper.load_model(args.model, device=device)
    print(f"모델 로딩 완료 (소요: {time.time() - start_time:.1f}초)")

    # 입력 파일 처리
    input_path = args.input
    if '*' in input_path:
        import glob
        audio_files = sorted(glob.glob(input_path))
        if not audio_files:
            print(f"파일을 찾을 수 없습니다: {input_path}")
            exit(1)
    else:
        if not os.path.exists(input_path):
            print(f"파일이 존재하지 않습니다: {input_path}")
            exit(1)
        audio_files = [input_path]

    # 출력 폴더 생성
    output_dir = os.path.abspath(args.output)
    os.makedirs(output_dir, exist_ok=True)
    print(f"출력 폴더: {output_dir}")

    total_files = len(audio_files)
    print(f"처리할 파일 수: {total_files}")

    # 파일 처리
    overall_start = time.time()
    
    for i, audio_path in enumerate(audio_files, 1):
        process_file(audio_path, output_dir, model, device, i, total_files)

    total_time = time.time() - overall_start
    print(f"\n전체 처리 완료! (총 소요시간: {total_time/60:.1f}분)")