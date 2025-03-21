#!/bin/bash

# inference_results/celebamask 경로를 BASE_DIR로 설정
BASE_DIR="inference_results/celebamask"

# 모을 폴더 이름 설정 (원하는 경우 경로까지 포함 가능)
OUTPUT_DIR="GTViews"

# base 디렉토리로 이동
cd "$BASE_DIR" || { echo "디렉토리 $BASE_DIR 를 찾을 수 없습니다."; exit 1; }

# GTView 이미지들을 모을 폴더 생성
mkdir -p "$OUTPUT_DIR"

# 00000, 00001, 00002 등으로 시작하는 모든 디렉토리를 순회
for d in 000*/ ; do
    # 실제 디렉토리인지 확인
    if [ -d "$d" ]; then
        # 해당 디렉토리 안의 *GTView.png 파일을 찾음
        for f in "$d"/*GTView.png; do
            # 파일이 존재하면 복사
            if [ -f "$f" ]; then
                # 예) 00001 폴더의 seg2face_2811802697_GTView.png
                #   -> GTViews/00001_seg2face_2811802697_GTView.png 으로 복사
                filename=$(basename "$f")
                dirname="${d%/}"  # 끝의 / 문자 제거
                cp "$f" "$OUTPUT_DIR/${dirname}_${filename}"
            fi
        done
    fi
done

echo "모든 GTView 파일을 '$OUTPUT_DIR' 폴더로 복사했습니다."
