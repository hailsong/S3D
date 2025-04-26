#!/bin/bash

model_name="pix2pix3d_edge2face"

# 폴더 경로
SOURCE="./results/${model_name}_sgdv/"
TARGET="./inference_results/model_output/${model_name}/gtview/"

# target 폴더가 없으면 생성합니다.
mkdir -p "$TARGET"

# 현재 폴더 내의 모든 디렉토리를 순회합니다.
for dir in "$SOURCE"/*; do
    if [ -d "$dir" ]; then
        # extract folder name
        folder_name=$(basename "$dir")
        # 디렉토리 내에서 "GTView"가 포함된 .png 파일들을 오름차순 정렬한 후 첫번째 파일 선택
        first_file=$(ls "$dir"/*GTView*.png 2>/dev/null | sort | head -n 1)
        if [ -n "$first_file" ]; then
            new_file="$TARGET/${folder_name}_$(basename "$first_file")"
            cp "$first_file" "$new_file"
            echo "[$dir] $first_file 를 $new_file 폴더로 복사하였습니다."
        else
            echo "[$dir] GTView 파일을 찾을 수 없습니다."
        fi
    fi
done
