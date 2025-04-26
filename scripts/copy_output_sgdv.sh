#!/bin/bash

model_name="pix2pix3d_edge2face"

# 폴더 경로
SOURCE="./results/${model_name}_sgdv/"
TARGET="./inference_results/model_output/${model_name}/sgdv/"

# 소스 폴더가 존재하는지 확인
if [ ! -d "$SOURCE" ]; then
    echo "Source folder does not exist: $SOURCE"
    exit 1
fi

# 타겟 폴더가 없으면 생성
mkdir -p "$TARGET"

# 소스 폴더 내의 모든 디렉토리 목록을 가져옴 (최대 1단계 깊이)
mask_dirs=($(find "$SOURCE" -maxdepth 1 -type d -name "*"))
total=${#mask_dirs[@]}

# 만약 mask 디렉토리 개수가 1000개보다 많으면 랜덤하게 1000개 선택
if [ "$total" -gt 1000 ]; then
    selected_dirs=($(printf "%s\n" "${mask_dirs[@]}" | shuf | head -n 1000))
else
    selected_dirs=("${mask_dirs[@]}")
fi

# 선택된 각 mask 폴더에 대해
for dir in "${selected_dirs[@]}"; do
    base=$(basename "$dir")
    target_dir="$TARGET/$base"
    mkdir -p "$target_dir"
    
    # 해당 폴더 내의 GTView가 포함된 파일만 복사
    for file in "$dir"/*GTView*; do
        if [ -f "$file" ]; then
            cp "$file" "$target_dir/"
            echo "$file 를 $target_dir 폴더로 복사하였습니다."
        fi
    done
done

echo "파일 복사가 완료되었습니다."
