#!/bin/bash

model_name="pix2pix3d_edge2face"

# 폴더 경로
src="./results/${model_name}/"
dest="./inference_results/model_output/${model_name}/fvv/"

if [ -z "$src" ] || [ -z "$dest" ]; then
    echo "Usage: $0 source_folder target_folder"
    exit 1
fi

# 타겟 폴더가 없으면 생성
mkdir -p "$dest"

# 소스 폴더 내의 모든 폴더 목록 (즉시 하위 디렉토리)
dirs=("$src"/*)
mask_dirs=()
for d in "${dirs[@]}"; do
    if [ -d "$d" ]; then
        mask_dirs+=("$d")
    fi
done

# 모든 폴더 개수 확인
count=${#mask_dirs[@]}

# 폴더 개수가 1000개를 초과하면 랜덤하게 1000개 선택
if [ "$count" -gt 1000 ]; then
    selected_dirs=($(printf "%s\n" "${mask_dirs[@]}" | shuf -n 1000))
else
    selected_dirs=("${mask_dirs[@]}")
fi

# 선택할 프레임 번호 (총 15장)
# 예시: 0000, 0007, 0015, 0023, ... , 0111
frame_indices=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14)

# 각 mask 폴더에 대해
for mask_dir in "${selected_dirs[@]}"; do
    base=$(basename "$mask_dir")
    target_mask_dir="$dest/$base"
    mkdir -p "$target_mask_dir"
    
    # 각 지정된 프레임 번호에 대해
    for idx in "${frame_indices[@]}"; do
        frame=$(printf "%04d" "$idx")
        # 예: *frame_0000.png
        files=("$mask_dir"/*frame_"$frame".png)
        # 파일이 여러 개 있으면 오름차순 정렬 후 첫 번째 파일만 선택
        if [ -e "${files[0]}" ]; then
            file=$(printf "%s\n" "${files[@]}" | sort | head -n 1)
            cp "$file" "$target_mask_dir/"
            echo "$file 를 $target_mask_dir 폴더로 복사하였습니다."
        fi
    done
done

echo "선택한 이미지 복사가 완료되었습니다."
