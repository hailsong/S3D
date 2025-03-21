#!/bin/bash

SRC_DIR="inference_results/celebamask"
DST_DIR="inference_results/celebamask_fvv"

# 1) 결과를 모을 상위 폴더 생성
mkdir -p "$DST_DIR"

# 2) 00000~02823 사이 폴더 목록 만들기
folders=()
for i in $(seq -f "%05g" 0 2823); do
    if [ -d "$SRC_DIR/$i" ]; then
        folders+=("$i")
    fi
done

# folders 배열이 비어있지 않은지 확인
if [ ${#folders[@]} -eq 0 ]; then
    echo "해당 범위 내에 디렉토리가 없습니다."
    exit 1
fi

echo "총 폴더 수: ${#folders[@]}"

# 3) 폴더 목록을 셔플(shuf)하여 랜덤 추출
shuffled=($(printf '%s\n' "${folders[@]}" | shuf))

# 4) 처음 1000개 폴더만 선택
selected=("${shuffled[@]:0:1000}")
echo "선택된 폴더 수: ${#selected[@]}"

# 5) 선택된 각 폴더에 대해
for folder in "${selected[@]}"; do
    # 결과를 저장할 하위 폴더 생성
    mkdir -p "$DST_DIR/$folder"

    # 후보 파일(0000~0119 범위)만 골라 배열에 저장
    candidate_images=()
    for img in "$SRC_DIR/$folder"/seg2face_*_frame_*.png; do
        # 파일이 존재하지 않는 경우(글로브 확장 실패 시) 넘어감
        [ -f "$img" ] || continue

        # 예: seg2face_1234567890_frame_0057.png -> frame 숫자만 추출
        base=$(basename "$img")
        frame_part="${base##*frame_}"   # frame_ 뒷부분
        frame_part="${frame_part%.png}" # 확장자 제거 -> 0057 (문자열)

        # 10진수로 변환해 범위 검사 (0~119)
        frame_num=$((10#$frame_part))
        if [ "$frame_num" -ge 0 ] && [ "$frame_num" -le 119 ]; then
            candidate_images+=("$img")
        fi
    done

    # 후보가 15개 미만이면 스킵(또는 적게 있으면 전부 복사하도록 바꿀 수도 있음)
    if [ "${#candidate_images[@]}" -lt 15 ]; then
        echo "폴더 $folder: 후보 이미지가 15개 미만이므로 스킵합니다."
        continue
    fi

    # 후보 이미지를 셔플한 뒤 15개만 복사
    candidate_shuffled=($(printf '%s\n' "${candidate_images[@]}" | shuf))
    picks=("${candidate_shuffled[@]:0:15}")

    for p in "${picks[@]}"; do
        cp "$p" "$DST_DIR/$folder/"
    done

    echo "폴더 $folder: ${#picks[@]}개 이미지를 복사했습니다."
done

echo "작업이 완료되었습니다."
