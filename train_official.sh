#!/bin/bash

#OBJECTS=("s0126_heart" "s0002_kidney_left" "s0468_lung_left" "s0564_vertebrae")
#OBJECTS=("s0468_lung_left")
#OBJECTS=("s0126_heart")
OBJECTS=("s0564_vertebrae")
for OBJECT in "${OBJECTS[@]}"; do
    DIR="$OBJECT"
    DATA=

    echo "====================================================="
    echo " Processing: $OBJECT"
    echo " Data path : $DATA"
    echo " Output dir: $DIR"
    echo "====================================================="

    # Check dataset path exists
    if [[ ! -d "$DATA" ]]; then
        echo "Error: dataset folder not found: $DATA"
        continue
    fi

    #Clean old run
    rm -rf "$DIR"
  
    #  Train
    python3 train.py -s "$DATA" -m "$DIR" --poly_degree 1 --pipeline seg

    # Render
    echo "Rendering..."
    python3 render.py --model_path "$DIR/" --interp 8 || {
       echo "Rendering failed for $OBJECT"
        continue
    }

    # Make video
    python3 video.py \
        --input_folder "$DIR/render" \
        --output_folder "$DIR" \
        --output_name "omg.mp4" \
        --fps 15

  
    echo "Finished: $OBJECT (results in $DIR/omg.mp4)"
done

