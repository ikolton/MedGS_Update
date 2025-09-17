MASK_PATH="prostate_mask"
MODEL_PATH="prostate"

#generate tensors with center points from mask video to use with knn on original video
python3 render.py --model_path $MASK_PATH --interp 1 --generate_points_path $MASK_PATH
#use those tensors to create masked video
python3 render.py --model_path $MODEL_PATH --interp 1 --mask_path $MASK_PATH