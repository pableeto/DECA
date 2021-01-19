CUDA_VISIBLE_DEVICES=0 python demos/run_video.py video_list_$(($1 + 0)).pkl &
CUDA_VISIBLE_DEVICES=1 python demos/run_video.py video_list_$(($1 + 1)).pkl &
CUDA_VISIBLE_DEVICES=2 python demos/run_video.py video_list_$(($1 + 2)).pkl &
CUDA_VISIBLE_DEVICES=3 python demos/run_video.py video_list_$(($1 + 3)).pkl &

wait
echo "finished"