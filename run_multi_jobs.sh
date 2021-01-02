CUDA_VISIBLE_DEVICES=0 python demos/run_video.py video_list_0.pkl &
CUDA_VISIBLE_DEVICES=1 python demos/run_video.py video_list_1.pkl &
CUDA_VISIBLE_DEVICES=2 python demos/run_video.py video_list_2.pkl &
CUDA_VISIBLE_DEVICES=3 python demos/run_video.py video_list_3.pkl &

wait
echo "finished"