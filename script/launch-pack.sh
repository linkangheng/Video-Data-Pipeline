export session_name="pack"

for i in {0..7}; do
    tmux new-session -d -s "$session_name-$i"
    tmux send-keys -t "$session_name-$i" 'rlaunch --cpu 64 --memory=$((1024*32)) --charged-group data --private-machine=yes --positive-tags feature/gpfs=yes --i-know-i-am-wasting-resource --mount=juicefs+s3://oss.i.shaipower.com/kanelin-jfs:/mnt/jfs-test -- zsh' C-m
    tmux send-keys -t "$session_name-$i" "export MACHINE_ID=$i" C-m
    tmux send-keys -t "$session_name-$i" "echo MACHINE_ID: \$MACHINE_ID" C-m
    tmux send-keys -t "$session_name-$i" 'conda activate webvid' C-m
    tmux send-keys -t "$session_name-$i" 'ip a' C-m
    tmux send-keys -t "$session_name-$i" "bash /data/video_pack/script/unicontrol.sh" C-m
done