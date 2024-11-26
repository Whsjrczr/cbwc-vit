nohup bash script-copy-copy.sh >output_script-copy-copy.out 2>&1 &
wait
nohup bash script-copy.sh >output_script-copy.out 2>&1 &
wait
nohup bash script.sh >output_script.out 2>&1 &
wait