#!/bin/bash
batch_sizes=(32)
methods=(cbwc ori rms)
lrs=(1e-4)
epochs=100
wd=0
num_classes=100
datasetroot="~/autodl-tmp/imagenet100/"
dumppath="~/autodl-tmp/result/"
seeds=1
patchsizes=8


l=${#batch_sizes[@]}
n=${#methods[@]}
m=${#seeds[@]}
t=${#lrs[@]}
f=${#wd[@]}

for ((a=0;a<$l;++a))
do 
   for ((b=0;b<$f;++b))
   do 
      for ((j=0;j<$m;++j))
      do	
        for ((k=0;k<$t;++k))
        do

          for ((i=0;i<$n;++i))
          do
                baseString="execute_b${batch_sizes[$a]}_${methods[$i]}_lr${lrs[$k]}_wd${wd[$b]}_s${seeds[$j]}_"
                fileName="${baseString}.sh"
   	            echo "${baseString}"
                touch "${fileName}"
                echo  "#!/usr/bin/env bash
cd \"\$(dirname \$0)/..\" 
CUDA_VISIBLE_DEVICES=0 python train.py \\
 --arch vit_small \\
 --m ${methods[$i]} \\
 --lr ${lrs[$k]} \\
 --batch_size ${batch_sizes[$a]} \\
 --epochs ${epochs} \\
 --wd ${wd} \\
 --num_classes ${num_classes} \\
 --data_path ${datasetroot} \\
 --dump_path ${dumppath} \\
 --patch_size ${patchsizes}\\
 --seed ${seeds[$b]}\\
 --wandb True\\" >> ${fileName}
                echo  "nohup bash ${fileName} >output_${baseString}.out 2>&1 & wait" >> z_bash_excute.sh
           done
         done
      done
   done
done