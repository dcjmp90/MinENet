#!/usr/bin/bash

training="train-keras-openeds"
test_data="keras-dataset-openeds"

if [ $1 == "-h" ]
then
    echo -e "Commands Available:\n $training \n $test_data"
fi

if [ $1 == "train-keras-openeds" ]
then
    docker run -i -t \
                --gpus all \
                --cpus 16 \
                -v /home/$USER/perry/Projects/rebased/MinENet/:/app \
                -w /app \
                utsavisionailab/cityscape:torch \
                python3 train.py \
                --framework=tf.keras \
                --num-classes=4 \
                --output-shape=640,400,4 \
                --input-shape=640,400,1 \
                --batch-size=32 \
                --model-name=openEDS_BCE_JACCARD \
                --dropout-rate=0.30
fi 

if [ $1 == "keras-dataset-openeds" ]
then 
    docker run -i -t \
                --gpus all \
                --cpus 16 \
                -v /home/$USER/perry/Projects/rebased/MinENet/:/app \
                -w /app \
                utsavisionailab/cityscape:torch \
                python3 utils/dataloaders.py
fi