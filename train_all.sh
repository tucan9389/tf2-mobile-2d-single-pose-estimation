# python train.py \
# --dataset_config config/dataset/receipt5982-gpu.cfg \
# --experiment_config config/training/receipt-experiment04-cpm-sg5-gpu.cfg; \
# python train.py \
# --dataset_config config/dataset/receipt5982-gpu.cfg \
# --experiment_config config/training/receipt-experiment04-cpm-sg3-gpu.cfg; \
# python train.py \
# --dataset_config config/dataset/receipt5982-gpu.cfg \
# --experiment_config config/training/receipt-experiment04-cpm-sg2-gpu.cfg; \
# python train.py \
# --dataset_config config/dataset/receipt5982-gpu.cfg \
# --experiment_config config/training/receipt-experiment04-cpm-sg1-gpu.cfg; \
# python train.py \
# --dataset_config config/dataset/receipt5982-gpu.cfg \
# --experiment_config config/training/receipt-experiment04-hg-gpu.cfg; 
python train.py \
--dataset_config config/dataset/receipt5982-gpu.cfg \
--experiment_config config/training/receipt-experiment05-cpm-sg4-onlyupsampling_1-gpu.cfg; \
python train.py \
--dataset_config config/dataset/receipt5982-gpu.cfg \
--experiment_config config/training/receipt-experiment05-cpm-sg4-onlyupsampling_2-gpu.cfg; \
python train.py \
--dataset_config config/dataset/receipt5982-gpu.cfg \
--experiment_config config/training/receipt-experiment05-cpm-sg4-onlyupsampling_3-gpu.cfg; \
python train.py \
--dataset_config config/dataset/receipt5982-gpu.cfg \
--experiment_config config/training/receipt-experiment05-cpm-sg4-onlyupsampling_4-gpu.cfg; \
python train.py \
--dataset_config config/dataset/receipt5982-gpu.cfg \
--experiment_config config/training/receipt-experiment05-cpm-sg4-onlyupsampling_5-gpu.cfg; \
python train.py \
--dataset_config config/dataset/receipt5982-gpu.cfg \
--experiment_config config/training/receipt-experiment05-cpm-sg4-onlyupsampling_6-gpu.cfg;