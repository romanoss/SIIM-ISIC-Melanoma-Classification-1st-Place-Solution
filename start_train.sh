# 2 gpus
#python train.py --kernel-type 9c_b3_384_384_ext_10ep_us4 --init-lr 9e-5 --batch-size 128 --data-dir ./data/ --data-folder 384 --image-size 384 --enet-type efficientnet_b3 --n-epochs 10 --num_workers 32 --CUDA_VISIBLE_DEVICES 0,1

# 1 gpu
python train.py --kernel-type 9c_b3_384_384_ext_10ep_us4 --init-lr 9e-4 --batch-size 32 --data-dir ./data/ --data-folder 384 --image-size 384 --enet-type efficientnet_b3 --n-epochs 10 --num_workers 16 --CUDA_VISIBLE_DEVICES 0
#python train_xla_bf16.py --kernel-type 9c_b6_384_384_ext_12ep --init-lr 9e-5 --data-dir /kaggle/input --data-folder 384 --batch-size 32 --image-size 384 --enet-type tf_efficientnet_b6.ns_jft_in1k --n-epochs 12
#python train_xla_bf16.py --kernel-type 9c_beit_384_384_ext_18ep --data-dir /kaggle/input --data-folder 384 --batch-size 4 --image-size 384 --enet-type beit_large_patch16_384.in22k_ft_in22k_in1k --n-epochs 10
#python train_xla_bf16.py --kernel-type 9c_eva_512_448_ext_10ep --init-lr 4e-5 --data-dir /kaggle/input --data-folder 512 --batch-size 16 --image-size 448 --enet-type eva02_base_patch14_448.mim_in22k_ft_in22k_in1k --n-epochs 10