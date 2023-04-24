python osNet_finetune.py --lr_factor 3e-06 --lr 3e-06 --batch 64 --patience 10 --head_T_F --epochs 200 --w1_path '/DATA/source/ij/pytorch_classfication/weights_fine_tune/3e-06_10FT_loss.pt' --w2_path '/DATA/source/ij/pytorch_classfication/weights_fine_tune/3e-06_10FT_last.pt'

python osNet_finetune.py --lr_factor 3e-06 --lr 3e-06 --batch 64 --patience 5 --head_T_F --epochs 200 --w1_path '/DATA/source/ij/pytorch_classfication/weights_fine_tune/3e-06_5FT_loss.pt' --w2_path '/DATA/source/ij/pytorch_classfication/weights_fine_tune/3e-06_5FT_last.pt'

python osNet_finetune.py --lr_factor 3e-04 --lr 3e-04 --batch 64 --patience 10 --head_T_F --epochs 200 --w1_path '/DATA/source/ij/pytorch_classfication/weights_fine_tune/3e-04_10FT_loss.pt' --w2_path '/DATA/source/ij/pytorch_classfication/weights_fine_tune/3e-04_10FT_last.pt'

python osNet_finetune.py --lr_factor 3e-04 --lr 3e-04 --batch 64 --patience 5 --head_T_F --epochs 200 --w1_path '/DATA/source/ij/pytorch_classfication/weights_fine_tune/3e-04_5FT_loss.pt' --w2_path '/DATA/source/ij/pytorch_classfication/weights_fine_tune/3e-04_5FT_last.pt'



python osNet_finetune_total.py --size_h 256 --size_w 128 --lr_factor 0.01 --batch 32 --patience 10 --epochs 200 --w1_path '/DATA/source/ij/pytorch_classfication/weights_osNet/001_total_loss1.pt' --w2_path '/DATA/source/ij/pytorch_classfication/weights_osNet/001_total_last1.pt' 

python osNet_finetune_total.py --size_h 256 --size_w 128 --lr_factor 0.05 --batch 32 --patience 10 --epochs 200 --w1_path '/DATA/source/ij/pytorch_classfication/weights_osNet/005_total_loss1.pt' --w2_path '/DATA/source/ij/pytorch_classfication/weights_osNet/005_total_last1.pt'  

