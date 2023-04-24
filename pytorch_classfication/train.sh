python osNet_parser.py --size_h 50 --size_w 50 --lr_factor 0.3 --rotate 0 --patience 10 --epochs 500 --w1_path '/DATA/source/ij/pytorch_classfication/weights_osNet/head_cls2_loss.pt' --w2_path '/DATA/source/ij/pytorch_classfication/weights_osNet/head_cls2_acc.pt' --w3_path '/DATA/source/ij/pytorch_classfication/weights_osNet/head_cls2_last.pt' 

python osNet_parser.py --size_h 50 --size_w 50 --lr_factor 0.3 --rotate 10 --patience 10 --epochs 500 --w1_path '/DATA/source/ij/pytorch_classfication/weights_osNet/head_cls2_r_20_loss.pt' --w2_path '/DATA/source/ij/pytorch_classfication/weights_osNet/head_cls2_r_20_acc.pt' --w3_path '/DATA/source/ij/pytorch_classfication/weights_osNet/head_cls2_r_20_last.pt' 

python osNet_parser.py --size_h 50 --size_w 50 --lr_factor 0.3 --rotate 30 --patience 10 --epochs 500 --w1_path '/DATA/source/ij/pytorch_classfication/weights_osNet/head_cls2_r_30_loss.pt' --w2_path '/DATA/source/ij/pytorch_classfication/weights_osNet/head_cls2_r_30_acc.pt' --w3_path '/DATA/source/ij/pytorch_classfication/weights_osNet/head_cls2_r_30_last.pt' 
