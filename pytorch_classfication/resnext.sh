python3 resNext_parser.py --size_h 256 --size_w 128 --batch 32 --lr_factor 0.3 --patience 10 --epochs 500 --w1_path '/DATA/source/ij/pytorch_classfication/resNext_weights/1_loss.pt' --w2_path '/DATA/source/ij/pytorch_classfication/resNext_weights/1_acc.pt' --w3_path '/DATA/source/ij/pytorch_classfication/resNext_weights/1_last.pt'

python3 osNet_ain.py --size_h 256 --size_w 128 --batch 32 --lr_factor 0.3 --patience 10 --epochs 300 --w1_path '/DATA/source/ij/pytorch_classfication/resNext_weights/o1_loss.pt' --w2_path '/DATA/source/ij/pytorch_classfication/resNext_weights/o1_acc.pt' --w3_path '/DATA/source/ij/pytorch_classfication/resNext_weights/o1_last.pt'

python3 resNext_parser.py --size_h 256 --size_w 128 --batch 32 --lr_factor 0.5 --patience 10 --epochs 500 --w1_path '/DATA/source/ij/pytorch_classfication/resNext_weights/2_loss.pt' --w2_path '/DATA/source/ij/pytorch_classfication/resNext_weights/2_acc.pt' --w3_path '/DATA/source/ij/pytorch_classfication/resNext_weights/2_last.pt'


python3 resNext_parser.py --size_h 256 --size_w 128 --batch 32 --lr_factor 0.03 --patience 10 --epochs 500 --w1_path '/DATA/source/ij/pytorch_classfication/resNext_weights/3_loss.pt' --w2_path '/DATA/source/ij/pytorch_classfication/resNext_weights/3_acc.pt' --w3_path '/DATA/source/ij/pytorch_classfication/resNext_weights/3_last.pt'

python3 resNext_parser.py --size_h 256 --size_w 128 --batch 32 --lr_factor 0.03 --patience 15 --epochs 500 --w1_path '/DATA/source/ij/pytorch_classfication/resNext_weights/4_loss.pt' --w2_path '/DATA/source/ij/pytorch_classfication/resNext_weights/4_acc.pt' --w3_path '/DATA/source/ij/pytorch_classfication/resNext_weights/4_last.pt'
