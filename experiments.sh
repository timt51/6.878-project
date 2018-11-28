source ~/.bashrc
#python train.py genomic-only 1000 10 > genomic-only_1000_10.txt
#python train.py piq-only 1000 10 > piq-only_1000_10.txt
#python train.py genomic-piq 1000 10 > genomic-piq_1000_10.txt

#python train.py genomic-only 4000 5 > genomic-only_4000_05.txt
#python train.py piq-only 4000 5 > piq-only_4000_05.txt
#python train.py genomic-piq 4000 5 > genomic-piq_4000_05.txt

#python train.py genomic-only 4000 10 > genomic-only_4000_10.txt
#python train.py piq-only 4000 10 > piq-only_4000_10.txt
#python train.py genomic-piq 4000 10 > genomic-piq_4000_10.txt

python train.py genomic-only 1000 10 HUVEC > ./data/HUVEC/results/genomic-only_1000_10.txt 
python train.py piq-only 1000 10 HUVEC > ./data/HUVEC/results/piq-only_1000_10.txt 
python train.py genomic-piq 1000 10 HUVEC > ./data/HUVEC/results/genomic-piq_1000_10.txt

python train.py genomic-only 4000 5 HUVEC > ./data/HUVEC/results/genomic-only_4000_05.txt
python train.py piq-only 4000 5 HUVEC > ./data/HUVEC/results/piq-only_4000_05.txt
python train.py genomic-piq 4000 5 HUVEC > ./data/HUVEC/results/genomic-piq_4000_05.txt

python train.py genomic-only 4000 10 HUVEC > ./data/HUVEC/results/genomic-only_4000_10.txt
python train.py piq-only 4000 10 HUVEC > ./data/HUVEC/results/piq-only_4000_10.txt
python train.py genomic-piq 4000 10 HUVEC > ./data/HUVEC/results/genomic-piq_4000_10.txt

