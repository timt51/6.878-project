source ~/.bashrc

python augment_train.py HUVEC

python -W ignore train.py --mode genomic-only --cell_line HUVEC --fixed_params True > ./data/HUVEC/results/genomic-only-fixed.txt
python -W ignore train.py --mode piq-only --cell_line HUVEC --fixed_params True > ./data/HUVEC/results/piq-only-fixed.txt
python -W ignore train.py --mode genomic-piq --cell_line HUVEC --fixed_params True > ./data/HUVEC/results/genomic-piq-fixed.txt

python -W ignore train.py --mode genomic-only --cell_line HUVEC --fixed_params False > ./data/HUVEC/results/genomic-only.txt
python -W ignore train.py --mode piq-only --cell_line HUVEC --fixed_params False > ./data/HUVEC/results/piq-only.txt
python -W ignore train.py --mode genomic-piq --cell_line HUVEC --fixed_params False > ./data/HUVEC/results/genomic-piq.txt


python augment_train.py K562

python -W ignore train.py --mode genomic-only --cell_line K562 --fixed_params True > ./data/K562/results/genomic-only-fixed.txt
python -W ignore train.py --mode piq-only --cell_line K562 --fixed_params True > ./data/K562/results/piq-only-fixed.txt
python -W ignore train.py --mode genomic-piq --cell_line K562 --fixed_params True > ./data/K562/results/genomic-piq-fixed.txt

python -W ignore train.py --mode genomic-only --cell_line K562 --fixed_params False > ./data/K562/results/genomic-only.txt
python -W ignore train.py --mode piq-only --cell_line K562 --fixed_params False > ./data/K562/results/piq-only.txt
python -W ignore train.py --mode genomic-piq --cell_line K562 --fixed_params False > ./data/K562/results/genomic-piq.txt
