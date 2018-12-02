source ~/.bashrc

#echo "Augment HUVEC"
#python augment_train.py HUVEC

#echo "Fixed HUVEC"
#python -W ignore train.py --mode genomic-only --cell_line HUVEC --fixed_params True > ./data/HUVEC/results/genomic-only-fixed.txt
#python -W ignore train.py --mode piq-only --cell_line HUVEC --fixed_params True > ./data/HUVEC/results/piq-only-fixed.txt
#python -W ignore train.py --mode genomic-piq --cell_line HUVEC --fixed_params True > ./data/HUVEC/results/genomic-piq-fixed.txt

echo "All HUVEC"
python -W ignore train.py --mode genomic-only --cell_line HUVEC --fixed_params False > ./data/HUVEC/results/genomic-only.txt
python -W ignore train.py --mode piq-only --cell_line HUVEC --fixed_params False > ./data/HUVEC/results/piq-only.txt
python -W ignore train.py --mode genomic-piq --cell_line HUVEC --fixed_params False > ./data/HUVEC/results/genomic-piq.txt

echo "Augment K562"
python augment_train.py K562

echo "Fixed K562"
python -W ignore train.py --mode genomic-only --cell_line K562 --fixed_params True > ./data/K562/results/genomic-only-fixed.txt
python -W ignore train.py --mode piq-only --cell_line K562 --fixed_params True > ./data/K562/results/piq-only-fixed.txt
python -W ignore train.py --mode genomic-piq --cell_line K562 --fixed_params True > ./data/K562/results/genomic-piq-fixed.txt

echo "ALL K562"
python -W ignore train.py --mode genomic-only --cell_line K562 --fixed_params False > ./data/K562/results/genomic-only.txt
python -W ignore train.py --mode piq-only --cell_line K562 --fixed_params False > ./data/K562/results/piq-only.txt
python -W ignore train.py --mode genomic-piq --cell_line K562 --fixed_params False > ./data/K562/results/genomic-piq.txt
