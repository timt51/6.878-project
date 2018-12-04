source ~/.bashrc

echo "EPW HUVEC"
python -W ignore train_epw.py --mode genomic-only --cell_line HUVEC --fixed_params False > ./data/HUVEC/results/genomic-only-epw.txt

echo "ALL K562"
python -W ignore train_epw.py --mode genomic-only --cell_line K562 --fixed_params False > ./data/K562/results/genomic-only-epw.txt
