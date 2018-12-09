source ~/.bashrc

python -W i verify.py --cell_line HUVEC --mode genomic-only --fixed_params True > HUVEC-genomic-only.txt &
python -W i verify.py --cell_line HUVEC --mode piq-only --fixed_params True > HUVEC-piq-only.txt &
python -W i verify.py --cell_line HUVEC --mode genomic-piq --fixed_params True > HUVEC-genomic-piq.txt &

python -W i verify.py --cell_line K562 --mode genomic-only --fixed_params True > K562-genomic-only.txt &
python -W i verify.py --cell_line K562 --mode piq-only --fixed_params True > K562-piq-only.txt &
python -W i verify.py --cell_line K562 --mode genomic-piq --fixed_params True > K562-genomic-piq.txt &
