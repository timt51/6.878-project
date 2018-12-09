source ~/.bashrc

python -W i verify_epw.py --cell_line HUVEC --mode genomic-only --fixed_params True > HUVEC-genomic-only-epw.txt
python -W i verify_epw.py --cell_line HUVEC --mode piq-no-window-only --fixed_params True > HUVEC-piq-no-window-only-epw.txt
python -W i verify_epw.py --cell_line HUVEC --mode piq-window-only --fixed_params True > HUVEC-piq-window-only-epw.txt
python -W i verify_epw.py --cell_line HUVEC --mode genomic-piq-no-window --fixed_params True > HUVEC-genomic-piq-no-window-epw.txt
python -W i verify_epw.py --cell_line HUVEC --mode genomic-piq-window --fixed_params True > HUVEC-genomic-piq-window-epw.txt

python -W i verify_epw.py --cell_line K562 --mode genomic-only --fixed_params True > K562-genomic-only-epw.txt
python -W i verify_epw.py --cell_line K562 --mode piq-no-window-only --fixed_params True > K562-piq-no-window-only-epw.txt
python -W i verify_epw.py --cell_line K562 --mode piq-window-only --fixed_params True > K562-piq-window-only-epw.txt
python -W i verify_epw.py --cell_line K562 --mode genomic-piq-no-window --fixed_params True > K562-genomic-piq-no-window-epw.txt
python -W i verify_epw.py --cell_line K562 --mode genomic-piq-window --fixed_params True > K562-genomic-piq-window-epw.txt

# gcloud compute scp 