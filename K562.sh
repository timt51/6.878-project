idx=0
N=6
for pwmid in {1..1316}
do
    ((idx=idx%N)); ((idx++==0)) && wait
    sudo Rscript ./piq-single/pwmmatch.exact.r ./piq-single/common.r ./piq-single/pwms/jasparfix.txt $pwmid ./data/K562/motif.matches &
done

sudo Rscript ./piq-single/bam2rdata.r ./piq-single/common.r ./data/K562/d0.Rdata ./data/K562/ENCFF441RET.bam

task() {
    sudo mkdir ./tmp$1/
    sudo Rscript ./piq-single/pertf.r ./piq-single/common.r ./data/K562/motif.matches/ ./tmp$1/ ./data/K562/calls/ ./data/K562/d0.Rdata $1
    sudo rm -rf ./tmp$1/
}

idx=0
N=6
for pwmid in {1..1316}
do
    ((idx=idx%N)); ((idx++==0)) && wait
    task "$pwmid" &
done
