# idx=0
# N=6
# for pwmid in {972..1316}
# do
#     ((idx=idx%N)); ((idx++==0)) && wait
#     sudo Rscript pwmmatch.exact.r ./common.r ./pwms/jasparfix.txt $pwmid ../data/piq/motif.matches &
# done

# sudo Rscript bam2rdata.r ./common.r ../data/piq/K562/d0.Rdata ../data/dnase-seq/K562/ENCFF441RET.bam

task() {
    sudo mkdir ../tmp$1/
    sudo Rscript pertf.r ./common.r ../data/piq/motif.matches/ ../tmp$1/ ../data/piq/K562/calls/ ../data/piq/K562/d0.Rdata $1
    sudo rm -rf ../tmp$1/
}

idx=0
N=6
for pwmid in {105..1316}
do
    ((idx=idx%N)); ((idx++==0)) && wait
    task "$pwmid" &
done
