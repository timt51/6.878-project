cd ./piq-single

idx=0
N=70
for pwmid in {1..1316}
do
    ((idx=idx%N)); ((idx++==0)) && wait
    sudo Rscript ./pwmmatch.exact.r ./common.r ./pwms/jasparfix.txt $pwmid ../data/HUVEC/motif.matches &
done

#sudo Rscript ./bam2rdata.r ./common.r ../data/HUVEC/d0.Rdata ../data/HUVEC/ENCFF818BFC.bam

open_sem(){
    mkfifo pipe-$$
    exec 3<>pipe-$$
    rm pipe-$$
    local i=$1
    for((;i>0;i--)); do
        printf %s 000 >&3
    done
}
run_with_lock(){
    local x
    read -u 3 -n 3 x && ((0==x)) || exit $x
    (
    "$@" 
    printf '%.3d' $? >&3
    )&
}
task() {
    sudo mkdir ../tmp$1/
    sudo Rscript ./pertf.r ./common.r ../data/HUVEC/motif.matches/ ../tmp$1/ ../data/HUVEC/calls/ ../data/HUVEC/d0.Rdata $1
    sudo rm -rf ../tmp$1/
}

idx=0
N=35
open_sem $N
for pwmid in {1..1316}; do
    run_with_lock task $pwmid
done 