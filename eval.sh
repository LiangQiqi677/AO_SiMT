fairseq_dir=fairseq_waitk
data_path=data/wmt15_deen
export PYTHONPATH=$fairseq_dir:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

model_dir=models/models_deen_cl_aw
results_path=results/results_deen_cl_aw
if [ ! -d ${results_path} ];then
    mkdir ${results_path}
    chmod -R 777 ${results_path}
fi

waitk=1
for step in `seq 150000 5000 200000`;
do
  PYTHONIOENCODING=utf-8 python3 ${fairseq_dir}/fairseq_cli/generate_waitk.py \
    ${data_path} \
    --path ${model_dir}/checkpoint_${step}.pt \
    --left-pad-source False \
    --task waitk_translation --eval-waitk ${waitk} \
    --user-dir ${fairseq_dir}/examples/waitk \
    --remove-bpe --beam 4 --batch-size 256 > ${results_path}/result.${waitk}.${step}
  cd ${results_path}
  grep ^H result.${waitk}.${step} | cut -f2- | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > result.bleu.${waitk}.${step}
  multi-bleu.perl data/test.norm.tok.en < result.bleu.${waitk}.${step}
  grep ^C result.${waitk}.${step} | cut -f2- | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > result.al.${waitk}.${step}
  python ${code_dir}/metricAL.py result.al.${waitk}.${step}
done
