fairseq_dir=fairseq_waitk
data_dir=data/wmt15_deen
export PYTHONPATH=${fairseq_dir}:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1,2,3

output_dir=${fairseq_dir}/data/wmt15_deen
if [ ! -d ${output_dir} ]; then
    mkdir ${output_dir}
    chmod -R 777 ${output_dir}
fi

cp data/wmt15_deen_AO ${data_dir}/train.norm.tok.bpe.align
python ${fairseq_dir}/fairseq_cli/preprocess.py \
    --source-lang de --target-lang en \
    --trainpref ${data_dir}/train.norm.tok.bpe \
    --validpref ${data_dir}/valid.norm.tok.bpe \
    --testpref ${data_dir}/test.norm.tok.bpe \
    --srcdict ${data_dir}/vocab.share.32000 \
    --tgtdict ${data_dir}/vocab.share.32000 \
    --destdir ${output_dir} \
    --align-suffix align \
    --workers 100

