fairseq_dir=fairseq_waitk
data_path=data/wmt15_deen
model_dir=models/models_deen_cl
export PYTHONPATH=${fairseq_dir}:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

waitk=1
cl=(0 1 2 3 4 5 6 7 8 9)
step=20000
for ((i=0; i<${#mask[@]}; i++));
do
  python3 ${fairseq_dir}/fairseq_cli/train.py ${data_path} \
    --source-lang de --target-lang en \
    --left-pad-source False \
    --left-pad-target False \
    --user-dir ${fairseq_dir}/examples/waitk --arch waitk_transformer \
    --optimizer adam --adam-betas '(0.9, 0.998)' \
    --clip-norm 0.0  --lr-scheduler inverse_sqrt \
    --warmup-init-lr 1e-07 --warmup-updates 4000 \
    --lr 5e-4 --min-lr 1e-09 --criterion label_smoothed_cross_entropy_cl \
    --label-smoothing 0.1 --weight-decay 0.0001 \
    --dropout 0.1 --attention-dropout 0.1 \
    --share-decoder-input-output-embed \
    --save-dir ${model_dir} --no-epoch-checkpoints \
    --update-freq 1 --max-tokens 4096 \
    --log-interval 100 --save-interval-updates 5000 \
    --max-update $((${i}*${step}+${step})) \
    --waitk ${waitk} --seed $RANDOM \
    --fp16 --ddp-backend=no_c10d \
    --load-alignments \
    --alignment-mask ${cl[${i}]}
done