python config.py --seed 0 --project-name INPs_temperature_test --dataset temperature --input-dim 1 --output-dim 1 --run-name-prefix np_beta_25 --beta 25 --use-knowledge False --noise 0 --min-num-context 0 --max-num-context 15 --num-targets 288 --batch-size 64 --num-epochs 3000 --x-sampler random-uniform-15 --knowledge-merge sum --data-agg-func cross-attention --knowledge-type min_max
python models/train.py

python config.py --seed 0 --project-name INPs_temperature_test --dataset temperature --input-dim 1 --output-dim 1 --run-name-prefix inp_min_max_beta_25 --beta 25 --use-knowledge True --noise 0 --min-num-context 0 --max-num-context 15 --num-targets 288 --batch-size 64 --num-epochs 3000 --x-sampler random-uniform-15  --knowledge-merge sum --data-agg-func cross-attention --knowledge-type min_max
python models/train.py

python config.py --seed 0 --project-name INPs_temperature_test --dataset temperature --input-dim 1 --output-dim 1 --run-name-prefix inp_llama_embed_beta_25 --beta 25 --use-knowledge True --noise 0 --min-num-context 0 --max-num-context 15 --num-targets 288 --batch-size 64 --num-epochs 3000 --x-sampler random-uniform-15  --knowledge-merge sum --data-agg-func cross-attention --knowledge-type llama_embed
python models/train.py


python config.py --seed 0 --project-name INPs_temperature_test --dataset temperature --input-dim 1 --output-dim 1 --run-name-prefix inp_desc_beta_25 --beta 25 --use-knowledge True --noise 0 --min-num-context 0 --max-num-context 15 --num-targets 288 --batch-size 64 --num-epochs 3000 --x-sampler random-uniform-15 --knowledge-type desc --knowledge-merge sum --knowledge-type desc --text-encoder roberta --freeze-llm True --tune-llm-layer-norms True --data-agg-func cross-attention   
python models/train.py
