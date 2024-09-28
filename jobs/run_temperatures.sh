# Train NP without knowledge
python config.py  --project-name INPs_temperature --dataset temperature  --run-name-prefix np --use-knowledge False --noise 0 --min-num-context 0 --max-num-context 15 --num-targets 288 --batch-size 64 --num-epochs 1000 --x-sampler random-uniform-15 --knowledge-merge sum --knowledge-type min_max --data-agg-func cross-attention --beta 25  --seed 1 --lr 1e-4 --hidden-dim 128 --input-dim 1 --output-dim 1
python models/train.py

# Train INP with knowledge as min_max_temperature
python config.py  --project-name INPs_temperature --dataset temperature  --run-name-prefix inp_min_max --use-knowledge True --noise 0 --min-num-context 0 --max-num-context 15 --num-targets 288 --batch-size 64 --num-epochs 1000 --x-sampler random-uniform-15  --knowledge-merge sum --knowledge-type min_max --data-agg-func cross-attention --beta 25  --seed 1 --lr 1e-4 --hidden-dim 128 --input-dim 1 --output-dim 1
python models/train.py

# Train INP with knowledge as text description
python config.py  --project-name INPs_temperature --dataset temperature  --run-name-prefix inp_desc --use-knowledge True --noise 0 --min-num-context 0 --max-num-context 15 --num-targets 288 --batch-size 64 --num-epochs 1000 --x-sampler random-uniform-15 --knowledge-merge sum --knowledge-type desc --text-encoder roberta --freeze-llm True --tune-llm-layer-norms True --data-agg-func cross-attention --beta 25  --seed 1 --lr 1e-4 --hidden-dim 128 --input-dim 1 --output-dim 1
python models/train.py
