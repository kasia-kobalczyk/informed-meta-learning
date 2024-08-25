# SET KNOWLEDGE
# =============

# BASE ========

python config.py --seed 0 --project-name inp_sinusoids --dataset set-trending-sinusoids --num-epochs 10 --input-dim 1 --output-dim 1 --run-name-prefix trend_sinusoid_base --use-knowledge False --noise 0.2 --min-num-context 0 --max-num-context 10 --num-targets 100 --batch-size 64 --num-epochs 1000 --text-encoder set --knowledge-merge sum 
python models/train_inp.py

# INFORMED =====

python config.py --seed 0 --project-name inp_sinusoids --dataset set-trending-sinusoids --num-epochs 10 --input-dim 1 --output-dim 1 --run-name-prefix trend_sinusoid_informed --use-knowledge True --noise 0.2 --min-num-context 0 --max-num-context 10 --num-targets 100 --batch-size 64 --num-epochs 1000 --text-encoder set --knowledge-merge sum --knowledge-type abc2 --test-num-z-samples 32
python models/train_inp.py