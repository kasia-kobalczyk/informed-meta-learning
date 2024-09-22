# BASE ========
python config.py --seed 1 --project-name INPs_sinusoids --dataset set-trending-sinusoids --input-dim 1 --output-dim 1 --run-name-prefix np --use-knowledge False --noise 0.2 --min-num-context 0 --max-num-context 10 --num-targets 100 --batch-size 64 --num-epochs 600 --text-encoder set --knowledge-merge sum 
python models/train_inp.py

python config.py --seed 1 --project-name INPs_sinusoids --dataset set-trending-sinusoids-dist-shift --input-dim 1 --output-dim 1 --run-name-prefix np_dist_shift --use-knowledge False --noise 0.2 --min-num-context 0 --max-num-context 10 --num-targets 100 --batch-size 64 --num-epochs 600 --text-encoder set --knowledge-merge sum 
python models/train_inp.py

# INFORMED =====
python config.py --seed 1 --project-name INPs_sinusoids --dataset set-trending-sinusoids  --input-dim 1 --output-dim 1 --run-name-prefix inp_abc2 --use-knowledge True --noise 0.2 --min-num-context 0 --max-num-context 10 --num-targets 100 --batch-size 64 --num-epochs 600 --text-encoder set --knowledge-merge sum --knowledge-type abc2 --test-num-z-samples 32
python models/train_inp.py

python config.py --seed 1 --project-name INPs_sinusoids --dataset set-trending-sinusoids-dist-shift  --input-dim 1 --output-dim 1 --run-name-prefix inp_b_dist_shift --use-knowledge True --noise 0.2 --min-num-context 0 --max-num-context 10 --num-targets 100 --batch-size 64 --num-epochs 600 --text-encoder set --knowledge-merge sum --knowledge-type b --test-num-z-samples 32
python models/train_inp.py