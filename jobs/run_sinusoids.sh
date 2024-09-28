# Base experiment for INPs on sinusoids
# ======================================
# Train NP without knowledge
python config.py  --project-name INPs_sinusoids --dataset set-trending-sinusoids --input-dim 1 --output-dim 1 --run-name-prefix np --use-knowledge False --noise 0.2 --min-num-context 0 --max-num-context 10 --num-targets 100 --batch-size 64 --num-epochs 400 --text-encoder set --knowledge-merge sum --seed 1
python models/train.py

# Train INP with knowledge as one or two parameters a, b, c
python config.py  --project-name INPs_sinusoids --dataset set-trending-sinusoids  --input-dim 1 --output-dim 1 --run-name-prefix inp_abc2 --use-knowledge True --noise 0.2 --min-num-context 0 --max-num-context 10 --num-targets 100 --batch-size 64 --num-epochs 400 --text-encoder set --knowledge-merge sum --knowledge-type abc2 --test-num-z-samples 32 --seed 1
python models/train.py


# Distribution shift experiment for INPs on sinusoids
# ===================================================
# Train NP without knowlege, train & eval: b ~ N(2, 1), test: b ~ N(3, 1)
python config.py  --project-name INPs_sinusoids --dataset set-trending-sinusoids-dist-shift --input-dim 1 --output-dim 1 --run-name-prefix np_dist_shift --use-knowledge False --noise 0.2 --min-num-context 0 --max-num-context 10 --num-targets 100 --batch-size 64 --num-epochs 400 --text-encoder set --knowledge-merge sum --seed 1
python models/train.py

# Train INP with knowledge as b, train & eval: b ~ N(2, 1), test: b ~ N(3, 1)
python config.py  --project-name INPs_sinusoids --dataset set-trending-sinusoids-dist-shift  --input-dim 1 --output-dim 1 --run-name-prefix inp_b_dist_shift --use-knowledge True --noise 0.2 --min-num-context 0 --max-num-context 10 --num-targets 100 --batch-size 64 --num-epochs 400 --text-encoder set --knowledge-merge sum --knowledge-type b --test-num-z-samples 32 --seed 1
python models/train.py