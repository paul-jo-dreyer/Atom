
# Fresh run

python -m AtomGym.training.train --run-name l1c_baseline --total-timesteps 16_000_000 --n-envs 15 --render-every 2_000_000 --render-grid 3x3 --render-frame-stride 4 --render-max-seconds 8 --checkpoint-every 2_000_000 --max-episode-steps 400 --ent-coef 0.01 --log-std-init 0.3 --batch-size 1024 --n-steps 1024  --stall-penalty 0.5 --obstacle-contact-penalty 0.1 --static-field-penalty 0.1 --ball-alignment 0.005  --manipulator default_pusher --use-subproc 

# Resuming a run

python -m AtomGym.training.train --run-name l1b_baseline --total-timesteps 16_000_000 --n-envs 16 --render-every 2_000_000  --render-frame-stride 4 --render-max-seconds 8 --checkpoint-every 1_000_000 --max-episode-steps 400 --ent-coef 0.01 --render-grid 3x3 --log-std-init 0.3 --stall-penalty 0.5 --obstacle-contact-penalty 0.1 --batch-size 1024 --n-steps 1024 --ball-alignment 0.005 --use-subproc --resume training_runs/l1b_baseline/checkpoints/ppo_12000000_steps.zip 


