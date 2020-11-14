# habitat_rl
habitat rl code without PointNavdataset


### Look into
- trainer/algo/ppo/ppo_trainer_memory.py line94
- env_utils/habitat_env.py
- rl_configs/example.yaml
- habitat_homing_rl/env_utils/habitat_env.py

### train
```
python train_rl.py --run-type train --exp-config rl_configs/example.yaml --gpu 0
```
### refer
- https://github.com/obin-hero/Vistarget
