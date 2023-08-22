import subprocess
devices = [0]
displays = [':0.0']
device_idx = 0
display_idx = 0
for seed in [0]:
    for task in ['take_lid_off_saucepan']:
        for prompt in ['r3mp2e_langprompt2']:
            for extr, intr in [(0, 1)]:
                if prompt == 'r3mp2e_langprompt2' or prompt == 'r3mp2e_langprompt6' or prompt == 'r3mp2e_langprompt2_zest':
                    extr, intr = 0.1, 0.9
                for ts in [290200]:
                    _out = _err = f"/home/ademi_adeniji/amber_vlsp_logs2/{prompt}_seed{seed}_{task}_extr{extr}_intr{intr}_ts{ts}_ft.log"
                    if prompt == "from_scratch":
                        command = f'TF_CPP_MIN_LOG_LEVEL=0 CUDA_VISIBLE_DEVICES={devices[device_idx%len(devices)]} TF_XLA_FLAGS=--tf_xla_auto_jit=2 ' +\
                        f'vglrun -d {displays[display_idx%len(displays)]} python /home/ademi_adeniji/amber_vlsp_mwm/train.py ' +\
                        f'--logdir /home/ademi_adeniji/amber_vlsp_experiments/finetune/0628_{task}/{prompt}_seed{seed} ' +\
                        f'--task {task} --seed {seed} ' +\
                        f'--configs front_wrist vlsp ' +\
                        f'--shaped_rewards True' 
                    else:
                        # loaddir = f"/home/amberxie/CurrProj/vlsp_mwm/experiments/pretrain/0510_r3m/{prompt}"
                        # loaddir = f"/home/ademi_adeniji/amber_vlsp_experiments/pretrain/0802/{prompt}_seed1_extr{extr}_intr{intr}"
                        # loaddir = f"/home/ademi_adeniji/amber_vlsp_experiments/pretrain/0628/{prompt}_seed1_extr{extr}_intr{intr}"
                        loaddir = f"/home/ademi_adeniji/amber_vlsp_experiments/finetune/0628_{task}/{prompt}_seed{seed}_extr{extr}_intr{intr}_ts290200"
                        # if prompt == 'r3mp2e_langprompt2' or prompt == 'r3mp2e_langprompt6':
                        #     loaddir = f"/home/ademi_adeniji/amber_vlsp_experiments/pretrain/0618/{prompt}_seed1_extr{extr}_intr{intr}"
                        command = f'TF_CPP_MIN_LOG_LEVEL=0 CUDA_VISIBLE_DEVICES={devices[device_idx%len(devices)]} TF_XLA_FLAGS=--tf_xla_auto_jit=2 ' +\
                        f'vglrun -d {displays[display_idx%len(displays)]} python /home/ademi_adeniji/amber_vlsp_mwm/train.py ' +\
                        f'--logdir /home/ademi_adeniji/amber_vlsp_experiments/finetune/0628_{task}/{prompt}_seed{seed}_extr{extr}_intr{intr}_ts290200 ' +\
                        f'--task {task} --seed {seed} --device cuda:0 --vidlang_model_device cuda:0 ' +\
                        f'--use_lang_embeddings True --configs front_wrist vlsp --curriculum.use False ' +\
                        f'--critic_linear_probe True ' +\
                        f'--loaddir {loaddir} --ts {ts} ' +\
                        f'--plan2explore True --expl_intr_scale 0 --expl_extr_scale 1 --shaped_rewards True' 
                    with open(_out,"wb") as out, open(_err,"wb") as err:
                        print(devices[device_idx%len(devices)], command)
                        device_idx += 1
                        display_idx += 1
                        p = subprocess.Popen(command, shell=True, stdout=out, stderr=err, bufsize=0)
p.wait()
print('returncode', p.returncode)

#  f'--loaddir /home/ademi_adeniji/amber_vlsp_experiments/finetune/0628_{task}/{prompt}_seed{seed} ' +\
#                         f'--ts {ts} ' +\