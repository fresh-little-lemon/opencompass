from mmengine.config import read_base
from opencompass.models import HuggingFaceCausalLM

models = [
    dict(
        type=HuggingFaceCausalLM,
        path='/root/finetune/ultra/script/swift_output/InternLM2_5-7B-Lora-SFT/v1-20250604-103655/checkpoint-1500-merged',
        tokenizer_path='/root/finetune/ultra/script/swift_output/InternLM2_5-7B-Lora-SFT/v1-20250604-103655/checkpoint-1500-merged',
        tokenizer_kwargs=dict(padding_side='left', truncation_side='left'),
        model_kwargs=dict(device_map='auto'),
        max_seq_len=32768,
        max_out_len=10,
        batch_size=2,
        run_cfg=dict(num_gpus=1),
    )
]

datasets = [
    {"path": "/root/finetune/swift/datasets/train/eval_oc_data.csv", "data_type": "mcq", "infer_method": "gen"},
]