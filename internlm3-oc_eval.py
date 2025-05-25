from mmengine.config import read_base
from opencompass.models import HuggingFaceCausalLM

models = [
    dict(
        type=HuggingFaceCausalLM,
        path='',
        tokenizer_path='',
        tokenizer_kwargs=dict(padding_side='left', truncation_side='left'),
        model_kwargs=dict(device_map='auto'),
        max_seq_len=32768,
        max_out_len=10,
        batch_size=2,
        run_cfg=dict(num_gpus=1),
    )
]

datasets = [
    {"path": "./newformat_sft_test_data.csv", "data_type": "mcq", "infer_method": "gen"},
]