from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS
import re

@ICL_EVALUATORS.register_module()
class PaperCLSEvaluator(BaseEvaluator):
    """Exact match evaluator for paper classification."""

    def __init__(self) -> None:
        super().__init__()

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different '
                'length'
            }

        cnt = 0
        details = []
        pattern = re.compile(
            r"<category>\s*(.*?)\s*</category>", 
            re.IGNORECASE | re.DOTALL  # 忽略大小写、允许跨行
        )

        for pred, ans in zip(predictions, references):
            # answers = list(set(ans + origin_ans))
            match = pattern.search(pred)
            if match:
                pred = match.group(1)
            else:
                pred = 'unknown'
            detail = {'pred': pred, 'answer': ans}
            if ans in pred:
                cnt += 1
                detail['correct'] = True
            else:
                detail['correct'] = False
            details.append(detail)

        score = cnt / len(predictions) * 100

        return {'score': score, 'details': details}