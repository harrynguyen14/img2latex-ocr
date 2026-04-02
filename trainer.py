import torch
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from transformers import Trainer, TrainerCallback, TrainingArguments, EvalPrediction

from encode import IMAGE_HEIGHT, PATCH_SIZE
from evaluate import compute_metrics as _compute_metrics


# ---------------------------------------------------------------------------
# Data collator
# ---------------------------------------------------------------------------

class LaTeXDataCollator:
    """
    Nhận list các dict từ LaTeXDataset (mỗi item có pixel_values tensor (3,H,W)).
    Pad width về cùng kích thước và tạo patch_mask.
    """

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        pixel_list = [item["pixel_values"] for item in batch]

        max_w = max(t.shape[-1] for t in pixel_list)
        padded, masks = [], []
        for t in pixel_list:
            w = t.shape[-1]
            pad_w = max_w - w
            t_pad = torch.nn.functional.pad(t, (0, pad_w), value=1.0)
            padded.append(t_pad)
            ph = IMAGE_HEIGHT // PATCH_SIZE
            pw_valid = w // PATCH_SIZE
            pw_total = max_w // PATCH_SIZE
            mask = torch.zeros(ph, pw_total, dtype=torch.bool)
            mask[:, :pw_valid] = True
            masks.append(mask.reshape(-1))

        pixel_values = torch.stack(padded)
        patch_mask = torch.stack(masks)

        input_ids = torch.stack([item["input_ids"] for item in batch])
        attention_mask = torch.stack([item["attention_mask"] for item in batch])
        labels = torch.stack([item["labels"] for item in batch])
        return {
            "pixel_values": pixel_values,
            "patch_mask": patch_mask,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


# ---------------------------------------------------------------------------
# compute_metrics adapter cho Trainer
# ---------------------------------------------------------------------------

def make_compute_metrics(tokenizer):
    """
    Trả về hàm compute_metrics tương thích với Trainer.
    predictions và label_ids là token id tensors/arrays.
    """
    def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
        preds, refs = eval_pred.predictions, eval_pred.label_ids

        if isinstance(preds, np.ndarray):
            pred_ids = preds
        else:
            pred_ids = np.array(preds)

        if isinstance(refs, np.ndarray):
            ref_ids = refs
        else:
            ref_ids = np.array(refs)

        # Replace -100 padding trước khi decode
        ref_ids = np.where(ref_ids == -100, tokenizer.pad_token_id, ref_ids)

        pred_strs = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        ref_strs  = tokenizer.batch_decode(ref_ids,  skip_special_tokens=True)

        return _compute_metrics(pred_strs, ref_strs)

    return compute_metrics


# ---------------------------------------------------------------------------
# LaTeXOCRTrainer
# ---------------------------------------------------------------------------

class LaTeXOCRTrainer(Trainer):
    """
    Override compute_loss và prediction_step để:
    - Truyền pixel_values và patch_mask vào forward()
    - Dùng model.generate() khi eval thay vì greedy forward
    """

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(
            pixel_values=inputs["pixel_values"],
            patch_mask=inputs.get("patch_mask"),
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"],
        )
        return (outputs.loss, outputs) if return_outputs else outputs.loss

    def prediction_step(
        self,
        model,
        inputs: Dict[str, Any],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:

        model.eval()
        with torch.no_grad():
            loss_outputs = model(
                pixel_values=inputs["pixel_values"],
                patch_mask=inputs.get("patch_mask"),
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=inputs["labels"],
            )
            loss = loss_outputs.loss.detach()

            if prediction_loss_only:
                return (loss, None, None)

            # Generate predictions (list of strings)
            generated_strs = model.generate(
                pixel_values=inputs["pixel_values"],
                patch_mask=inputs.get("patch_mask"),
            )

            tokenizer = model.tokenizer
            # Encode strings → ids để Trainer truyền vào compute_metrics
            pred_enc = tokenizer(
                generated_strs,
                padding=True,
                truncation=True,
                max_length=model.config.max_new_tokens,
                return_tensors="pt",
            )
            pred_ids = pred_enc.input_ids.cpu()

            # label_ids: -100 padding giữ nguyên để compute_metrics decode đúng
            label_ids = inputs["labels"].detach().cpu()

        return (loss, pred_ids, label_ids)
