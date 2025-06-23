"""
Tool for metrics calculation through data and label (string and string).
 * Calculation from Optical Character Recognition (OCR) metrics with editdistance.
"""

import string
import unicodedata
import editdistance
import numpy as np


def ocr_metrics(predicts, ground_truth, norm_accentuation=False, norm_punctuation=False):
    """Calculate Character Error Rate (CER), Word Error Rate (WER), Sequence Error Rate (SER)
    and their corresponding Accuracies (Char Acc, Word Acc, Seq Acc)"""

    if len(predicts) == 0 or len(ground_truth) == 0:
        # Return 1 for error rates, 0 for accuracies (no correct predictions)
        return (1, 1, 1, 0, 0, 0)

    cer, wer, ser = [], [], []
    char_acc, word_acc, seq_acc = [], [], []

    for (pd, gt) in zip(predicts, ground_truth):
        if norm_accentuation:
            pd = unicodedata.normalize("NFKD", pd).encode("ASCII", "ignore").decode("ASCII")
            gt = unicodedata.normalize("NFKD", gt).encode("ASCII", "ignore").decode("ASCII")

        if norm_punctuation:
            pd = pd.translate(str.maketrans("", "", string.punctuation))
            gt = gt.translate(str.maketrans("", "", string.punctuation))

        # Character Error Rate & Accuracy
        pd_cer, gt_cer = list(pd), list(gt)
        dist_cer = editdistance.eval(pd_cer, gt_cer)
        max_cer = max(len(pd_cer), len(gt_cer))
        cer.append(dist_cer / max_cer if max_cer > 0 else 0)
        char_acc.append(1.0 - (dist_cer / max_cer) if max_cer > 0 else 1.0)

        # Word Error Rate & Accuracy
        pd_wer, gt_wer = pd.split(), gt.split()
        dist_wer = editdistance.eval(pd_wer, gt_wer)
        max_wer = max(len(pd_wer), len(gt_wer))
        wer.append(dist_wer / max_wer if max_wer > 0 else 0)
        word_acc.append(1.0 - (dist_wer / max_wer) if max_wer > 0 else 1.0)

        # Sequence Error Rate & Accuracy
        pd_ser, gt_ser = [pd], [gt]
        dist_ser = editdistance.eval(pd_ser, gt_ser)
        max_ser = max(len(pd_ser), len(gt_ser))
        ser.append(dist_ser / max_ser if max_ser > 0 else 0)
        seq_acc.append(1.0 if pd == gt else 0.0)

    metrics = [cer, wer, ser, char_acc, word_acc, seq_acc]
    metrics = np.mean(metrics, axis=1)

    # Return: CER, WER, SER, Char Acc, Word Acc, Seq Acc
    return tuple(metrics)
