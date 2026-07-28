from pathlib import Path

from scripts.evaluate_test import FROZEN_TEST_ASSETS


def test_e4_frozen_test_asset_is_exact() -> None:
    asset = FROZEN_TEST_ASSETS["E4-ImageWPE-K9"]

    assert asset.experiment_id == "E4-ImageWPE-K9"
    assert asset.checkpoint == Path(
        "06_models/retrain_outputs/maize_model_research/"
        "e4_tsvit_image_nine_point_mexican_hat_k9_seed42/best.pt"
    )
    assert asset.sha256 == "E30E912C7ECEA90FAEFCCCF8444849F7F36A1137084FDDC9F70EC0F1204ABE14"
    assert asset.epoch == 11
    assert asset.validation_maize_iou == 0.9389875395835213
