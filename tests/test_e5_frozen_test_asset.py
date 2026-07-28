from pathlib import Path

from scripts.evaluate_test import FROZEN_TEST_ASSETS


def test_e5_frozen_test_asset_is_exact() -> None:
    asset = FROZEN_TEST_ASSETS["E5-DOY-K9WPE"]

    assert asset.experiment_id == "E5-DOY-K9WPE"
    assert asset.checkpoint == Path(
        "06_models/retrain_outputs/maize_model_research/"
        "e5_tsvit_doy_nine_point_mexican_hat_k9_seed42/best.pt"
    )
    assert asset.sha256 == "1648958D1563941EEE427B6222FCB4ACB283E6E2A84C76B83E1616B8C2181561"
    assert asset.epoch == 14
    assert asset.validation_maize_iou == 0.934533052755867
