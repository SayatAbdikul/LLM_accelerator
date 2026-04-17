from PIL import Image
import pytest

from images.download_imagenet_class import save_class_images


class TestDownloadImagenetClass:
    @staticmethod
    def _sample(label: int):
        return {
            "label": label,
            "image": Image.new("RGB", (8, 8), color=(label % 255, 0, 0)),
        }

    def test_save_class_images_only_saves_requested_class(self, tmp_path):
        dataset = [
            self._sample(7),
            self._sample(3),
            self._sample(7),
            self._sample(7),
        ]

        count = save_class_images(dataset, class_id=7, num_images=2, out_dir=tmp_path)

        assert count == 2
        saved = sorted(path.name for path in tmp_path.glob("*.jpg"))
        assert saved == ["007_00000.jpg", "007_00001.jpg"]

    def test_save_class_images_raises_when_labels_are_missing(self, tmp_path):
        dataset = [{"image": Image.new("RGB", (8, 8), color=(0, 0, 0))}]

        with pytest.raises(RuntimeError, match="does not expose labels"):
            save_class_images(dataset, class_id=7, num_images=1, out_dir=tmp_path)
