import io
import unittest
from unittest.mock import patch
from PIL import Image

from valor_a_la_vista.gallery import add_photos, inspection_inputs


def image(color):
    out = io.BytesIO()
    Image.new("RGB", (32, 32), color).save(out, "PNG")
    return out.getvalue()


class GalleryTests(unittest.TestCase):
    def test_incremental_add_preserves_previous(self):
        first, _ = add_photos({}, [("a.png", image("red"))])
        second, notes = add_photos(first, [("b.png", image("blue"))])
        self.assertEqual((len(first), len(second), notes), (1, 2, []))

    def test_exact_duplicates_not_added_under_other_names(self):
        raw = image("red")
        gallery, notes = add_photos({}, [("a.png", raw), ("otra.png", raw)])
        self.assertEqual(len(gallery), 1)
        self.assertIn("ya está añadida", notes[0])

    def test_same_filename_different_content_is_allowed(self):
        gallery, _ = add_photos({}, [("a.png", image("red")), ("a.png", image("blue"))])
        photos, categories = inspection_inputs(gallery)
        self.assertEqual(len(photos), 2)
        self.assertEqual(set(categories.values()), {"Detalle adicional"})

    def test_invalid_image_rejected_without_losing_previous(self):
        gallery, _ = add_photos({}, [("a.png", image("red"))])
        second, notes = add_photos(gallery, [("bad.png", b"corrupt")])
        self.assertEqual(second, gallery)
        self.assertTrue(notes)

    def test_sixteen_photo_limit(self):
        uploads = [(str(i), image((i, 10, 20))) for i in range(17)]
        gallery, notes = add_photos({}, uploads)
        self.assertEqual(len(gallery), 16)
        self.assertIn("16", notes[0])

    def test_total_size_limit(self):
        with patch("valor_a_la_vista.gallery.MAX_TOTAL_PHOTO_BYTES", 1):
            gallery, notes = add_photos({}, [("a.png", image("red"))])
        self.assertFalse(gallery)
        self.assertTrue(notes)

    def test_per_photo_limit(self):
        gallery, notes = add_photos({}, [("grande.png", b"x" * (12 * 1024 * 1024 + 1))])
        self.assertFalse(gallery)
        self.assertIn("12 MB", notes[0])
