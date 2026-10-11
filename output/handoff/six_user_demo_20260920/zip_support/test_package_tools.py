import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SPEC = importlib.util.spec_from_file_location(
    "package_tools", Path(__file__).with_name("package_tools.py")
)
TOOLS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOLS)


class PackageToolsTest(unittest.TestCase):
    def test_relocation_preserves_nonpath_values(self):
        with tempfile.TemporaryDirectory() as folder:
            project = Path(folder) / "project"
            before = {
                "audio_path": "H:\\H-CODE\\speechtokenizer\\data\\VCTK\\p225.flac",
                "codes_path": "D:/tmp/speechtokenizer-exp23/tmp/exp23_formal/cache/a.npz",
                "label": "p225",
                "temperature": 1.0188,
                "hash": "abc123",
            }
            after = TOOLS.rewrite_paths(before, project)
            self.assertEqual(after["audio_path"], str((project / "data/VCTK/p225.flac").resolve()))
            self.assertEqual(after["codes_path"], str((project / "tmp/exp23_formal/cache/a.npz").resolve()))
            self.assertEqual(after["temperature"], before["temperature"])
            self.assertEqual(after["hash"], before["hash"])
            self.assertEqual(before["audio_path"], "H:\\H-CODE\\speechtokenizer\\data\\VCTK\\p225.flac")

    def test_archive_members_reject_traversal(self):
        with tempfile.TemporaryDirectory() as folder:
            for name in ("../escape", "/absolute", "C:/escape", "x/../../escape"):
                with self.assertRaises(ValueError):
                    TOOLS.safe_member(Path(folder), name)

    def test_prepare_writes_copies_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            project = root / "project"
            formal = project / "tmp/exp23_formal/cache_lca_l3/cache_manifest"
            formal.mkdir(parents=True)
            audio = project / "data/VCTK/a.flac"
            audio.parent.mkdir(parents=True)
            audio.write_bytes(b"audio")
            row = {"audio_path": "H:/H-CODE/speechtokenizer/data/VCTK/a.flac", "speaker_id": "p225"}
            records = formal / "cache_records.json"
            records.write_text(json.dumps([row]), encoding="utf-8")
            (formal / "split_manifest.json").write_text(json.dumps({"splits": {"train": [row]}}), encoding="utf-8")
            (root / "handoff").mkdir()
            (root / "handoff/text_controlled_109_split.generated.json").write_text(json.dumps([row]), encoding="utf-8")
            original = records.read_bytes()
            TOOLS.prepare_paths(root)
            self.assertEqual(records.read_bytes(), original)
            output = project / "tmp/exp23_portable/cache_records.json"
            self.assertEqual(json.loads(output.read_text(encoding="utf-8"))[0]["audio_path"], str(audio.resolve()))
            with self.assertRaises(FileExistsError):
                TOOLS.prepare_paths(root)


if __name__ == "__main__":
    unittest.main()
