import os
from pathlib import Path
import sys
import subprocess
import tempfile
import types
import unittest
from unittest.mock import patch
import zipfile
from tools.remote import reu_release as release
from tools.remote import reu_remote_job as job


class RemoteWorkflowTests(unittest.TestCase):
    def fixture(self, root):
        (root / "src").mkdir()
        (root / "src/model.py").write_text("value=1\n")
        (root / "main.py").write_text("print(1)\n")
        (root / ".env").write_text("FAKE_TEST_SECRET=do-not-bundle\n")
        (root / "outputs").mkdir()
        (root / "outputs/result.json").write_text("{}")
        (root / "AGENTS.md").write_text("local agent instructions")
        (root / "skills").mkdir()
        (root / "skills/SKILL.md").write_text("local skill")
        (root / "tools/remote").mkdir(parents=True)
        (root / "tools/remote/reu_ssh.py").write_text("local credential client")

    def test_bundle_roundtrip_excludes_private_and_generated_files(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "repo"
            root.mkdir()
            self.fixture(root)
            z = Path(d) / "release.zip"
            expected = release.build(root, z)
            target = Path(d) / "target"
            target.mkdir()
            with zipfile.ZipFile(z) as archive:
                self.assertNotIn("AGENTS.md", archive.namelist())
                self.assertNotIn("skills/SKILL.md", archive.namelist())
                self.assertNotIn("tools/remote/reu_ssh.py", archive.namelist())
                self.assertNotIn(".env", archive.namelist())
                self.assertNotIn("outputs/result.json", archive.namelist())
                archive.extractall(target)
            release.verify(target, target / "release-manifest.json", expected)
            (target / "src/model.py").write_text("changed")
            with self.assertRaises(ValueError):
                release.verify(target, target / "release-manifest.json", expected)

    def test_extra_source_and_wrong_manifest_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.fixture(root)
            z = root / "bundle.zip"
            sha = release.build(root, z)
            with zipfile.ZipFile(z) as archive:
                (root / "release-manifest.json").write_bytes(archive.read("release-manifest.json"))
            with self.assertRaises(ValueError):
                release.verify(root, root / "release-manifest.json", "wrong")
            (root / "src/extra.py").write_text("")
            with self.assertRaises(ValueError):
                release.verify(root, root / "release-manifest.json", sha)

    def test_no_overwrite_bundle(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.fixture(root)
            output = root / "release.zip"
            release.build(root, output)
            with self.assertRaises(FileExistsError):
                release.build(root, output)

    def test_verified_release_pin_removes_repeated_hash_argument(self):
        with tempfile.TemporaryDirectory() as d:
            source = Path(d) / "source"
            source.mkdir()
            self.fixture(source)
            bundle = Path(d) / "bundle.zip"
            sha = release.build(source, bundle)
            target = Path(d) / "releases" / "v1"
            target.mkdir(parents=True)
            with zipfile.ZipFile(bundle) as archive:
                archive.extractall(target)
            with self.assertRaises(FileNotFoundError):
                release.approved_hash(target)
            pin = release.pin_verified_release(target, sha)
            self.assertEqual(pin.parent, target.parent / ".approved-releases")
            self.assertEqual(release.approved_hash(target), sha)
            self.assertEqual(release.approved_hash(target, sha), sha)
            self.assertEqual(release.pin_verified_release(target, sha), pin)
            with self.assertRaises(ValueError):
                release.approved_hash(target, "0" * 64)
            (target / "src/model.py").write_text("changed")
            with self.assertRaises(ValueError):
                release.verify(
                    target, target / "release-manifest.json", release.approved_hash(target)
                )

    def test_pin_cannot_be_created_for_wrong_manifest(self):
        with tempfile.TemporaryDirectory() as d:
            source = Path(d) / "source"
            source.mkdir()
            self.fixture(source)
            bundle = Path(d) / "bundle.zip"
            release.build(source, bundle)
            target = Path(d) / "releases" / "v1"
            target.mkdir(parents=True)
            with zipfile.ZipFile(bundle) as archive:
                archive.extractall(target)
            with self.assertRaises(ValueError):
                release.pin_verified_release(target, "0" * 64)
            self.assertFalse(release.approval_path(target).exists())

    def test_path_boundary(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.assertEqual(job.within_home(root / "runs/new", root), root / "runs/new")
            with self.assertRaises(ValueError):
                job.within_home(root / "../outside", root)

    def test_lock_success_keeps_descriptor_open(self):
        fake = types.SimpleNamespace(LOCK_EX=2, LOCK_NB=4, flock=lambda fd, op: None)
        with tempfile.TemporaryDirectory() as d, patch.dict(sys.modules, fcntl=fake), patch.object(
            os, "O_NOFOLLOW", 0, create=True
        ), patch.object(os, "open", return_value=81), patch.object(os, "close") as close:
            self.assertEqual(job.acquire_lock(Path(d) / "lock"), 81)
            close.assert_not_called()

    def test_lock_contention_closes_descriptor(self):
        def busy(fd, op):
            raise BlockingIOError()

        fake = types.SimpleNamespace(LOCK_EX=2, LOCK_NB=4, flock=busy)
        with tempfile.TemporaryDirectory() as d, patch.dict(sys.modules, fcntl=fake), patch.object(
            os, "O_NOFOLLOW", 0, create=True
        ), patch.object(os, "open", return_value=81), patch.object(os, "close") as close:
            with self.assertRaises(BlockingIOError):
                job.acquire_lock(Path(d) / "lock")
            close.assert_called_once_with(81)


class RemoteSessionTests(unittest.TestCase):
    def test_cross_domain_workflow_reaches_session_guard(self):
        argv = [
            "reu_remote_job.py",
            "--record-dir",
            "unused",
            "--gpu",
            "1",
            "--",
            "cross-domain",
            "--stage",
            "pilot",
            "--prepare-cache",
        ]
        with patch.object(sys, "argv", argv), patch.object(sys, "platform", "linux"), patch.object(
            job, "require_tmux_session", side_effect=RuntimeError("session guard reached")
        ):
            with self.assertRaisesRegex(RuntimeError, "session guard reached"):
                job.main()

    def test_required_session_accepted(self):
        with patch.dict(os.environ, {"TMUX": "socket", "TMUX_PANE": "%7"}), patch.object(
            subprocess, "check_output", return_value="sugar_run\n"
        ) as run:
            self.assertEqual(job.require_tmux_session(), "sugar_run")
            self.assertEqual(
                run.call_args[0][0],
                ["tmux", "display-message", "-p", "-t", "%7", "#{session_name}"],
            )

    def test_wrong_session_rejected(self):
        with patch.dict(os.environ, {"TMUX": "socket", "TMUX_PANE": "%7"}), patch.object(
            subprocess, "check_output", return_value="other\n"
        ):
            with self.assertRaises(RuntimeError):
                job.require_tmux_session()

    def test_missing_session_rejected(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError):
                job.require_tmux_session()


if __name__ == "__main__":
    unittest.main()
