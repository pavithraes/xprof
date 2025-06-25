"""Tests for the XProf server."""

import os

from etils import epath

from google3.testing.pybase import googletest
from xprof import server


class ServerTest(googletest.TestCase):

  def test_get_abs_gcs_path(self):
    # Arrange
    input_gcs_path = "gs://xprof/"

    # Act
    actual_path = server.get_abs_path(input_gcs_path)

    # Assert
    self.assertEqual(actual_path, input_gcs_path)

  def test_get_abs_path_absolute(self):
    # Arrange
    temp_dir = epath.Path(self.create_tempdir().full_path)
    self.addCleanup(temp_dir.rmtree)
    input_path = temp_dir / "log"
    input_path.mkdir(parents=True)

    # Act
    actual_path = server.get_abs_path(str(input_path))

    # Assert
    self.assertEqual(actual_path, str(input_path))

  def test_get_abs_path_relative(self):
    # Arrange
    base_temp_dir = epath.Path(self.create_tempdir().full_path)
    relative_part = "xprof"
    full_path = base_temp_dir / relative_part
    full_path.mkdir(parents=True, exist_ok=True)

    original_cwd = os.getcwd()
    os.chdir(base_temp_dir)
    self.addCleanup(os.chdir, original_cwd)

    # Act
    actual_path = server.get_abs_path(relative_part)

    # Assert
    self.assertEqual(actual_path, str(full_path.resolve()))

  def test_get_abs_path_home(self):
    # Arrange
    input_path = "~/xprof"

    # Act
    actual_path_str = server.get_abs_path(input_path)
    actual_path = epath.Path(actual_path_str)

    # Assert
    self.assertTrue(actual_path.is_absolute())
    # Check that the path is within the expanded home directory
    self.assertEqual(actual_path.parent, epath.Path("~").expanduser())
    self.assertEqual(actual_path.name, "xprof")


if __name__ == "__main__":
  googletest.main()
