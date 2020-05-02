from __future__ import print_function
from enum import Enum
import os
import sys
import pathlib
import shutil
import psutil
import subprocess
import time
import re

disk_info = psutil.disk_usage("/")

recommended_space_in_gb = 26 

download_path = "/home/datasets"

print("Welcome to datasets downloader!")
time.sleep(1)
print("Checking disk free space...")
time.sleep(1)

free_space_in_gb = int((disk_info.free / (2**30)))

if free_space_in_gb < recommended_space_in_gb:
  print("*** WARNING ***")
  print("You have less than required free space in your system.")
  print(f"Your free space in system for / path: {free_space_in_gb}")
  print("*** WARNING ***")
  r = input("If you are sure that this is some kind of mistake and wish to proceed, please enter YES or Y: ")

  if "Y" not in r and "YES" not in r:
    print("Aborting")
    time.sleep(1)
    sys.exit(0)


r = input(
  f"You are about to download huge datasets into your system. One of the archives is 18GB. Are you sure? [Y/n]: "
)

positive = ["yes", "y"]

proceed = False

for p in positive:
  if p.upper() in r or p.lower() in r:
    proceed = True
    break

if not proceed:
  print("Aborting")
  time.sleep(1)
  sys.exit(0)

time.sleep(1)

downloader_program_name = "wget"

try:
  print("Checking program to download datasets... It's path:")
  result_code = subprocess.call(["which", downloader_program_name])
except subprocess.CalledProcessError as e:
  print("There seems to be the problem with checking downloader program installation. Aborting. See more info below.")
  print(f"Shell output: {e.output}")
  sys.exit(0)

if result_code != 0:
  print(f"Unfortunately, {downloader_program_name} not found. Please, install {downloader_program_name} into your system")
  sys.exit(0)


class UnpackingType(Enum):
  Zip = "Zip"
  Tar = "Tar"


class DownloadRequest:
  def __init__(self, name, url, path):
    self.name = name
    self.url = url
    self.path = path

    self.filename = self.url.split("/")[:-1][0]

  def get_unpack_command(self):
    if ".zip" in self.filename
      return "unzip"
    elif "tar.gz" in self.filename:
      return "tar xf"
    else:
      raise Exception(f"Could not work out unpacking command for provided type! Filename: {self.filename")

  def get_url_file_size(self):

    try:
      output = subprocess.check_output(["wget", "--spider", self.url])

      regex_result = re.search("\((.*)\)\s\[.*\/.*]", output.decode("utf-8"))

      return regex_result.group()
    except Exception:
      return "ERROR GETTING SIZE"

  def download(self):
    filepath = os.path.join(self.path, self.filename)
    extensionless_filename, extension = os.path.splitext(self.filename)
    directory_path = os.path.join(self.path, extensionless_filename)

    if Path(filepath).exists():
      r = input(f"""Uh-oh.
It seems file OR folder exists with this dataset. Check file archive path:
{filepath}
and folder:
{directory_path}

Please choose how should I proceed by typing NUMBER of preferred option:

1) SKIP - I will move to next dataset (if any) and do nothing with this.
2) DELETE AND REDOWNLOAD - I will remove archive with dataset and folder (if exists)
""")

      if r == 2:
        try:
          print(f"Going to remove dataset file and directory in 1 second for dataset: {self.name}")

          Path(filepath).unlink()

          directory_obj = Path(directory_path)
        
          if directory_obj.exists() and directory_obj.is_dir():
            print("Found directory for removal. Removing it.")
        except Exception as e:
          print(f"Error info: {e}")
          print("*** WARNING ***")
          print("ERROR REMOVING DATASET ARCHIVE AND FOLDER, SKIPPING...")
          print("*** WARNING ***")


download_requests = [
  DownloadRequest("annotations_trainval2017", "http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip", download_path),
  DownloadRequest("val2017", "http://images.cocodataset.org/zips/val2017.zip", download_path),
  DownloadRequest("train2017", "http://images.cocodataset.org/zips/train2017.zip", download_path),
]



