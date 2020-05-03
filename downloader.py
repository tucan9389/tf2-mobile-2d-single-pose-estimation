from __future__ import print_function
from enum import Enum
import os
import datetime
import traceback
import sys
from pathlib import Path
import shutil
import psutil
import subprocess
import time
import re


argv = sys.argv

def log(msg):
  print(str_to_log(msg))

def str_to_log(msg):
  dt = datetime.datetime.now()

  return f"[{dt.strftime('%d.%m.%Y %H:%M:%S')}]: {msg}"

skip_sleep = "--quick" in argv

def sleep(*args):
  """ Args for backwards compatibility
  """
  if skip_sleep:
    return
  time.sleep(1)

debug = "--debug" in argv

disk_info = psutil.disk_usage("/")

recommended_space_in_gb = 36

download_path = "/home/datasets/"

try:
  arg_download_path = [a for a in argv if "--download-path" in a]

  if arg_download_path:
    arg_download_path = arg_download_path[0]
    if "=" in arg_download_path:
      blob = arg_download_path.split("=")
    else:
      raise Exception("Proper format for option is: --download-path=/example/for/your/path")
  
    download_path = blob[1]
except Exception as e:
  if debug:
    traceback.print_exc()
  log("It seems there was a problem with understanding your --download-path option. Please use --debug option to see the error")
  sys.exit(0)

log("Welcome to datasets downloader!")
sleep(1)
log("Checking disk free space...")
sleep(1)

free_space_in_gb = int((disk_info.free / (2**30)))

if free_space_in_gb < recommended_space_in_gb:
  log("*** WARNING ***")
  log("You have less than required free space in your system.")
  log(f"Your free space in system for / path: {free_space_in_gb} GB")
  log("*** WARNING ***")
  r = input(
    str_to_log("If you are sure that this is some kind of mistake and wish to proceed, please enter YES or Y: ")
  )

  if "Y" not in r and "YES" not in r:
    log("Aborting")
    sleep(1)
    sys.exit(0)
else:
  log(f"It seems you have recommended amount of free space for datasets. RECOMMENDED: {recommended_space_in_gb} GB. YOUR SPACE: {free_space_in_gb} GB. Proceeding...")


r = input(
  str_to_log(f"You are about to download huge datasets into your system. One of the archives is 18GB. Are you sure? [Y/n]: ")
)

positive = ["yes", "y"]

proceed = False

for p in positive:
  if p.upper() in r or p.lower() in r:
    proceed = True
    break

if not proceed:
  log("Aborting")
  sys.exit(0)

sleep(1)

downloader_program_name = "wget"

downloader_program_command = "wget URL -P DIR"

try:
  log("Checking program to download datasets... It's path:")
  result_code = subprocess.call(["which", downloader_program_name])
except subprocess.CalledProcessError as e:
  log("There seems to be the problem with checking downloader program installation. Aborting. See more info below.")
  log(f"Shell output: {e.output}")
  sys.exit(0)

if result_code != 0:
  log(f"Unfortunately, {downloader_program_name} not found. Please, install {downloader_program_name} into your system")
  sys.exit(0)


class UnpackingType(Enum):
  Zip = "Zip"
  Tar = "Tar"


class DownloadRequest:
  def __init__(self, name, url, path, check_for_unpacking_folder=False):
    self.name = name
    self.url = url
    self.path = path

    self.filename = self.url.split("/")[-1]
    self.filepath = os.path.join(self.path, self.filename)
    self.check_for_unpacking_folder = check_for_unpacking_folder
 
  @property
  def remote_filesize(self):
    try:
      process = subprocess.Popen(["wget", "--spider", self.url],
                                 text=True,
                                 universal_newlines=True,
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE
                                )

      output = process.stdout.read()

      if not output:
        output = process.stderr.read()

      regex_result = re.search("Length: \d* \((.*)\)\s\[.*\/.*]", output)

      return regex_result.groups(0)[0]
    except Exception as e:
      if debug:
        traceback.print_exc()
      return "ERROR GETTING SIZE"

  def create_unpack_command(self):
    if ".zip" in self.filename:
      return f"unzip -q {self.filepath} -d {self.path}"
    elif "tar.gz" in self.filename:
      return f"tar xf {self.filepath}"
    else:
      raise Exception(f"Could not work out unpacking command for provided type! Filename: {self.filename}")
 
  def create_download_command(self):
    return downloader_program_command.replace(
      "URL",
      self.url
    ).replace(
      "DIR",
      download_path
    ).replace(
      "FIENAME",
      self.filename
    )

  def download(self):
    filepath = self.filepath
    extensionless_filename, extension = os.path.splitext(self.filename)
    directory_path = os.path.join(self.path, extensionless_filename)

    log(f"Dataset archive will be saved at: {filepath}")

    if Path(filepath).exists():
      r = input(str_to_log(f"""Uh-oh.
It seems file OR folder exists with this dataset. Check file archive path:
{filepath}
and folder:
{directory_path}

Please choose how should I proceed by typing NUMBER of preferred option:

1) SKIP - I will move to next dataset (if any) and do nothing with this.
2) DELETE AND REDOWNLOAD - I will remove archive with dataset and folder (if exists)
"""))

      if r == "2":
        try:
          log(f"Going to remove dataset file and directory in 1 second for dataset: {self.name}")

          Path(filepath).unlink()

          directory_obj = Path(directory_path)
        
          if directory_obj.exists() and directory_obj.is_dir():
            log("Found directory for removal. Removing it.")

          log("Success! Now on to download...")
        except Exception as e:
          if debug:
            traceback.print_exc()
          log("*** WARNING ***")
          log("ERROR REMOVING DATASET ARCHIVE AND FOLDER, SKIPPING...")
          log("*** WARNING ***")
      elif r == "1":
        log("Skipping...")
        return
      else:
        log("Uncrecognized option, aborting...")
        sys.exit(0)
    
    try:
      download_command = self.create_download_command()
      subprocess.run(download_command,
                              shell=True,
                              stderr=subprocess.STDOUT)
    except Exception as e:
      if debug:
        traceback.print_exc()

      log("ERROR downloading dataset. Skipping... Please use --debug to show error")
      return

    log("Download success! Going to unpack archive")

    unpack_command = self.create_unpack_command()

    try:
      subprocess.run(unpack_command,
                     shell=True,
                     stderr=subprocess.STDOUT)     
      
      if self.check_for_unpacking_folder:
        if not Path(directory_path).exists():
          raise Exception(
            f"""It seems there is a problem with unpacking!
Folder with inner files not created after unpacking!
There should be folder: {directory_path} after unpacking, but it's not there!
Please check it for yourself"""
          )
    except Exception as e:
      if debug:
        traceback.print_exc()
      log("ERROR unpacking! Skipping... Please use --debug to show error")
      return


download_requests = [
  DownloadRequest("annotations_trainval2017", "http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip", download_path),
#  DownloadRequest("zipTest", "https://github.com/marmelroy/Zip/blob/master/examples/Sample/Sample/Images.zip?raw=true", download_path, check_for_unpacking_folder=False)
  DownloadRequest("val2017", "http://images.cocodataset.org/zips/val2017.zip", download_path),
  DownloadRequest("train2017", "http://images.cocodataset.org/zips/train2017.zip", download_path),
]

log("Checking path for downloading...")

if not Path(download_path).exists():
  log(f"Path {download_path} does not exist! Creating it...")

  try:
    Path(download_path).mkdir(parents=True)
  except Exception as e:
    if debug:
      traceback.print_exc()
    log("Critical error creating path! Aborting. (more info with --debug option)")
    sys.exit(0)
else:
  log("Path seems to bi ok!")

log("Going to download needed datasets!")
sleep(1)

for dr in download_requests:
  log("")
  log("")
  log(f"Dataset [{dr.name}] with size: {dr.remote_filesize}")
  log("*** DOWNLOADING ***")
  
  dr.download()
  
log("*** SUCCESS ***")
log("All datasets downloads complete!")
log("*** SUCCESS ***")
sys.exit(0)
