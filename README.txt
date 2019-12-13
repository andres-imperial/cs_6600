=====================================================================
Andres Imperial
CS_6000
Project_2
=====================================================================

Unzip project's structure need to be unchanged for program to run.

I used linux ubuntu 16.04 with python2.7 tensorflow-gpu == 1.9 and cuda 9.0 to
train and test. I also was able to run on MX Linux with
tensorflow == 1.9 and utilizing the cpu.

Tensorflow can be installed using:
  pip install --user tensorflow==1.9

MISC Libs maybe needed:
  pip install --user matplotlib
  pip install --user pillow
  sudo apt-get install python-tk

To run network trained on Waldo images that are were not blurred use
the following command:
  python findWaldoNoBlur.py ./images/<your choice>.jpg

To run network trained on Waldo images that are were not blurred along
with images that were blurred use the following command:
  python findWaldoBlurred.py ./images/<your choice>.jpg
