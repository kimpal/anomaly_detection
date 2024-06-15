# just a small main.py to run the requirement file and welcome you
import subprocess
import sys
import time
import os

chose = ""


def install_requerment():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])


def print_hi(name):
    print(f'Hi, {name}')


def main():
    print_hi('Welcome to the Anomaly-detection application\n')
    print('wil you install all the required packages ')
    print('Yes or NO chose whit Y, y or N:')
    choice = input()
    print('Chose' + choice)
    if choice == 'Y' or choice == 'y':
        print('Now the requirement is getting installed')
        install_requerment()
        time.sleep(1)
        print('Success! the requirement install is Finish\n')
        print('next step is ot run the pytho file..')
        print("form the pre-processing-file")
    else:
        print("Okay, I can't install all the required packages since you did not accepts")


if __name__ == '__main__':
    main()
