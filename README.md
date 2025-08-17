# How to run prediction

## Step 1: Prepare virtual environment
The exact steps may look different depending on your python installation.

### Create virtual environment:
   Ex. python -m venv venv

### Activate environment
   Ex. ./venv/Scripts/activate

### Install dependencies
   Ex. pip install -r requirements.txt

## Step 2: Use predict.py
There are several prediction options.


### Input interactively in terminal:
`python .\predict.py`

### Non-JSON plain output:
`python .\predict.py --x 12.5 --yz 15 --v 3000`

### JSON output:
`python .\predict.py --x 12.5 --yz 15 --v 3000 --json`