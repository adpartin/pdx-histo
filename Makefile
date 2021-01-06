# stackoverflow.com/questions/24736146/how-to-use-virtualenv-in-makefile
# https://docs.zhengyuan.sg/snippets/makefile.html#python

# https://stackoverflow.com/questions/60115420/check-for-existing-conda-environment-in-makefile
# http://blog.ianpreston.ca/2020/05/13/conda_envs.html
venv:
	: # virtualenv venv
	: # Create venv if it doesn't exist
	test -d venv || virtualenv venv

requirements: venv
	: # ./venv/bin/python -m pip install -r requirements.txt
	: # Activate venv and install smthing inside
	. venv/bin/activate && pip install -r requirements.txt

clean:
	# rm -rf venv
	# find -iname "*.pyc" -delete
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

build:
	python src/get_meta_from_slides.py
	python src/merge_meta_files.py
	python src/build_df.py

tile:
	python src/tiling.py
