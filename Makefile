# Refs:
# https://stackoverflow.com/questions/24736146/how-to-use-virtualenv-in-makefile
# https://docs.zhengyuan.sg/snippets/makefile.html#python
# https://stackoverflow.com/questions/60115420/check-for-existing-conda-environment-in-makefile
# http://blog.ianpreston.ca/2020/05/13/conda_envs.html

# Production env (TF 2.4 — paper-canonical; see PAPER.md)
venv_tf24:
	test -d venv_tf24 || virtualenv venv_tf24
	. venv_tf24/bin/activate && pip install -r requirements_tf24.txt

# Development env (adds flake8, ipython, ipdb, jupyterlab, lsp servers)
dev-venv_tf24: venv_tf24
	. venv_tf24/bin/activate && pip install -r requirements-dev_tf24.txt

clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
