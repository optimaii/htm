import os
os.system("git clone https://github.com/htm-community/htm.core")
os.system("cd htm.core && python setup.py install --user --force")
os.system("cd htm.core && python setup.py test")
os.system("pip install -i https://test.pypi.org/simple/ htm.core[examples]")
