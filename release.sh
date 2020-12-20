pytest || exit 1
pip3 install twine
rm -rf dist
python3 setup.py sdist bdist_wheel
twine upload dist/*