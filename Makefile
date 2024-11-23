.PHONY: all clean build upload

all: clean build upload

clean:
	rm -rf dist/

build:
	python -m build

upload:
	python -m twine upload dist/*
