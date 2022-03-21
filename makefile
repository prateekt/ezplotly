build:
	rm -rf dist
	rm -rf build
	rm -rf ezplotly.egg*
	python setup.py sdist bdist_wheel

deploy:
	twine upload dist/*

run_notebook_tests:
	jupyter nbconvert --to notebook --execute EZPlotlyExamples.ipynb
	jupyter nbconvert --to notebook --execute EZPlotlyBioExamples.ipynb

clean:
	rm -rf *.nbconvert*
	rm -rf test_figs