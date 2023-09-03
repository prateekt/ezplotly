conda_dev:
	conda env remove -n ezplotly_env
	conda env create -f conda.yaml

build:
	rm -rf dist
	hatch build

publish:
	hatch publish

run_notebook_tests:
	jupyter nbconvert --to notebook --execute EZPlotlyExamples.ipynb
	jupyter nbconvert --to notebook --execute EZPlotlyBioExamples.ipynb

clean:
	rm -rf test_figs
	rm -rf dist
	rm -rf .ipynb_checkpoints
	rm *.nbconvert.ipynb