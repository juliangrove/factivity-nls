# factivity-nls

This repository contains the data and models associated with the submission *Factivity, presupposition projection, and the role of discrete knowledge in gradient inference judgments*.
We also include the data from [Degen and Tonhauser 2021](https://direct.mit.edu/opmi/article/doi/10.1162/opmi_a_00042/106927/Prior-Beliefs-Modulate-Projection), described in the paper, as a submodule.

All models reported in the paper were run on a Lenovo ThinkPad X1 Carbon Gen 12 with 64GB of memory and a 64GB swap partition.
They require about 128GB of hard drive space altogether.

## Installation

Two R packages are required:
 - [`cmdstanr`](https://mc-stan.org/cmdstanr/reference/cmdstanr-package.html) (to fit the models)
 - [`loo`](https://cran.r-project.org/web/packages/loo/index.html) (to analyze the results)
 
If you use [Nix](https://nixos.org/), you can install R and the relevant packages by running `nix-shell` in this directory.

## The modeling pipeline

All scripts should be run from this directory.
To run the variants of the norming and factivity models which use ordered beta likelihoods instead of truncated normal likelihoods, follow the instructions below, replacing any occurrences of 'truncation' with 'inflation' throughout.
E.g.,

	Rscript fits/norming/truncation/norming-models.r

becomes

	Rscript fits/norming/inflation/norming-models.r

### The norming models

To fit the norming models and extract the posteriors of the norming-gradient model, run:

	Rscript fits/norming/truncation/norming-models.r

You can check and compare the ELPDs of the norming-gradient and norming-discrete models in R:
	
	library(loo);
	model_dir <- "fits/norming/truncation/results/";
	model_names <- c("norming-gradient","norming-discrete");
	model_waic <- list();
	for (n in model_names) {
		model_path <- paste0(model_dir,n,".rds");
		model_fit <- readRDS(model_path);
		model_waic <- c(model_waic,list(waic(model_fit$draws("ll"))));
	}
	loo_compare(model_waic);

### The factivity models

To fit the four factivity models and extract their posteriors, run:

	Rscript fits/factivity/truncation/factivity-models.r

You can check and compare the ELPDs of the four models in R:
	
	library(loo);
	model_dir <- "fits/factivity/truncation/results/";
	model_names <- c("discrete-factivity","wholly-gradient","discrete-world","wholly-discrete");
	model_waic <- list();
	for (n in model_names) {
		model_path <- paste0(model_dir,n,".rds");
		model_fit <- readRDS(model_path);
		model_waic <- c(model_waic,list(waic(model_fit$draws("ll"))));
	}
	loo_compare(model_waic);

### Modeling the replication data

To fit the four models of the replication data, run:

	Rscript fits/evaluation/replication/replication-models.r

You can check and compare the ELPDs of the four model evaluations in R as above, but instead use:

	model_dir <- "fits/evaluation/replication/results/";

### Modeling the bleached/templatic data

Make sure to un-comment line 6 or 7 in `non-contentful-models.r`, depending on whether you want to use the bleached or templatic data.
To fit the four models of the data, run:

	Rscript fits/evaluation/non-contentful/non-contentful-models.r

You can check and compare the ELPDs of the four model evaluations on the bleached/templatic data in R as above.
To define the relevant directory, use the following, (un-)commenting as appropriate:

	which <- "bleached";
	# which <- "templatic";
	model_dir <- paste0("fits/evaluation/non-contentful/results-",which,"/");
