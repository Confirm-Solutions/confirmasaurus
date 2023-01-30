# Figures

## Making reproducible figures

1. Prepare the data so that it is 100% ready for plotting.
2. Push that data to S3. (TODO: standardize a location, add a folder?)
3. Create a notebook that automatically loads that data from S3 and makes a figure. Ideally this notebook only uses standard tools like pandas/numpy/matplotlib so we don’t need to worry about installing old versions of imprint.
4. Save the resulting PDF in the repo adjacent to the notebook.

## Making pretty figures

The presets from `imprint.nb_util.setup_nb()` are useful for making default-pretty figures.