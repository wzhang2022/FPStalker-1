This repo is a fork of the code for FPStalker's paper and contains the extensions done by William Zhang, Eryk Pecyna, and Barron Wei for their CS263 final project.


# Create virtual environment and install dependencies
Run the command below to create a virtual environment.
```ruby
virtualenv --python=/usr/bin/python3 myvenv
```

Then activate the virtual environment.
```ruby
. myvenv/bin/activate
```

Finally, install the dependencies.
```ruby
pip install -r requirements.txt
```

# Database
Create a database that will contain the table that stores the fingerprints.
Then, you have two solutions:
- Run the command below to generate a sql file tableFingerprints.sql with few fingerprints. It contains 15k fingerprints in this table that were randomly sampled from the first half of the raw dataset, i.e. with no filter.
The reason we split the table in two files is to overcome the Github storage limit.
```ruby
tar zxvf extension1.txt.tar.gz; tar zxvf extension2.txt.tar.gz; cat extension1.txt extension2.txt > tableFingerprints.sql
```
- Import extensionDataScheme.sql that contains only the scheme of the table to stores the fingerprints.

Change the connection to the database at the top of the main with your credentials.

We used MySQL community edition with a local server set up. After adding a new schema and table we ran the sql files generated above.
There are multiple connections and table names to edit in algo.py and main.py so for greater ease of use, use the following specifications.

host="127.0.0.1", port=3306, user="stalker", passwd="baddy", db="canvas_fp_project"

When setting up the SQL server create an administrative user named "stalker" with password "baddy"
Make sure the schema name is "canvas_fp_project" and the table name is "extensiondatascheme"

Using these specifications should ensure compatibility with the current settings defined in main.py and algo.py.

# Get ids of browser instances with countermeasures
```ruby
python main.py getids
```

It generates a file called "consistent_extension_ids.csv" in data folder.

# Launch evaluation process of a linking algorithm

The multiple extensions we added made it easier to move the argument passing into main so we could save some copies of the code to run specific experiments.

main() takes argv as its first argument. For evaluating a linking algorithm you must put the following call into the end of main.py:

main(["auto", experimentname, "hybridalgo", "6"], lambda_threshold=0.994, diff=0.10, model_type="randomforest",
         model_path="./saved_models/my_ml_model", train_round_2=False, load=True)

where experiment is an arbitrary experiment name. model_path must point to a saved model, the following are the model paths for the models used in our paper:

Our neural network:
"./saved_models/nn100x100dp5"

Original random forest:
"./saved_models/my_ml_model"

model_type must equal either "randomforest" or "neuralnet"

Lambda threshold and diff are currently set to the parameters defined in FPStalker, for our model they must be (lambda_threshold = 0.9) and (diff=0).

# Benchmark

Running the evaluation process will print the average time and percentiles for how long it took the model to link two fingerprints.