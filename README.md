# Description

This is the first in a series of challenge projects where I work 
from a task list provided by GPT and translate each task 
into programming logic. This differs from previous "vibe-coding" 
methods where code is hand-typed line by line from GPT scripts, 
and instead I now only use my notes and published GitHub 
projects as references to execute tasks. 

Since I am still moving slow through the procedures, this will also be
exercise in updating commits to GitHub on a singular project, compared
to previous repos where I waited until project completion.

# Reflections

Schema Validation: I began to get a grasp of the purpose of schema
validation today as I smoke tested the various validation mechanisms 
(`smoke_tests.py`). 
I came to understand today that validation is highly important because data
formatting and schema-change issues are a major source of training trouble 
with machine learning models. This helped me to understand why I needed to
pass my data source and defined constants through validation functions for 
comparison. This understanding helped me to identify the necessary data 
structures and logic needed to pass through my smoke tests. I also further
internalized how to cascade errors up through a logic chain in `schema.py`.

Feature Building: Today's task was to build the features for the training pipeline. An interesting
cognitive process observation I noticed is that my mind attempts to make things much more complicated
than they actually are. Because I am working from previous project references, my mind sees 
all of the complex logic used in those projects, and thinks it also applies to the task list for 
this self-built project as well. But, this forces me to truly understand the code I am referencing
so I can be sure I am building the correct logic into my project. Today, I got a better grasp of 
how the `@dataclass` decorator helps to package data from multiple functions into a single object
for use at runtime. I also further internalized the utility of creating local variables with
built functions in `features.py`, so I could access that created data in a composite function. When I first 
ran my `smoke_tests.py` for building features, the `torch.Tensor` informed me that it couldn't
directly convert a `pd.DataFrame` to a tensor, so I also now fully understand why I need to convert
`to_numpy()` first. I also found it very cool that I utilized a context manager to dump my meta python object
into a JSON object, which I knew how to store in my `constants.py` file. Lastly, I finally got familiar with pushing 
updates to a repo, and figured out that if you run `git add .` and skip directly to `git push` without `git commit`,
the changes won't be pushed. I then went in to see how I would access previous commits if I ever need to see what an 
original file looked like. 

Loader Build: Today it was very clear why I was pulling my build functions
from their various files, and I felt so good with understanding how the 
loader function was bringing all of my generated data objects into one place.
This is probably the most confident I've felt in understanding object-oriented 
programming. Especially in determining the best way to catch
validation errors when bringing `validate_schema()` over to `loaders.py `
For this Claude recommended building a class that inherits from `ValueError`
but presents my custom message, made up of the errors list from
the `validate_schema()` function. It wasn't even a second thought
why it was a good idea to create a bundle out of the return objects
for the `make_loaders()` function. This packaging made it easy to iterate
my loader objects in the smoke test, and access the dictionary with 
the same loader object. I also learned something about `git push` which is 
that if you make changes in the browser, as I did yesterday to the README
file, those changes have to be merged in the IDE with `git pull`, before
pushing other changes. I learned that best practice is to not make 
changes in the browser. Something else I intuited is that trying to hold
all the data objects in your mind at one time is not a valid way to
build a pipeline. It is easier to focus on a single piece of the pipeline
at a time, and just know where data is coming from and where it needs to 
go. It's like you mentally pass it along as you go, and enhancements can
be made once it's fully constructed and you have clarity about each piece
you've built. 

Model & Metrics: This metrics section has been the most difficult 
section to wrap my head around conceptually. 
Even though these metric concepts were touched in my DataCamp coursework,
I really had to spend extra cognitive effort to understand how these 
metrics would be computed, but not so much why, since it is 
intuitive to measure `correct/total correct`. It turned out that 
the accuracy helper function was much simpler than I expected, but
it left me wondering what the use-case for the `torchmetrics.Accuracy`
would be, if we can easily just build our own helper to use in a loop. 
I started out trying to use the `torchmetrics` functions since that 
is what I learned to compute accuracy for multiclass on DataCamp, but
it brought in complexity that wasn't necessary. When it came to smoke
testing the metrics functions, I felt like I had to take some steps
backwards to understand how `logits` and `y` are compared to compute
the accuracies. Because I didn't want to initiate the model training
I had to construct arbitrary tensors to test with and produce results
that would prove the metric function's functionality. This was good 
exercise because I came to understand how the predictions, `logits`
are checked for the `argmax` across each row, `dim=1`, and accuracy
is computed by checking `True` and `False` when the ground truth `y`
index represents the column with the max value in the logit tensor. 
I also had to clarify how `unsqueeze(1)` reshapes the `y` tensor so 
it aligns with each row of the `logit` tensor. 
I can be proud that these things actually makes sense to me. 
This also led me to dig further into how accuracy computations 
align with the error in loss functions, and how gradients 
are used to adjust the weights going forward. I now have a more intuitive
understanding of how the loss function is also comparing the `logits` and
individual ground truth batches `yb` when it comes to the training loop.