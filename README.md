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
