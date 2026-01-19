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