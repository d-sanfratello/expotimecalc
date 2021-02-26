# expotimecalc
This code has been written by me to complete an assignment for the Astrophysics Observation
course at University of Pisa. The course was held during the Academic Year 2020/2021 for the
M.Sc. in Physics (Astronomy and Astrophysics).

It features an Exposure Time Calculator (hence the name), to be also used during the year for
planning of observations.

## Requirements
* `astropy 4.0.2`: This code uses this version but, since no specific function is required, any
  version of `astropy` released just before that one should be enough. See `astropy`'s website
  for their installation instructions.

## Usage
To run the program, you just need to call `python main.py` inside your terminal. A set of
arguments will determine the outcome of the software.

### File arguments
#### Without file arguments
If you call the module without any file argument, you will be asked for the coordinates of the
observatory, the time of observation and target object.

#### With file arguments
If you don't feel being annoyed by a terminal, you can always write the observatory data in two
rows of a `.txt` file, whose path (either absolute or relative) can be passed as the first
argument to `main.py`. In a second `.txt` file you will, then, insert the coordinates for the
target object, and you will pass its absolute (relative) path as the second argument of the
module call.