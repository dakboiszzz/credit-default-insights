# credit-default-insights
My project for ML and Feature Engineering
# Agenda
## Some words from the newbie (me)
When doing data stuff, we really need a clear to-do list, step by step. 
At first I didn't thought of this and I'm just kinda chilling out with a bunch of stuff,
then when I put my hands into these kinds of projects, I didn't even know where to start.
It took me hours to binge-watch some Youtube videos, read some books (I highly recommend 
Hands-on Machine Learning with Scikit-Learn, Keras and Tensorflow), and asking Gemini to 
sketch out some of the things in this data-exploration agenda.

It's my first step in this data journey, so the agenda here can be silly sometimes.
But well, everyone starts somewhere, so if you are not comfortable with this agenda,
then you can give some friendly comments, I will gracefully take all of those.

Also, I will leave insightful notes along the way for exploring this dataset, and some tips and tricks.
If you want a more detailed analysis, please checkout the `notebooks` folder and find my notebook there.

## The agenda

### **Cleaning up the Data**
The data is not always perfect. There are mistakes along the way, usually from the part of collecting data, and that could result in something I would call "_dirty dataset_". 

These "dirty dataset" can have problems ranging from missing values, outliers, duplicates, or even some incosistency when reporting data. Those things can really lead to problems when training our model.

So the rule of thumb here, is to clean up the data right in the first place, so that we can deal with a fresh-and-clean dataset afterwards.
1. ***Take the first look at our data/ Check the type***


There are some methods in `pandas` library that help us 
to see some cool things about our data:
- `head()`: shows the top five rows 
- `columns`: shows all the columns (which are also the
attributes), if you want to see the number of attributes then you can apply the `len()` function
- `info()`: show a quick description of the data, like the total number of rows, the **type** of each attribute (which is important), and the **number of non-null values** (this is also important because in the next step we will learn how to handle missing values)
- `describe()`: give the summary of only the numerical attributes (mean, min, max, std...). I don't really see the importance of this, but it looks cool so make sure to include that.
- Also, in `matplotlib` library, we can plot out the histogram (use `hist()`) for the whole dataset (each attribute at a time) to have the first visualization of our data.

> ***Friendly Reminder***: Remember to checkout every functions that I gave throughout this note, you may need to read the documentation to figure out how to use them. Honestly, even professionals still have to check how to use those functions (I really think so), so be comfortable with that. The main idea is that you have a good understanding of each function and when to use it, so that when you ask **questions** (which is an advanced technique), you know which functions should be used. Understand it first, and then you can easily search out the syntax.

To-do: Check datatype