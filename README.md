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

### Read the Description First
We must understand every number/text that is included in our dataset. I mean that is obvious, right? We must know what we're dealing with, the domain that our dataset is in, and even figure out the problems we're trying to solve.

It's better to take some time pause and ponder about the data, reading about the metrics that the collector specified when reporting these data (this is important, because people can come up with creative ways for reporting stuff). And my tip here is, try to tell some stories with the data yourself! Remember that we're **exploring** it, so why don't we have some fun, right?

Let's look our data to understand what I mean:

First, the dataset is about (i  need to do some research about this)

Second, we need to look at the metrics used when reporting our data. And actually, if you sneak-peak our data, you can see that everything is in numerical format, eventhough there are some categorical attributes (like `SEX` or `MARRIAGE`). Why's that? Well, when we scrutinize the description, we can see that the collector actually _encode_ those categorical attributes for us (God bless him!). That's of great convenience because we **have to** encode the text/categorical attributes to feed in our model later in the course (remember: the model knows nothing but numbers). And in the `PAY_0` or other similar categories, we might see values ranging from -1 to 2 or even more, those are actually representing the **grades** that the bank gave for each customer considering his or her _behaviour_ in the month. What _behaviour_ means here is that: Did the customer pay the bank enough in that month? Then we must look at the `BILL_AMT` and `PAY_AMT` corresponding to that month, which show the _money owe_ and the _money paid back_ respectively.

I know you're still getting confused with these metrics. Don't worry too much, mate, we're crawling into the world of finance, and I gotta tell you, finance is really complicated. Now let's take my third tip in the bucket list, which is **telling some stories about the data**. You might imagine yourself as a brilliant detective who is having some clients coming to his house (try!).

Let's look at one of our clients:
Well, my dear Watson, I suggest that this single, 24-year-old,
graduated woman is having some troubles with the bank, as the `defaul.payment.next.month` is `1.0`. Specifically, she got out of control during the last months. Why is that? Oh well, look at what she spent and what she paid back for the bank during the first three months (`BILL_AMT 6 to 4`, `PAY_AMT 6 to 4`, we gotta trace back in time, check the description). Nothing, right? She wasn't using the credit card at all, she owed nothing and bought nothing. That inactive state gave her a point `-2` in the `PAY_6 to 4`. Now let's look at the recent months, you might notice that things got spiralled out quickly. She had a bill of `689` in July (`BILL_AMT3`), but paid in full in August (`PAY_AMT2`), so she did a good job at that. Then in the later months, she owed a lot of money without paying back (`BILL` > 3000 while `PAY_AMT` equal 0), thus received bad grades from the bank (`PAY` is `2.0`). Eventually, despite her high credit at the first place (`LIMIT_BAL` up to `20000`), she got defaulted by the bank. Case end.

Elementary, my dear Watson.

And that's it. Another thing to mention is that we have the `PAY_4` equals to `-1.0`, not `-2.0` as we deduced. This represents an incosistency in the dataset itself, maybe some errors when reporting the data. So we must be careful with that.

Now we have some intial explorations of our data, let's jump right in the coding part.

### **Cleaning up the Data**
The data is not always perfect. There are mistakes along the way, usually from the part of collecting data, and that could result in something I would call "_dirty dataset_". 

These "dirty dataset" can have problems ranging from missing values, outliers, duplicates, or even some incosistency when reporting data. Those things can really lead to problems when training our model.

So the rule of thumb here, is to clean up the data right in the first place, so that we can deal with a fresh-and-clean dataset afterwards.
1. ***Take the first look at our data/ Check the type***


There are some methods in `pandas` library that help us 
to see some cool things about our data:
- `head()`: shows the top five rows 
- `shape`: show the shape of the dataframe, so as you might guess, it shows the number of instances (rows) and the number of attributes (columns)
- `columns`: shows all the columns (which are also the
attributes), if you want to see the number of attributes then you can apply the `len()` function
- `info()`: show a quick description of the data, like the total number of rows, the **type** of each attribute (which is important), and the **number of non-null values** (this is also important because in the next step we will learn how to handle missing values)
- `describe()`: give the summary of only the numerical attributes (mean, min, max, std...). I don't really see the importance of this, but it looks cool so make sure to include that.
- Also, in `matplotlib` library, we can plot out the histogram (use `hist()`) for the whole dataset (each attribute at a time) to have the first visualization of our data.

> ***Friendly Reminder***: Remember to checkout every functions that I gave throughout this note, you may need to read the documentation to figure out how to use them. Honestly, even professionals still have to check how to use those functions (I really think so), so be comfortable with that. The main idea is that you have a good understanding of each function and when to use it, so that when you ask **questions** (which is an advanced technique), you know which functions should be used. Understand it first, and then you can easily search out the syntax.

Now we really need to **check our datatype**, using `dtypes` or `info()`, and here are some datatypes that I encountered:
- `int64`/ `float64`: These are for numerical attributes, but make sure to clear out `int` and `float`

> For instance, in my dataset, the `LIMI_BAL` or the `BILL_AMT` and `PAY_AMT` are all `float65`, understandable because those are amounts of money (in NT dollars). There might be questions over some other categories like `SEX` or `EDUCATION`, those cannot be integers. But remember that those integers are just representing some convenient ways of encoding data, so it's fine.

- `object`: These things can be anything, and we must be really careful. The most common thing to notice here is when we have 
some data represent **time**. Those are actually of `datetime` datatype or the **time-series** feature, but oftentimes be misclassified as `object`.

One convenient way to cast our data to its correct type is using the function `astype()` in pandas, or you can use the function `to_datetime()` if you want to specifically convert the data to a `datetime` object.
> In my example I don't have to worry about those things, but I think in the future when I'm dealing with another dataset, that could be useful.

2. *** Handling Missing Values***
- Find/count/count the percentage of missing values
- Handle: 3 ways -> drop, or impute

3. *** Check duplicates ***
4. *** Detect outliers ***
5. *** Handling Text/Categorical Attributes ***
6. *** Feature Scaling/Normalization ***
 

To-do: Check datatype