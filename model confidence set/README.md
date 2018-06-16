Here are the MATLAB scripts i developped for predicting a model confidence set (henceforth, MCS) of superior volatility models. Special thanks to Dr. Ilze Kalnina (University of Montreal), for supervising the work.

## Project background
I wanted to predict a MCS of 10% size, 500 moving block bootstrap replications and 2 blocks, to identify a set of superior models (both statistically and economically) among 9 of the most predictive and recent volatility forecasting models (2014). I was mostly interested to test the results of Barndorf-Niesen and al. (2008) and Patton and Sheppard (2013) at predicting the daily component of volatility from various underlying components.

## Method
The motivation to use the MCS approach is the lack of power of traditional approaches and the limitations of pairwise comparison when assessing the best model. The MCS solves use instead a systemicand more robust approach thanks to boostraping. The procedure is iterative, each step eliminating inferior models until a set of superior models -and relatively close, is accepted. I tested three different loss-functions : the mean absolute error, the negative of gaussian quasi likelihood, and the expected trading loss from an option.
For more details on my setup of the procedure, see file ‘model-confidence-set-volatility-forecast-comparison-2014_FR’ or report to the original model confidence set work Hansen, Lunde, and Nason (2011)

## Data 
I worked 5-minutes intraday sp500 index data, from 04/01/1993 to 10/26/2012, with trading hours from 09h30 AM to 04h00 PM. After data exploration and cleaning, i worked over 380000 data points covering more than 4800 trading days.

## Results
My results identify Barndorf-Niesen and al. (2008) model as top performer within my initial set. With regards to Patton and Sheppard (2013) model, only the negative component of Jump variation is deemed significant for one day-ahead forecasting. Also, i saw that the classic leverage effect parameter does not significantly increase the predictive power of these models. 

In term of economic gain, Corsi (2009) model performs better, hence reflecting better the theory of market segmentation. The fact that models accounting for jumps in volatility performed less is also intuitive with respect to investors losses due to negative events affecting volatility.

## Guide through the files
* bstrap.m : Moving block bootstraping function
* loss_comput.m : Loss function computing
* loss_comput_b.m : Loss function computing on boostrap samples
* main.m : Main function
* mcs_iter.m : Iteration computing
* mcs.m : Second main function

The procedure is full replicable. First place all mfiles in a Matlab folder, then, Run 'mcs.m', wich prompt you to select the index datafile and wait for results display. Thanks for your interest.
