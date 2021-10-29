Research Project: Forecasting and Trading the Market Open of Highly Volatile U.S. Listed Stocks
---

The project is divided into the following steps:
1. Building and researching predictor variables (including fundamental variables, "technical analysis" variables, and novel variables built from the prior overnight trading session).
2. Researching the price percent change in each stock between [9:30, 10:00]. After performing extensive EDA, target variables for the predictive modelling process are chosen.
3. Developing a predictive model which incorporates both steps (1) and (2).
4. Developing a simple trading strategy which builds upon prior steps.

---

Code Notes:
* The underlying code for this project is located in ```wqu_capstone_research_shared.py```
* ```main.ipynb``` contains the main research of this project.
* Library dependencies ```stock_data_dealer``` and ```trading_research_core``` are used for loading data. They are proprietary to this author and not provided. Pickled versions of the data are provided, so the research in ```main.ipynb``` can be carried out without requiring reference to the proprietary libraries. However, the ```import``` statements in ```wqu_capstone_research_shared.py``` will prevent the code from running without issue and must be dealt with first.