# Glacier Mass Balance Over Time

This project investigates the changes in glacier mass balance over time to better understand the impacts of climate change on glacier dynamics. By analyzing historical data, we aim to uncover trends and patterns in glacier mass balance, utilizing various regression models to forecast future changes. The findings will contribute to a greater understanding of the implications of glacial changes on ecosystems and water resources.

## Dataset Description

The dataset used in this project contains information on glacier mass balance measurements from various locations worldwide. It includes columns such as "Year" and "Mean cumulative mass balance," which represent the time frame of the data and the average mass balance of glaciers, respectively.

**DATASET URL** - https://datahub.io/core/glacier-mass-balance (Average cumulative mass balance of reference Glaciers worldwide)

## Summary of Findings

The analysis indicates a marked decline in glacier mass balance over the years, revealing a clear trend of melting and reduced ice accumulation associated with climate change. Through various regression models, the project demonstrates differing levels of predictive accuracy in relation to mass balance changes.

- **Linear Regression** showed a strong correlation with an R² value of 92.23%, effectively explaining a significant portion of the variance in the data.

- **Decision Tree** and **Random Forest** models exhibited exceptional accuracy, with R² values of 99.57% and 99.71%, respectively, underscoring their capability to capture complex patterns in the dataset with low Mean Squared Errors (MSE).

- **Support Vector Machine (SVM)**, while still performing reasonably well with an R² of 90.03%, indicated limitations compared to the other models, with a higher MSE of 7.63.

These findings underscore the urgent need for continued monitoring and research into the impacts of climate change on glacial regions, emphasizing that historical data can inform future climate models and mitigation strategies.

## Data Preprocessing

Data preprocessing involved checking for missing values and removing any incomplete records to ensure data integrity. The dataset was then examined for data types, and a statistical summary was provided to understand the distribution of values across different features.

## Exploratory Data Analysis

### Visualization

#### 1. Glacier Mass Balance Over Time

![alt text](https://i.imgur.com/jNaAeCa.png)

This line plot illustrates the trend of glacier mass balance over the years, highlighting a general decline in mass balance.
Year-over-Year Change in Glacier Mass Balance.

#### 2. Year-over-Year Change in Glacier Mass Balance

![alt text](https://i.imgur.com/ndkmbFU.png)

The bar plot reveals fluctuations in the annual mass balance of glaciers over time, showcasing both positive and negative changes. In recent years, there is a notable trend of declining mass balance, with several years exhibiting significantly negative changes, indicating accelerated glacier loss. These patterns suggest that climate factors are increasingly impacting glacier stability, leading to more pronounced annual losses. The variability in mass balance changes also reflects the influence of different climatic conditions, such as variations in temperature and precipitation. Overall, the plot underscores the urgency of addressing climate change, as the continued decline in glacier mass balance poses serious implications for ecosystems and water resources.

#### 3. Correlation Heatmap

![alt text](https://i.imgur.com/5SdNZst.png)

This heatmap reveals the relationships between different features in the dataset, showing strong correlations among certain variables that can influence mass balance. There is a strong negative correlation of -0.96 between the year and mean cumulative mass balance, indicating that as time progresses, glacier mass balance decreases. A positive correlation of 0.92 between the year and number of observations suggests that more recent years have more recorded observations. The mean cumulative mass balance positively correlates at 0.75 with annual change in mass balance, showing that years with higher balances also have greater annual changes. Conversely, a negative correlation of -0.78 between mean cumulative mass balance and the number of observations implies that higher balances correspond to fewer observations. Lastly, the annual change in mass balance negatively correlates at -0.68 with the year, indicating a trend of increasing glacier loss over time.

## Model Development

The model development process involved selecting multiple regression algorithms, including Linear Regression, Decision Tree, Random Forest, and Support Vector Machine. The dataset was split into training and testing sets to train the models and evaluate their performance effectively.

## Model Evaluation

The performance of each model used to predict the mean cumulative mass balance was assessed using Mean Squared Error (MSE) and R-squared (R²) metrics. The following evaluations detail the results obtained for each model:

### 1. Linear Regression

- **MSE**: 5.95
- **R²**: 92.23%

The Linear Regression model demonstrates a strong performance with an R² value of 92.23%, indicating that approximately 92% of the variance in the mean cumulative mass balance can be explained by the year. This model meets the performance requirement, suggesting it is suitable for predicting glacier mass balance trends over time.

### 2. Decision Tree

- **MSE**: 0.33
- **R²**: 99.57%

The Decision Tree model shows exceptional accuracy with an R² of 99.57%, meaning it explains nearly all the variability in the dataset. The low MSE of 0.33 further reinforces its predictive capability. This model also meets the performance requirement, indicating its robustness in capturing the complexities in the data.

### 3. Random Forest

- **MSE**: 0.22
- **R²**: 99.71%

The Random Forest model outperforms the others with an R² of 99.71% and an MSE of 0.22, suggesting it provides highly accurate predictions. Its ability to account for variations in the data and minimize errors makes it an excellent choice for modeling glacier mass balance trends. This model also meets the performance requirement, reflecting its high reliability.

### 4. Support Vector Machine (SVM)

- **MSE**: 7.63
- **R²**: 90.03%

The Support Vector Machine model, while still performing well with an R² of 90.03%, lags behind the others with a higher MSE of 7.63. This indicates that it explains only about 90% of the variance in the dataset. Although it meets the performance requirement, it is less effective compared to the Decision Tree and Random Forest models in this specific context.

## Conclusion

This project successfully highlights the significant decline in glacier mass balance over time, emphasizing the urgent implications of climate change on glacial regions. The analysis demonstrated the varying effectiveness of different regression models in predicting mass balance trends, with both Decision Tree and Random Forest models achieving exceptional accuracy. These findings underscore the necessity for ongoing monitoring and research into climate impacts on glaciers. Furthermore, leveraging historical data will be crucial in enhancing future climate models and strategies aimed at mitigating climate change effects.
