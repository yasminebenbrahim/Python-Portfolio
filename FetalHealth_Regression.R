library(stargazer)

# load the dataset
health = readxl::read_excel("fetal_health.xlsx")

# shape of the data
dim(health)

# automatically attach the dataset to future functions that i will run
attach(health)

# names of columns in dataset
names(health)

# summary of the variables
summary(health)
stargazer(health, title = "Summary Statistics", style = 'asq' , type = 'html', out = 'summarystats.html')

# Looking at potential outliers
par(mfrow=c(1,1))
plot(severe_decelerations, main="Scatterplot of Severe Decelerations")

# Removing outliers
health = health[!(health$severe_decelerations > 0.0008 ),]

# New shape of the data after removing the outliers
dim(health)
summary(health)

# Multi Linear Regression Model
lm.fits = lm(fetal_health~.,data = health)
summary(lm.fits)
formula(lm.fits)
par(mfrow = c(2,2))
plot(lm.fit)

# Stepwise Backward Selection

# we find that the p value for age is the largest one, we use backward selection method to delete one variable with the largest p value
lm.fit1 = lm(fetal_health~.-fetal_movement,data=health)
summary(lm.fit1)
# Perform backward selection once more for the one variable left with a large p-value
lm.fit2=update(lm.fit1,~.-severe_decelerations, data=health)
summary(lm.fit2)

# Now that all the variables are statistically significant, we can make some assumptions on the model
par(mfrow = c(2,2))
plot(lm.fit2)

stargazer(lm.fits, title = "Multiple Linear Regression Results", type = "html", out = 'regressionresults.html')
stargazer(lm.fit2, title = "Multiple Linear Regression After Backwards Selection", type = "html", out = 'backwardsselection.html')

# Findings: The Multiple R^2 score wasn't lowered by much after performing backward selection
