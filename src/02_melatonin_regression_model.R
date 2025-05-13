

# Required packages
#install.packages(c("lme4", "lmerTest", "ggplot2", "performance", "readr"),,dependencies = TRUE)
#install.packages("ggResidpanel",dependencies = TRUE)

# Load libraries-------------
library(lme4)
library(lmerTest)
library(ggplot2)
library(performance)
library(readr)
require(ggResidpanel)


# Import CSV data-----------
data <- read_csv("D:\\NextCloud\\BinIntMel\\data_main\\derivatives\\VR_paper_melatonin_results.csv")
str(data)

filtered_data <- subset(data, condition != "dark")
filtered_data$condition <- relevel(filtered_data$condition)

# Condition and participant as factors
data$participant_id <- as.factor(data$participant_id)
data$condition <- as.factor(data$condition)
data$condition <- relevel(data$condition, ref = "dark")

filtered_data$participant_id <- as.factor(filtered_data$participant_id)
filtered_data$condition <- as.factor(filtered_data$condition)

#--------------
# AUC fit

# Fit the linear mixed-effects model
mx <- lmer(post_auc_norm ~ condition * pupil_dilation  + (1 | participant_id), data = data)
summary(mx)

# 95% confidence intervals via boostrap
conf_intervals <- confint(mx, method = "boot")

# Model diagnostics

# check residuals
resid_panel(mx, plots = c("resid", "hist","qq","yvp"), qqbands=TRUE)


# extract model results
mx.fixedEff = as.data.frame(summary(mx)$coefficients)
mx.randEff = as.data.frame(ranef(mx))
mx.varCorr = as.data.frame(summary(mx)$varcor)
mx.confint = as.data.frame(confint(mx,method = "Wald"))


# export
write.table(mx.fixedEff, sep = ".", paste("LMM_final_fixedEffects.csv", sep=",")) 
write.table(mx.randEff, sep = ",", paste("LMM_final_randomEffects.csv", sep=",")) 
write.table(mx.varCorr, sep = ",", paste("LMM_final_varCorr.csv", sep=",")) 
write.table(conf_intervals, sep = ",", paste("LMM_final_confint.csv", sep=",")) 


