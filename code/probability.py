
from scipy.stats import t
import math
import numpy as np

##########################   Confidence Intervals   ##########################

#Confidence Interval for Means ($n \ge 30$)
# $$(\bar{x} \pm Z \frac{\sigma}{\sqrt{n}})$$

def confidence_interval_for(samples=[], confidence=0.95):
  sample_size = len(samples)
  degrees_freedom = sample_size - 1
  outlier_tails = (1.0 - confidence) / 2.0
  t_distribution_number = -1 * t.ppf(outlier_tails, degrees_freedom)

  step_1 = np.std(samples)/math.sqrt(sample_size)
  step_2 = step_1 * t_distribution_number

  low_end = np.mean(samples) - step_2
  high_end = np.mean(samples) + step_2

  return low_end, high_end
  
#Confidence Interval for Means ($n < 30$)
#$$(\bar{x} \pm t \frac{s}{\sqrt{n}})$$
#Confidence Interval for Proportions
#$$(\hat{p} \pm Z \sqrt{\frac{\hat{p} - (1 - \hat{p})}{n}})$$
