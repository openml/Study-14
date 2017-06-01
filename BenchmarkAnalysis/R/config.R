#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------

# library('ggplot2')
# library('reshape2')
# library('gridExtra')  
# library('mlr')
# library('OpenML')
# library('dplyr')

# apikey = Sys.getenv('OPENMLKEY')
setOMLConfig(arff.reader = "farff") #, apikey = apikey)

AVAILABLE.MEASURES = c("f.measure", "kappa", "mean.absolute.error", "precision", "recall", 
  "usercpu.time.millis", "area.under.roc.curve", "predictive.accuracy", "root.mean.squared.error")

#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
