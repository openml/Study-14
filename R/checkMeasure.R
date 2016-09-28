#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------

checkMeasure = function(measure){

  allowed.measures = c("f.measure", "kappa", "mean.absolute.error", "precision", "recall", 
    "usercpu.time.millis", "area.under.roc.curve", "predictive.accuracy", "root.mean.squared.error")
  if (!( measure %in% allowed.measures)) {
    stop(paste0(" - Please, choose one of the following measures: ", 
      paste(allowed.measures, collapse=', '), " \n"))  
  } else {
    return(TRUE)
  }
}

#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
