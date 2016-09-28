#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------

getExperimentsData = function(tag, numRuns = 10000) {

  if(is.null(tag)) {
    stop("You should specifiy a tag to get your OpenML runs!")
  }

  if(numRuns < 1) {
    stop("You should specifiy a positive integer number of runs.")
  }

  # getting run results from OpenML
  results = do.call("rbind", 
    lapply(0:floor(numRuns/10000), function(i) {
      return(listOMLRunEvaluations(tag = tag, limit = 10000, offset = (10000 * i) + 1))
    })
  )

  sub.datasets = dplyr::select(.data = listOMLDataSets(tag = tag), data.id, name, NumberOfInstances, 
    NumberOfFeatures, NumberOfClasses, MajorityClassSize)
  colnames(sub.datasets)[2] = "data.name"

  temp = merge(results, sub.datasets, by = "data.name")
  temp$perMajClass = temp$MajorityClassSize / temp$NumberOfInstances
  temp$flow.name = sub("\\(.*", "", temp$flow.name)
  
  return(temp)
}

#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------