#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------

getExperimentsData = function(tag, n.runs = 9999) {

  if(is.null(tag)) {
    stop("You should specifiy a tag to get your OpenML runs!")
  }

  if(n.runs < 1) {
    stop("You should specifiy a positive integer number of runs.")
  }

  # getting run results from OpenML
  results = do.call("rbind", 
    lapply(0:floor(n.runs/10000), function(i) {
      return(OpenML::listOMLRunEvaluations(tag = tag, limit = 10000, offset = (10000 * i) + 1))
    })
  )

  datasets = OpenML::listOMLDataSets(tag = tag)
  sub.datasets = dplyr::select(.data = datasets, data.id, name, number.of.features, 
    number.of.classes, majority.class.size)
  colnames(sub.datasets)[2] = "data.name"

  temp = merge(results, sub.datasets, by = "data.name")
  temp$per.majority.class = temp$majority.class.size / temp$number.of.instances
  temp$flow.name = sub("\\(.*", "", temp$flow.name)
  
  return(temp)
}

#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------