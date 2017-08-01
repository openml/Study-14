#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------

getExperimentsData = function(tag, n.runs = 10000) {

  if(is.null(tag)) {
    stop("You should specifiy a tag to get your OpenML runs!")
  }

  if(n.runs < 1) {
    stop("You should specifiy a positive integer number of runs.")
  }

  results = OpenML::listOMLRunEvaluations(tag = tag, limit = 10000)
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