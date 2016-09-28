#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------

getAvgRuntimeData = function(data) {

  temp = dplyr::select(.data = data, task.id, flow.name, usercpu.time.millis.training, 
    usercpu.time.millis.testing, usercpu.time.millis)

  algos = unique(temp$flow.name)
  aux = lapply(algos, function(alg) {
    # TO DO: how to handle missing data here?
    d = na.omit(temp[which(temp$flow.name == alg),])
    return(colMeans(d[,3:ncol(d)]))
  })

  temp = data.frame(do.call("rbind", aux))
  temp$alg = algos
  return(temp)
}


#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
