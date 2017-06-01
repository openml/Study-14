#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------

getAvgPerformance = function(data, measure) {

  temp = na.omit(data[, c("flow.name", measure)])
  algos = unique(temp$flow.name)

  aux = lapply(algos, function(alg) {
    d = temp[which(temp$flow.name == alg),]
    ret = mean(d[,2])
    return(ret)
  })

  temp = data.frame(do.call("rbind", aux))
  temp$alg = algos
  temp = temp[, c(2,1)]
  colnames(temp) = c("algo", measure)
  return(temp)
}

#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
