#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------

getRankFrequencyPlot = function(rk, data, k = 5, version = "percentage"){

  data.runtime = getAvgRuntimeData(data = data)
  all.learners = unique(data$flow.name)
  mat = rk$rk

  # k-best algorithms
  aux = lapply(1:nrow(mat), function(i){
    algos.ids = which(!is.na(mat[i,]))
    temp = mat[i, algos.ids]
    ret = sort(temp, decreasing = FALSE)

    if( length(algos.ids) == 1) {
      alg.names = colnames(mat)[algos.ids]
    } else{
      alg.names = names(ret)
    }

    obj = c(alg.names, rep(x=NA, times=(length(all.learners) - length(temp))))
    return(obj)
  })
  aux = data.frame(do.call("rbind", aux))
  
  #list of k-best
  temp = lapply(1:k, function(i){
    return(table(factor(aux[,i], levels = all.learners)))
  })

  # df with the best learners
  rk.df = data.frame(do.call("cbind", temp))
  colnames(rk.df) = c(1:ncol(rk.df))
  rk.df$alg = rownames(rk.df)
  rownames(rk.df) = NULL

  # Adding runtime on the data frame
  rk.df$runtime = 0
  for(i in 1:nrow(rk.df)){
    id = which(data.runtime$alg == rk.df$alg[i])
    if(length(id) != 0) {
      value = data.runtime$usercpu.time.millis[id]
      rk.df$runtime[i] = value
    }
  }

  df = melt(rk.df, id.vars = c(ncol(rk.df) - 1, ncol(rk.df)))
  colnames(df) = c("learner", "runtime", "rank", "value")
  df = df[which(df$value != 0),]

  if(version == "percentage") {
    df$value = df$value / length(unique(data$task.id))
  }
  
  g = ggplot(data = df, aes(x=learner, y=value, fill=runtime)) 
  g = g + geom_bar(position="dodge",stat="identity") + guides(fill=FALSE)
  g = g + scale_y_continuous(limits = c(0, max(df$value))) + facet_grid(rank ~ .)
  g = g + theme(text = element_text(size=10), axis.text.x = element_text(angle=90, vjust=.5, hjust=1)) 
  g = g + xlab("Algorithms")
  g = g + scale_fill_gradient(high="red", low="grey40")

  if(version == "percentage") {
    g = g + ylab("% of Occurences per Rank Position") 
  } else {
    g = g + ylab("Occurences per Rank Position") 
  }

  return(g)
}


#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
