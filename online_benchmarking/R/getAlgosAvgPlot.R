#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------

handleData = function(data, measures.list) {

  aux = lapply(measures.list, function(meas){
    checkMeasure(measure = meas)
  })
  meas.names = do.call("c", measures.list)
 
  # Average performance by learner
  perf = lapply(measures.list, function(meas){
    return(getAvgPerformance(data = data, measure = meas))
  })
  perf = Reduce(function(...) merge(..., all=T), perf)
  new.names = gsub(meas.names, pattern = "\\.", replacement = "_")
  colnames(perf) = c("algo", new.names)
  perf$chart = "avg performance"

  # Computing the Average Ranking
  rk.list = lapply(measures.list, function(meas){
    mat = getPerfMatrix(data = data, measure = meas)
    rk  = getRanking(mat = mat)$rk.avg 
    colnames(rk) = c("algo", gsub(meas, pattern = "\\.", replacement = "_"))
    return(rk)
  })
  rks = Reduce(function(...) merge(..., all=T), rk.list)
  rks$chart = "avg ranking"
  rks$algo = as.factor(rks$algo)

  rks = rks[order(rks[,2], decreasing = FALSE), ]
  perf$algo = factor(perf$algo, levels = rks$algo)
 
  covr = getAlgoCoverage(data = data)
  df = melt(rbind(perf, rks), id.vars = c(1, ncol(perf)))
  colnames(df) = c("algo", "chart", "measure", "value") 
  df = merge(df, covr, by = "algo")
  return(df)

}


#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------


getAlgosAvgLinePlot = function(data, measures.list) {

  df = handleData(data = data, measures.list = measures.list)
  
  g = ggplot(data = df, aes(x = algo, y = value, group = measure, 
    colour = measure, linetype = measure))
  g = g + geom_line() + geom_point(aes(size = coverage)) 
  g = g + scale_size(range = c(0.3, 5))
  g = g + facet_grid(chart ~ ., scales="free")

  g = g + theme(text = element_text(size=10),
   axis.text.x = element_text(angle=90, vjust=1, hjust=1))
  g = g + ylab("Average value") + xlab("Algorithms")
  g = g + scale_colour_brewer(palette="Dark2")
  g

  return(g)
}

#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------

getAlgosAvgBarPlot = function(data, measures.list) {

  df = handleData(data = data, measures.list = measures.list)
  
  g = ggplot(data = df, aes(x = algo, y = value, fill = measure,
    alpha = coverage))
  g = g + facet_grid(chart ~ measure, scales="free")
  g = g + geom_bar(stat='identity', position="dodge")
  g = g + scale_alpha(range = c(0.3, 1))
  g = g + scale_fill_brewer(palette="Dark2")
  g = g + theme(text = element_text(size=10),
   axis.text.x = element_text(angle=90, vjust=1, hjust=1))
  g = g + ylab("Average value") + xlab("Algorithms")

  return(g)

}

#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
