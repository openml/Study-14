#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------

getPercMaxPerfLinePlot = function(matrices.list, measures.names) {
  
  if(length(matrices.list) != length(measures.names)) {
    stop("The length of the matrices list does not match with the measures names length!")
  }

  aux = lapply(matrices.list, function(mat){
    temp = scaleMatrix(mat = mat) 
    inner.tmp = do.call("rbind",lapply(1:ncol(temp), function(j) {
      ids = which(!is.na(temp[,j]))
      return(mean(temp[ids,j]))
    }))
    return(inner.tmp)
 })

  df = data.frame(do.call("cbind", aux))
  df$algo = colnames(matrices.list[[1]])
  colnames(df)[1:length(matrices.list)] = paste0("perc_max_",measures.names)

  temp = df[order(df[,1], decreasing = TRUE), ]
  temp$algo = factor(temp$algo, levels = temp$algo)
 
  df.p = melt(temp, id.vars = ncol(temp))
  colnames(df.p)[2] = "Measure"
  
  g = ggplot(data=df.p, aes(x=algo, y=value, group=Measure, colour=Measure, linetype=Measure, shape=Measure)) 
  g = g + geom_line() + geom_point() 
  g = g + guides(fill = FALSE)
  g = g + theme(text = element_text(size = 10), axis.text.x = element_text(angle = 90, vjust = .5, hjust = 1))
  g = g + scale_y_continuous(limits = c(0.4, 1))
  g = g + scale_colour_brewer(palette = "Set2")
  g = g + ylab("% of Max. Performance") + xlab("Algorithms")

  return(g)
}

#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
