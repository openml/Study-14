#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------

getMatrixBoxPlot = function(mat, data, landscape = FALSE, prefix = NULL) {

  # TODO: show tasks order tasks according to the type ?

  df = na.omit(melt(mat))
  colnames(df)[1] = "algo"
 
  g = ggplot(data = df, mapping = aes(x = as.factor(algo), y = value))
  g = g + stat_boxplot(geom ='errorbar')
  g = g + geom_boxplot(outlier.colour = "black", outlier.size = 0.5)
  g = g + theme(text = element_text(size = 10), 
    axis.text.x = element_text(angle = 90, vjust = .5, hjust = 1))
  g = g + ylab(paste("% of the maximum", prefix)) + xlab("Algorithms")
  g = g + scale_y_continuous(limits = c(0, 1))
  
  if(landscape) {
    g = g + coord_flip()
  }
  return(g)

}

#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------

getMatrixViolinPlot = function(mat, landscape = FALSE, prefix = NULL) {

  df = na.omit(melt(mat))
  colnames(df)[1] = "algo"
 
  g = ggplot(data = df, mapping = aes(x = as.factor(algo), y = value, fill = algo))
  g = g + geom_violin(trim = TRUE, scale = "width")
  g = g + geom_boxplot(outlier.colour = "black", outlier.size = 0.5, width = 0.1, fill = "white")
  g = g + scale_y_continuous(limits = c(0, 1))
  g = g + theme(text = element_text(size = 10), 
    axis.text.x = element_text(angle = 90, vjust = .5, hjust = 1))
  g = g + theme_bw() + theme(legend.position="none")
  g = g + ylab(paste("% of the maximum", prefix)) + xlab("Algorithms")

  if(landscape) {
    g = g + coord_flip()
  }
  return(g)
}

#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------

getMatrixHeatMap = function(mat, landscape = FALSE, prefix = NULL) {

  mat$task = paste0("OML_Task_",rownames(mat))
  # mat[is.na(mat)] = 0

  df = melt(mat, id.vars = ncol(mat))
  colnames(df) = c("task", "algo", "percentage")

  g = ggplot(df, aes(x = task, y = as.factor(algo), fill=percentage, colour=percentage))
  g = g + geom_tile() 
  g = g + scale_fill_gradient(low = "white", high = "black", na.value = "salmon")
  g = g + scale_colour_gradient(low = "white", high = "black", na.value = "salmon")
  g = g + scale_x_discrete(breaks = FALSE)
  g = g + theme(text = element_text(size = 10), axis.text.x = element_blank()) 
  g = g + xlab("Tasks") + ylab("Algorithms")
  g = g + ggtitle(paste("Percentage of the maximum", prefix, "over all the tasks"))

  return(g)
}

#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------