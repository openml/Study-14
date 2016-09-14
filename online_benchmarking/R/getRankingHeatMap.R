#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------

getRankingHeatMap = function(data, prefix = NULL) {

  data$task = rownames(data)
  rownames(data) = NULL

  df = melt(data, id.vars=ncol(data))
  colnames(df) = c("task", "algo", "Rank") 

  g = ggplot(df, aes(x = task, y = as.factor(algo), fill = Rank, colour = Rank))
  g = g + geom_tile()
  g = g + scale_fill_gradient(low = "white", high = "black", na.value = "salmon")
  g = g + scale_colour_gradient(low = "white", high = "black")
  g = g + scale_x_discrete(breaks = FALSE)
  g = g + theme(text = element_text(size = 10), axis.text.x = element_blank()) 
  g = g + xlab("Tasks") + ylab("Algorithms")

  return(g)
}

#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
