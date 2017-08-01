#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------

getRankingHeatMap = function(rk, prefix = NULL) {

  data = rk$rk
  data$task = rownames(data)
  rownames(data) = NULL

  df = melt(data, id.vars=ncol(data))
  colnames(df) = c("task", "algo", "Rank") 

  expression = "\\.classif|.preproc|.tuned|.model_selection|._search"
  df$algo = gsub(x = df$algo, pattern = expression, replacement = "")

  rk$rk.avg[,1] = gsub(x = rk$rk.avg[,1], pattern = expression, replacement = "")
  df$algo = factor(df$algo, levels = rk$rk.avg[order(rk$rk.avg$rk.avg),1])
  df$task = as.factor(df$task)
  
  g = ggplot(df, aes(x = as.factor(task), y = as.factor(algo), fill = Rank, colour = Rank))
  g = g + geom_tile()
  g = g + scale_fill_gradient2(low = "blue", high = "red", na.value = "gray", mid = "white",
    midpoint = nrow(rk$rk.avg)/2, limits = c(1, nrow(rk$rk.avg)))
  g = g + scale_colour_gradient2(low = "blue", high = "red", na.value = "gray", mid = "white",
    midpoint = nrow(rk$rk.avg)/2, limits = c(1, nrow(rk$rk.avg))) 
  g = g + xlab("OpenML task ids") + ylab("Algorithms")
  g = g + theme_classic()
  g = g + theme(axis.text.x = element_text(angle = 90, vjust = .5, hjust = 1, size = 5))

  return(g)
}

#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
