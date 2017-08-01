#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------

getBestLinePlot = function(mat, algo, measure = "predictive accuracy") {

  max.values  = apply(mat, 1, max, na.rm = TRUE)
  algo.values = mat[, algo]

  expression = "\\.classif|.preproc|.tuned|.model_selection|._search"
  algo.name = gsub(x = algo, pattern = expression, replacement = "")

  df = data.frame(cbind(rownames(mat), max.values, algo.values))
  rownames(df) = NULL
  colnames(df) = c("Task","Max", algo.name)
  df$Max = as.numeric(as.character(df$Max))
  df[,3] = as.numeric(as.character(df[,3]))

  # ordering tasks 
  df$Task = factor(df$Task, levels = df$Task[order(df$Max, decreasing = TRUE)])

  df.melt = melt(df, id.vars = 1)
  g = ggplot(df.melt, mapping = aes(x = Task, y = value, colour = variable, 
    linetype = variable, shape = variable, group = variable))
  g = g + geom_line()+ geom_point() + theme_bw()
  g = g + theme(axis.text.x = element_text(angle = 90, vjust = .5, hjust = 1, size = 5))
  g = g + ylab(measure) + xlab("OpenML task ids")
  g = g + labs(linetype = "Algorithm") + labs(shape = "Algorithm") + 
    labs(colour = "Algorithm")
  g = g + scale_y_continuous(limits = c(0, 1), breaks = c(0, 0.25, 0.5, 0.75, 1))
  return(g)

}

#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
